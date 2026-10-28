"""Command line tool that turns the challenge readings into ``predictions.csv``.

The forecast is built from the training rows only, and the file is written before the holdout
values are looked at. The holdout is then used once, to report how the three models and the
count forecaster did; it never changes the forecast. Run it with
``python -m cwfc.predict --data challenge_data --output predictions.csv``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.analysis import elapsed_hours, save_figure, save_table, set_plot_style
from cwfc.data import ACTIVE_VM_COUNT, AGGREGATE_CPU, CPU_MAX, CPU_MIN, HOLDOUT_HORIZON, build_targets, load_readings, split_holdout
from cwfc.evaluate import metrics_table, run_cv
from cwfc.features import check_history, check_horizon, future_timestamps
from cwfc.metrics import mae, normalized_mae, rmse
from cwfc.models import ARIMA_NAME, BASELINE_NAME, FOREST_NAME, baseline_forecast, candidate_forecasters, forest_forecast, history_mean, persistence

DEFAULT_OUTPUT = "predictions.csv"
OUTPUT_COLUMNS = ("timestamp", ACTIVE_VM_COUNT, AGGREGATE_CPU)
PERSISTENCE_NAME = "Persistence"
COUNT_FORECASTERS = {PERSISTENCE_NAME: persistence, "History mean": history_mean, BASELINE_NAME: baseline_forecast}
HOLDOUT_TABLE_NAME = "holdout_metrics"
HOLDOUT_FIGURE_NAME = "holdout_forecast"
TAIL_STEPS = 36  # training points shown before the holdout in the figure (3 hours)


def choose_count_forecaster(counts: pd.Series, horizon: int = HOLDOUT_HORIZON) -> str:
    """Return the name of the count forecaster with the lowest rolling-origin MAE, persistence on ties.

    A history not longer than the horizon cannot be cross-validated, so persistence is used.
    """
    if len(counts) <= horizon:
        return PERSISTENCE_NAME
    table = metrics_table(run_cv(counts.astype(float), COUNT_FORECASTERS, horizon=horizon), reference=PERSISTENCE_NAME)
    return str(table["MAE"].idxmin())


def forecast_cpu(history: pd.Series, horizon: int) -> tuple[np.ndarray, bool]:
    """Forecast the CPU series with ARIMA, clipped to the valid range, and say whether ARIMA fell back to the baseline."""
    arima = candidate_forecasters()[ARIMA_NAME](history, horizon)
    fell_back = bool(np.array_equal(arima, baseline_forecast(history, horizon)))
    return np.clip(arima, CPU_MIN, CPU_MAX), fell_back


def forecast_count(history: pd.Series, horizon: int, name: str) -> np.ndarray:
    """Forecast the VM count with the named forecaster, rounded to whole VMs and never below zero."""
    raw = COUNT_FORECASTERS[name](history.astype(float), horizon)
    return np.clip(np.rint(raw), 0, None).astype("int64")


def build_predictions(train: pd.DataFrame, count_name: str, horizon: int = HOLDOUT_HORIZON) -> pd.DataFrame:
    """Build the forecast table for the ``horizon`` timestamps after the training rows, from the training rows only.

    Raises ``ValueError`` if the training CPU series has missing values (a timestamp where every
    reading was missing or out of range) or is not on the five-minute grid. Warns if ARIMA fell
    back to the baseline, so the submitted CPU forecast is never silently the baseline.
    """
    check_horizon(horizon)
    cpu_history = train[AGGREGATE_CPU]
    missing = cpu_history.index[cpu_history.isna().to_numpy()]
    if len(missing):
        raise ValueError(f"{AGGREGATE_CPU} is missing at {len(missing)} training timestamps (first: {missing[0]}); every reading there was missing or out of range")
    check_history(cpu_history)
    cpu, fell_back = forecast_cpu(cpu_history, horizon)
    if fell_back:
        warnings.warn("ARIMA fell back to the baseline (short history or the fit failed); the CPU forecast is the baseline", RuntimeWarning, stacklevel=2)
    return pd.DataFrame(
        {
            "timestamp": future_timestamps(train.index[-1], horizon),
            ACTIVE_VM_COUNT: forecast_count(train[ACTIVE_VM_COUNT], horizon, count_name),
            AGGREGATE_CPU: cpu,
        }
    )


def validate_predictions(predictions: pd.DataFrame, holdout_timestamps) -> None:
    """Raise ``ValueError`` listing every way the table breaks the submission format.

    It must have exactly the output columns, one row per holdout timestamp in the same order
    (so no duplicates, as the holdout timestamps are unique), finite values, a non-negative count and a CPU value in [0, 100].
    """
    problems = []
    expected = np.asarray(holdout_timestamps)
    if tuple(predictions.columns) != OUTPUT_COLUMNS:
        problems.append(f"columns must be {OUTPUT_COLUMNS}, got {tuple(predictions.columns)}")
    elif len(predictions) != len(expected):
        problems.append(f"expected {len(expected)} rows, got {len(predictions)}")
    else:
        if not np.array_equal(predictions["timestamp"].to_numpy(), expected):
            problems.append("timestamps differ from the holdout timestamps")
        values = predictions[[ACTIVE_VM_COUNT, AGGREGATE_CPU]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            problems.append("values must be finite")
        else:
            if (predictions[ACTIVE_VM_COUNT] < 0).any():
                problems.append(f"{ACTIVE_VM_COUNT} must not be negative")
            if not predictions[AGGREGATE_CPU].between(CPU_MIN, CPU_MAX).all():
                problems.append(f"{AGGREGATE_CPU} must be in [{CPU_MIN}, {CPU_MAX}]")
    if problems:
        raise ValueError("invalid predictions: " + "; ".join(problems))


def write_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write the table as CSV without an index column, creating the folder if needed, and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(target, index=False)
    return target


def comparison_forecasts(train_cpu: pd.Series, predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the CPU forecasts of the baseline, ARIMA (as submitted), the forest and persistence, from the training rows only."""
    horizon = len(predictions)
    return {
        BASELINE_NAME: baseline_forecast(train_cpu, horizon),
        ARIMA_NAME: predictions[AGGREGATE_CPU].to_numpy(dtype=float),
        FOREST_NAME: forest_forecast(train_cpu, horizon),
        PERSISTENCE_NAME: persistence(train_cpu, horizon),
    }


def holdout_table(holdout: pd.DataFrame, predictions: pd.DataFrame, comparison: dict[str, np.ndarray], count_name: str) -> pd.DataFrame:
    """Score the forecasts on the holdout (for reporting only): MAE and RMSE, and for the CPU the MAE ratio to the baseline.

    Holdout CPU values that are missing are left out of the scores, which are NaN if none is left.
    """
    actual = holdout[AGGREGATE_CPU].to_numpy(dtype=float)
    seen = np.isfinite(actual)
    rows = [
        {
            "series": AGGREGATE_CPU,
            "model": name,
            "MAE": mae(actual[seen], forecast[seen]) if seen.any() else np.nan,
            "RMSE": rmse(actual[seen], forecast[seen]) if seen.any() else np.nan,
            "MAE ratio": normalized_mae(actual[seen], forecast[seen], comparison[BASELINE_NAME][seen]) if seen.any() else np.nan,
        }
        for name, forecast in comparison.items()
    ]
    counts = holdout[ACTIVE_VM_COUNT].to_numpy(dtype=float)
    forecast = predictions[ACTIVE_VM_COUNT].to_numpy(dtype=float)
    rows.append({"series": ACTIVE_VM_COUNT, "model": count_name, "MAE": mae(counts, forecast), "RMSE": rmse(counts, forecast), "MAE ratio": np.nan})
    return pd.DataFrame(rows).set_index(["series", "model"])


def plot_holdout(train: pd.DataFrame, holdout: pd.DataFrame, comparison: dict[str, np.ndarray]):
    """Plot the end of the training CPU series, the holdout values and the forecasts, and return the figure."""
    stamps = np.concatenate([train.index.to_numpy(), holdout.index.to_numpy()])
    hours = elapsed_hours(stamps)
    tail = slice(len(train) - TAIL_STEPS, len(train))
    future = slice(len(train), len(stamps))
    fig, ax = plt.subplots()
    ax.plot(hours[tail], train[AGGREGATE_CPU].to_numpy()[-TAIL_STEPS:], color="grey", label="Training")
    ax.plot(hours[future], holdout[AGGREGATE_CPU].to_numpy(), color="black", linewidth=2, label="Holdout (actual)")
    for name, forecast in comparison.items():
        ax.plot(hours[future], forecast, label=name)
    ax.axvline(hours[len(train)], color="grey", linestyle=":")
    ax.set_xlabel("Hours since the start of the trace")
    ax.set_ylabel("Aggregate CPU utilization (%)")
    ax.set_title("Holdout forecast against the observed values")
    ax.legend(fontsize=8)
    return fig


def main(argv: list[str] | None = None) -> None:
    """Read the data, write ``predictions.csv`` from the training rows, then score the holdout once for reporting."""
    parser = argparse.ArgumentParser(description="Forecast the final two hours of active_vm_count and aggregate_cpu_utilization.")
    parser.add_argument("--data", default=None, help="folder with the challenge data (default: challenge_data)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="where to write the predictions (default: predictions.csv)")
    parser.add_argument("--save-report", action="store_true", help="also save the holdout table and figure under reports/")
    args = parser.parse_args(argv)
    try:
        train, holdout = split_holdout(build_targets(load_readings(args.data)))
        count_name = choose_count_forecaster(train[ACTIVE_VM_COUNT])
        predictions = build_predictions(train, count_name)
        validate_predictions(predictions, holdout.index.to_numpy())
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    try:
        path = write_predictions(predictions, args.output)
    except OSError as error:
        parser.exit(1, f"error: cannot write {args.output}: {error}\n")
    print(f"wrote {len(predictions)} rows to {path} (CPU: {ARIMA_NAME}, VM count: {count_name})")
    comparison = comparison_forecasts(train[AGGREGATE_CPU], predictions)
    table = holdout_table(holdout, predictions, comparison, count_name)
    print("Holdout scores (reporting only, the forecast above is not changed by them):")
    print(table.round(4).to_string())
    if args.save_report:
        set_plot_style()
        saved = [save_table(table, HOLDOUT_TABLE_NAME), save_figure(plot_holdout(train, holdout, comparison), HOLDOUT_FIGURE_NAME)]
        print("saved:", ", ".join(Path(p).name for p in saved))


if __name__ == "__main__":
    main()
