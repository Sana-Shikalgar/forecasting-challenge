"""Rolling-origin evaluation of forecasters on the training window.

At each origin a forecaster sees only the observations before it and forecasts the next
``horizon`` steps directly. The origins start after ``MIN_TRAIN_POINTS`` observations and move
one step at a time, so neighbouring origins overlap and are not independent. All thresholds
(for the high-load periods) come from each origin's own history.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cwfc.analysis import save_figure, save_table, set_plot_style
from cwfc.data import AGGREGATE_CPU, HOLDOUT_HORIZON, build_targets, load_readings, split_holdout
from cwfc.features import check_history, check_horizon
from cwfc.metrics import mae, normalized_mae, rmse
from cwfc.models import BASELINE_NAME, FOREST_PARAMS, Forecaster, ForestParams, candidate_forecasters, forest_fit_error, reference_forecasters
from cwfc.validation import origin_positions

HIGH_LOAD_QUANTILE = 0.8
MIN_TRAIN_POINTS = 48
METRICS_TABLE_NAME = "cv_metrics"
HORIZON_TABLE_NAME = "cv_horizon_mae"
HORIZON_RMSE_TABLE_NAME = "cv_horizon_rmse"
HORIZON_FIGURE_NAME = "cv_horizon_mae"


@dataclass(frozen=True)
class CVResult:
    """Out-of-sample forecasts of every forecaster at every rolling origin.

    ``origins`` holds the number of history points at each origin, ``timestamps`` the raw
    timestamp of the first forecast step, and ``actual`` and each array in ``predictions``
    have one row per origin and one column per forecast step. ``high_load`` marks the actual
    values at or above the 80th percentile of that origin's own history.
    """

    origins: np.ndarray
    timestamps: np.ndarray
    actual: np.ndarray
    predictions: dict[str, np.ndarray]
    high_load: np.ndarray


def default_min_train(n_obs: int, horizon: int) -> int:
    """Return the history length of the first origin: 48 points, or less when the series is short."""
    return max(1, min(MIN_TRAIN_POINTS, n_obs - horizon))


def _cv_origins(values: np.ndarray, horizon: int, min_train: int | None, step: int) -> list[int]:
    """Return the rolling origins for a series of ``values``, or raise ``ValueError`` if there are none."""
    check_horizon(horizon)
    if len(values) <= horizon:
        raise ValueError(f"need more than {horizon} observations, got {len(values)}")
    start = default_min_train(len(values), horizon) if min_train is None else min_train
    origins = origin_positions(len(values), horizon, start, step)
    if not origins:
        raise ValueError("the series is too short for the requested first origin and horizon")
    return origins


def run_cv(
    series: pd.Series,
    forecasters: dict[str, Forecaster],
    horizon: int = HOLDOUT_HORIZON,
    min_train: int | None = None,
    step: int = 1,
    n_jobs: int = 1,
) -> CVResult:
    """Run every forecaster at every rolling origin and collect the out-of-sample forecasts.

    Each forecaster is called with ``series.iloc[:origin]`` only; ``n_jobs`` runs origins in
    parallel and does not change the result. Raises ``ValueError`` if no forecaster is given,
    the series is invalid or not longer than ``horizon``, or a forecast does not hold
    ``horizon`` finite values.
    """
    values = check_history(series)
    if not forecasters:
        raise ValueError("at least one forecaster is required")
    origins = _cv_origins(values, horizon, min_train, step)
    actual = np.array([values[origin : origin + horizon] for origin in origins])
    high_load = np.array([actual[i] >= np.quantile(values[:origin], HIGH_LOAD_QUANTILE) for i, origin in enumerate(origins)])
    predictions = {}
    for name, forecaster in forecasters.items():
        histories = [series.iloc[:origin] for origin in origins]
        rows = [np.asarray(row, dtype=float) for row in Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(forecaster)(history, horizon) for history in histories)]
        if any(row.shape != (horizon,) or not np.isfinite(row).all() for row in rows):
            raise ValueError(f"forecaster {name!r} must return {horizon} finite values")
        predictions[name] = np.array(rows)
    return CVResult(np.array(origins), series.index.to_numpy()[origins], actual, predictions, high_load)


def metrics_table(result: CVResult, reference: str = BASELINE_NAME) -> pd.DataFrame:
    """Summarise every forecaster over all origins and steps, with the ``reference`` as the baseline.

    Columns: MAE, RMSE, MAE ratio and improvement against the reference, MAE, RMSE and bias
    on high-load values (bias is actual minus forecast, so positive means under-forecasting),
    the standard deviation of the per-origin MAE, and the share of origins better than the
    reference. High-load columns are NaN if no value was high-load.
    """
    if reference not in result.predictions:
        raise ValueError(f"unknown reference forecaster {reference!r}")
    reference_prediction = result.predictions[reference]
    reference_fold_mae = np.abs(result.actual - reference_prediction).mean(axis=1)
    high = result.high_load
    rows = {}
    for name, prediction in result.predictions.items():
        fold_mae = np.abs(result.actual - prediction).mean(axis=1)
        ratio = normalized_mae(result.actual, prediction, reference_prediction)
        has_high = high.any()
        rows[name] = {
            "MAE": mae(result.actual, prediction),
            "RMSE": rmse(result.actual, prediction),
            "MAE ratio": ratio,
            "Improvement %": (1 - ratio) * 100,
            "High-load MAE": mae(result.actual[high], prediction[high]) if has_high else np.nan,
            "High-load RMSE": rmse(result.actual[high], prediction[high]) if has_high else np.nan,
            "High-load bias": float(np.mean(result.actual[high] - prediction[high])) if has_high else np.nan,
            "Fold MAE std": float(fold_mae.std()),
            "Folds better %": float((fold_mae < reference_fold_mae).mean() * 100),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def horizon_table(result: CVResult, metric: str = "mae") -> pd.DataFrame:
    """Return the error at each forecast step (rows) for each forecaster (columns); ``metric`` is ``mae`` or ``rmse``."""
    if metric not in ("mae", "rmse"):
        raise ValueError("metric must be 'mae' or 'rmse'")
    columns = {}
    for name, prediction in result.predictions.items():
        errors = result.actual - prediction
        columns[name] = np.abs(errors).mean(axis=0) if metric == "mae" else np.sqrt((errors**2).mean(axis=0))
    return pd.DataFrame(columns, index=pd.RangeIndex(1, result.actual.shape[1] + 1, name="step"))


def fold_table(result: CVResult) -> pd.DataFrame:
    """Return the MAE of each forecaster at each origin, indexed by the first forecast timestamp."""
    columns = {name: np.abs(result.actual - prediction).mean(axis=1) for name, prediction in result.predictions.items()}
    return pd.DataFrame(columns, index=pd.Index(result.timestamps, name="timestamp"))


def in_sample_table(
    series: pd.Series,
    horizon: int = HOLDOUT_HORIZON,
    params: ForestParams = FOREST_PARAMS,
    min_train: int | None = None,
    step: int = 1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Return the in-sample error of the baseline and the forest at each step, averaged over the rolling origins.

    This is the error on the rows each forest was trained on. Set against the out-of-sample
    error from ``run_cv`` it shows over- or underfitting. Raises ``ValueError`` if no origin
    has enough rows for any step.
    """
    origins = _cv_origins(check_history(series), horizon, min_train, step)
    tables = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(forest_fit_error)(series.iloc[:origin], horizon, params) for origin in origins)
    tables = [table for table in tables if not table.empty]
    if not tables:
        raise ValueError("no origin has enough rows to fit the forest")
    return pd.concat(tables).groupby(level=0).mean().rename_axis("step")


def plot_horizon_errors(table: pd.DataFrame, ylabel: str = "MAE (pp)"):
    """Plot the error against the forecast step for every column of ``table`` and return the figure."""
    fig, ax = plt.subplots()
    for name in table.columns:
        ax.plot(table.index, table[name], marker="o", markersize=3, label=name)
    ax.set_xlabel("Forecast step (5 minutes each)")
    ax.set_ylabel(ylabel)
    ax.set_title("Error by forecast step, rolling-origin CV on the training window")
    ax.legend()
    return fig


def main(argv: list[str] | None = None) -> None:
    """Run the CV on the training window for all models and save the metrics and the horizon MAE and RMSE outputs."""
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation of the baseline, ARIMA and random forest.")
    parser.add_argument("--data-dir", default=None, help="folder with the challenge data (default: challenge_data)")
    args = parser.parse_args(argv)
    train, _ = split_holdout(build_targets(load_readings(args.data_dir)))
    result = run_cv(train[AGGREGATE_CPU], {**candidate_forecasters(), **reference_forecasters()})
    metrics, horizon, horizon_rmse = metrics_table(result), horizon_table(result), horizon_table(result, "rmse")
    set_plot_style()
    paths = [
        save_table(metrics, METRICS_TABLE_NAME),
        save_table(horizon, HORIZON_TABLE_NAME),
        save_table(horizon_rmse, HORIZON_RMSE_TABLE_NAME),
        save_figure(plot_horizon_errors(horizon), HORIZON_FIGURE_NAME),
    ]
    print(metrics.round(3).to_string())
    print("saved:", ", ".join(Path(p).name for p in paths))


if __name__ == "__main__":
    main()
