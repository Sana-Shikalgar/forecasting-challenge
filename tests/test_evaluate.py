import numpy as np
import pandas as pd
import pytest
from conftest import make_series
from matplotlib.figure import Figure

from cwfc import evaluate as ev
from cwfc.models import ForestParams, forest_fit_error, history_mean, persistence


def _result(actual, predictions, high_load=None):
    """Build a CVResult by hand with two origins and the given arrays."""
    actual = np.asarray(actual, dtype=float)
    mask = np.zeros(actual.shape, dtype=bool) if high_load is None else np.asarray(high_load, dtype=bool)
    return ev.CVResult(np.array([10, 11]), np.array([3000, 3300]), actual, {k: np.asarray(v, dtype=float) for k, v in predictions.items()}, mask)


@pytest.mark.parametrize("n_obs, horizon, expected", [(110, 24, 48), (60, 24, 36), (30, 24, 6), (25, 24, 1), (24, 24, 1)])
def test_default_min_train_is_48_or_what_the_series_allows(n_obs, horizon, expected):
    """The first origin has 48 points, fewer when the series is short, and never less than one."""
    assert ev.default_min_train(n_obs, horizon) == expected


def test_run_cv_origins_and_shapes_on_the_training_window_size():
    """110 points, a 24-step horizon and a first origin at 48 give 39 origins with 24-step paths."""
    series = make_series(np.arange(110.0))
    result = ev.run_cv(series, {"persistence": persistence})
    assert result.origins.tolist() == list(range(48, 87)) and result.actual.shape == (39, 24)
    assert result.predictions["persistence"].shape == (39, 24) and result.high_load.shape == (39, 24)


def test_run_cv_actuals_and_timestamps_come_from_the_series():
    """Each row of actuals is the 24 values after the origin, and the timestamp is that of the first one."""
    series = make_series(np.arange(80.0))
    result = ev.run_cv(series, {"persistence": persistence})
    assert result.actual[0].tolist() == list(np.arange(48.0, 72.0))
    assert result.timestamps[0] == series.index[48] and result.timestamps[-1] == series.index[56]


def test_run_cv_gives_each_forecaster_only_the_history_before_the_origin():
    """A forecaster is called with exactly the observations before each origin."""
    seen = []

    def spy(history, horizon):
        """Record the history length and last value, then forecast zeros."""
        seen.append((len(history), history.iloc[-1]))
        return np.zeros(horizon)

    ev.run_cv(make_series(np.arange(70.0)), {"spy": spy}, horizon=6, min_train=50, step=5)
    assert seen == [(50, 49.0), (55, 54.0), (60, 59.0)]


def test_run_cv_high_load_threshold_comes_from_each_origins_own_history():
    """The high-load mask compares the actual values with the 80th percentile of that origin's history."""
    series = make_series(np.arange(60.0))
    result = ev.run_cv(series, {"persistence": persistence}, horizon=4, min_train=40, step=10)
    for row, origin in enumerate(result.origins):
        threshold = np.quantile(np.arange(float(origin)), ev.HIGH_LOAD_QUANTILE)
        assert (result.high_load[row] == (result.actual[row] >= threshold)).all()


def test_run_cv_adapts_to_a_short_series():
    """A series just longer than the horizon gives a single origin that starts early."""
    result = ev.run_cv(make_series(np.arange(30.0)), {"mean": history_mean})
    assert result.origins.tolist() == [6]


def test_run_cv_parallel_run_gives_the_same_result():
    """Running origins in parallel does not change any forecast."""
    series = make_series(np.sin(np.arange(70.0)))
    serial = ev.run_cv(series, {"mean": history_mean, "last": persistence}, horizon=6, min_train=50)
    parallel = ev.run_cv(series, {"mean": history_mean, "last": persistence}, horizon=6, min_train=50, n_jobs=2)
    assert all(np.array_equal(serial.predictions[k], parallel.predictions[k]) for k in serial.predictions)


def test_run_cv_rejects_bad_inputs():
    """No forecasters, a short series, a bad horizon, a bad step and a too-late first origin raise ValueError."""
    series = make_series(np.arange(50.0))
    with pytest.raises(ValueError, match="at least one forecaster"):
        ev.run_cv(series, {})
    with pytest.raises(ValueError, match="need more than"):
        ev.run_cv(make_series(np.arange(24.0)), {"p": persistence})
    with pytest.raises(ValueError, match="positive integer"):
        ev.run_cv(series, {"p": persistence}, horizon=0)
    with pytest.raises(ValueError, match="step must be positive"):
        ev.run_cv(series, {"p": persistence}, step=0)
    with pytest.raises(ValueError, match="too short"):
        ev.run_cv(series, {"p": persistence}, min_train=40)
    with pytest.raises(ValueError, match="empty"):
        ev.run_cv(series.iloc[:0], {"p": persistence})


def test_run_cv_rejects_a_series_with_missing_values_or_gaps_and_a_non_positive_first_origin():
    """NaN values, a timestamp gap and a first origin of zero all raise ValueError before any forecasting."""
    series = make_series(np.arange(60.0))
    with pytest.raises(ValueError, match="finite"):
        ev.run_cv(series.where(series.index != 3000), {"p": persistence})
    with pytest.raises(ValueError, match="evenly spaced"):
        ev.run_cv(series.drop(series.index[10]), {"p": persistence})
    with pytest.raises(ValueError, match="must be positive"):
        ev.run_cv(series, {"p": persistence}, min_train=0)


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_run_cv_rejects_forecasts_of_the_wrong_length_or_not_finite(bad):
    """A forecast with the wrong length or non-finite values raises ValueError naming the forecaster."""
    with pytest.raises(ValueError, match="'broken'"):
        ev.run_cv(make_series(np.arange(60.0)), {"broken": lambda history, horizon: bad}, horizon=6, min_train=50)


def test_metrics_table_values_on_a_hand_made_result():
    """MAE, RMSE, ratio, improvement and share of better folds are right on a small example."""
    actual = [[1.0, 3.0], [5.0, 7.0]]
    result = _result(actual, {"ref": [[0.0, 0.0], [0.0, 0.0]], "exact": actual, "half": [[0.5, 1.5], [2.5, 3.5]]})
    table = ev.metrics_table(result, "ref")
    assert table.loc["ref", ["MAE", "RMSE", "MAE ratio", "Improvement %", "Folds better %"]].tolist() == pytest.approx([4.0, np.sqrt(21), 1.0, 0.0, 0.0])
    assert table.loc["exact", ["MAE", "RMSE", "MAE ratio", "Improvement %", "Folds better %"]].tolist() == pytest.approx([0.0, 0.0, 0.0, 100.0, 100.0])
    assert table.loc["half", "MAE ratio"] == pytest.approx(0.5) and table.loc["half", "Folds better %"] == 100.0


def test_metrics_table_high_load_columns_use_only_flagged_values():
    """High-load MAE, RMSE and bias come from the flagged values only, with bias as actual minus forecast."""
    result = _result([[1.0, 3.0], [5.0, 7.0]], {"ref": [[0.0, 0.0], [0.0, 0.0]], "m": [[1.0, 1.0], [5.0, 4.0]]}, high_load=[[False, True], [False, True]])
    row = ev.metrics_table(result, "ref").loc["m"]
    assert row["High-load MAE"] == pytest.approx(2.5) and row["High-load RMSE"] == pytest.approx(np.sqrt(6.5)) and row["High-load bias"] == pytest.approx(2.5)


def test_metrics_table_high_load_is_nan_when_nothing_is_high_load():
    """Without any high-load value the high-load columns are NaN instead of failing."""
    result = _result([[1.0, 2.0], [3.0, 4.0]], {"ref": [[0.0, 0.0], [0.0, 0.0]]})
    assert ev.metrics_table(result, "ref").loc["ref", ["High-load MAE", "High-load RMSE", "High-load bias"]].isna().all()


def test_metrics_table_with_an_exact_reference_has_no_relative_improvement():
    """An exact reference gives infinity for a worse model and zero for an exact one."""
    actual = [[1.0, 2.0], [3.0, 4.0]]
    result = _result(actual, {"ref": actual, "worse": [[2.0, 2.0], [3.0, 4.0]], "same": actual})
    table = ev.metrics_table(result, "ref")
    assert table.loc["worse", "MAE ratio"] == np.inf and table.loc["same", "MAE ratio"] == 0.0


def test_metrics_table_rejects_an_unknown_reference():
    """A reference name that is not a forecaster raises ValueError."""
    with pytest.raises(ValueError, match="unknown reference"):
        ev.metrics_table(_result([[1.0], [2.0]], {"a": [[1.0], [2.0]]}), "b")


def test_horizon_table_mae_and_rmse_per_step():
    """The step-wise MAE and RMSE are computed across origins for each step."""
    result = _result([[1.0, 2.0], [3.0, 6.0]], {"m": [[0.0, 0.0], [0.0, 0.0]]})
    assert ev.horizon_table(result, "mae")["m"].tolist() == [2.0, 4.0]
    assert ev.horizon_table(result, "rmse")["m"].tolist() == pytest.approx([np.sqrt(5), np.sqrt(20)])
    assert ev.horizon_table(result).index.tolist() == [1, 2]


def test_horizon_table_rejects_an_unknown_metric():
    """Only mae and rmse are accepted."""
    with pytest.raises(ValueError, match="metric must be"):
        ev.horizon_table(_result([[1.0], [2.0]], {"a": [[1.0], [2.0]]}), "mape")


def test_fold_table_is_indexed_by_the_first_forecast_timestamp():
    """One row per origin with the MAE of each forecaster."""
    table = ev.fold_table(_result([[1.0, 3.0], [5.0, 7.0]], {"m": [[0.0, 0.0], [5.0, 5.0]]}))
    assert table.index.tolist() == [3000, 3300] and table["m"].tolist() == [2.0, 1.0]


def test_plot_horizon_errors_returns_a_figure_with_one_line_per_model():
    """The figure holds a line for each column of the table and labelled axes."""
    table = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}, index=[1, 2])
    fig = ev.plot_horizon_errors(table)
    ax = fig.axes[0]
    assert isinstance(fig, Figure) and len(ax.lines) == 2 and ax.get_xlabel() and ax.get_ylabel()


def test_main_runs_the_cv_and_saves_the_outputs(tmp_path, monkeypatch, capsys):
    """The command line entry point evaluates, prints the metrics and writes three tables and a figure."""
    stamps = np.arange(80) * 300
    readings = pd.DataFrame({"timestamp": np.repeat(stamps, 2), "vm_id": np.tile(["a", "b"], 80), "avg_cpu": 10 + np.repeat(np.sin(np.arange(80)), 2)})
    monkeypatch.setattr(ev, "load_readings", lambda data_dir=None: readings)
    monkeypatch.setattr(ev, "candidate_forecasters", lambda: {"Baseline": persistence})
    monkeypatch.setattr(ev, "reference_forecasters", lambda: {"Mean": history_mean})
    monkeypatch.setattr("cwfc.analysis.reports_dir", lambda: tmp_path)
    ev.main([])
    assert sorted(path.name for path in tmp_path.rglob("*.*")) == sorted([f"{ev.HORIZON_FIGURE_NAME}.png", f"{ev.HORIZON_TABLE_NAME}.csv", f"{ev.HORIZON_RMSE_TABLE_NAME}.csv", f"{ev.METRICS_TABLE_NAME}.csv"])
    assert "Baseline" in capsys.readouterr().out


def test_in_sample_table_averages_the_fit_error_over_origins(noisy_series):
    """The table is the mean of the per-origin in-sample errors, one row per step, with the forest below the baseline."""
    params = ForestParams(n_estimators=10)
    table = ev.in_sample_table(noisy_series, horizon=3, params=params, min_train=80, step=8)
    manual = pd.concat([forest_fit_error(noisy_series.iloc[:origin], 3, params) for origin in (80, 88)]).groupby(level=0).mean()
    assert table.index.tolist() == [1, 2, 3] and np.allclose(table.to_numpy(), manual.to_numpy())
    assert (table["forest"] < table["baseline"]).all()


def test_in_sample_table_rejects_a_short_series_and_origins_without_enough_rows():
    """A series not longer than the horizon, or origins whose histories give no usable rows, raise ValueError."""
    with pytest.raises(ValueError, match="need more than"):
        ev.in_sample_table(make_series(np.arange(10.0)), horizon=10)
    with pytest.raises(ValueError, match="no origin has enough rows"):
        ev.in_sample_table(make_series(np.arange(30.0)), horizon=24, min_train=5)
