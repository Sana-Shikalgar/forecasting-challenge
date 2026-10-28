import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cwfc import predict
from cwfc.data import ACTIVE_VM_COUNT, AGGREGATE_CPU, INTERVAL_SECONDS, READINGS_FILE

START = 1_700_000_100  # not a multiple of the interval, to check that raw timestamps are kept


def _readings(n_timestamps=70, n_vms=4, seed=0, start=START, drop=None, holdout_shift=0.0) -> pd.DataFrame:
    """Build readings of ``n_vms`` VMs with an hourly pattern and noise; ``drop`` removes (vm, row) readings."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_timestamps):
        for vm in range(n_vms):
            level = 10 + 2 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=0.5) + (holdout_shift if t >= n_timestamps - 24 else 0.0)
            rows.append({"timestamp": start + t * INTERVAL_SECONDS, "vm_id": f"vm{vm}", "avg_cpu": level})
    frame = pd.DataFrame(rows)
    if drop is not None:
        frame = frame[~frame.apply(lambda r: (r["vm_id"], int((r["timestamp"] - start) // INTERVAL_SECONDS)) in drop, axis=1)]
    return frame


def _save(folder, readings):
    """Write readings to the file name the loader expects and return the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    readings.to_csv(folder / READINGS_FILE, index=False)
    return folder


def _targets(n=70, cpu=None, counts=None) -> pd.DataFrame:
    """Build a targets table on the five-minute grid with the given CPU and count series."""
    cpu = 10 + np.sin(2 * np.pi * np.arange(n) / 12) + np.random.default_rng(1).normal(scale=0.2, size=n) if cpu is None else cpu
    counts = np.full(n, 10) if counts is None else counts
    index = pd.Index(START + np.arange(n) * INTERVAL_SECONDS, name="timestamp")
    return pd.DataFrame({ACTIVE_VM_COUNT: counts, AGGREGATE_CPU: cpu}, index=index)


def _valid_predictions(holdout_index) -> pd.DataFrame:
    """Build a predictions table that satisfies the submission format for the given timestamps."""
    return pd.DataFrame({"timestamp": holdout_index, ACTIVE_VM_COUNT: 10, AGGREGATE_CPU: 9.5})


def test_main_writes_a_valid_file_on_a_smoke_dataset(tmp_path, capsys):
    """A tiny dataset gives 24 ordered rows with the raw holdout timestamps, the right columns and no index column."""
    data = _save(tmp_path / "data", _readings())
    output = tmp_path / "out" / "predictions.csv"
    predict.main(["--data", str(data), "--output", str(output)])
    frame = pd.read_csv(output)
    assert list(frame.columns) == ["timestamp", ACTIVE_VM_COUNT, AGGREGATE_CPU]
    assert len(frame) == 24
    assert frame["timestamp"].tolist() == [START + t * INTERVAL_SECONDS for t in range(46, 70)]
    assert (frame[ACTIVE_VM_COUNT] == 4).all()
    assert frame[AGGREGATE_CPU].between(0, 100).all()
    assert "Holdout scores" in capsys.readouterr().out


def test_the_predictions_do_not_depend_on_the_holdout_values(tmp_path):
    """Two datasets that differ only in the holdout readings give the identical predictions file."""
    first = _save(tmp_path / "a", _readings())
    second = _save(tmp_path / "b", _readings(holdout_shift=5.0))
    predict.main(["--data", str(first), "--output", str(tmp_path / "a.csv")])
    predict.main(["--data", str(second), "--output", str(tmp_path / "b.csv")])
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_main_copes_with_a_varying_vm_count(tmp_path):
    """When VMs come and go the count forecast is a non-negative whole number, not a hard-coded value."""
    drop = {(f"vm{vm}", t) for vm in (1, 2) for t in range(0, 70, 3)}
    data = _save(tmp_path / "data", _readings(drop=drop))
    predict.main(["--data", str(data), "--output", str(tmp_path / "p.csv")])
    counts = pd.read_csv(tmp_path / "p.csv")[ACTIVE_VM_COUNT]
    assert counts.between(0, 4).all() and counts.dtype.kind == "i"


def test_main_runs_on_a_dataset_too_short_to_fit_arima(tmp_path):
    """With 6 training rows ARIMA falls back to the baseline with a warning and the file is still valid."""
    data = _save(tmp_path / "data", _readings(n_timestamps=30))
    with pytest.warns(RuntimeWarning, match="fell back"):
        predict.main(["--data", str(data), "--output", str(tmp_path / "p.csv")])
    assert len(pd.read_csv(tmp_path / "p.csv")) == 24


def test_main_exits_with_a_message_for_missing_data_gaps_and_a_broken_holdout_grid(tmp_path, capsys):
    """A missing folder, a gap in the training rows and a gap inside the holdout all stop the run with an error message."""
    with pytest.raises(SystemExit) as stop:
        predict.main(["--data", str(tmp_path / "nothing"), "--output", str(tmp_path / "p.csv")])
    assert stop.value.code == 1 and "error:" in capsys.readouterr().err
    for missing_row in (10, 60):
        readings = _readings()
        gap = readings[readings["timestamp"] != START + missing_row * INTERVAL_SECONDS]
        with pytest.raises(SystemExit):
            predict.main(["--data", str(_save(tmp_path / f"gap{missing_row}", gap)), "--output", str(tmp_path / "p.csv")])
    assert not (tmp_path / "p.csv").exists()


def test_main_saves_the_report_only_when_asked(tmp_path, monkeypatch):
    """--save-report writes the holdout table and figure; without it nothing is written under reports."""
    monkeypatch.setattr("cwfc.analysis.reports_dir", lambda: tmp_path / "reports")
    data = _save(tmp_path / "data", _readings())
    predict.main(["--data", str(data), "--output", str(tmp_path / "p.csv")])
    assert not (tmp_path / "reports").exists()
    predict.main(["--data", str(data), "--output", str(tmp_path / "p.csv"), "--save-report"])
    saved = sorted(path.name for path in (tmp_path / "reports").rglob("*.*"))
    assert saved == [f"{predict.HOLDOUT_FIGURE_NAME}.png", f"{predict.HOLDOUT_TABLE_NAME}.csv"]


def test_count_forecaster_is_persistence_for_a_constant_or_short_series_and_not_for_noise():
    """Ties and series too short to cross-validate go to persistence; a noisy count prefers an average."""
    assert predict.choose_count_forecaster(pd.Series(np.full(70, 10), index=_targets(70).index)) == predict.PERSISTENCE_NAME
    assert predict.choose_count_forecaster(pd.Series(np.full(20, 10), index=_targets(20).index)) == predict.PERSISTENCE_NAME
    noisy = 100 + np.random.default_rng(0).normal(scale=5, size=110).round()
    assert predict.choose_count_forecaster(pd.Series(noisy, index=_targets(110).index)) != predict.PERSISTENCE_NAME


def test_count_forecast_is_rounded_and_never_negative(monkeypatch):
    """Fractional forecasts become whole VMs and a negative forecast becomes zero."""
    history = pd.Series(np.full(30, 10.0), index=_targets(30).index)
    monkeypatch.setitem(predict.COUNT_FORECASTERS, "Persistence", lambda h, n: np.full(n, 9.6))
    assert predict.forecast_count(history, 3, "Persistence").tolist() == [10, 10, 10]
    monkeypatch.setitem(predict.COUNT_FORECASTERS, "Persistence", lambda h, n: np.full(n, -3.4))
    assert predict.forecast_count(history, 3, "Persistence").tolist() == [0, 0, 0]


def test_cpu_forecast_is_clipped_to_the_valid_range(monkeypatch):
    """A forecast outside 0 to 100 is clipped, and that is not reported as a fallback."""
    history = _targets(40)[AGGREGATE_CPU]
    monkeypatch.setattr(predict, "candidate_forecasters", lambda: {predict.ARIMA_NAME: lambda h, n: np.full(n, 150.0)})
    forecast, fell_back = predict.forecast_cpu(history, 4)
    assert forecast.tolist() == [100.0] * 4 and not fell_back


def test_a_constant_cpu_series_makes_arima_fall_back_and_warns():
    """The fallback to the baseline is announced instead of being silent."""
    train = _targets(60, cpu=np.full(60, 9.0))
    with pytest.warns(RuntimeWarning, match="fell back"):
        predictions = predict.build_predictions(train, predict.PERSISTENCE_NAME)
    assert np.allclose(predictions[AGGREGATE_CPU], 9.0)


def test_build_predictions_continues_the_training_grid_and_uses_the_chosen_count_forecaster():
    """The timestamps follow the last training timestamp by one interval each, and the count comes from the chosen forecaster."""
    train = _targets(60, counts=np.full(60, 7))
    predictions = predict.build_predictions(train, predict.PERSISTENCE_NAME, horizon=6)
    assert predictions["timestamp"].tolist() == [train.index[-1] + INTERVAL_SECONDS * k for k in range(1, 7)]
    assert predictions[ACTIVE_VM_COUNT].tolist() == [7] * 6
    with pytest.raises(ValueError, match="horizon"):
        predict.build_predictions(train, predict.PERSISTENCE_NAME, horizon=0)


def test_validate_predictions_accepts_a_correct_table():
    """A table that follows the format passes without an error."""
    holdout = _targets(30).index[-24:]
    predict.validate_predictions(_valid_predictions(holdout), holdout.to_numpy())


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda f: f.rename(columns={AGGREGATE_CPU: "cpu"}), "columns"),
        (lambda f: f.iloc[:-1], "rows"),
        (lambda f: f.iloc[::-1].reset_index(drop=True), "differ"),
        (lambda f: f.assign(timestamp=f["timestamp"] + 1), "differ"),
        (lambda f: f.assign(**{AGGREGATE_CPU: np.where(f.index == 3, np.nan, f[AGGREGATE_CPU])}), "finite"),
        (lambda f: f.assign(**{ACTIVE_VM_COUNT: -1}), "negative"),
        (lambda f: f.assign(**{AGGREGATE_CPU: 100.5}), r"\[0.0, 100.0\]"),
    ],
)
def test_validate_predictions_rejects_each_broken_rule(change, message):
    """Wrong columns, row count, order, timestamps, NaN, a negative count and an out-of-range CPU each raise ValueError."""
    holdout = _targets(30).index[-24:]
    with pytest.raises(ValueError, match=message):
        predict.validate_predictions(change(_valid_predictions(holdout)), holdout.to_numpy())


def test_write_predictions_has_no_index_column_and_creates_the_folder(tmp_path):
    """The CSV holds only the three columns and survives a round trip."""
    frame = _valid_predictions(_targets(30).index[-24:])
    path = predict.write_predictions(frame, tmp_path / "new" / "predictions.csv")
    assert path.read_text().splitlines()[0] == "timestamp,active_vm_count,aggregate_cpu_utilization"
    assert pd.read_csv(path).equals(frame)


def test_holdout_table_scores_each_model_and_the_count():
    """The table holds the MAE, RMSE and ratio to the baseline for the CPU models and absolute errors for the count."""
    holdout = pd.DataFrame({ACTIVE_VM_COUNT: [10, 10], AGGREGATE_CPU: [10.0, 12.0]})
    predictions = pd.DataFrame({"timestamp": [1, 2], ACTIVE_VM_COUNT: [10, 12], AGGREGATE_CPU: [11.0, 12.0]})
    comparison = {"Baseline": np.array([8.0, 12.0]), "ARIMA": np.array([11.0, 12.0])}
    table = predict.holdout_table(holdout, predictions, comparison, "Persistence")
    assert table.loc[(AGGREGATE_CPU, "Baseline"), "MAE"] == 1.0
    assert table.loc[(AGGREGATE_CPU, "ARIMA"), "MAE"] == 0.5 and table.loc[(AGGREGATE_CPU, "ARIMA"), "MAE ratio"] == 0.5
    assert table.loc[(ACTIVE_VM_COUNT, "Persistence"), "MAE"] == 1.0 and np.isnan(table.loc[(ACTIVE_VM_COUNT, "Persistence"), "MAE ratio"])


def test_plot_holdout_draws_the_training_tail_the_actual_values_and_each_forecast():
    """The figure has one line for the training tail, one for the holdout and one per forecast."""
    targets = _targets(70)
    train, holdout = targets.iloc[:46], targets.iloc[46:]
    comparison = {"A": np.full(24, 9.0), "B": np.full(24, 10.0)}
    fig = predict.plot_holdout(train, holdout, comparison)
    assert isinstance(fig, Figure) and len(fig.axes[0].lines) == 2 + 2 + 1  # tail, holdout, two forecasts, origin line


def test_comparison_forecasts_cover_the_four_cpu_models_and_keep_the_submitted_arima():
    """The comparison holds the baseline, ARIMA as submitted, the forest and persistence, each with the full horizon."""
    train = _targets(60)
    predictions = predict.build_predictions(train, predict.PERSISTENCE_NAME, horizon=6)
    comparison = predict.comparison_forecasts(train[AGGREGATE_CPU], predictions)
    assert list(comparison) == ["Baseline", "ARIMA", "Random forest", "Persistence"]
    assert all(len(forecast) == 6 for forecast in comparison.values())
    assert np.array_equal(comparison["ARIMA"], predictions[AGGREGATE_CPU].to_numpy())


def test_a_training_timestamp_without_a_valid_reading_stops_the_run_with_a_clear_message(tmp_path, capsys):
    """When every reading at a training timestamp is invalid the error names the series, how many timestamps and the first one."""
    readings = _readings()
    readings.loc[readings["timestamp"] == START + 20 * INTERVAL_SECONDS, "avg_cpu"] = -5.0
    with pytest.raises(SystemExit):
        predict.main(["--data", str(_save(tmp_path / "data", readings)), "--output", str(tmp_path / "p.csv")])
    message = capsys.readouterr().err
    assert "missing at 1 training timestamps" in message and str(START + 20 * INTERVAL_SECONDS) in message


def test_a_missing_holdout_value_does_not_stop_the_report_after_the_file_is_written(tmp_path, capsys):
    """A holdout timestamp with no valid reading is left out of the scores while the predictions are still written."""
    readings = _readings()
    readings.loc[readings["timestamp"] == START + 50 * INTERVAL_SECONDS, "avg_cpu"] = -5.0
    predict.main(["--data", str(_save(tmp_path / "data", readings)), "--output", str(tmp_path / "p.csv")])
    assert len(pd.read_csv(tmp_path / "p.csv")) == 24 and "Holdout scores" in capsys.readouterr().out


def test_holdout_table_is_nan_when_no_holdout_value_is_available():
    """With every holdout CPU value missing the CPU scores are NaN instead of an error."""
    holdout = pd.DataFrame({ACTIVE_VM_COUNT: [10, 10], AGGREGATE_CPU: [np.nan, np.nan]})
    predictions = pd.DataFrame({"timestamp": [1, 2], ACTIVE_VM_COUNT: [10, 10], AGGREGATE_CPU: [11.0, 12.0]})
    table = predict.holdout_table(holdout, predictions, {"Baseline": np.array([8.0, 12.0])}, "Persistence")
    assert table.loc[(AGGREGATE_CPU, "Baseline"), ["MAE", "RMSE", "MAE ratio"]].isna().all()


def test_an_output_path_that_cannot_be_written_exits_with_a_message(tmp_path, capsys):
    """Pointing --output at a folder gives an error message and exit code 1, not a traceback."""
    data = _save(tmp_path / "data", _readings())
    with pytest.raises(SystemExit) as stop:
        predict.main(["--data", str(data), "--output", str(tmp_path)])
    assert stop.value.code == 1 and "cannot write" in capsys.readouterr().err
