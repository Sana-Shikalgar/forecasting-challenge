from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc import data
from cwfc.data import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    METADATA_FILE,
    READINGS_FILE,
    build_targets,
    check_readings,
    load_metadata,
    load_readings,
    require_columns,
    select_timestamps,
    split_holdout,
)
from cwfc.download import default_data_dir

REAL_DATA_AVAILABLE = (default_data_dir() / READINGS_FILE).is_file()


def _readings(rows: list[tuple]) -> pd.DataFrame:
    """Build a readings frame from (timestamp, vm_id, avg_cpu) tuples."""
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def _write_readings(folder: Path, frame: pd.DataFrame) -> Path:
    """Write a readings frame as the gzip CSV the loader expects."""
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / READINGS_FILE, index=False)
    return folder


def _full_readings(rows: list[tuple]) -> pd.DataFrame:
    """Build a five-column readings frame from (timestamp, vm_id, avg_cpu) tuples."""
    frame = _readings(rows)
    frame["min_cpu"] = frame["avg_cpu"] - 1.0
    frame["max_cpu"] = frame["avg_cpu"] + 1.0
    return frame[["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]]


def _metadata(**overrides) -> pd.DataFrame:
    """Build a two-VM metadata frame, optionally overriding columns."""
    frame = pd.DataFrame(
        {
            "vm_id": ["001", "002"],
            "subscription_id": ["s1", "s2"],
            "deployment_id": ["d1", "d2"],
            "timestamp_vm_created": [0, 0],
            "timestamp_vm_deleted": [40500, 50000],
            "max_cpu": [50.0, 60.0],
            "avg_cpu": [5.0, 6.0],
            "p95_max_cpu": [20.0, 30.0],
            "vm_category": ["Interactive", "Unknown"],
            "virtual_core_bucket": ["2", ">24"],
            "memory_gb_bucket": ["4", ">64"],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


# shared constants


def test_time_constants_are_consistent():
    """The 5-minute step divides the hour evenly and the holdout covers two hours."""
    assert data.STEPS_PER_HOUR == 12 and data.STEPS_PER_HOUR * data.INTERVAL_SECONDS == data.SECONDS_PER_HOUR
    assert data.SECONDS_PER_DAY == 86400 and data.HOLDOUT_HORIZON == 2 * data.STEPS_PER_HOUR


# require_columns


def test_require_columns_passes_when_all_present():
    """No error is raised when every required column exists."""
    require_columns(["a", "b", "c"], ["a", "b"], "table")


def test_require_columns_lists_every_missing_column():
    """The error names the table and all missing columns."""
    with pytest.raises(ValueError, match=r"table is missing required columns: \['x', 'y'\]"):
        require_columns(["a"], ["a", "x", "y"], "table")


# load_readings


def test_load_readings_reads_gzip_csv_with_expected_types(tmp_path):
    """A gzip CSV loads with integer timestamps, string ids and float CPU values."""
    folder = tmp_path / "data"
    folder.mkdir()
    pd.DataFrame(
        {"timestamp": [0, 300], "vm_id": ["001", "002"], "min_cpu": [1.0, 2.0], "max_cpu": [9.0, 8.0], "avg_cpu": [4.5, 5.5]}
    ).to_csv(folder / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    frame = load_readings(folder)
    assert list(frame.columns) == ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    assert frame["timestamp"].dtype == "int64"
    assert frame["avg_cpu"].dtype == "float64"
    assert frame["vm_id"].tolist() == ["001", "002"]
    assert frame["avg_cpu"].tolist() == [4.5, 5.5]


def test_load_readings_accepts_string_directory(tmp_path):
    """The data directory may be given as a plain string."""
    folder = _write_readings(tmp_path / "data", _full_readings([(0, "a", 1.0)]))
    assert len(load_readings(str(folder))) == 1


def test_load_readings_uses_default_directory_when_none_given(tmp_path, monkeypatch):
    """Without an argument the loader reads from the default data directory."""
    folder = _write_readings(tmp_path / "data", _full_readings([(0, "a", 1.0)]))
    monkeypatch.setattr(data, "default_data_dir", lambda: folder)
    assert len(load_readings()) == 1


def test_load_readings_missing_file_raises(tmp_path):
    """A missing readings file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="data file not found"):
        load_readings(tmp_path)


def test_load_readings_missing_required_column_raises(tmp_path):
    """A file without avg_cpu is rejected and the column is named."""
    folder = _write_readings(tmp_path / "data", _full_readings([(0, "a", 1.0)]).drop(columns="avg_cpu"))
    with pytest.raises(ValueError, match="avg_cpu"):
        load_readings(folder)


def test_load_readings_works_without_min_and_max_columns(tmp_path):
    """min_cpu and max_cpu are optional."""
    folder = _write_readings(tmp_path / "data", _readings([(0, "a", 1.0)]))
    assert list(load_readings(folder).columns) == ["timestamp", "vm_id", "avg_cpu"]


def test_load_readings_ignores_unknown_columns(tmp_path):
    """Columns the loader does not know about are dropped."""
    frame = _full_readings([(0, "a", 1.0)])
    frame["junk"] = 1
    folder = _write_readings(tmp_path / "data", frame)
    assert "junk" not in load_readings(folder).columns


def test_load_readings_keeps_file_order(tmp_path):
    """Rows come back in file order, even if timestamps are unsorted."""
    folder = _write_readings(tmp_path / "data", _full_readings([(600, "a", 1.0), (0, "a", 2.0), (300, "a", 3.0)]))
    assert load_readings(folder)["timestamp"].tolist() == [600, 0, 300]


def test_load_readings_header_only_file_gives_empty_frame(tmp_path):
    """A file with only a header loads as an empty frame."""
    folder = _write_readings(tmp_path / "data", _full_readings([(0, "a", 1.0)]).iloc[0:0])
    frame = load_readings(folder)
    assert frame.empty and "avg_cpu" in frame.columns


def test_load_readings_empty_file_raises(tmp_path):
    """A zero-byte file raises ValueError."""
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / READINGS_FILE).write_bytes(b"")
    with pytest.raises(ValueError):
        load_readings(folder)


def test_load_readings_non_numeric_cpu_raises(tmp_path):
    """Text in avg_cpu raises ValueError."""
    frame = _full_readings([(0, "a", 1.0)])
    frame["avg_cpu"] = ["not a number"]
    with pytest.raises(ValueError):
        load_readings(_write_readings(tmp_path / "data", frame))


def _write_raw_readings(folder: Path, text: str) -> Path:
    """Write raw CSV text as the gzip readings file."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / READINGS_FILE).write_bytes(gzip.compress(text.encode()))
    return folder


def test_load_readings_empty_timestamp_cell_raises(tmp_path):
    """An empty timestamp cell raises ValueError."""
    folder = _write_raw_readings(tmp_path / "data", "timestamp,vm_id,avg_cpu\n0,a,1.0\n,a,2.0\n")
    with pytest.raises(ValueError, match="NA values"):
        load_readings(folder)


def test_load_readings_fractional_timestamp_raises(tmp_path):
    """A timestamp that is not a whole number of seconds raises ValueError."""
    folder = _write_raw_readings(tmp_path / "data", "timestamp,vm_id,avg_cpu\n0.5,a,1.0\n300,a,2.0\n")
    with pytest.raises(ValueError):
        load_readings(folder)


def test_load_readings_accepts_whole_timestamps_written_as_floats(tmp_path):
    """Timestamps like 300.0 (as pandas writes them) load as integers."""
    folder = _write_raw_readings(tmp_path / "data", "timestamp,vm_id,avg_cpu\n0.0,a,1.0\n300.0,a,2.0\n")
    frame = load_readings(folder)
    assert frame["timestamp"].tolist() == [0, 300] and frame["timestamp"].dtype == "int64"


# load_metadata


def test_load_metadata_keeps_all_columns_and_string_buckets(tmp_path):
    """Metadata loads whole, with ids and buckets as strings (">24" stays text)."""
    folder = tmp_path / "data"
    folder.mkdir()
    _metadata().to_csv(folder / METADATA_FILE, index=False)
    frame = load_metadata(folder)
    assert list(frame.columns) == list(_metadata().columns)
    assert frame["vm_id"].tolist() == ["001", "002"]
    assert frame["virtual_core_bucket"].tolist() == ["2", ">24"]


def test_load_metadata_missing_file_raises(tmp_path):
    """A missing metadata file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_load_metadata_requires_vm_id(tmp_path):
    """Metadata without vm_id is rejected."""
    folder = tmp_path / "data"
    folder.mkdir()
    _metadata().drop(columns="vm_id").to_csv(folder / METADATA_FILE, index=False)
    with pytest.raises(ValueError, match="vm_id"):
        load_metadata(folder)


def test_load_metadata_rejects_duplicate_vm_ids(tmp_path):
    """The same vm_id twice is ambiguous and raises ValueError."""
    folder = tmp_path / "data"
    folder.mkdir()
    _metadata(vm_id=["001", "001"]).to_csv(folder / METADATA_FILE, index=False)
    with pytest.raises(ValueError, match="more than once"):
        load_metadata(folder)


def test_load_metadata_does_not_require_other_columns(tmp_path):
    """Only vm_id is needed to load; callers check the rest."""
    folder = tmp_path / "data"
    folder.mkdir()
    _metadata()[["vm_id"]].to_csv(folder / METADATA_FILE, index=False)
    assert list(load_metadata(folder).columns) == ["vm_id"]


# check_readings


def test_check_readings_clean_data():
    """Clean, regular data reports no problems."""
    rows = [(t, vm, 10.0) for t in (0, 300, 600) for vm in ("a", "b")]
    quality = check_readings(_readings(rows))
    assert quality.n_rows == 6 and quality.n_vms == 2 and quality.n_timestamps == 3
    assert quality.n_duplicate_vm_timestamps == 0
    assert quality.n_missing_avg_cpu == 0 and quality.n_out_of_range_avg_cpu == 0
    assert quality.n_irregular_steps == 0
    assert quality.sorted_by_timestamp and quality.vm_count_is_constant


def test_check_readings_counts_duplicates():
    """A repeated (vm_id, timestamp) pair is counted once as a duplicate."""
    quality = check_readings(_readings([(0, "a", 1.0), (0, "a", 2.0), (300, "a", 3.0)]))
    assert quality.n_duplicate_vm_timestamps == 1


def test_check_readings_counts_missing_and_out_of_range_but_not_boundaries():
    """NaN and values outside [0, 100] are counted; 0 and 100 are valid."""
    rows = [(0, "a", np.nan), (0, "b", -0.1), (0, "c", 100.1), (0, "d", 0.0), (0, "e", 100.0)]
    quality = check_readings(_readings(rows))
    assert quality.n_missing_avg_cpu == 1
    assert quality.n_out_of_range_avg_cpu == 2


def test_check_readings_counts_missing_vm_ids():
    """Rows without a vm_id are counted."""
    quality = check_readings(_readings([(0, None, 1.0), (0, "a", 2.0), (300, np.nan, 3.0)]))
    assert quality.n_missing_vm_id == 2


def test_check_readings_flags_gap_in_timestamps():
    """A missing 5-minute step counts as one irregular step."""
    quality = check_readings(_readings([(0, "a", 1.0), (300, "a", 1.0), (900, "a", 1.0)]))
    assert quality.n_irregular_steps == 1


def test_check_readings_respects_custom_interval():
    """Regularity is judged against the interval that is passed in."""
    frame = _readings([(0, "a", 1.0), (60, "a", 1.0), (120, "a", 1.0)])
    assert check_readings(frame, interval_seconds=60).n_irregular_steps == 0
    assert check_readings(frame).n_irregular_steps == 2


def test_check_readings_detects_unsorted_rows():
    """Rows out of chronological order are flagged."""
    quality = check_readings(_readings([(300, "a", 1.0), (0, "a", 1.0)]))
    assert not quality.sorted_by_timestamp
    assert quality.n_irregular_steps == 0


def test_check_readings_detects_changing_vm_count():
    """A VM that appears late means the per-timestamp VM count is not constant."""
    quality = check_readings(_readings([(0, "a", 1.0), (300, "a", 1.0), (300, "b", 1.0)]))
    assert not quality.vm_count_is_constant


def test_check_readings_single_timestamp_has_no_steps():
    """One timestamp has no steps to be irregular."""
    quality = check_readings(_readings([(0, "a", 1.0), (0, "b", 1.0)]))
    assert quality.n_irregular_steps == 0 and quality.vm_count_is_constant


def test_check_readings_handles_empty_frame():
    """An empty frame gives zero counts instead of failing."""
    quality = check_readings(_readings([]).astype({"timestamp": "int64", "avg_cpu": "float64"}))
    assert quality.n_rows == 0 and quality.n_vms == 0 and quality.n_timestamps == 0
    assert quality.n_irregular_steps == 0


def test_check_readings_requires_columns():
    """Missing required columns raise ValueError."""
    with pytest.raises(ValueError, match="vm_id"):
        check_readings(pd.DataFrame({"timestamp": [0], "avg_cpu": [1.0]}))


# build_targets


def test_build_targets_hand_computed_example():
    """Count and mean per timestamp match a hand calculation."""
    targets = build_targets(_readings([(0, "a", 10.0), (0, "b", 20.0), (300, "a", 30.0), (300, "b", 50.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2, 2]
    assert targets[AGGREGATE_CPU].tolist() == [15.0, 40.0]
    assert list(targets.columns) == [ACTIVE_VM_COUNT, AGGREGATE_CPU]


def test_build_targets_index_dtypes_and_order():
    """The result is indexed by raw sorted timestamps with integer counts."""
    targets = build_targets(_readings([(600, "a", 1.0), (0, "a", 2.0), (300, "a", 3.0)]))
    assert targets.index.name == "timestamp"
    assert targets.index.tolist() == [0, 300, 600]
    assert targets[ACTIVE_VM_COUNT].dtype == "int64"
    assert targets[AGGREGATE_CPU].dtype == "float64"
    assert targets[AGGREGATE_CPU].tolist() == [2.0, 3.0, 1.0]


def test_build_targets_counts_duplicate_vm_once_and_averages_its_values():
    """Two rows for one VM at one timestamp count once and contribute their mean."""
    targets = build_targets(_readings([(0, "a", 10.0), (0, "a", 30.0), (0, "b", 40.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2]
    assert targets[AGGREGATE_CPU].tolist() == [30.0]


def test_build_targets_ignores_readings_without_vm_id():
    """A reading with no vm_id cannot be attributed to a VM, so it is ignored."""
    targets = build_targets(_readings([(0, "a", 10.0), (0, None, 90.0), (300, None, 50.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [1, 0]
    assert targets[AGGREGATE_CPU].iloc[0] == 10.0 and np.isnan(targets[AGGREGATE_CPU].iloc[1])


def test_build_targets_duplicate_with_one_invalid_reading_uses_the_valid_one():
    """If one of a VM's duplicate rows is invalid, only the valid row counts."""
    targets = build_targets(_readings([(0, "a", 40.0), (0, "a", np.nan), (0, "b", 20.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2]
    assert targets[AGGREGATE_CPU].tolist() == [30.0]


def test_build_targets_ignores_missing_values():
    """A NaN reading is neither counted nor averaged."""
    targets = build_targets(_readings([(0, "a", 10.0), (0, "b", np.nan), (0, "c", 30.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2]
    assert targets[AGGREGATE_CPU].tolist() == [20.0]


def test_build_targets_keeps_timestamp_with_only_invalid_readings():
    """A timestamp with no valid reading stays, with count 0 and a NaN aggregate."""
    targets = build_targets(_readings([(0, "a", 10.0), (300, "a", np.nan), (300, "b", 150.0), (600, "a", 20.0)]))
    assert targets.index.tolist() == [0, 300, 600]
    assert targets[ACTIVE_VM_COUNT].tolist() == [1, 0, 1]
    assert np.isnan(targets[AGGREGATE_CPU].iloc[1])


def test_build_targets_excludes_out_of_range_but_keeps_boundaries():
    """Values outside [0, 100] are invalid; exactly 0 and 100 are valid."""
    targets = build_targets(_readings([(0, "a", -1.0), (0, "b", 100.5), (0, "c", 0.0), (0, "d", 100.0)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2]
    assert targets[AGGREGATE_CPU].tolist() == [50.0]


def test_build_targets_follows_vms_that_start_late_or_end_early():
    """The active count changes as VMs appear and disappear."""
    rows = [(0, "a", 10.0), (0, "b", 10.0), (300, "a", 10.0), (300, "c", 10.0), (600, "c", 10.0)]
    assert build_targets(_readings(rows))[ACTIVE_VM_COUNT].tolist() == [2, 2, 1]


def test_build_targets_does_not_invent_missing_timestamps():
    """A gap in the 5-minute grid stays a gap."""
    targets = build_targets(_readings([(0, "a", 1.0), (300, "a", 1.0), (900, "a", 1.0)]))
    assert targets.index.tolist() == [0, 300, 900]


def test_build_targets_single_row():
    """One VM at one timestamp works."""
    targets = build_targets(_readings([(0, "a", 7.5)]))
    assert targets[ACTIVE_VM_COUNT].tolist() == [1] and targets[AGGREGATE_CPU].tolist() == [7.5]


def test_build_targets_needs_only_the_three_required_columns():
    """No metadata, min_cpu or max_cpu are needed to build the targets."""
    frame = _readings([(0, "a", 1.0)])
    assert list(frame.columns) == ["timestamp", "vm_id", "avg_cpu"]
    assert len(build_targets(frame)) == 1


def test_build_targets_does_not_modify_its_input():
    """The input frame is left unchanged."""
    frame = _readings([(300, "b", 5.0), (0, "a", np.nan)])
    before = frame.copy()
    build_targets(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_build_targets_empty_input_raises():
    """No rows at all raises ValueError."""
    with pytest.raises(ValueError, match="no rows"):
        build_targets(_readings([]).astype({"timestamp": "int64", "avg_cpu": "float64"}))


def test_build_targets_missing_column_raises():
    """A frame without avg_cpu raises ValueError."""
    with pytest.raises(ValueError, match="avg_cpu"):
        build_targets(pd.DataFrame({"timestamp": [0], "vm_id": ["a"]}))


def test_build_targets_aggregate_stays_within_bounds_for_random_data():
    """For random valid and invalid readings the aggregate stays in [0, 100]."""
    rng = np.random.default_rng(0)
    n = 500
    frame = pd.DataFrame(
        {"timestamp": rng.integers(0, 5, n) * 300, "vm_id": rng.integers(0, 40, n).astype(str), "avg_cpu": rng.uniform(-20, 120, n)}
    )
    aggregate = build_targets(frame)[AGGREGATE_CPU].dropna()
    assert ((aggregate >= 0) & (aggregate <= 100)).all()


# select_timestamps


def test_select_timestamps_keeps_only_listed_timestamps():
    """Only readings at the listed timestamps are kept, in their original order."""
    frame = _readings([(0, "a", 1.0), (300, "a", 2.0), (600, "a", 3.0), (300, "b", 4.0)])
    kept = select_timestamps(frame, [300, 600])
    assert kept["avg_cpu"].tolist() == [2.0, 3.0, 4.0]
    assert select_timestamps(frame, []).empty and len(frame) == 4


# end to end


def test_readings_file_to_targets_without_metadata(tmp_path):
    """End to end: gzip CSV in, target series out, no metadata present."""
    rows = [(t, vm, 10.0 * (i + 1)) for i, t in enumerate((0, 300)) for vm in ("a", "b")]
    folder = _write_readings(tmp_path / "data", _full_readings(rows))
    targets = build_targets(load_readings(folder))
    assert targets[ACTIVE_VM_COUNT].tolist() == [2, 2]
    assert targets[AGGREGATE_CPU].tolist() == [10.0, 20.0]


# split_holdout


def test_split_holdout_default_is_last_24_rows():
    """With 134 rows the split is 110 training rows and 24 holdout rows."""
    frame = pd.DataFrame({"x": range(134)}, index=pd.Index(np.arange(134) * 300, name="timestamp"))
    train, holdout = split_holdout(frame)
    assert len(train) == 110 and len(holdout) == 24
    assert train.index[-1] + 300 == holdout.index[0]
    assert holdout.index[-1] == frame.index[-1]


def test_split_holdout_is_derived_from_length_not_hard_coded():
    """A shorter series still holds out the last 24 rows."""
    train, holdout = split_holdout(pd.Series(range(30), index=np.arange(30)))
    assert len(train) == 6 and len(holdout) == 24


def test_split_holdout_custom_horizon():
    """A different horizon holds out that many rows."""
    train, holdout = split_holdout(pd.Series(range(10), index=np.arange(10)), horizon=3)
    assert train.tolist() == list(range(7)) and holdout.tolist() == [7, 8, 9]


def test_split_holdout_leaves_one_training_row_at_minimum():
    """Horizon plus one rows is the smallest series that can be split."""
    train, holdout = split_holdout(pd.Series(range(25), index=np.arange(25)))
    assert len(train) == 1 and len(holdout) == 24


@pytest.mark.parametrize("length", [0, 1, 23, 24])
def test_split_holdout_rejects_series_without_training_rows(length):
    """A series no longer than the horizon cannot be split."""
    with pytest.raises(ValueError, match="need more than 24"):
        split_holdout(pd.Series(range(length), index=np.arange(length), dtype="float64"))


@pytest.mark.parametrize("horizon", [0, -3])
def test_split_holdout_rejects_non_positive_horizon(horizon):
    """Zero or negative horizons are rejected."""
    with pytest.raises(ValueError, match="positive"):
        split_holdout(pd.Series(range(50), index=np.arange(50)), horizon=horizon)


def test_split_holdout_rejects_unsorted_index():
    """A series that is not chronological is rejected instead of split wrongly."""
    with pytest.raises(ValueError, match="sorted"):
        split_holdout(pd.Series(range(30), index=np.arange(30)[::-1]))


# real data


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="challenge data not downloaded; run python -m cwfc.download")
def test_real_data_has_expected_shape_and_split():
    """The real release has 134 regular timestamps, 10,000 VMs and a 110/24 split."""
    readings = load_readings()
    quality = check_readings(readings)
    assert quality.n_rows == 1_340_000 and quality.n_vms == 10_000 and quality.n_timestamps == 134
    assert quality.n_duplicate_vm_timestamps == 0 and quality.n_missing_vm_id == 0 and quality.n_missing_avg_cpu == 0
    assert quality.n_out_of_range_avg_cpu == 0 and quality.n_irregular_steps == 0 and quality.vm_count_is_constant
    targets = build_targets(readings)
    assert targets.index.tolist() == list(range(0, 134 * 300, 300))
    assert targets[ACTIVE_VM_COUNT].eq(10_000).all()
    assert targets[AGGREGATE_CPU].between(0, 100).all()
    train, holdout = split_holdout(targets)
    assert len(train) == 110 and len(holdout) == 24


# edge cases found in an audit


def test_build_targets_rejects_missing_timestamps():
    """A NaN timestamp cannot be placed in the series, so ValueError is raised."""
    frame = pd.DataFrame({"timestamp": [0.0, np.nan], "vm_id": ["a", "b"], "avg_cpu": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing timestamps"):
        build_targets(frame)


@pytest.mark.parametrize("horizon", [24.0, True, "24", None])
def test_split_holdout_rejects_non_integer_horizons(horizon):
    """A float, boolean, string or None horizon raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        split_holdout(pd.Series(range(40), index=np.arange(40)), horizon=horizon)


def test_split_holdout_accepts_numpy_integer_horizon():
    """A numpy integer is a valid horizon."""
    train, holdout = split_holdout(pd.Series(range(40), index=np.arange(40)), horizon=np.int64(5))
    assert len(train) == 35 and len(holdout) == 5
