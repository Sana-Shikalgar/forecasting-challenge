"""Load the challenge files and build the two forecasting target series.

``load_readings`` and ``load_metadata`` only read and validate one file each.
``build_targets`` needs the readings alone, so forecasting never depends on the
metadata. Callers (the predict command, the notebooks) decide what they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.download import default_data_dir

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"
READINGS_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")
REQUIRED_READINGS_COLUMNS = ("timestamp", "vm_id", "avg_cpu")
_READINGS_DTYPES = {"timestamp": "int64", "vm_id": str, "min_cpu": "float64", "max_cpu": "float64", "avg_cpu": "float64"}
_METADATA_STRING_COLUMNS = ("vm_id", "subscription_id", "deployment_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket")
INTERVAL_SECONDS = 300
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
STEPS_PER_HOUR = SECONDS_PER_HOUR // INTERVAL_SECONDS
HOLDOUT_HORIZON = 24
ACTIVE_VM_COUNT = "active_vm_count"
AGGREGATE_CPU = "aggregate_cpu_utilization"
CPU_MIN = 0.0
CPU_MAX = 100.0


@dataclass(frozen=True)
class ReadingsQuality:
    """Data-quality counts for a table of VM readings."""

    n_rows: int
    n_vms: int
    n_timestamps: int
    n_duplicate_vm_timestamps: int
    n_missing_vm_id: int
    n_missing_avg_cpu: int
    n_out_of_range_avg_cpu: int
    n_irregular_steps: int
    sorted_by_timestamp: bool
    vm_count_is_constant: bool


def require_columns(columns, required, name: str) -> None:
    """Raise ``ValueError`` naming every column in ``required`` that ``columns`` lacks."""
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _read_table(path: Path, required: tuple[str, ...], wanted: tuple[str, ...] | None, dtypes: dict) -> pd.DataFrame:
    """Read a CSV, check ``required`` columns exist, and keep only ``wanted`` ones if given."""
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    require_columns(header, required, path.name)
    usecols = [column for column in header if wanted is None or column in wanted]
    return pd.read_csv(path, usecols=usecols, dtype={c: t for c, t in dtypes.items() if c in usecols})


def load_readings(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the VM CPU readings in file order, keeping only the known columns.

    Needs ``timestamp``, ``vm_id`` and ``avg_cpu``; ``min_cpu`` and ``max_cpu`` are
    kept when present. Raises ``FileNotFoundError`` if the file is missing and
    ``ValueError`` for missing columns or values that are not numeric.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    return _read_table(root / READINGS_FILE, REQUIRED_READINGS_COLUMNS, READINGS_COLUMNS, _READINGS_DTYPES)


def load_metadata(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the VM metadata, keeping every column in the file.

    Only ``vm_id`` is required here; code that needs other columns should call
    ``require_columns``. Raises ``FileNotFoundError`` if the file is missing and
    ``ValueError`` if ``vm_id`` is missing or appears more than once.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    metadata = _read_table(root / METADATA_FILE, ("vm_id",), None, dict.fromkeys(_METADATA_STRING_COLUMNS, str))
    if metadata["vm_id"].duplicated().any():
        raise ValueError(f"{METADATA_FILE} lists some vm_id more than once")
    return metadata


def _valid_cpu(avg_cpu: pd.Series) -> pd.Series:
    """Mark readings that are present and within the 0 to 100 percent range."""
    return avg_cpu.between(CPU_MIN, CPU_MAX)


def check_readings(readings: pd.DataFrame, interval_seconds: int = INTERVAL_SECONDS) -> ReadingsQuality:
    """Count duplicates, missing ids and values, out-of-range values and irregular timestamp steps."""
    require_columns(readings.columns, REQUIRED_READINGS_COLUMNS, "readings")
    avg_cpu = readings["avg_cpu"]
    steps = np.diff(np.sort(readings["timestamp"].unique()))
    vms_per_timestamp = readings.groupby("timestamp")["vm_id"].nunique()
    return ReadingsQuality(
        n_rows=len(readings),
        n_vms=int(readings["vm_id"].nunique()),
        n_timestamps=int(readings["timestamp"].nunique()),
        n_duplicate_vm_timestamps=int(readings.duplicated(["vm_id", "timestamp"]).sum()),
        n_missing_vm_id=int(readings["vm_id"].isna().sum()),
        n_missing_avg_cpu=int(avg_cpu.isna().sum()),
        n_out_of_range_avg_cpu=int((avg_cpu.notna() & ~_valid_cpu(avg_cpu)).sum()),
        n_irregular_steps=int((steps != interval_seconds).sum()),
        sorted_by_timestamp=bool(readings["timestamp"].is_monotonic_increasing),
        vm_count_is_constant=bool(vms_per_timestamp.nunique() == 1),
    )


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Build ``active_vm_count`` and ``aggregate_cpu_utilization`` per timestamp.

    A reading is valid if it has a ``vm_id`` and ``avg_cpu`` is present and within
    [0, 100]; invalid readings are ignored, not filled. A VM with several valid readings at one
    timestamp counts once and contributes the mean of them. A timestamp whose
    readings are all invalid keeps its row with count 0 and a NaN aggregate.
    Timestamps absent from the readings are not invented. The result is sorted
    by timestamp and indexed by the raw timestamp values.
    """
    require_columns(readings.columns, REQUIRED_READINGS_COLUMNS, "readings")
    if readings.empty:
        raise ValueError("readings contain no rows")
    if readings["timestamp"].isna().any():
        raise ValueError("readings contain missing timestamps")
    valid = readings.loc[_valid_cpu(readings["avg_cpu"]) & readings["vm_id"].notna(), ["timestamp", "vm_id", "avg_cpu"]]
    per_vm = valid.groupby(["timestamp", "vm_id"], sort=False)["avg_cpu"].mean()
    by_timestamp = per_vm.groupby(level="timestamp")
    targets = pd.DataFrame({ACTIVE_VM_COUNT: by_timestamp.size(), AGGREGATE_CPU: by_timestamp.mean()})
    targets = targets.reindex(pd.Index(np.sort(readings["timestamp"].unique()), name="timestamp"))
    targets[ACTIVE_VM_COUNT] = targets[ACTIVE_VM_COUNT].fillna(0).astype("int64")
    return targets


def select_timestamps(readings: pd.DataFrame, timestamps) -> pd.DataFrame:
    """Keep only the readings whose timestamp is in ``timestamps`` (for example the training window)."""
    return readings[readings["timestamp"].isin(timestamps)]


def split_holdout(series: pd.DataFrame | pd.Series, horizon: int = HOLDOUT_HORIZON) -> tuple:
    """Split a chronological series into training rows and the final ``horizon`` rows.

    The split is by position from the end, so it works for any series length.
    Raises ``ValueError`` if ``horizon`` is not positive, the index is not sorted,
    or there are not more than ``horizon`` rows (at least one is left to train on).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series must be sorted chronologically")
    if len(series) <= horizon:
        raise ValueError(f"need more than {horizon} observations to hold out {horizon}, got {len(series)}")
    return series.iloc[:-horizon], series.iloc[-horizon:]
