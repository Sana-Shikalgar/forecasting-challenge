"""Past-only features for the models.

Every feature of a row is computed from the observations up to that row's time and
nothing later. Timestamps are integer seconds, and the position in the hour is the
five-minute slot (0 to 11) that a timestamp falls in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.data import INTERVAL_SECONDS, STEPS_PER_HOUR

FEATURE_NAMES = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "mean_last_hour",
    "last_change",
    "profile_at_target",
    "target_sin",
    "target_cos",
)
MIN_FEATURE_ROW = STEPS_PER_HOUR - 1
PROFILE_WINDOW = STEPS_PER_HOUR + 1
PROFILE_MIN_PERIODS = STEPS_PER_HOUR // 2 + 1
PROFILE_MIN_STEPS = 2 * STEPS_PER_HOUR
TARGET = "target"


def check_history(history: pd.Series) -> np.ndarray:
    """Return the values of a history series after checking it is usable for forecasting.

    The series must be non-empty and finite, with a strictly increasing integer index of
    timestamps in seconds spaced exactly one interval apart. Raises ``ValueError`` otherwise.
    """
    if not isinstance(history, pd.Series):
        raise ValueError("history must be a pandas Series indexed by timestamp")
    if history.empty:
        raise ValueError("history must not be empty")
    if history.index.dtype.kind not in "iu":
        raise ValueError("history index must hold integer timestamps in seconds")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("history must be finite")
    if not (np.diff(history.index.to_numpy()) == INTERVAL_SECONDS).all():
        raise ValueError(f"history timestamps must be evenly spaced {INTERVAL_SECONDS} seconds apart")
    return values


def check_horizon(horizon: int) -> None:
    """Raise ``ValueError`` unless ``horizon`` is a positive integer."""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")


def position_in_hour(timestamps) -> np.ndarray:
    """Return the five-minute slot within the hour (0 to 11) of each timestamp in seconds."""
    stamps = np.asarray(timestamps)
    if stamps.dtype.kind not in "iu":
        raise ValueError("timestamps must be integers in seconds")
    return (stamps // INTERVAL_SECONDS) % STEPS_PER_HOUR


def future_timestamps(last_timestamp: int, horizon: int) -> np.ndarray:
    """Return the timestamps of the ``horizon`` steps after ``last_timestamp``."""
    check_horizon(horizon)
    return last_timestamp + INTERVAL_SECONDS * np.arange(1, horizon + 1)


def hourly_profile(values, positions) -> pd.Series:
    """Return the average deviation from a one-hour centered rolling mean, by position in the hour.

    Uses only the given values. With less than two hours of values the profile is empty,
    and ``profile_at`` then gives zero for every position.
    """
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions)
    if values.shape != positions.shape:
        raise ValueError("values and positions must have the same shape")
    if len(values) < PROFILE_MIN_STEPS:
        return pd.Series(dtype=float)
    trend = pd.Series(values).rolling(PROFILE_WINDOW, center=True, min_periods=PROFILE_MIN_PERIODS).mean().to_numpy()
    return pd.Series(values - trend).groupby(positions).mean()


def profile_at(profile: pd.Series, positions) -> np.ndarray:
    """Look up the profile at each position, using zero where the profile has no value."""
    return profile.reindex(np.asarray(positions)).fillna(0.0).to_numpy()


def feature_row(values: np.ndarray, row: int, profile_value: float, target_position: int) -> np.ndarray:
    """Return the features at index ``row`` of ``values`` for a target ``target_position`` in the hour.

    Lag k is the k-th most recent observation at ``row`` (lag 1 is the observation at ``row``),
    so lag 12 lies 11 steps before ``row``: one hour before the first forecast step.
    Only values up to and including ``row`` are read. Raises ``ValueError`` if ``row`` is too
    early to hold a full hour of history.
    """
    if row < MIN_FEATURE_ROW or row >= len(values):
        raise ValueError(f"row must be between {MIN_FEATURE_ROW} and {len(values) - 1}")
    angle = 2 * np.pi * target_position / STEPS_PER_HOUR
    return np.array(
        [
            values[row],
            values[row - 1],
            values[row - 2],
            values[row - STEPS_PER_HOUR + 1],
            values[row - STEPS_PER_HOUR + 1 : row + 1].mean(),
            values[row] - values[row - 1],
            profile_value,
            np.sin(angle),
            np.cos(angle),
        ]
    )


def training_table(history: pd.Series, step: int, correction: bool = True) -> pd.DataFrame:
    """Build the feature rows and target for forecasting ``step`` steps ahead from the history alone.

    Each row is a time ``t`` in the history with the features known at ``t``, and the target
    is the observation ``step`` steps later. With ``correction`` the target is that value minus
    the history mean and the hourly profile at the target position, which is what the forest
    learns. The result is empty if the history is too short for a full hour of features.
    """
    values = check_history(history)
    check_horizon(step)
    positions = position_in_hour(history.index)
    mean = values.mean()
    profile = hourly_profile(values, positions)
    rows = range(MIN_FEATURE_ROW, len(values) - step)
    features, targets = [], []
    for row in rows:
        target_position = positions[row + step]
        profile_value = profile_at(profile, [target_position])[0]
        features.append(feature_row(values, row, profile_value, target_position))
        targets.append(values[row + step] - (mean + profile_value if correction else 0.0))
    table = pd.DataFrame(features, columns=list(FEATURE_NAMES), index=history.index[list(rows)])
    table[TARGET] = targets
    return table


def correlation_with_target(history: pd.Series, steps, correction: bool = True) -> pd.DataFrame:
    """Return the correlation of each feature with the target, one column per forecast step."""
    columns = {}
    for step in steps:
        table = training_table(history, step, correction)
        if len(table) < 3:
            raise ValueError(f"not enough rows to correlate at step {step}")
        columns[step] = table[list(FEATURE_NAMES)].corrwith(table[TARGET])
    return pd.DataFrame(columns)
