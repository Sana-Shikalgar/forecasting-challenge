from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def origin_positions(n_obs: int, horizon: int, min_train_size: int, step: int | None = None) -> list[int]:
    """Return the number of history points at each rolling origin (an expanding window).

    The first origin has ``min_train_size`` points and each next one ``step`` more (default
    ``horizon``), as long as ``horizon`` further points exist to forecast. The list is empty
    if the series is too short. Raises ``ValueError`` if ``horizon``, ``min_train_size`` or
    ``step`` is not positive.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    if step is not None and step <= 0:
        raise ValueError("step must be positive")
    return list(range(min_train_size, n_obs - horizon + 1, horizon if step is None else step))


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence. Raises ``ValueError`` if ``horizon``,
    ``min_train_size`` or ``step`` is not positive, or the timestamps are not sorted.
    """
    positions = origin_positions(len(frame), horizon, min_train_size, step)
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    return [TimeSplit(ts.iloc[end - 1], ts.iloc[end], ts.iloc[end + horizon - 1]) for end in positions]
