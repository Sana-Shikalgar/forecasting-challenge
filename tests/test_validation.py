import pandas as pd
import pytest

from cwfc.validation import origin_positions, rolling_origin_splits


def _frame(n=20):
    """Build a frame with n hourly timestamps."""
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")})


def test_rolling_origin_splits_are_temporal():
    """Each split trains before it validates, and validation windows move forward."""
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_default_step_equals_the_horizon_and_windows_do_not_overlap():
    """With no step, validation windows tile the series without overlap."""
    splits = rolling_origin_splits(_frame(), horizon=4, min_train_size=8)
    assert len(splits) == 3
    assert all(a.validation_end < b.validation_start for a, b in zip(splits, splits[1:]))


def test_small_step_gives_overlapping_windows_and_an_expanding_train():
    """A step below the horizon gives more splits, with the training window growing each time."""
    splits = rolling_origin_splits(_frame(), horizon=4, min_train_size=8, step=1)
    assert len(splits) == 9
    assert all(a.train_end < b.train_end for a, b in zip(splits, splits[1:]))


def test_last_split_ends_exactly_at_the_last_timestamp_when_it_fits():
    """When the series fits whole horizons, the last validation window ends at the final timestamp."""
    frame = _frame(16)
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert splits[-1].validation_end == frame["timestamp"].iloc[-1]


def test_no_split_when_the_series_is_too_short():
    """A series shorter than the minimum training size plus one horizon gives no splits."""
    assert rolling_origin_splits(_frame(11), horizon=4, min_train_size=8) == []
    assert rolling_origin_splits(_frame(5), horizon=4, min_train_size=8) == []


def test_exactly_one_split_when_the_series_just_fits():
    """Minimum training size plus one horizon gives exactly one split."""
    assert len(rolling_origin_splits(_frame(12), horizon=4, min_train_size=8)) == 1


@pytest.mark.parametrize("step", [0, -1])
def test_non_positive_step_raises_instead_of_looping_forever(step):
    """A step of zero or less raises ValueError (zero used to loop forever)."""
    with pytest.raises(ValueError, match="step must be positive"):
        rolling_origin_splits(_frame(), horizon=4, min_train_size=8, step=step)


@pytest.mark.parametrize("horizon, min_train", [(0, 8), (-1, 8), (4, 0), (4, -2)])
def test_non_positive_horizon_or_minimum_train_raises(horizon, min_train):
    """A horizon or minimum training size of zero or less raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        rolling_origin_splits(_frame(), horizon=horizon, min_train_size=min_train)


def test_unsorted_timestamps_raise():
    """Timestamps out of order raise ValueError."""
    frame = _frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        rolling_origin_splits(frame, horizon=4, min_train_size=8)


def test_missing_timestamp_column_raises_key_error():
    """A frame without the timestamp column raises KeyError."""
    with pytest.raises(KeyError):
        rolling_origin_splits(pd.DataFrame({"t": [1, 2]}), horizon=1, min_train_size=1)


def test_origin_positions_default_step_is_the_horizon():
    """Origins start at the minimum training size and move one horizon at a time."""
    assert origin_positions(20, horizon=4, min_train_size=8) == [8, 12, 16]


def test_origin_positions_with_step_one_gives_every_possible_origin():
    """A step of one gives every origin that still has a full horizon ahead of it."""
    assert origin_positions(110, horizon=24, min_train_size=48, step=1) == list(range(48, 87))


def test_origin_positions_is_empty_when_the_series_is_too_short():
    """No origin exists when the minimum training size plus one horizon exceeds the series length."""
    assert origin_positions(11, horizon=4, min_train_size=8) == []
    assert origin_positions(0, horizon=4, min_train_size=8) == []


@pytest.mark.parametrize("horizon, min_train, step", [(0, 8, None), (4, 0, None), (4, 8, 0), (4, 8, -2)])
def test_origin_positions_rejects_non_positive_settings(horizon, min_train, step):
    """A non-positive horizon, minimum training size or step raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        origin_positions(20, horizon=horizon, min_train_size=min_train, step=step)


def test_rolling_origin_splits_agree_with_the_origin_positions():
    """Each split's training window ends right before the origin position it comes from."""
    frame = _frame()
    positions = origin_positions(len(frame), horizon=4, min_train_size=8, step=2)
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=2)
    assert [frame["timestamp"].iloc[p - 1] for p in positions] == [s.train_end for s in splits]
