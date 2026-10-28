import numpy as np
import pandas as pd
import pytest
from conftest import HOURLY_PATTERN, make_series

from cwfc import features as ft
from cwfc.data import INTERVAL_SECONDS


def test_position_in_hour_cycles_through_twelve_slots():
    """Timestamps five minutes apart give slots 0 to 11 and then start again."""
    stamps = np.arange(14) * INTERVAL_SECONDS
    assert ft.position_in_hour(stamps).tolist() == list(range(12)) + [0, 1]


def test_position_in_hour_depends_on_the_raw_timestamp_not_the_row_number():
    """A series that starts mid-hour gets the matching slot for its first row."""
    assert ft.position_in_hour(pd.Index([7 * INTERVAL_SECONDS]))[0] == 7


def test_position_in_hour_rejects_non_integer_timestamps():
    """Float or datetime timestamps raise instead of being silently misread as seconds."""
    with pytest.raises(ValueError, match="integers"):
        ft.position_in_hour(np.array([0.0, 300.0]))
    with pytest.raises(ValueError, match="integers"):
        ft.position_in_hour(pd.date_range("2026-01-01", periods=2, freq="5min"))


def test_future_timestamps_continue_one_interval_after_the_last():
    """The future timestamps start one interval after the last one and step by one interval."""
    assert ft.future_timestamps(900, 3).tolist() == [1200, 1500, 1800]


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "3"])
def test_horizon_must_be_a_positive_integer(horizon):
    """Zero, negative, boolean, float and string horizons raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        ft.future_timestamps(0, horizon)


def test_check_history_returns_float_values(periodic_series):
    """A valid history returns its values as floats."""
    assert ft.check_history(periodic_series).dtype == float


@pytest.mark.parametrize(
    "bad, message",
    [
        (pd.Series([], dtype=float), "empty"),
        (pd.Series([1.0, np.nan], index=[0, 300]), "finite"),
        (pd.Series([1.0, np.inf], index=[0, 300]), "finite"),
        (pd.Series([1.0, 2.0], index=[0.0, 300.0]), "integer"),
        (pd.Series([1.0, 2.0, 3.0], index=[0, 300, 900]), "evenly spaced"),
        (pd.Series([1.0, 2.0], index=[300, 300]), "evenly spaced"),
        (pd.Series([1.0, 2.0], index=[300, 0]), "evenly spaced"),
        ([1.0, 2.0], "Series"),
    ],
)
def test_check_history_rejects_unusable_input(bad, message):
    """Empty, non-finite, non-integer-index, gapped, duplicated, unsorted and non-Series input raises."""
    with pytest.raises(ValueError, match=message):
        ft.check_history(bad)


def test_check_history_accepts_a_single_point():
    """One observation is a valid history."""
    assert ft.check_history(pd.Series([5.0], index=[0])).tolist() == [5.0]


def test_hourly_profile_recovers_the_pattern(periodic_series):
    """The profile of an exact hourly pattern matches the pattern up to the small smoothing bias."""
    profile = ft.hourly_profile(periodic_series.to_numpy(), ft.position_in_hour(periodic_series.index))
    assert profile.index.tolist() == list(range(12))
    assert np.abs(profile.to_numpy() - HOURLY_PATTERN).max() < 0.15


def test_hourly_profile_is_empty_with_less_than_two_hours():
    """Less than 24 values give an empty profile, and the lookup then gives zeros."""
    values = np.arange(23.0)
    positions = ft.position_in_hour(np.arange(23) * INTERVAL_SECONDS)
    profile = ft.hourly_profile(values, positions)
    assert profile.empty and ft.profile_at(profile, [0, 5]).tolist() == [0.0, 0.0]


def test_hourly_profile_rejects_mismatched_inputs():
    """Values and positions of different lengths raise ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        ft.hourly_profile(np.arange(30.0), np.arange(29) % 12)


def test_hourly_profile_of_a_constant_series_is_zero():
    """A constant series has no hourly pattern."""
    values = np.full(48, 7.0)
    profile = ft.hourly_profile(values, ft.position_in_hour(np.arange(48) * INTERVAL_SECONDS))
    assert np.allclose(profile, 0.0)


def test_profile_at_uses_zero_for_positions_not_in_the_profile():
    """A position without a profile value gets zero."""
    profile = pd.Series({0: 1.5, 1: -0.5})
    assert ft.profile_at(profile, [1, 0, 7]).tolist() == [-0.5, 1.5, 0.0]


def test_feature_row_values_on_a_known_sequence():
    """Lags, the one-hour mean and the last change follow their definitions on 0, 1, 2, ..."""
    values = np.arange(30.0)
    row = ft.feature_row(values, 20, profile_value=0.25, target_position=3)
    expected = [20, 19, 18, 9, np.mean(np.arange(9, 21)), 1, 0.25, np.sin(2 * np.pi * 3 / 12), np.cos(2 * np.pi * 3 / 12)]
    assert dict(zip(ft.FEATURE_NAMES, row)) == pytest.approx(dict(zip(ft.FEATURE_NAMES, expected)))


def test_feature_row_needs_a_full_hour_of_history_and_a_valid_row():
    """A row before the twelfth observation or past the end raises ValueError."""
    values = np.arange(30.0)
    with pytest.raises(ValueError, match="row must be between"):
        ft.feature_row(values, ft.MIN_FEATURE_ROW - 1, 0.0, 0)
    with pytest.raises(ValueError, match="row must be between"):
        ft.feature_row(values, 30, 0.0, 0)
    assert len(ft.feature_row(values, ft.MIN_FEATURE_ROW, 0.0, 0)) == len(ft.FEATURE_NAMES)


def test_training_table_shape_index_and_columns(noisy_series):
    """The table has one row per usable time, the feature columns plus the target, indexed by row time."""
    table = ft.training_table(noisy_series, step=3)
    n_rows = len(noisy_series) - 3 - ft.MIN_FEATURE_ROW
    assert table.shape == (n_rows, len(ft.FEATURE_NAMES) + 1)
    assert list(table.columns) == [*ft.FEATURE_NAMES, ft.TARGET]
    assert table.index[0] == noisy_series.index[ft.MIN_FEATURE_ROW]
    assert table.index[-1] == noisy_series.index[len(noisy_series) - 4]


def test_training_table_target_is_the_correction_to_the_baseline(noisy_series):
    """With correction the target is the later value minus the history mean and the profile at its position."""
    values = noisy_series.to_numpy()
    profile = ft.hourly_profile(values, ft.position_in_hour(noisy_series.index))
    table = ft.training_table(noisy_series, step=2)
    row = ft.MIN_FEATURE_ROW
    position = ft.position_in_hour(noisy_series.index[[row + 2]])
    expected = values[row + 2] - values.mean() - ft.profile_at(profile, position)[0]
    assert table[ft.TARGET].iloc[0] == pytest.approx(expected)


def test_training_table_raw_target_is_the_later_value(noisy_series):
    """Without correction the target is the observation `step` steps later."""
    table = ft.training_table(noisy_series, step=4, correction=False)
    assert table[ft.TARGET].iloc[0] == noisy_series.iloc[ft.MIN_FEATURE_ROW + 4]


def test_training_table_is_empty_when_the_history_is_too_short():
    """A history shorter than a full hour plus the step gives no rows."""
    table = ft.training_table(make_series(np.arange(10.0)), step=2)
    assert table.empty and list(table.columns) == [*ft.FEATURE_NAMES, ft.TARGET]


def test_training_table_rejects_bad_step(noisy_series):
    """A step of zero or less raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        ft.training_table(noisy_series, step=0)


def test_correlation_with_target_has_one_column_per_step_and_one_row_per_feature(noisy_series):
    """The correlation table is features by steps, with values in [-1, 1]."""
    table = ft.correlation_with_target(noisy_series, steps=[1, 6, 12], correction=False)
    assert list(table.columns) == [1, 6, 12] and list(table.index) == list(ft.FEATURE_NAMES)
    assert table.abs().max().max() <= 1.0 + 1e-9


def test_correlation_with_target_is_high_for_the_latest_value_of_a_trend():
    """On a straight line the latest value predicts the next one almost perfectly."""
    table = ft.correlation_with_target(make_series(np.arange(60.0)), steps=[1], correction=False)
    assert table.loc["lag_1", 1] == pytest.approx(1.0)


def test_correlation_with_target_needs_enough_rows():
    """Too few rows to correlate raise ValueError naming the step."""
    with pytest.raises(ValueError, match="step 20"):
        ft.correlation_with_target(make_series(np.arange(30.0)), steps=[20])


def test_correlation_with_a_constant_feature_is_nan_not_an_error():
    """A constant series gives NaN correlations without raising or warning."""
    table = ft.correlation_with_target(make_series(np.full(50, 4.0)), steps=[1], correction=False)
    assert table[1].isna().all()

