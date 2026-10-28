import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast, seasonal_naive_forecast


def _train(values):
    """Build a training frame with one target column."""
    return pd.DataFrame({"y": values}, index=pd.RangeIndex(len(values)) * 300)


def _template(n, start=10_000):
    """Build a template frame whose index is the future timestamps."""
    return pd.DataFrame(index=pd.RangeIndex(n) * 300 + start)


def test_persistence_repeats_last_value_on_the_template_index():
    """Every future row gets the last observed value and keeps the template index."""
    template = _template(4)
    forecast = persistence_forecast(_train([1.0, 2.0, 7.5]), template, "y")
    assert forecast.tolist() == [7.5] * 4 and forecast.index.equals(template.index)


def test_persistence_with_empty_template_gives_empty_forecast():
    """An empty template gives an empty forecast."""
    assert persistence_forecast(_train([1.0]), _template(0), "y").empty


def test_persistence_rejects_empty_train_and_missing_last_value():
    """An empty training frame or a NaN last value raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        persistence_forecast(_train([]), _template(2), "y")
    with pytest.raises(ValueError, match="missing"):
        persistence_forecast(_train([1.0, np.nan]), _template(2), "y")


def test_persistence_unknown_target_column_raises_key_error():
    """A target column that does not exist raises KeyError."""
    with pytest.raises(KeyError):
        persistence_forecast(_train([1.0]), _template(1), "nope")


def test_seasonal_naive_repeats_the_last_season():
    """The last season repeats, wrapping around for horizons longer than one season."""
    forecast = seasonal_naive_forecast(_train([9.0, 1.0, 2.0, 3.0]), _template(7), "y", 3)
    assert forecast.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]


def test_seasonal_naive_works_when_history_equals_one_season():
    """Exactly one season of history is enough."""
    assert seasonal_naive_forecast(_train([4.0, 5.0]), _template(3), "y", 2).tolist() == [4.0, 5.0, 4.0]


def test_seasonal_naive_season_of_one_is_persistence():
    """A season length of one repeats the last value."""
    assert seasonal_naive_forecast(_train([1.0, 2.0, 3.0]), _template(3), "y", 1).tolist() == [3.0] * 3


def test_seasonal_naive_ignores_missing_values_before_the_last_season():
    """A NaN earlier than the last season does not matter."""
    assert seasonal_naive_forecast(_train([np.nan, 1.0, 2.0]), _template(2), "y", 2).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("season", [0, -3])
def test_seasonal_naive_rejects_non_positive_season(season):
    """A season length of zero or less raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        seasonal_naive_forecast(_train([1.0, 2.0]), _template(2), "y", season)


def test_seasonal_naive_rejects_short_history_and_missing_last_season():
    """Less history than one season, or a NaN in the last season, raises ValueError."""
    with pytest.raises(ValueError, match="not enough history"):
        seasonal_naive_forecast(_train([1.0]), _template(2), "y", 2)
    with pytest.raises(ValueError, match="missing"):
        seasonal_naive_forecast(_train([1.0, np.nan]), _template(2), "y", 2)
