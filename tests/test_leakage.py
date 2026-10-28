from functools import partial

import numpy as np
import pytest
from conftest import make_series

from cwfc import features as ft
from cwfc import models
from cwfc.evaluate import run_cv

FAST_FOREST = models.ForestParams(n_estimators=10)
FORECASTERS = {
    "baseline": models.baseline_forecast,
    "arima": partial(models.arima_forecast, order=models.ARIMA_ORDER),
    "forest": partial(models.forest_forecast, params=FAST_FOREST),
    "persistence": models.persistence,
    "seasonal-naive": models.seasonal_naive,
    "history mean": models.history_mean,
}
HORIZON = 6
CHANGE_AT = 60


def _pair():
    """Return two series that are identical before position CHANGE_AT and very different after it."""
    rng = np.random.default_rng(3)
    base = 10 + np.sin(2 * np.pi * np.arange(72) / 12) + rng.normal(scale=0.2, size=72)
    changed = base.copy()
    changed[CHANGE_AT:] += 25.0
    return make_series(base), make_series(changed)


def test_feature_row_ignores_everything_after_its_row():
    """Changing values after a row leaves the features of that row unchanged."""
    values = np.random.default_rng(0).normal(size=40)
    altered = values.copy()
    altered[26:] += 100.0
    row = 25
    assert np.array_equal(ft.feature_row(values, row, 0.1, 4), ft.feature_row(altered, row, 0.1, 4))


def test_hourly_profile_depends_only_on_the_values_it_is_given():
    """A profile built from a history prefix is the same whatever comes after the prefix."""
    series, changed = _pair()
    prefix = series.iloc[:CHANGE_AT]
    same_prefix = changed.iloc[:CHANGE_AT]
    positions = ft.position_in_hour(prefix.index)
    assert ft.hourly_profile(prefix.to_numpy(), positions).equals(ft.hourly_profile(same_prefix.to_numpy(), positions))


@pytest.mark.parametrize("name", FORECASTERS)
def test_a_forecast_from_a_history_does_not_depend_on_later_values(name):
    """For every forecaster, forecasts made at origins before the change are identical on both series."""
    series, changed = _pair()
    before = run_cv(series, {name: FORECASTERS[name]}, horizon=HORIZON, min_train=48)
    after = run_cv(changed, {name: FORECASTERS[name]}, horizon=HORIZON, min_train=48)
    safe = before.origins <= CHANGE_AT
    assert safe.sum() >= 10
    assert np.array_equal(before.predictions[name][safe], after.predictions[name][safe])


@pytest.mark.parametrize("name", ["baseline", "persistence", "history mean"])
def test_forecasts_after_the_change_do_use_the_new_history(name):
    """The leakage check is not vacuous: forecasts at origins after the change do move."""
    series, changed = _pair()
    before = run_cv(series, {name: FORECASTERS[name]}, horizon=HORIZON, min_train=48)
    after = run_cv(changed, {name: FORECASTERS[name]}, horizon=HORIZON, min_train=48)
    late = before.origins > CHANGE_AT
    assert not np.array_equal(before.predictions[name][late], after.predictions[name][late])


def test_training_rows_never_use_a_target_beyond_the_history():
    """Every training target of a table is an observation inside the history, none beyond its end."""
    series, _ = _pair()
    history = series.iloc[:CHANGE_AT]
    table = ft.training_table(history, step=6, correction=False)
    assert table[ft.TARGET].isin(history.to_numpy()).all()
    assert table[ft.TARGET].index.max() + 6 * 300 <= history.index.max()
