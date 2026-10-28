from functools import partial

import numpy as np
import pandas as pd
import pytest
from conftest import HOURLY_PATTERN, make_series

from cwfc import models
from cwfc.data import INTERVAL_SECONDS, STEPS_PER_HOUR
from cwfc.evaluate import metrics_table, run_cv

FAST_FOREST = models.ForestParams(n_estimators=10)
ALL_FORECASTERS = {**models.candidate_forecasters(), **models.reference_forecasters()}


def test_baseline_forecast_continues_the_hourly_pattern(periodic_series):
    """On an exact hourly pattern the baseline repeats the pattern from the next slot on."""
    forecast = models.baseline_forecast(periodic_series, 24)
    expected = 10 + np.tile(HOURLY_PATTERN, 2)
    assert np.abs(forecast - expected).max() < 0.2
    assert np.allclose(forecast[:12], forecast[12:])


def test_baseline_forecast_follows_the_raw_timestamps_not_the_row_count(periodic_series):
    """Dropping the first three rows keeps each forecast step tied to its own position in the hour."""
    shifted = periodic_series.iloc[3:]
    expected = 10 + HOURLY_PATTERN
    assert np.abs(models.baseline_forecast(shifted, 12) - expected).max() < 0.2


def test_baseline_forecast_with_less_than_two_hours_is_the_mean():
    """Without enough history for a profile the baseline is the plain mean."""
    series = make_series(np.arange(10.0))
    assert np.allclose(models.baseline_forecast(series, 5), 4.5)


def test_baseline_forecast_of_a_constant_series_is_that_constant():
    """A constant history gives a constant forecast."""
    assert np.allclose(models.baseline_forecast(make_series(np.full(40, 3.0)), 24), 3.0)


@pytest.mark.parametrize("forecaster", ALL_FORECASTERS.values(), ids=ALL_FORECASTERS.keys())
def test_every_forecaster_rejects_a_bad_horizon_and_a_bad_history(forecaster, noisy_series):
    """All forecasters raise ValueError for a non-positive horizon, an empty history or NaN values."""
    with pytest.raises(ValueError):
        forecaster(noisy_series, 0)
    with pytest.raises(ValueError):
        forecaster(noisy_series.iloc[:0], 3)
    with pytest.raises(ValueError):
        forecaster(noisy_series.where(noisy_series.index != 600), 3)


@pytest.mark.parametrize("forecaster", ALL_FORECASTERS.values(), ids=ALL_FORECASTERS.keys())
def test_every_forecaster_returns_horizon_finite_values_and_leaves_the_history_alone(forecaster, noisy_series):
    """Each forecast has the requested length, is finite, and the input history is not modified."""
    before = noisy_series.copy()
    forecast = forecaster(noisy_series, 24)
    assert forecast.shape == (24,) and np.isfinite(forecast).all()
    pd.testing.assert_series_equal(noisy_series, before)


def test_persistence_repeats_the_last_value(noisy_series):
    """Persistence forecasts the last observation at every step."""
    assert np.allclose(models.persistence(noisy_series, 5), noisy_series.iloc[-1])


def test_seasonal_naive_repeats_the_last_hour(noisy_series):
    """Seasonal-naive repeats the last twelve observations in order."""
    forecast = models.seasonal_naive(noisy_series, 14)
    last_hour = noisy_series.to_numpy()[-STEPS_PER_HOUR:]
    assert np.allclose(forecast, np.resize(last_hour, 14))


def test_seasonal_naive_needs_one_hour_of_history():
    """Less than twelve observations cannot be repeated hourly."""
    with pytest.raises(ValueError, match="not enough history"):
        models.seasonal_naive(make_series(np.arange(5.0)), 3)


def test_history_mean_is_the_mean(noisy_series):
    """The history-mean reference is the mean of all observations."""
    assert np.allclose(models.history_mean(noisy_series, 4), noisy_series.mean())


def test_arima_forecast_tracks_a_noisy_hourly_series(noisy_series):
    """ARIMA with the profile beats the history mean on a short horizon of a persistent noisy series."""
    history, future = noisy_series.iloc[:84], noisy_series.iloc[84:90].to_numpy()
    arima_error = np.abs(models.arima_forecast(history, 6) - future).mean()
    mean_error = np.abs(models.history_mean(history, 6) - future).mean()
    assert arima_error < mean_error


def test_arima_forecast_without_the_profile_regressor_still_works(noisy_series):
    """The ARIMA fit also runs without the profile regressor and differs from the one with it."""
    with_profile = models.arima_forecast(noisy_series, 12)
    without_profile = models.arima_forecast(noisy_series, 12, use_profile=False)
    assert np.isfinite(without_profile).all() and not np.allclose(with_profile, without_profile)


def test_arima_forecast_falls_back_to_the_baseline_on_a_constant_series():
    """A constant series cannot be fitted, so the forecast is the baseline."""
    series = make_series(np.full(40, 5.0))
    assert np.allclose(models.arima_forecast(series, 6), models.baseline_forecast(series, 6))


def test_arima_forecast_falls_back_with_less_than_two_hours():
    """Short histories use the baseline instead of fitting ARIMA."""
    series = make_series(np.arange(15.0))
    assert np.allclose(models.arima_forecast(series, 4), models.baseline_forecast(series, 4))


def test_arima_forecast_falls_back_on_an_invalid_order(noisy_series):
    """An order that statsmodels rejects gives the baseline instead of an exception."""
    forecast = models.arima_forecast(noisy_series, 6, order=(-1, 0, 0))
    assert np.allclose(forecast, models.baseline_forecast(noisy_series, 6))


def test_arima_forecast_falls_back_when_the_fit_warns(monkeypatch, noisy_series):
    """A non-finite forecast from the fit is replaced by the baseline."""

    class FakeFit:
        """Stand-in for a fitted model that returns NaN."""

        def forecast(self, steps, exog=None):
            """Return NaN for every step."""
            return np.full(steps, np.nan)

    class FakeArima:
        """Stand-in for ARIMA whose fit returns the NaN model."""

        def __init__(self, *args, **kwargs):
            """Accept and ignore the usual arguments."""

        def fit(self):
            """Return the NaN model."""
            return FakeFit()

    monkeypatch.setattr(models, "ARIMA", FakeArima)
    assert np.allclose(models.arima_forecast(noisy_series, 5), models.baseline_forecast(noisy_series, 5))


def test_arima_residuals_have_the_expected_length(noisy_series):
    """The residuals drop the first `p` start-up values and are finite."""
    residuals = models.arima_residuals(noisy_series, order=(2, 0, 0))
    assert len(residuals) == len(noisy_series) - 2 and np.isfinite(residuals).all()


def test_forest_forecast_is_deterministic_and_close_to_the_baseline_on_a_clean_pattern(periodic_series):
    """The same call twice gives the same forecast, and an exact pattern leaves nothing for the forest to correct."""
    first = models.forest_forecast(periodic_series, 12, FAST_FOREST)
    second = models.forest_forecast(periodic_series, 12, FAST_FOREST)
    assert np.array_equal(first, second)
    assert np.abs(first - models.baseline_forecast(periodic_series, 12)).max() < 0.1


def test_forest_forecast_seed_changes_the_forecast(noisy_series):
    """A different seed gives a different forest, hence a different forecast."""
    a = models.forest_forecast(noisy_series, 6, models.ForestParams(n_estimators=10, seed=1))
    b = models.forest_forecast(noisy_series, 6, models.ForestParams(n_estimators=10, seed=2))
    assert not np.array_equal(a, b)


def test_forest_forecast_keeps_the_baseline_where_there_are_too_few_rows():
    """Steps whose training table has fewer than the minimum rows keep the baseline value."""
    rng = np.random.default_rng(1)
    series = make_series(10 + np.tile(HOURLY_PATTERN, 4)[:40] + rng.normal(scale=0.1, size=40))
    forecast = models.forest_forecast(series, 24, FAST_FOREST)
    baseline = models.baseline_forecast(series, 24)
    n_rows = [len(models.training_table(series, step)) for step in range(1, 25)]
    assert min(n_rows) < models.MIN_TRAINING_ROWS < max(n_rows)
    short = [i for i, rows in enumerate(n_rows) if rows < models.MIN_TRAINING_ROWS]
    assert np.allclose(forecast[short], baseline[short])


def test_forest_forecast_on_a_very_short_history_is_the_baseline():
    """With no usable rows at all the forest forecast is the baseline."""
    series = make_series(np.arange(10.0))
    assert np.allclose(models.forest_forecast(series, 3, FAST_FOREST), models.baseline_forecast(series, 3))


def test_forest_forecast_uses_only_the_chosen_features(noisy_series):
    """Different feature subsets give different forecasts."""
    small = models.forest_forecast(noisy_series, 6, models.ForestParams(n_estimators=10, features=("lag_1",)))
    full = models.forest_forecast(noisy_series, 6, models.ForestParams(n_estimators=10, features=models.FEATURE_NAMES))
    assert not np.allclose(small, full)


@pytest.mark.parametrize("features", [(), ("lag_1", "not_a_feature")])
def test_forest_params_reject_empty_or_unknown_features(features):
    """An empty feature list or an unknown feature name raises ValueError."""
    with pytest.raises(ValueError, match="non-empty subset"):
        models.ForestParams(features=features)


def test_forest_fit_error_compares_the_forest_with_the_baseline_in_sample(noisy_series):
    """The in-sample table has one row per usable step and the forest fits its own rows better than the baseline."""
    table = models.forest_fit_error(noisy_series, 6, FAST_FOREST)
    assert list(table.index) == [1, 2, 3, 4, 5, 6] and list(table.columns) == ["baseline", "forest"]
    assert (table["forest"] < table["baseline"]).all()


def test_forest_fit_error_skips_steps_with_too_few_rows():
    """Steps without enough rows are left out of the table."""
    series = make_series(10 + np.sin(np.arange(30)))
    table = models.forest_fit_error(series, 24, FAST_FOREST)
    assert 1 in table.index and 24 not in table.index


def test_forest_fit_error_rejects_a_bad_horizon(noisy_series):
    """A non-positive horizon raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        models.forest_fit_error(noisy_series, 0)


def test_forecaster_groups_have_the_expected_names():
    """The compared models and the references are returned under their report names."""
    assert list(models.candidate_forecasters()) == [models.BASELINE_NAME, models.ARIMA_NAME, models.FOREST_NAME]
    assert list(models.reference_forecasters()) == ["Persistence", "Seasonal-naive", "History mean"]


def test_forecasts_start_one_interval_after_the_history_ends():
    """The forecast for step 1 is for the slot right after the last timestamp, even for a late start."""
    series = pd.Series(10 + np.tile(HOURLY_PATTERN, 4), index=pd.Index(np.arange(48) * INTERVAL_SECONDS + 5 * INTERVAL_SECONDS))
    forecast = models.baseline_forecast(series, 1)
    next_slot = ((series.index[-1] + INTERVAL_SECONDS) // INTERVAL_SECONDS) % STEPS_PER_HOUR
    series_slot = (series.index // INTERVAL_SECONDS) % STEPS_PER_HOUR
    in_history = series[series_slot == next_slot].mean()
    assert abs(forecast[0] - in_history) < 0.2


def _noisy_hourly_series(phi: float, seed: int, n: int = 150) -> pd.Series:
    """Return a level of 10, the hourly pattern and AR(phi) noise with standard deviation 0.2 for ``n`` points."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(scale=0.2)
    return make_series(10 + np.tile(HOURLY_PATTERN, n // STEPS_PER_HOUR + 1)[:n] + noise)


def _cross_validated_mae(phi: float, horizon: int) -> pd.Series:
    """Return the rolling-origin MAE of the three models, averaged over three seeded series."""
    candidates = {
        models.BASELINE_NAME: models.baseline_forecast,
        models.ARIMA_NAME: models.candidate_forecasters()[models.ARIMA_NAME],
        models.FOREST_NAME: partial(models.forest_forecast, params=FAST_FOREST),
    }
    tables = [metrics_table(run_cv(_noisy_hourly_series(phi, seed), candidates, horizon=horizon, min_train=48, step=4))["MAE"] for seed in range(3)]
    return pd.concat(tables, axis=1).mean(axis=1)


def test_arima_and_the_forest_beat_the_baseline_when_the_noise_is_persistent():
    """When noise carries over between steps, the next step is forecast clearly better than by the baseline."""
    mae = _cross_validated_mae(phi=0.9, horizon=1)
    assert mae[models.ARIMA_NAME] < 0.8 * mae[models.BASELINE_NAME]
    assert mae[models.FOREST_NAME] < 0.95 * mae[models.BASELINE_NAME]


def test_arima_and_the_forest_do_not_invent_skill_on_pure_noise():
    """With nothing to learn beyond the hourly pattern, neither model is much worse than the baseline."""
    mae = _cross_validated_mae(phi=0.0, horizon=3)
    assert mae[models.ARIMA_NAME] < 1.05 * mae[models.BASELINE_NAME]
    assert mae[models.FOREST_NAME] < 1.05 * mae[models.BASELINE_NAME]


def test_arima_and_the_forest_move_towards_a_new_level_but_the_baseline_does_not():
    """After a level jump near the end of the history, the baseline stays at the old mean and both models move up."""
    rng = np.random.default_rng(2)
    values = 10 + rng.normal(scale=0.1, size=60)
    values[50:] += 3.0
    series = make_series(values)
    new_level = 13.0
    baseline_gap = abs(models.baseline_forecast(series, 6).mean() - new_level)
    assert abs(models.arima_forecast(series, 6).mean() - new_level) < 0.5 * baseline_gap
    assert abs(models.forest_forecast(series, 6, FAST_FOREST).mean() - new_level) < 0.7 * baseline_gap


@pytest.mark.parametrize("name", [models.BASELINE_NAME, models.ARIMA_NAME, models.FOREST_NAME])
def test_forecasts_follow_the_scale_of_the_values_and_ignore_a_shift_in_time(name, noisy_series):
    """A linear change of the values changes the forecast the same way, and moving all timestamps by whole steps changes nothing."""
    forecaster = {**models.candidate_forecasters(), models.FOREST_NAME: partial(models.forest_forecast, params=FAST_FOREST)}[name]
    history = noisy_series.iloc[:72]
    reference = forecaster(history, 12)
    assert np.allclose(forecaster(history * 7.0 + 3.0, 12), 7.0 * reference + 3.0, atol=1e-3)
    shifted = pd.Series(history.to_numpy(), index=history.index + 5 * INTERVAL_SECONDS)
    assert np.allclose(forecaster(shifted, 12), reference)
