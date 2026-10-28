"""Forecasters for one series: a baseline, an ARIMA model and a random forest.

Every forecaster has the same signature, ``forecaster(history, horizon) -> array``. The
history is a series indexed by raw timestamps in seconds, and the result holds the
``horizon`` forecasts that follow its last timestamp. A forecaster reads only the history it
is given, so it cannot see later data. Models that cannot be fitted on a short or awkward
history fall back to the baseline instead of failing.

Multi-step strategy: the forest is direct (a separate model for each step ahead, trained on the
target at that step), so errors are not fed back into later steps and no future inputs are needed.
ARIMA forecasts its horizon by iterating its own one-step equation, with the hourly profile as a
known future input.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

from cwfc.baselines import persistence_forecast, seasonal_naive_forecast
from cwfc.data import STEPS_PER_HOUR
from cwfc.features import (
    FEATURE_NAMES,
    PROFILE_MIN_STEPS,
    TARGET,
    check_history,
    check_horizon,
    feature_row,
    future_timestamps,
    hourly_profile,
    position_in_hour,
    profile_at,
    training_table,
)

MIN_TRAINING_ROWS = 8
FOREST_FEATURES = ("lag_1", "lag_12", "mean_last_hour")
BASELINE_NAME = "Baseline"
ARIMA_NAME = "ARIMA"
FOREST_NAME = "Random forest"
Forecaster = Callable[[pd.Series, int], np.ndarray]


@dataclass(frozen=True)
class ForestParams:
    """Settings of the random forest."""

    n_estimators: int = 100
    min_samples_leaf: int | float = 0.15
    max_depth: int | None = None
    max_features: float = 1.0
    features: tuple[str, ...] = FOREST_FEATURES
    seed: int = 0

    def __post_init__(self) -> None:
        """Raise ``ValueError`` if no feature is given or a feature name is unknown."""
        unknown = [name for name in self.features if name not in FEATURE_NAMES]
        if not self.features or unknown:
            raise ValueError(f"features must be a non-empty subset of {FEATURE_NAMES}, got {self.features}")


ARIMA_ORDER = (1, 0, 0)
FOREST_PARAMS = ForestParams()


def _baseline_parts(history: pd.Series, horizon: int) -> tuple[np.ndarray, float, pd.Series, np.ndarray]:
    """Return the history values, their mean, the hourly profile and the future positions in the hour."""
    values = check_history(history)
    check_horizon(horizon)
    profile = hourly_profile(values, position_in_hour(history.index))
    future_positions = position_in_hour(future_timestamps(history.index[-1], horizon))
    return values, float(values.mean()), profile, future_positions


def baseline_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast the history mean plus the average hourly profile at each future position in the hour.

    With less than two hours of history the profile is empty and this is the plain history mean.
    """
    _, mean, profile, future_positions = _baseline_parts(history, horizon)
    return mean + profile_at(profile, future_positions)


def persistence(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast the last observed value for every step (reference at short horizons)."""
    check_history(history)
    check_horizon(horizon)
    frame = history.to_frame("value")
    return persistence_forecast(frame, pd.DataFrame(index=range(horizon)), "value").to_numpy()


def seasonal_naive(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast by repeating the last hour of observations (reference)."""
    check_history(history)
    check_horizon(horizon)
    frame = history.to_frame("value")
    return seasonal_naive_forecast(frame, pd.DataFrame(index=range(horizon)), "value", STEPS_PER_HOUR).to_numpy()


def history_mean(history: pd.Series, horizon: int) -> np.ndarray:
    """Forecast the mean of the whole history for every step (reference)."""
    values = check_history(history)
    check_horizon(horizon)
    return np.full(horizon, values.mean())


def _arima_inputs(history: pd.Series, horizon: int, use_profile: bool):
    """Return the history values and, when used, the hourly profile as an extra regressor for history and future."""
    values, _, profile, future_positions = _baseline_parts(history, horizon)
    if not use_profile:
        return values, None, None
    return values, profile_at(profile, position_in_hour(history.index))[:, None], profile_at(profile, future_positions)[:, None]


def arima_forecast(history: pd.Series, horizon: int, order: tuple[int, int, int] = ARIMA_ORDER, use_profile: bool = True) -> np.ndarray:
    """Forecast with an ARIMA model of the given order and a constant, optionally with the hourly profile as a regressor.

    The baseline is returned if the history is shorter than two hours, or if the fit fails,
    raises a warning (for example non-convergence) or gives a non-finite forecast.
    """
    values, exog, exog_future = _arima_inputs(history, horizon, use_profile)
    fallback = baseline_forecast(history, horizon)
    if len(values) < PROFILE_MIN_STEPS:
        return fallback
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted = ARIMA(values, exog=exog, order=order, trend="c").fit()
            forecast = np.asarray(fitted.forecast(horizon, exog=exog_future), dtype=float)
    except (ValueError, np.linalg.LinAlgError, Warning):
        return fallback
    return forecast if np.isfinite(forecast).all() else fallback


def arima_residuals(history: pd.Series, order: tuple[int, int, int] = ARIMA_ORDER, use_profile: bool = True) -> np.ndarray:
    """Return the in-sample one-step residuals of the ARIMA fit, without the first ``order[0]`` start-up values."""
    values, exog, _ = _arima_inputs(history, 1, use_profile)
    fitted = ARIMA(values, exog=exog, order=order, trend="c").fit()
    return np.asarray(fitted.resid, dtype=float)[order[0] :]


def _make_forest(params: ForestParams) -> RandomForestRegressor:
    """Build an unfitted forest from its settings."""
    return RandomForestRegressor(
        n_estimators=params.n_estimators,
        min_samples_leaf=params.min_samples_leaf,
        max_depth=params.max_depth,
        max_features=params.max_features,
        random_state=params.seed,
        n_jobs=1,
    )


def _fit_step(history: pd.Series, step: int, params: ForestParams):
    """Fit the forest for one forecast step, returning it with its training table, or None if there are too few rows."""
    table = training_table(history, step)
    if len(table) < MIN_TRAINING_ROWS:
        return None, table
    return _make_forest(params).fit(table[list(params.features)], table[TARGET]), table


def forest_forecast(history: pd.Series, horizon: int, params: ForestParams = FOREST_PARAMS) -> np.ndarray:
    """Forecast with one random forest per step, each learning the correction to the baseline.

    The forecast at a step is the baseline plus the forest's predicted correction. A step
    with fewer than ``MIN_TRAINING_ROWS`` training rows keeps the baseline value.
    """
    values, _, profile, future_positions = _baseline_parts(history, horizon)
    forecast = baseline_forecast(history, horizon)
    last_position = len(values) - 1
    for step in range(1, horizon + 1):
        forest, _ = _fit_step(history, step, params)
        if forest is None:
            continue
        origin = feature_row(values, last_position, profile_at(profile, future_positions[step - 1 : step])[0], future_positions[step - 1])
        origin_frame = pd.DataFrame([origin], columns=list(FEATURE_NAMES))[list(params.features)]
        forecast[step - 1] += forest.predict(origin_frame)[0]
    return forecast


def forest_fit_error(history: pd.Series, horizon: int, params: ForestParams = FOREST_PARAMS) -> pd.DataFrame:
    """Return the in-sample mean absolute error of the baseline and the forest at each step.

    This is the error on the rows the forest was trained on, used to compare with the
    out-of-sample error and so to see over- or underfitting. Steps with too few rows are skipped.
    """
    check_history(history)
    check_horizon(horizon)
    rows = {}
    for step in range(1, horizon + 1):
        forest, table = _fit_step(history, step, params)
        if forest is None:
            continue
        fitted = forest.predict(table[list(params.features)])
        rows[step] = {"baseline": np.abs(table[TARGET]).mean(), "forest": np.abs(table[TARGET] - fitted).mean()}
    return pd.DataFrame.from_dict(rows, orient="index")


def candidate_forecasters() -> dict[str, Forecaster]:
    """Return the three models that are compared: baseline, ARIMA and random forest."""
    return {
        BASELINE_NAME: baseline_forecast,
        ARIMA_NAME: partial(arima_forecast, order=ARIMA_ORDER),
        FOREST_NAME: partial(forest_forecast, params=FOREST_PARAMS),
    }


def reference_forecasters() -> dict[str, Forecaster]:
    """Return the simple reference forecasters: persistence, seasonal-naive and history mean."""
    return {"Persistence": persistence, "Seasonal-naive": seasonal_naive, "History mean": history_mean}
