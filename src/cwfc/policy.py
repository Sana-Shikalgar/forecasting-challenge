"""Error analysis, prediction intervals and a headroom policy built from rolling-origin forecast errors.

An error is the actual value minus the forecast, so a positive error means the forecast was too
low. Intervals and margins are calibrated only on errors that were already known when the
forecast was issued: the error of a forecast made at an earlier origin at a given step is known
once that step's value has been observed. This keeps the simulation honest, and it means the
early origins have little or nothing to calibrate on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.data import INTERVAL_SECONDS
from cwfc.evaluate import HIGH_LOAD_QUANTILE, CVResult
from cwfc.features import position_in_hour

MIN_CALIBRATION_ERRORS = 8
UNDER_PROVISIONING_COST = 5.0
MARGIN_PROBABILITIES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
TRIGGER_RECENT_STEPS = 3  # the peak-risk trigger looks at the latest 15 minutes
HIGH_LOAD_TARGET_SHARE = 0.05  # at most this share of high-load values may exceed the planned capacity


def forecast_errors(result: CVResult, name: str) -> np.ndarray:
    """Return actual minus forecast for one forecaster, one row per origin and one column per step."""
    if name not in result.predictions:
        raise ValueError(f"unknown forecaster {name!r}")
    return result.actual - result.predictions[name]


def _check_probabilities(probabilities) -> np.ndarray:
    """Return the probabilities as an array after checking that each lies strictly between 0 and 1."""
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or ((values <= 0) | (values >= 1)).any():
        raise ValueError("probabilities must be non-empty and strictly between 0 and 1")
    return values


def realised_error_quantiles(errors, origins, probabilities, min_errors: int = MIN_CALIBRATION_ERRORS, absolute: bool = False) -> np.ndarray:
    """Return, for the forecast issued at each origin, the error quantiles at each step that were known by then.

    The result has shape (origins, steps, probabilities). The error of the forecast made at
    history length ``h`` for step ``k`` is known once ``h + k`` is no later than the current
    history length. Entries with fewer than ``min_errors`` known errors are NaN. With ``absolute``
    the quantiles are of the absolute errors, which do not depend on whether the forecast has
    recently been too high or too low.
    """
    errors = np.abs(np.asarray(errors, dtype=float)) if absolute else np.asarray(errors, dtype=float)
    origins = np.asarray(origins)
    probabilities = _check_probabilities(probabilities)
    if errors.ndim != 2 or len(origins) != errors.shape[0]:
        raise ValueError("errors must be a 2-D array with one row per origin")
    if min_errors < 1:
        raise ValueError("min_errors must be at least 1")
    quantiles = np.full((errors.shape[0], errors.shape[1], len(probabilities)), np.nan)
    for current, history_length in enumerate(origins):
        for step in range(1, errors.shape[1] + 1):
            known = origins + step <= history_length
            if known.sum() >= min_errors:
                quantiles[current, step - 1] = np.quantile(errors[known, step - 1], probabilities)
    return quantiles


def realised_margins(errors, origins, probabilities, min_errors: int = MIN_CALIBRATION_ERRORS) -> np.ndarray:
    """Return the headroom margins: the realised error quantiles, never below zero, with NaN where there is too little to calibrate on.

    Headroom adds capacity to the forecast; a negative quantile (the forecast has recently been
    too high) means no margin is needed, not that less than the forecast should be planned.
    """
    return np.maximum(realised_error_quantiles(errors, origins, probabilities, min_errors), 0.0)


def final_error_quantiles(errors, probabilities, absolute: bool = False) -> np.ndarray:
    """Return the quantiles of all errors (or of the absolute errors) at each step, for a forecast issued after every error is known.

    The result has shape (steps, probabilities).
    """
    errors = np.abs(np.asarray(errors, dtype=float)) if absolute else np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[0] == 0:
        raise ValueError("errors must be a non-empty 2-D array")
    return np.quantile(errors, _check_probabilities(probabilities), axis=0).T


def coverage(actual, forecast, lower, upper) -> tuple[float, int]:
    """Return the share of actual values inside ``forecast + lower`` to ``forecast + upper``, and how many values were checked.

    Entries where either bound is NaN are skipped. The share is NaN if nothing was checked.
    """
    actual, forecast, lower, upper = (np.asarray(a, dtype=float) for a in (actual, forecast, lower, upper))
    if not actual.shape == forecast.shape == lower.shape == upper.shape:
        raise ValueError("actual, forecast, lower and upper must have the same shape")
    checked = np.isfinite(lower) & np.isfinite(upper)
    if not checked.any():
        return float("nan"), 0
    inside = (actual >= forecast + lower) & (actual <= forecast + upper)
    return float(inside[checked].mean()), int(checked.sum())


def coverage_by_step(actual, forecast, lower, upper) -> pd.DataFrame:
    """Return, for each step, the number of checked values, the coverage and the mean interval width."""
    actual, forecast, lower, upper = (np.asarray(a, dtype=float) for a in (actual, forecast, lower, upper))
    rows = {}
    for step in range(actual.shape[1]):
        share, count = coverage(actual[:, step], forecast[:, step], lower[:, step], upper[:, step])
        rows[step + 1] = {"checked": count, "coverage": share, "mean width": float(np.nanmean(upper[:, step] - lower[:, step])) if count else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("step")


def peak_risk_trigger(values, origins, recent: int = TRIGGER_RECENT_STEPS) -> np.ndarray:
    """Return, for each origin, whether the mean of the latest ``recent`` observations is at or above the 80th percentile of the history.

    Only the ``origins`` first observations are read for an origin, so the trigger is known when the forecast is issued.
    """
    values = np.asarray(values, dtype=float)
    origins = np.asarray(origins)
    if recent < 1:
        raise ValueError("recent must be at least 1")
    if (origins < recent).any() or (origins > len(values)).any():
        raise ValueError("every origin must lie between recent and the number of values")
    return np.array([values[h - recent : h].mean() >= np.quantile(values[:h], HIGH_LOAD_QUANTILE) for h in origins])


def apply_trigger(margins, trigger) -> np.ndarray:
    """Keep the margins of the origins where ``trigger`` is true and use zero for the others; missing margins stay missing."""
    margins = np.asarray(margins, dtype=float)
    trigger = np.asarray(trigger, dtype=bool)
    if margins.ndim != 2 or trigger.shape != (margins.shape[0],):
        raise ValueError("margins must be 2-D with one trigger value per origin")
    return np.where(trigger[:, None], margins, np.where(np.isfinite(margins), 0.0, np.nan))


def headroom_curve(actual, forecast, margins, labels, high_load=None) -> pd.DataFrame:
    """Simulate capacity = forecast + margin for each margin option and summarise the outcome.

    ``margins`` has shape (options, origins, steps), one array of margins per label. Only values
    where every option has a margin are used, so the options are compared on the same values.
    The columns are the share of values where demand exceeded capacity, the mean shortfall (demand
    above capacity), the mean excess (idle capacity) and the mean margin, all in the units of the
    series. With a ``high_load`` mask (same shape as ``actual``) the share of high-load values
    that exceeded capacity is added, and is NaN if no used value is high-load.
    """
    actual, forecast, margins = (np.asarray(a, dtype=float) for a in (actual, forecast, margins))
    if margins.ndim != 3 or margins.shape[0] != len(labels) or margins.shape[1:] != actual.shape or forecast.shape != actual.shape:
        raise ValueError("margins must have shape (options, origins, steps) matching actual and forecast")
    usable = np.isfinite(margins).all(axis=0)
    if not usable.any():
        raise ValueError("no value has a margin for every option")
    peaks = None
    if high_load is not None:
        high_load = np.asarray(high_load, dtype=bool)
        if high_load.shape != actual.shape:
            raise ValueError("high_load must have the same shape as actual")
        peaks = usable & high_load
    rows = {}
    for label, margin in zip(labels, margins):
        capacity = (forecast + margin)[usable]
        demand = actual[usable]
        rows[label] = {
            "under-provisioned share": float((demand > capacity).mean()),
            "mean shortfall": float(np.maximum(demand - capacity, 0).mean()),
            "mean excess": float(np.maximum(capacity - demand, 0).mean()),
            "mean margin": float(margin[usable].mean()),
        }
        if peaks is not None:
            rows[label]["high-load under-provisioned share"] = float((actual[peaks] > (forecast + margin)[peaks]).mean()) if peaks.any() else float("nan")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.attrs["values used"] = int(usable.sum())
    return table


def provisioning_cost(curve: pd.DataFrame, under_cost: float = UNDER_PROVISIONING_COST) -> pd.Series:
    """Return the mean cost of each option when one unit of shortfall costs ``under_cost`` units of idle capacity."""
    if under_cost <= 0:
        raise ValueError("under_cost must be positive")
    return under_cost * curve["mean shortfall"] + curve["mean excess"]


def choose_option(curve: pd.DataFrame, target: float = HIGH_LOAD_TARGET_SHARE):
    """Return the label of the option with the smallest mean margin whose high-load under-provisioned share is at most ``target``.

    If no option meets the target, the label of the option with the smallest share is returned.
    """
    if "high-load under-provisioned share" not in curve:
        raise ValueError("the curve needs the high-load under-provisioned share (pass high_load to headroom_curve)")
    if not 0 <= target <= 1:
        raise ValueError("target must be between 0 and 1")
    shares = curve["high-load under-provisioned share"]
    meets = curve[shares <= target]
    return meets["mean margin"].idxmin() if len(meets) else shares.idxmin()


def error_by_position(result: CVResult, name: str) -> pd.DataFrame:
    """Return the number of values, MAE and mean error (bias) by the target's position in the hour (0 to 11)."""
    errors = forecast_errors(result, name)
    stamps = result.timestamps[:, None] + INTERVAL_SECONDS * np.arange(errors.shape[1])[None, :]
    positions = position_in_hour(stamps.ravel())
    frame = pd.DataFrame({"position": positions, "error": errors.ravel(), "absolute error": np.abs(errors.ravel())})
    grouped = frame.groupby("position")
    return pd.DataFrame({"values": grouped.size(), "MAE": grouped["absolute error"].mean(), "bias": grouped["error"].mean()})


def worst_origins(result: CVResult, name: str, count: int = 5) -> pd.DataFrame:
    """Return the ``count`` origins with the largest MAE: history length, first forecast timestamp, MAE and bias."""
    if count < 1:
        raise ValueError("count must be at least 1")
    errors = forecast_errors(result, name)
    table = pd.DataFrame({"history points": result.origins, "first timestamp": result.timestamps, "MAE": np.abs(errors).mean(axis=1), "bias": errors.mean(axis=1)})
    return table.sort_values("MAE", ascending=False).head(count).reset_index(drop=True)


def capacity_plan(forecast, timestamps, errors, level: float, probability: float, margin_active: bool = True) -> pd.DataFrame:
    """Return the forecast with its central interval and the capacity to plan for, one row per forecast timestamp.

    The interval is the forecast plus and minus the ``level`` quantile of the absolute errors at
    each step. The capacity adds the one-sided ``probability`` quantile of the signed errors (at
    least zero), or is the forecast alone when ``margin_active`` is false (the peak-risk trigger is off).
    """
    forecast = np.asarray(forecast, dtype=float)
    half_width = final_error_quantiles(errors, [level], absolute=True)[:, 0]
    margin = np.maximum(final_error_quantiles(errors, [probability])[:, 0], 0.0)
    if len(half_width) != len(forecast) or len(timestamps) != len(forecast):
        raise ValueError("forecast, timestamps and errors must cover the same number of steps")
    return pd.DataFrame(
        {"forecast": forecast, "lower": forecast - half_width, "upper": forecast + half_width, "capacity": forecast + (margin if margin_active else 0.0)},
        index=pd.Index(timestamps, name="timestamp"),
    )
