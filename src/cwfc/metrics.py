from __future__ import annotations

import numpy as np


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return both inputs as float arrays after checking they match, are non-empty and finite."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if a.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("y_true and y_pred must be finite")
    return a, b


def mae(y_true, y_pred) -> float:
    """Return the mean absolute error."""
    a, b = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    """Return the root mean squared error."""
    a, b = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    """Return the model MAE divided by the baseline MAE (below 1 means better than the baseline).

    If the baseline is exact the ratio is undefined: it returns 0.0 for an exact model and infinity otherwise.
    """
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)
