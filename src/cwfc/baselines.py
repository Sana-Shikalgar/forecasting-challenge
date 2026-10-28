from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Forecast every row of ``template`` as the last observed value of ``target``.

    Raises ``ValueError`` if ``train`` is empty or its last value is missing.
    """
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    if np.isnan(last_value):
        raise ValueError("the last observed value is missing")
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    """Forecast by repeating the last ``season_length`` observed values of ``target``.

    Raises ``ValueError`` if ``season_length`` is not positive, there is less history than one
    season, or the last season contains a missing value.
    """
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    if np.isnan(values[-season_length:]).any():
        raise ValueError("the last season contains a missing value")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
