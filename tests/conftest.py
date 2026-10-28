import matplotlib
import numpy as np
import pandas as pd
import pytest

from cwfc.data import INTERVAL_SECONDS, STEPS_PER_HOUR

matplotlib.use("Agg")  # figures are only saved in the tests, never shown

HOURLY_PATTERN = np.sin(2 * np.pi * np.arange(STEPS_PER_HOUR) / STEPS_PER_HOUR)


def make_series(values) -> pd.Series:
    """Build a series of values indexed by raw timestamps in seconds, one interval apart, starting at 0."""
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.Index(np.arange(len(values)) * INTERVAL_SECONDS, name="timestamp"))


@pytest.fixture
def periodic_series() -> pd.Series:
    """Return 96 points (8 hours) of a level of 10 plus an exact hourly sine pattern."""
    return make_series(10 + np.tile(HOURLY_PATTERN, 8))


@pytest.fixture
def noisy_series() -> pd.Series:
    """Return 96 points of a level of 10, an hourly pattern and AR(1) noise, with a fixed seed."""
    rng = np.random.default_rng(0)
    noise = np.zeros(96)
    for i in range(1, 96):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(scale=0.2)
    return make_series(10 + np.tile(HOURLY_PATTERN, 8) + noise)
