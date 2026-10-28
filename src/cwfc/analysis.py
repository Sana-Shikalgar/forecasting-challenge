"""Shared analysis helpers for the notebooks and the modelling code.

Statistics on a series, per-VM and per-segment summaries, and helpers that save
figures and tables under ``reports/``. Nothing here reads the data files; callers
pass in frames built by ``cwfc.data``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from cycler import cycler
from scipy import signal, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from cwfc.data import require_columns
from cwfc.download import default_data_dir

NO_METADATA = "No metadata"
# Okabe-Ito colorblind-safe palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#666666",
}


@dataclass(frozen=True)
class TrendFit:
    """Result of a straight-line fit of a series against its position."""

    slope: float
    intercept: float
    ci_low: float
    ci_high: float
    p_value: float
    r_squared: float


def set_plot_style() -> None:
    """Apply the shared figure style: clean axes, light grid, colorblind-safe colors."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=[PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "grey")]),
            "font.size": 10,
            "axes.titlesize": 11,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def reports_dir() -> Path:
    """Return the ``reports/`` folder at the repository root."""
    return default_data_dir().parent / "reports"


def _plain_name(name: str) -> str:
    """Return ``name`` if it is a plain file name; raise ``ValueError`` if it is empty or contains a path."""
    if not name or Path(name).name != name:
        raise ValueError(f"name must be a plain file name, got {name!r}")
    return name


def save_figure(fig, name: str, directory: Path | None = None) -> Path:
    """Save a matplotlib figure as ``<name>.png`` and return its path.

    Raises ``ValueError`` if ``name`` is empty or contains a path.
    """
    folder = Path(directory) if directory is not None else reports_dir() / "figures"
    path = folder / f"{_plain_name(name)}.png"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def save_table(table: pd.DataFrame | pd.Series, name: str, directory: Path | None = None) -> Path:
    """Save a table as ``<name>.csv`` and return its path.

    Raises ``ValueError`` if ``name`` is empty or contains a path.
    """
    folder = Path(directory) if directory is not None else reports_dir() / "tables"
    path = folder / f"{_plain_name(name)}.csv"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path


def elapsed_hours(timestamps) -> np.ndarray:
    """Convert timestamps in seconds to hours since the first one (an empty input gives an empty array)."""
    values = np.asarray(timestamps, dtype=float)
    if values.size == 0:
        return values
    return (values - values[0]) / 3600.0


def fit_trend(series: pd.Series, confidence: float = 0.95) -> TrendFit:
    """Fit a straight line to a series against its position (per step).

    Raises ``ValueError`` if fewer than three non-missing values are given.
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError("need at least 3 values to fit a trend")
    fit = stats.linregress(np.arange(len(values)), values)
    half_width = stats.t.ppf(0.5 + confidence / 2, df=len(values) - 2) * fit.stderr
    return TrendFit(fit.slope, fit.intercept, fit.slope - half_width, fit.slope + half_width, fit.pvalue, fit.rvalue**2)


def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS tests on the level and the first difference of a series.

    ADF assumes a unit root (non-stationary) and KPSS assumes stationarity, so
    they are read together. Raises ``ValueError`` for fewer than 12 values or a constant series.
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 12:
        raise ValueError("need at least 12 values for stationarity tests")
    if np.ptp(values) == 0:
        raise ValueError("series is constant")
    rows = {}
    for label, x in (("level", values), ("first difference", np.diff(values))):
        adf = adfuller(x, autolag="AIC", result_object=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)  # the p-value is clipped to the lookup table range
            kpss_result = kpss(x, regression="c", nlags="auto", result_object=True)
        rows[label] = {
            "adf_statistic": adf.statistic,
            "adf_p_value": adf.pvalue,
            "kpss_statistic": kpss_result.statistic,
            "kpss_p_value": kpss_result.pvalue,
        }
    return pd.DataFrame(rows).T


def autocorrelation_table(series: pd.Series, nlags: int = 30) -> pd.DataFrame:
    """Return the ACF and PACF of a series for lags 0 to ``nlags`` (capped by its length).

    Raises ``ValueError`` for fewer than 3 values or a constant series.
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError("need at least 3 values for autocorrelation")
    if np.ptp(values) == 0:
        raise ValueError("series is constant")
    nlags = max(1, min(nlags, len(values) // 2 - 1))
    table = pd.DataFrame({"acf": acf(values, nlags=nlags, fft=False), "pacf": pacf(values, nlags=nlags, method="ywm")})
    table.index.name = "lag"
    return table


def white_noise_band(n: int) -> float:
    """Return the 95% band for an autocorrelation of white noise with ``n`` observations (``n`` must be positive)."""
    if n < 1:
        raise ValueError("n must be positive")
    return 1.96 / np.sqrt(n)


def ljung_box(series: pd.Series, lags: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation up to each lag; a small p-value means not white noise.

    Lags that are not shorter than the series are left out; ``ValueError`` if none is left.
    """
    values = series.dropna().to_numpy(dtype=float)
    usable = [lag for lag in lags if lag < len(values)]
    if not usable:
        raise ValueError("every lag must be shorter than the series")
    result = acorr_ljungbox(values, lags=usable, return_df=True)
    result.index.name = "lag"
    return result.rename(columns={"lb_stat": "statistic", "lb_pvalue": "p_value"})


def periodogram_table(series: pd.Series, step_minutes: float = 5.0) -> pd.DataFrame:
    """Return power by period (in minutes) after removing a linear trend, longest period first."""
    values = series.dropna().to_numpy(dtype=float)
    frequency, power = signal.periodogram(values, detrend="linear")
    keep = frequency > 0
    table = pd.DataFrame({"period_minutes": step_minutes / frequency[keep], "power": power[keep]})
    return table.sort_values("period_minutes", ascending=False).reset_index(drop=True)


def stl_decompose(series: pd.Series, period: int, seasonal_window: int = 13) -> pd.DataFrame:
    """Split a series into trend, seasonal and residual parts with a robust STL decomposition.

    A larger ``seasonal_window`` keeps the seasonal pattern more stable across cycles.
    Raises ``ValueError`` if ``period`` is below 2, the series has fewer than two full periods,
    or it contains a missing value (STL needs a regular series, so values are never dropped).
    """
    values = series
    if period < 2 or len(values) < 2 * period:
        raise ValueError("need period of at least 2 and at least two full periods of data")
    if values.isna().any():
        raise ValueError("series contains missing values; STL needs a regular series")
    fit = STL(values.to_numpy(dtype=float), period=period, seasonal=seasonal_window, robust=True).fit()
    components = {"observed": values.to_numpy(dtype=float), "trend": fit.trend, "seasonal": fit.seasonal, "resid": fit.resid}
    return pd.DataFrame(components, index=values.index)


def stl_strengths(components: pd.DataFrame) -> pd.Series:
    """Return trend and seasonal strength (0 to 1) from an STL decomposition.

    Strength is ``max(0, 1 - Var(resid) / Var(component + resid))``; it is NaN when
    that variance is zero.
    """
    strengths = {}
    for name in ("trend", "seasonal"):
        spread = (components[name] + components["resid"]).var()
        strengths[f"{name}_strength"] = max(0.0, 1 - components["resid"].var() / spread) if spread > 0 else float("nan")
    return pd.Series(strengths)


def fold_by_period(series: pd.Series, period: int, step_seconds: int = 300) -> pd.DataFrame:
    """Fold a series indexed by timestamp (seconds) into one row per cycle and one column per position.

    The position is ``(timestamp // step_seconds) % period``, so it is tied to the
    trace clock, not to where the series happens to start. Incomplete cycles at
    either end have missing values. Raises ``ValueError`` if ``period`` or ``step_seconds`` is not
    positive, or the index has duplicate timestamps.
    """
    if period < 1 or step_seconds < 1:
        raise ValueError("period and step_seconds must be positive")
    if series.index.has_duplicates:
        raise ValueError("series index must be unique")
    steps = np.asarray(series.index, dtype="int64") // step_seconds
    folded = pd.DataFrame({"cycle": steps // period, "position": steps % period, "value": series.to_numpy()})
    return folded.pivot(index="cycle", columns="position", values="value")


def cycle_strength(series: pd.Series, periods, step_seconds: int = 300) -> pd.Series:
    """Score how well a repeating pattern of each candidate period explains a series.

    For each period the series is grouped by position in the cycle and the score is
    the adjusted share of variance explained by the group means. Periods as long as
    the series, and constant series, give NaN. Raises ``ValueError`` if a period or
    ``step_seconds`` is not positive.
    """
    periods = list(periods)
    if any(period < 1 for period in periods) or step_seconds < 1:
        raise ValueError("periods and step_seconds must be positive")
    values = series.dropna()
    steps = np.asarray(values.index, dtype="int64") // step_seconds
    total = ((values - values.mean()) ** 2).sum()
    scores = {}
    for period in periods:
        if total == 0 or period >= len(values):
            scores[period] = np.nan
            continue
        residual = values - values.groupby(steps % period).transform("mean")
        r_squared = 1 - (residual**2).sum() / total
        scores[period] = 1 - (1 - r_squared) * (len(values) - 1) / (len(values) - period)
    return pd.Series(scores, name="adjusted_r_squared").rename_axis("period_steps")


def horizon_change_std(series: pd.Series, horizons) -> pd.Series:
    """Return the standard deviation of h-step changes of a series for each horizon h.

    A horizon longer than the series gives NaN. Raises ``ValueError`` if a horizon is not positive.
    """
    horizons = list(horizons)
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive")
    values = series.dropna().reset_index(drop=True)
    return pd.Series({h: values.diff(h).std() for h in horizons}, name="change_std").rename_axis("horizon_steps")


def pivot_readings(readings: pd.DataFrame, value: str = "avg_cpu") -> pd.DataFrame:
    """Reshape readings to one row per timestamp and one column per VM (duplicates averaged)."""
    require_columns(readings.columns, ("timestamp", "vm_id", value), "readings")
    return readings.groupby(["timestamp", "vm_id"])[value].mean().unstack("vm_id").sort_index()


def variance_decomposition(wide: pd.DataFrame) -> pd.Series:
    """Split the variance of all values into between-VM and within-VM parts.

    Exact for complete data (no missing values): total = between + within.
    Raises ``ValueError`` for an empty frame.
    """
    if wide.empty:
        raise ValueError("need at least one VM and one timestamp")
    values = wide.to_numpy(dtype=float)
    between = float(np.nanvar(np.nanmean(values, axis=0)))
    within = float(np.nanmean(np.nanvar(values, axis=0)))
    return pd.Series({"between_vm": between, "within_vm": within, "total": float(np.nanvar(values))})


def implied_average_correlation(wide: pd.DataFrame) -> float:
    """Return the average pairwise correlation across VMs implied by how much their mean varies.

    Compares the variance of the cross-VM mean with the value it would have if the
    VMs were independent, and expresses the excess as an average covariance divided
    by the average VM variance. Raises ``ValueError`` for fewer than two VMs or when no VM varies.
    """
    n_vms = wide.shape[1]
    if n_vms < 2:
        raise ValueError("need at least two VMs")
    own_variance = wide.var()
    if own_variance.mean() == 0:
        raise ValueError("the VM series have no variance")
    excess = wide.mean(axis=1).var() - own_variance.sum() / n_vms**2
    return float(excess * n_vms / (n_vms - 1) / own_variance.mean())


def aggregate_without_top_variance_vms(wide: pd.DataFrame, shares) -> pd.DataFrame:
    """Show how the cross-VM mean changes when the highest-variance VMs are removed.

    For each share of VMs removed (largest own variance first) return the standard
    deviation of the mean of the remaining VMs and its correlation with the full
    mean. Raises ``ValueError`` if a share is outside [0, 1) or leaves no VM.
    """
    order = wide.var().sort_values(ascending=False).index
    full = wide.mean(axis=1)
    rows = {}
    for share in shares:
        removed = int(round(share * wide.shape[1]))
        if not 0 <= share < 1 or removed >= wide.shape[1]:
            raise ValueError("each share must be in [0, 1) and leave at least one VM")
        remaining = wide[order[removed:]].mean(axis=1)
        rows[share] = {"std": remaining.std(), "correlation_with_full": remaining.corr(full)}
    return pd.DataFrame(rows).T.rename_axis("share_removed")


def cumulative_share(values: pd.Series) -> pd.DataFrame:
    """Return the cumulative share of the total held by the top-ranked items.

    Columns: ``share_of_items`` and ``share_of_total``, with items ranked from largest to smallest.
    Raises ``ValueError`` for empty values or values that sum to zero.
    """
    ranked = values.sort_values(ascending=False)
    if ranked.empty or ranked.sum() == 0:
        raise ValueError("need values with a non-zero sum")
    n = len(ranked)
    return pd.DataFrame({"share_of_items": np.arange(1, n + 1) / n, "share_of_total": ranked.cumsum().to_numpy() / ranked.sum()})


def sampled_pairwise_correlations(wide: pd.DataFrame, sample_size: int = 300, seed: int = 0) -> np.ndarray:
    """Return correlations between pairs of randomly sampled VM series, skipping constant series.

    Missing values are skipped pair by pair, and pairs with fewer than two shared observations are
    dropped. Raises ``ValueError`` if fewer than two varying series are available or no pair remains.
    """
    rng = np.random.default_rng(seed)
    columns = rng.choice(wide.columns.to_numpy(), size=min(sample_size, wide.shape[1]), replace=False)
    sample = wide[list(columns)]
    sample = sample.loc[:, sample.std() > 0]
    if sample.shape[1] < 2:
        raise ValueError("need at least two varying series")
    correlation = sample.corr().to_numpy()
    pairs = correlation[np.triu_indices_from(correlation, k=1)]
    pairs = pairs[~np.isnan(pairs)]
    if pairs.size == 0:
        raise ValueError("no pair of series shares enough observations")
    return pairs


def lag1_autocorrelation(wide: pd.DataFrame) -> pd.Series:
    """Return each VM series' lag-1 autocorrelation (NaN for constant series)."""
    return wide.apply(lambda column: column.autocorr(1))


def peak_periods(series: pd.Series, threshold: float) -> pd.DataFrame:
    """List the runs of consecutive points above ``threshold`` with their start, end, length and maximum."""
    above = series > threshold
    run_id = (above != above.shift()).cumsum()
    rows = [
        {"start": run.index[0], "end": run.index[-1], "n_steps": len(run), "maximum": run.max()}
        for _, run in series[above].groupby(run_id[above])
    ]
    return pd.DataFrame(rows, columns=["start", "end", "n_steps", "maximum"])


def _segment_lookup(metadata: pd.DataFrame, column: str) -> pd.Series:
    """Return a ``vm_id`` to segment lookup from the metadata; raise ``ValueError`` on duplicate ids or a missing column."""
    require_columns(metadata.columns, ("vm_id", column), "metadata")
    if metadata["vm_id"].duplicated().any():
        raise ValueError("metadata lists some vm_id more than once")
    return metadata.set_index("vm_id")[column]


def segment_series(readings: pd.DataFrame, metadata: pd.DataFrame, column: str, value: str = "avg_cpu") -> pd.DataFrame:
    """Return the mean of ``value`` per timestamp for each segment defined by a metadata column.

    VMs without a metadata row, or with a missing segment value, fall into the ``No metadata`` segment.
    """
    segment = readings["vm_id"].map(_segment_lookup(metadata, column)).fillna(NO_METADATA)
    return readings.assign(segment=segment).groupby(["timestamp", "segment"])[value].mean().unstack("segment")


def segment_summary(readings: pd.DataFrame, metadata: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarise each segment: VM count, share of VMs, mean CPU per VM and share of total load."""
    vm_mean = readings.groupby("vm_id")["avg_cpu"].mean()
    segment = vm_mean.index.to_series().map(_segment_lookup(metadata, column)).fillna(NO_METADATA)
    grouped = vm_mean.groupby(segment.to_numpy())
    summary = pd.DataFrame({"n_vms": grouped.size(), "mean_cpu_per_vm": grouped.mean(), "load": grouped.sum()})
    summary["share_of_vms"] = summary["n_vms"] / summary["n_vms"].sum()
    summary["share_of_load"] = summary.pop("load") / vm_mean.sum()
    summary.index.name = "segment"
    return summary
