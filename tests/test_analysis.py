from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cwfc import analysis as an


def _ar1(n: int, phi: float, seed: int = 0) -> pd.Series:
    """Generate an AR(1) series with coefficient ``phi``."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + rng.normal()
    return pd.Series(values)


def _wide(columns: dict[str, list[float]]) -> pd.DataFrame:
    """Build a timestamp-by-VM frame from per-VM value lists."""
    frame = pd.DataFrame(columns)
    frame.index = pd.Index(np.arange(len(frame)) * 300, name="timestamp")
    return frame


def _long_readings() -> pd.DataFrame:
    """Build readings for three VMs over two timestamps."""
    return pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 300, 300, 300],
            "vm_id": ["a", "b", "c", "a", "b", "c"],
            "avg_cpu": [10.0, 20.0, 30.0, 14.0, 24.0, 40.0],
        }
    )


def _metadata() -> pd.DataFrame:
    """Build metadata that labels a and b as small and c as large."""
    return pd.DataFrame({"vm_id": ["a", "b", "c"], "size": ["small", "small", "large"]})


# plot style and saving


def test_set_plot_style_changes_defaults():
    """The shared style turns on the grid, hides the top spine and sets the figure size."""
    with mpl.rc_context():
        an.set_plot_style()
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["axes.spines.top"] is False
        assert tuple(mpl.rcParams["figure.figsize"]) == (9, 4)


def test_reports_dir_sits_next_to_pyproject():
    """The reports folder is at the repository root."""
    path = an.reports_dir()
    assert path.name == "reports" and path.parent.joinpath("pyproject.toml").is_file()


def test_save_figure_writes_png_in_created_folder(tmp_path):
    """A figure is saved as a PNG, creating the folder if needed."""
    fig = Figure()
    fig.subplots().plot([1, 2, 3])
    path = an.save_figure(fig, "demo", tmp_path / "new" / "folder")
    assert path.name == "demo.png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_table_writes_csv_for_frame_and_series(tmp_path):
    """Frames and series are saved as CSV files that read back unchanged."""
    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="k"))
    path = an.save_table(frame, "frame", tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(path, index_col="k"), frame)
    series_path = an.save_table(frame["a"], "series", tmp_path)
    assert series_path.name == "series.csv" and series_path.is_file()


def test_elapsed_hours_starts_at_zero():
    """Timestamps in seconds become hours since the first one."""
    np.testing.assert_allclose(an.elapsed_hours([1000, 1300, 4600]), [0.0, 300 / 3600, 1.0])


# trend and tests


def test_fit_trend_recovers_a_straight_line():
    """A noiseless line y = 2x + 1 has slope 2, r-squared 1 and a tiny p-value."""
    fit = an.fit_trend(pd.Series(2.0 * np.arange(20) + 1.0))
    assert fit.slope == pytest.approx(2.0) and fit.intercept == pytest.approx(1.0) and fit.r_squared == pytest.approx(1.0)
    assert fit.ci_low <= 2.0 <= fit.ci_high and fit.p_value < 1e-6


def test_fit_trend_on_a_constant_series_has_zero_slope():
    """A constant series has no trend."""
    fit = an.fit_trend(pd.Series([5.0] * 10))
    assert fit.slope == pytest.approx(0.0)


def test_fit_trend_wider_confidence_gives_wider_interval():
    """A 99% interval is wider than a 95% interval."""
    series = _ar1(60, 0.3)
    narrow, wide = an.fit_trend(series, 0.95), an.fit_trend(series, 0.99)
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_fit_trend_ignores_missing_values_and_needs_three_points():
    """NaNs are dropped, and fewer than three remaining values raises ValueError."""
    assert an.fit_trend(pd.Series([1.0, np.nan, 3.0, 5.0])).slope == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least 3"):
        an.fit_trend(pd.Series([1.0, np.nan, 2.0]))


def test_stationarity_tests_separate_noise_from_a_trending_series():
    """White noise looks stationary on both tests; a trending series does not, until differenced."""
    rng = np.random.default_rng(1)
    noise = an.stationarity_tests(pd.Series(rng.normal(size=200)))
    trending = an.stationarity_tests(pd.Series(0.3 * np.arange(200) + rng.normal(size=200)))
    assert list(noise.index) == ["level", "first difference"]
    assert list(noise.columns) == ["adf_statistic", "adf_p_value", "kpss_statistic", "kpss_p_value"]
    assert noise.loc["level", "adf_p_value"] < 0.01 and noise.loc["level", "kpss_p_value"] > 0.05
    assert trending.loc["level", "adf_p_value"] > 0.05 and trending.loc["level", "kpss_p_value"] < 0.05
    assert trending.loc["first difference", "adf_p_value"] < 0.01


def test_stationarity_tests_need_enough_values():
    """Short series raise ValueError."""
    with pytest.raises(ValueError, match="at least 12"):
        an.stationarity_tests(pd.Series(range(8), dtype=float))


# autocorrelation


def test_autocorrelation_table_matches_ar1_theory():
    """An AR(1) series with phi 0.8 has lag-1 autocorrelation near 0.8 and lag 0 equal to 1."""
    table = an.autocorrelation_table(_ar1(600, 0.8), nlags=10)
    assert table.index.name == "lag" and len(table) == 11
    assert table.loc[0, "acf"] == pytest.approx(1.0)
    assert table.loc[1, "acf"] == pytest.approx(0.8, abs=0.08)
    assert abs(table.loc[3, "pacf"]) < 0.15


def test_autocorrelation_table_caps_lags_for_short_series():
    """Asking for more lags than the series supports returns fewer lags."""
    assert len(an.autocorrelation_table(pd.Series(np.arange(10.0) ** 2), nlags=30)) <= 5


def test_white_noise_band_shrinks_with_length():
    """The band is 1.96 divided by the square root of the length."""
    assert an.white_noise_band(100) == pytest.approx(0.196)
    assert an.white_noise_band(400) < an.white_noise_band(100)


def test_ljung_box_flags_autocorrelation_but_not_noise():
    """Noise gives large p-values; a strongly autocorrelated series gives tiny ones."""
    noise = an.ljung_box(pd.Series(np.random.default_rng(2).normal(size=200)))
    correlated = an.ljung_box(_ar1(200, 0.9))
    assert list(noise.index) == [5, 10, 20] and list(noise.columns) == ["statistic", "p_value"]
    assert (noise["p_value"] > 0.05).all() and (correlated["p_value"] < 0.001).all()


def test_ljung_box_drops_lags_longer_than_the_series():
    """Lags that exceed the series length are left out."""
    result = an.ljung_box(pd.Series(np.random.default_rng(3).normal(size=15)), lags=(5, 10, 50))
    assert list(result.index) == [5, 10]


# periodogram and changes


def test_periodogram_finds_the_dominant_period():
    """A sine wave with a 12-step period peaks at 60 minutes."""
    series = pd.Series(np.sin(2 * np.pi * np.arange(120) / 12))
    table = an.periodogram_table(series, step_minutes=5.0)
    assert table.loc[table["power"].idxmax(), "period_minutes"] == pytest.approx(60.0)
    assert (table["period_minutes"].diff().dropna() <= 0).all()
    assert (table["period_minutes"] > 0).all()


def _repeating_series(period: int, cycles: int, start_step: int = 0, noise: float = 0.0, seed: int = 0) -> pd.Series:
    """Build a series that repeats a fixed pattern every ``period`` steps, indexed by timestamp."""
    rng = np.random.default_rng(seed)
    pattern = np.arange(period, dtype=float) ** 2 % 7
    steps = np.arange(start_step, start_step + period * cycles)
    return pd.Series(pattern[steps % period] + rng.normal(scale=noise, size=len(steps)) if noise else pattern[steps % period], index=steps * 300)


def test_fold_by_period_gives_identical_rows_for_a_repeating_series():
    """A perfectly repeating series folds into rows that all equal the pattern."""
    folded = an.fold_by_period(_repeating_series(12, 5), 12)
    assert folded.shape == (5, 12) and (folded.nunique() == 1).all()


def test_fold_by_period_uses_the_trace_clock_for_positions():
    """A series that starts mid-cycle keeps positions tied to timestamps, leaving gaps at the ends."""
    folded = an.fold_by_period(_repeating_series(12, 3, start_step=5), 12)
    assert folded.shape == (4, 12) and folded.iloc[0, :5].isna().all() and folded.iloc[0, 5:].notna().all()
    assert folded.iloc[-1, 5:].isna().all()


def test_cycle_strength_peaks_at_the_true_period():
    """A series repeating every 12 steps scores highest at 12, and low at nearby periods."""
    scores = an.cycle_strength(_repeating_series(12, 9, noise=0.2), range(5, 20))
    assert scores.idxmax() == 12 and scores.loc[12] > 0.9
    assert scores.loc[[7, 11, 13]].max() < 0.3
    assert scores.index.name == "period_steps"


def test_cycle_strength_is_nan_for_constant_series_and_too_long_periods():
    """A constant series, or a period as long as the series, gives NaN."""
    constant = pd.Series([3.0] * 30, index=np.arange(30) * 300)
    assert an.cycle_strength(constant, [4]).isna().all()
    assert np.isnan(an.cycle_strength(_repeating_series(12, 2), [24]).loc[24])


def test_stl_decompose_parts_add_up_and_find_the_seasonal_pattern():
    """The parts sum to the observed series, and a repeating pattern gets a high seasonal strength."""
    series = _repeating_series(12, 9, noise=0.1) + 0.05 * np.arange(108)
    parts = an.stl_decompose(series, 12)
    assert list(parts.columns) == ["observed", "trend", "seasonal", "resid"]
    np.testing.assert_allclose(parts[["trend", "seasonal", "resid"]].sum(axis=1), parts["observed"])
    strengths = an.stl_strengths(parts)
    assert strengths["seasonal_strength"] > 0.9 and strengths["trend_strength"] > 0.9


def test_stl_strengths_are_low_for_noise():
    """White noise has little trend or seasonal strength."""
    noise = pd.Series(np.random.default_rng(11).normal(size=120), index=np.arange(120) * 300)
    strengths = an.stl_strengths(an.stl_decompose(noise, 12))
    assert strengths["seasonal_strength"] < 0.5 and strengths["trend_strength"] < 0.5


def test_stl_decompose_rejects_short_series_and_bad_periods():
    """Fewer than two periods of data, or a period below 2, raises ValueError."""
    with pytest.raises(ValueError, match="two full periods"):
        an.stl_decompose(_repeating_series(12, 1), 12)
    with pytest.raises(ValueError, match="period of at least 2"):
        an.stl_decompose(_repeating_series(12, 5), 1)


def test_stl_strengths_known_values_and_clipping():
    """No residual gives strength 1, a residual that swamps the component clips to 0, and zero spread is NaN."""
    exact = pd.DataFrame({"trend": [1.0, 2.0, 3.0, 4.0], "seasonal": [1.0, -1.0, 1.0, -1.0], "resid": [0.0] * 4})
    assert an.stl_strengths(exact).tolist() == [1.0, 1.0]
    swamped = pd.DataFrame({"trend": [0.0] * 4, "seasonal": [0.0] * 4, "resid": [1.0, -1.0, 2.0, -2.0]})
    assert an.stl_strengths(swamped).tolist() == [0.0, 0.0]
    flat = pd.DataFrame({"trend": [1.0] * 4, "seasonal": [1.0] * 4, "resid": [0.0] * 4})
    assert an.stl_strengths(flat).isna().all()


def test_horizon_change_std_known_values():
    """Changes of an alternating series have a known standard deviation; a ramp has none."""
    alternating = an.horizon_change_std(pd.Series([0.0, 1.0, 0.0, 1.0]), [1, 2])
    assert alternating.loc[1] == pytest.approx(np.std([1, -1, 1], ddof=1))
    assert alternating.loc[2] == pytest.approx(0.0)
    assert alternating.index.name == "horizon_steps"
    assert an.horizon_change_std(pd.Series(np.arange(10.0)), [3]).loc[3] == pytest.approx(0.0)


# per-VM helpers


def test_pivot_readings_shape_and_duplicate_average():
    """Readings become a timestamp-by-VM table, averaging duplicate rows."""
    readings = _long_readings()
    readings = pd.concat([readings, pd.DataFrame({"timestamp": [0], "vm_id": ["a"], "avg_cpu": [30.0]})])
    wide = an.pivot_readings(readings)
    assert wide.shape == (2, 3) and list(wide.columns) == ["a", "b", "c"]
    assert wide.loc[0, "a"] == 20.0 and wide.loc[300, "c"] == 40.0


def test_pivot_readings_requires_columns():
    """A missing value column raises ValueError."""
    with pytest.raises(ValueError, match="avg_cpu"):
        an.pivot_readings(_long_readings().drop(columns="avg_cpu"))


def test_variance_decomposition_known_and_identity():
    """Two constant VMs have only between-VM variance, and parts add up for random data."""
    known = an.variance_decomposition(_wide({"a": [0.0, 0.0, 0.0], "b": [10.0, 10.0, 10.0]}))
    assert known["between_vm"] == pytest.approx(25.0) and known["within_vm"] == pytest.approx(0.0)
    random = an.variance_decomposition(pd.DataFrame(np.random.default_rng(4).normal(size=(30, 12))))
    assert random["total"] == pytest.approx(random["between_vm"] + random["within_vm"])


def test_implied_average_correlation_is_one_for_identical_vms_and_zero_for_independent_ones():
    """Identical VM series imply a correlation of 1; independent noise implies about 0."""
    base = np.random.default_rng(5).normal(size=60)
    assert an.implied_average_correlation(pd.DataFrame({i: base for i in range(8)})) == pytest.approx(1.0)
    independent = pd.DataFrame(np.random.default_rng(6).normal(size=(300, 200)))
    assert abs(an.implied_average_correlation(independent)) < 0.01


def test_implied_average_correlation_matches_a_shared_factor():
    """VMs that share a common factor with correlation 0.5 imply an average correlation near 0.5."""
    rng = np.random.default_rng(7)
    common = rng.normal(size=(500, 1))
    wide = pd.DataFrame(np.sqrt(0.5) * common + np.sqrt(0.5) * rng.normal(size=(500, 60)))
    assert an.implied_average_correlation(wide) == pytest.approx(0.5, abs=0.06)


def test_implied_average_correlation_needs_two_vms():
    """A single VM raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        an.implied_average_correlation(_wide({"a": [1.0, 2.0, 3.0]}))


def test_removing_the_only_varying_vm_flattens_the_aggregate():
    """Dropping the single high-variance VM leaves a constant mean; dropping nothing keeps the full mean."""
    wide = _wide({"a": [1.0, 5.0, 2.0, 8.0], "b": [3.0] * 4, "c": [3.0] * 4, "d": [3.0] * 4})
    result = an.aggregate_without_top_variance_vms(wide, [0, 0.25])
    assert result.index.name == "share_removed"
    assert result.loc[0, "std"] == pytest.approx(wide.mean(axis=1).std()) and result.loc[0, "correlation_with_full"] == pytest.approx(1.0)
    assert result.loc[0.25, "std"] == pytest.approx(0.0) and np.isnan(result.loc[0.25, "correlation_with_full"])


def test_std_falls_as_more_high_variance_vms_are_removed():
    """The highest-variance VMs are removed first, so for a skewed fleet the std falls as the removed share grows."""
    rng = np.random.default_rng(8)
    scales = np.linspace(5.0, 0.1, 40)
    wide = pd.DataFrame(rng.normal(size=(80, 40)) * scales)
    result = an.aggregate_without_top_variance_vms(wide, [0, 0.1, 0.5])
    assert result["std"].is_monotonic_decreasing


@pytest.mark.parametrize("share", [1.0, -0.1, 1.5])
def test_aggregate_without_top_variance_vms_rejects_bad_shares(share):
    """Shares outside [0, 1) raise ValueError."""
    with pytest.raises(ValueError, match="at least one VM"):
        an.aggregate_without_top_variance_vms(_wide({"a": [1.0, 2.0, 4.0], "b": [2.0, 1.0, 3.0]}), [share])


def test_cumulative_share_ranks_from_largest():
    """The top item holds its share first, and the totals reach 100%."""
    result = an.cumulative_share(pd.Series([1.0, 6.0, 3.0]))
    np.testing.assert_allclose(result["share_of_items"], [1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(result["share_of_total"], [0.6, 0.9, 1.0])


def test_pairwise_correlations_identical_and_independent():
    """Copies of one series correlate at 1; independent noise averages near 0."""
    base = np.random.default_rng(6).normal(size=50)
    same = an.sampled_pairwise_correlations(pd.DataFrame({i: base for i in range(6)}), sample_size=6)
    assert len(same) == 15 and np.allclose(same, 1.0)
    noise = an.sampled_pairwise_correlations(pd.DataFrame(np.random.default_rng(7).normal(size=(200, 60))), sample_size=60)
    assert abs(noise.mean()) < 0.02


def test_pairwise_correlations_skip_constant_series_and_are_reproducible():
    """Constant columns are skipped, a fixed seed repeats, and too few series raise ValueError."""
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.normal(size=(40, 10)))
    frame["flat"] = 1.0
    first = an.sampled_pairwise_correlations(frame, sample_size=5, seed=3)
    np.testing.assert_array_equal(first, an.sampled_pairwise_correlations(frame, sample_size=5, seed=3))
    with pytest.raises(ValueError, match="at least two"):
        an.sampled_pairwise_correlations(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 1.0]}))


def test_lag1_autocorrelation_known_patterns():
    """An alternating series gives -1, a ramp gives 1, and a constant series gives NaN."""
    result = an.lag1_autocorrelation(_wide({"alt": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], "ramp": list(range(6)), "flat": [2.0] * 6}))
    assert result["alt"] == pytest.approx(-1.0) and result["ramp"] == pytest.approx(1.0)
    assert np.isnan(result["flat"])


# peaks


def test_peak_periods_finds_runs_above_threshold():
    """Consecutive points above the threshold form one period with its length and maximum."""
    series = pd.Series([1.0, 5.0, 6.0, 1.0, 7.0, 1.0], index=[0, 300, 600, 900, 1200, 1500])
    result = an.peak_periods(series, 4.0)
    assert result["start"].tolist() == [300, 1200] and result["end"].tolist() == [600, 1200]
    assert result["n_steps"].tolist() == [2, 1] and result["maximum"].tolist() == [6.0, 7.0]


def test_peak_periods_edges_ties_and_empty_result():
    """Runs at the start and end count, equal-to-threshold and NaN do not, and none gives an empty table."""
    series = pd.Series([9.0, 1.0, np.nan, 4.0, 9.0], index=range(5))
    result = an.peak_periods(series, 4.0)
    assert result["start"].tolist() == [0, 4] and result["n_steps"].tolist() == [1, 1]
    empty = an.peak_periods(pd.Series([1.0, 2.0]), 10.0)
    assert empty.empty and list(empty.columns) == ["start", "end", "n_steps", "maximum"]


# segments


def test_segment_series_means_per_timestamp():
    """Each segment gets the mean of its VMs at every timestamp."""
    result = an.segment_series(_long_readings(), _metadata(), "size")
    assert result.loc[0, "small"] == 15.0 and result.loc[300, "small"] == 19.0 and result.loc[300, "large"] == 40.0


def test_segment_series_labels_vms_without_metadata():
    """A VM missing from the metadata is reported under its own segment."""
    result = an.segment_series(_long_readings(), _metadata().iloc[:2], "size")
    assert an.NO_METADATA in result.columns and result.loc[0, an.NO_METADATA] == 30.0


def test_segment_series_requires_the_column():
    """A metadata frame without the segment column raises ValueError."""
    with pytest.raises(ValueError, match="size"):
        an.segment_series(_long_readings(), _metadata().drop(columns="size"), "size")


def test_segment_summary_counts_and_shares():
    """Counts, shares and mean CPU per VM match a hand calculation."""
    summary = an.segment_summary(_long_readings(), _metadata(), "size")
    assert summary.loc["small", "n_vms"] == 2 and summary.loc["large", "n_vms"] == 1
    assert summary["share_of_vms"].sum() == pytest.approx(1.0) and summary["share_of_load"].sum() == pytest.approx(1.0)
    assert summary.loc["small", "mean_cpu_per_vm"] == pytest.approx((12.0 + 22.0) / 2)
    assert summary.loc["large", "share_of_load"] == pytest.approx(35.0 / (12.0 + 22.0 + 35.0))


def test_segment_summary_handles_vms_without_metadata():
    """VMs missing from the metadata are summarised under their own segment."""
    summary = an.segment_summary(_long_readings(), _metadata().iloc[:2], "size")
    assert summary.loc[an.NO_METADATA, "n_vms"] == 1


# edge cases found in an audit


@pytest.mark.parametrize("bad_name", ["../escape", "sub/name", ""])
def test_save_figure_and_table_reject_empty_or_path_like_names(tmp_path, bad_name):
    """A name that is empty or contains a path is refused, and nothing is written outside the folder."""
    folder = tmp_path / "out"
    with pytest.raises(ValueError, match="plain file name"):
        an.save_figure(Figure(), bad_name, folder)
    with pytest.raises(ValueError, match="plain file name"):
        an.save_table(pd.Series([1]), bad_name, folder)
    assert not (tmp_path / "escape.png").exists() and not (tmp_path / "escape.csv").exists()


def test_elapsed_hours_handles_empty_and_single_input():
    """An empty input gives an empty array and a single timestamp gives zero."""
    assert an.elapsed_hours([]).size == 0
    assert an.elapsed_hours([12345]).tolist() == [0.0]


@pytest.mark.parametrize("n", [0, -5])
def test_white_noise_band_rejects_non_positive_length(n):
    """A length below one raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        an.white_noise_band(n)


def test_constant_series_is_rejected_by_stationarity_and_autocorrelation():
    """A constant series raises a clear ValueError instead of NaN or a library error."""
    flat = pd.Series([5.0] * 30)
    with pytest.raises(ValueError, match="constant"):
        an.stationarity_tests(flat)
    with pytest.raises(ValueError, match="constant"):
        an.autocorrelation_table(flat)


def test_autocorrelation_table_rejects_fewer_than_three_values_and_works_with_three():
    """Two values raise ValueError; three is the smallest usable series."""
    with pytest.raises(ValueError, match="at least 3"):
        an.autocorrelation_table(pd.Series([1.0, 2.0]))
    assert an.autocorrelation_table(pd.Series([1.0, 2.0, 4.0]), nlags=30).shape == (2, 2)


def test_ljung_box_rejects_a_series_shorter_than_every_lag():
    """If no requested lag is shorter than the series, ValueError is raised."""
    with pytest.raises(ValueError, match="every lag"):
        an.ljung_box(pd.Series(np.random.default_rng(12).normal(size=6)), lags=(10, 20))


def test_periodogram_table_constant_and_single_value_series():
    """A constant series has no power and a single value has no frequencies."""
    assert an.periodogram_table(pd.Series([5.0] * 20))["power"].max() < 1e-20
    assert an.periodogram_table(pd.Series([5.0])).empty


def test_stl_decompose_rejects_missing_values_instead_of_dropping_them():
    """A NaN would silently shift the seasonal period if dropped, so it raises ValueError."""
    series = pd.Series([1.0, 2.0, np.nan] * 12, index=np.arange(36) * 300)
    with pytest.raises(ValueError, match="missing values"):
        an.stl_decompose(series, 3)


def test_stl_decompose_rejects_an_even_seasonal_window():
    """STL needs an odd seasonal window, and the library error is passed on."""
    with pytest.raises(ValueError, match="odd"):
        an.stl_decompose(pd.Series(np.random.default_rng(13).normal(size=40)), 4, seasonal_window=8)


@pytest.mark.parametrize("period", [0, -3])
def test_fold_by_period_and_cycle_strength_reject_non_positive_periods(period):
    """A period of zero or less raises ValueError."""
    series = pd.Series(np.arange(24.0), index=np.arange(24) * 300)
    with pytest.raises(ValueError, match="positive"):
        an.fold_by_period(series, period)
    with pytest.raises(ValueError, match="positive"):
        an.cycle_strength(series, range(period, period + 2))


def test_fold_by_period_and_cycle_strength_reject_non_positive_step_seconds():
    """A step of zero seconds raises ValueError."""
    series = pd.Series(np.arange(24.0), index=np.arange(24) * 300)
    with pytest.raises(ValueError, match="positive"):
        an.fold_by_period(series, 12, step_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        an.cycle_strength(series, [12], step_seconds=0)


def test_fold_by_period_rejects_duplicate_timestamps():
    """Two values at one timestamp cannot be folded, so ValueError is raised."""
    with pytest.raises(ValueError, match="unique"):
        an.fold_by_period(pd.Series(range(6), index=[0, 0, 300, 300, 600, 600]), 3)


def test_horizon_change_std_rejects_non_positive_horizons_and_gives_nan_for_long_ones():
    """A horizon of zero or less raises ValueError; one longer than the series gives NaN."""
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        an.horizon_change_std(series, [0])
    with pytest.raises(ValueError, match="positive"):
        an.horizon_change_std(series, [-1, 2])
    assert np.isnan(an.horizon_change_std(series, [5]).iloc[0])


def test_variance_decomposition_rejects_an_empty_frame():
    """No VMs or no timestamps raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        an.variance_decomposition(pd.DataFrame())


def test_implied_average_correlation_rejects_vms_without_variance():
    """If no VM varies, the correlation is undefined and ValueError is raised."""
    with pytest.raises(ValueError, match="no variance"):
        an.implied_average_correlation(pd.DataFrame({"a": [1.0] * 5, "b": [2.0] * 5}))


def test_cumulative_share_rejects_empty_and_zero_total_values():
    """Empty values, or values that sum to zero, raise ValueError."""
    with pytest.raises(ValueError, match="non-zero sum"):
        an.cumulative_share(pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="non-zero sum"):
        an.cumulative_share(pd.Series([0.0, 0.0]))


def test_pairwise_correlations_skip_missing_values_pair_by_pair():
    """Missing values do not turn the result into NaN; only complete pairs are used."""
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(rng.normal(size=(40, 6)))
    frame.iloc[3, 0] = np.nan
    frame.iloc[7, 2] = np.nan
    result = an.sampled_pairwise_correlations(frame, sample_size=6)
    assert len(result) == 15 and np.isfinite(result).all()


def test_pairwise_correlations_raise_when_no_pair_shares_observations():
    """If the series never overlap, no pair remains and ValueError is raised."""
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0, 5.0]})
    with pytest.raises(ValueError, match="enough observations"):
        an.sampled_pairwise_correlations(frame, sample_size=2)


def test_segment_helpers_reject_duplicate_metadata_ids():
    """Two metadata rows for one VM are ambiguous and raise ValueError in both helpers."""
    metadata = pd.DataFrame({"vm_id": ["a", "a"], "size": ["x", "y"]})
    with pytest.raises(ValueError, match="more than once"):
        an.segment_series(_long_readings(), metadata, "size")
    with pytest.raises(ValueError, match="more than once"):
        an.segment_summary(_long_readings(), metadata, "size")


def test_segment_helpers_put_missing_segment_values_under_no_metadata():
    """A VM whose segment value is missing is grouped with the VMs that have no metadata."""
    metadata = pd.DataFrame({"vm_id": ["a", "b", "c"], "size": ["small", None, "large"]})
    assert an.NO_METADATA in an.segment_series(_long_readings(), metadata, "size").columns
    assert an.segment_summary(_long_readings(), metadata, "size").loc[an.NO_METADATA, "n_vms"] == 1


def test_plot_heatmap_writes_each_value_and_labels_the_axes():
    """Every finite cell gets its value as text, the labels match the frame, and missing cells stay blank."""
    matrix = pd.DataFrame([[1.0, -0.5], [np.nan, 0.25]], index=["a", "b"], columns=["x", "y"])
    fig, ax = plt.subplots()
    image = an.plot_heatmap(ax, matrix, "Title")
    texts = sorted(t.get_text() for t in ax.texts)
    assert texts == ["-0.50", "0.25", "1.00"] and ax.get_title() == "Title"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x", "y"] and [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    assert image.get_clim() == (-1.0, 1.0)
    plt.close(fig)


def test_plot_heatmap_rejects_an_empty_matrix():
    """An empty matrix raises ValueError instead of drawing nothing."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must not be empty"):
        an.plot_heatmap(ax, pd.DataFrame(), "Title")
    plt.close(fig)
