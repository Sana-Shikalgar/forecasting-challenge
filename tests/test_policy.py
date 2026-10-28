import numpy as np
import pandas as pd
import pytest

from cwfc import policy
from cwfc.evaluate import CVResult


def _result(actual, predicted, origins=(10, 11), timestamps=(3000, 3300)) -> CVResult:
    """Build a CVResult by hand with one forecaster called ``m``."""
    actual = np.asarray(actual, dtype=float)
    return CVResult(np.array(origins), np.array(timestamps), actual, {"m": np.asarray(predicted, dtype=float)}, np.zeros(actual.shape, dtype=bool))


def test_forecast_errors_are_actual_minus_forecast_and_reject_an_unknown_name():
    """A positive error means the forecast was too low."""
    result = _result([[5, 5], [6, 6]], [[4, 6], [6, 3]])
    assert policy.forecast_errors(result, "m").tolist() == [[1, -1], [0, 3]]
    with pytest.raises(ValueError, match="unknown"):
        policy.forecast_errors(result, "other")


def test_realised_quantiles_use_only_errors_known_at_the_origin():
    """At history length 13, step 3 may only use the forecast made at history length 10, whose step 3 is already observed."""
    origins = np.array([10, 11, 12, 13])
    errors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    quantiles = policy.realised_error_quantiles(errors, origins, [0.5], min_errors=1)
    assert quantiles[3, 2, 0] == 3.0
    assert quantiles[3, 1, 0] == np.median([2.0, 5.0])  # step 2 is known for origins 10 and 11
    assert np.isnan(quantiles[0]).all()  # nothing is known at the first origin
    assert np.isnan(policy.realised_error_quantiles(errors, origins, [0.5], min_errors=2)[3, 2, 0])


def test_realised_margins_are_never_negative_and_keep_missing_values_missing():
    """A negative error quantile becomes a zero margin, and entries without enough known errors stay NaN."""
    origins = np.arange(20, 40)
    errors = np.full((20, 3), -1.0)
    margins = policy.realised_margins(errors, origins, [0.9], min_errors=3)
    assert np.nanmax(margins) == 0.0 and np.isnan(margins[0]).all() and not np.isnan(margins[-1]).any()
    assert np.nanmin(policy.realised_margins(-errors, origins, [0.9], min_errors=3)) == 1.0


def test_absolute_quantiles_do_not_depend_on_the_sign_of_the_errors():
    """Flipping the sign of every error leaves the absolute-error quantiles unchanged but moves the signed ones."""
    origins = np.arange(20, 40)
    errors = np.random.default_rng(1).normal(loc=0.5, size=(20, 3))
    same = policy.realised_error_quantiles(errors, origins, [0.9], min_errors=3, absolute=True)
    flipped = policy.realised_error_quantiles(-errors, origins, [0.9], min_errors=3, absolute=True)
    assert np.array_equal(same, flipped, equal_nan=True)
    assert not np.array_equal(policy.realised_error_quantiles(errors, origins, [0.9], min_errors=3), policy.realised_error_quantiles(-errors, origins, [0.9], min_errors=3), equal_nan=True)


def test_realised_quantiles_ignore_errors_that_are_not_known_yet():
    """Changing the error of a forecast whose step is not yet observed leaves the calibration unchanged."""
    origins = np.arange(20, 40)
    rng = np.random.default_rng(0)
    errors = rng.normal(size=(20, 4))
    altered = errors.copy()
    altered[15:] += 100.0  # origins 35 to 39
    before = policy.realised_error_quantiles(errors, origins, [0.1, 0.9], min_errors=3)
    after = policy.realised_error_quantiles(altered, origins, [0.1, 0.9], min_errors=3)
    assert np.array_equal(before[:12], after[:12], equal_nan=True)  # origins up to history 31 cannot see those errors


@pytest.mark.parametrize("probabilities", [[], [0.0], [1.0], [1.2]])
def test_quantile_functions_reject_bad_probabilities(probabilities):
    """Empty probabilities or ones outside (0, 1) raise ValueError."""
    with pytest.raises(ValueError):
        policy.final_error_quantiles(np.ones((3, 2)), probabilities)


def test_realised_quantiles_reject_bad_shapes_and_min_errors():
    """Errors must be 2-D with one row per origin, and at least one error is needed."""
    with pytest.raises(ValueError, match="2-D"):
        policy.realised_error_quantiles(np.ones(5), np.arange(5), [0.5])
    with pytest.raises(ValueError, match="2-D"):
        policy.realised_error_quantiles(np.ones((3, 2)), np.arange(4), [0.5])
    with pytest.raises(ValueError, match="min_errors"):
        policy.realised_error_quantiles(np.ones((3, 2)), np.arange(3), [0.5], min_errors=0)


def test_final_error_quantiles_have_one_row_per_step():
    """The quantiles are taken over all origins separately for each step."""
    errors = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    quantiles = policy.final_error_quantiles(errors, [0.5])
    assert quantiles.shape == (2, 1) and quantiles[:, 0].tolist() == [3.0, 20.0]
    assert policy.final_error_quantiles(-errors, [0.5], absolute=True)[:, 0].tolist() == [3.0, 20.0]
    with pytest.raises(ValueError):
        policy.final_error_quantiles(np.empty((0, 2)), [0.5])


def test_coverage_of_a_well_calibrated_interval_is_close_to_its_level():
    """For independent normal errors the 90% interval covers about 90% of the values."""
    rng = np.random.default_rng(0)
    errors = rng.normal(size=(2000, 3))
    low, high = np.quantile(errors, [0.05, 0.95], axis=0)
    share, checked = policy.coverage(errors, np.zeros_like(errors), np.broadcast_to(low, errors.shape), np.broadcast_to(high, errors.shape))
    assert checked == errors.size and abs(share - 0.9) < 0.02


def test_coverage_skips_missing_bounds_and_checks_shapes():
    """NaN bounds are not counted, an all-NaN input gives NaN, and mismatched shapes raise ValueError."""
    actual = np.array([[1.0, 5.0]])
    share, checked = policy.coverage(actual, np.zeros((1, 2)), np.array([[-2.0, np.nan]]), np.array([[2.0, np.nan]]))
    assert (share, checked) == (1.0, 1)
    share, checked = policy.coverage(actual, np.zeros((1, 2)), np.full((1, 2), np.nan), np.full((1, 2), np.nan))
    assert np.isnan(share) and checked == 0
    with pytest.raises(ValueError, match="same shape"):
        policy.coverage(actual, np.zeros((1, 2)), np.zeros((2, 2)), np.zeros((1, 2)))


def test_coverage_by_step_reports_count_coverage_and_width():
    """Each step gets its own count, coverage and mean width."""
    actual = np.array([[0.0, 9.0], [0.0, 0.0]])
    table = policy.coverage_by_step(actual, np.zeros((2, 2)), np.full((2, 2), -1.0), np.full((2, 2), 1.0))
    assert table["checked"].tolist() == [2, 2]
    assert table["coverage"].tolist() == [1.0, 0.5]
    assert table["mean width"].tolist() == [2.0, 2.0]


def test_headroom_curve_counts_shortfall_excess_and_margin_on_the_same_values():
    """With a forecast of 9 and demand of 10, a margin of 0.5 falls short by 0.5 and a margin of 1.5 leaves 0.5 idle."""
    actual = np.full((2, 2), 10.0)
    forecast = np.full((2, 2), 9.0)
    margins = np.array([np.full((2, 2), 0.5), np.full((2, 2), 1.5)])
    curve = policy.headroom_curve(actual, forecast, margins, ["low", "high"])
    assert curve.loc["low", "under-provisioned share"] == 1.0 and curve.loc["high", "under-provisioned share"] == 0.0
    assert curve.loc["low", "mean shortfall"] == 0.5 and curve.loc["high", "mean excess"] == 0.5
    assert curve["mean margin"].tolist() == [0.5, 1.5] and curve.attrs["values used"] == 4


def test_headroom_curve_skips_values_where_any_option_has_no_margin():
    """A value without a margin for one option is left out for every option."""
    margins = np.array([np.array([[1.0, np.nan]]), np.array([[2.0, 2.0]])])
    curve = policy.headroom_curve(np.array([[5.0, 5.0]]), np.array([[4.0, 4.0]]), margins, ["a", "b"])
    assert curve.attrs["values used"] == 1
    with pytest.raises(ValueError, match="no value"):
        policy.headroom_curve(np.array([[5.0]]), np.array([[4.0]]), np.array([[[np.nan]]]), ["a"])
    with pytest.raises(ValueError, match="shape"):
        policy.headroom_curve(np.array([[5.0]]), np.array([[4.0]]), np.ones((2, 1, 1)), ["a"])


def test_a_higher_cost_of_under_provisioning_makes_a_larger_margin_the_cheapest():
    """When running short costs more, the cheapest option moves to a bigger margin; a non-positive cost is rejected."""
    rng = np.random.default_rng(0)
    actual = rng.normal(size=(300, 4))
    forecast = np.zeros_like(actual)
    options = (0.0, 0.5, 1.0, 1.5, 2.0)
    curve = policy.headroom_curve(actual, forecast, np.array([np.full(actual.shape, m) for m in options]), options)
    cheap, dear = policy.provisioning_cost(curve, 1.0).idxmin(), policy.provisioning_cost(curve, 20.0).idxmin()
    assert dear > cheap
    assert policy.provisioning_cost(curve, 2.0).iloc[0] == 2.0 * curve["mean shortfall"].iloc[0] + curve["mean excess"].iloc[0]
    with pytest.raises(ValueError, match="positive"):
        policy.provisioning_cost(curve, 0)


def test_error_by_position_groups_the_target_timestamps_by_slot_in_the_hour():
    """Two origins of two steps cover slots 0 to 3 here, each with its own count, MAE and bias."""
    result = _result([[2.0, 4.0], [6.0, 8.0]], [[1.0, 5.0], [6.0, 5.0]], origins=(10, 11), timestamps=(0, 300))
    table = policy.error_by_position(result, "m")
    assert table.index.tolist() == [0, 1, 2]
    assert table.loc[1, "values"] == 2 and table.loc[1, "MAE"] == 0.5 and table.loc[1, "bias"] == -0.5
    assert table.loc[2, "bias"] == 3.0 and table.loc[0, "bias"] == 1.0


def test_worst_origins_are_sorted_by_mae():
    """The origin with the largest mean absolute error comes first, and a count below 1 is rejected."""
    result = _result([[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [4.0, 4.0]])
    worst = policy.worst_origins(result, "m", count=1)
    assert worst["history points"].tolist() == [11] and worst["MAE"].tolist() == [3.0] and worst["bias"].tolist() == [-3.0]
    with pytest.raises(ValueError):
        policy.worst_origins(result, "m", count=0)


def test_capacity_plan_orders_lower_forecast_upper_and_checks_lengths():
    """The interval widens around the forecast, the capacity adds a one-sided margin, and mismatched lengths raise ValueError."""
    errors = np.random.default_rng(0).normal(size=(50, 3))
    plan = policy.capacity_plan([10.0, 10.0, 10.0], [100, 400, 700], errors, level=0.9, probability=0.9)
    assert plan.index.tolist() == [100, 400, 700] and plan.index.name == "timestamp"
    assert (plan["lower"] < plan["forecast"]).all() and (plan["upper"] > plan["forecast"]).all()
    assert np.allclose(plan["upper"] - plan["forecast"], plan["forecast"] - plan["lower"])
    assert (plan["capacity"] <= plan["upper"] + 1e-12).all() and (plan["capacity"] > plan["forecast"]).all()
    with pytest.raises(ValueError, match="same number"):
        policy.capacity_plan([10.0, 10.0], [100, 400], errors, level=0.9, probability=0.9)


def test_peak_risk_trigger_is_on_when_the_latest_values_are_high_and_off_after_they_fall():
    """The trigger compares the mean of the latest values with the 80th percentile of the history so far."""
    rising = np.array([1.0] * 10 + [10.0] * 3)
    falling = np.array([10.0] * 10 + [1.0] * 3)
    assert policy.peak_risk_trigger(rising, [13], recent=3).tolist() == [True]
    assert policy.peak_risk_trigger(falling, [13], recent=3).tolist() == [False]


def test_peak_risk_trigger_reads_only_the_values_before_each_origin():
    """Changing values after an origin does not change the trigger at that origin."""
    values = np.random.default_rng(0).normal(size=60)
    altered = values.copy()
    altered[40:] += 50.0
    origins = np.arange(10, 40)
    assert np.array_equal(policy.peak_risk_trigger(values, origins), policy.peak_risk_trigger(altered, origins))


def test_peak_risk_trigger_rejects_bad_settings():
    """A window below one, or origins outside the series, raise ValueError."""
    values = np.arange(20.0)
    with pytest.raises(ValueError, match="recent"):
        policy.peak_risk_trigger(values, [10], recent=0)
    with pytest.raises(ValueError, match="origin"):
        policy.peak_risk_trigger(values, [2], recent=3)
    with pytest.raises(ValueError, match="origin"):
        policy.peak_risk_trigger(values, [21], recent=3)


def test_apply_trigger_zeroes_the_margins_of_untriggered_origins_and_keeps_missing_ones_missing():
    """Triggered origins keep their margins, the others get zero, and NaN stays NaN so the values are still skipped."""
    margins = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, np.nan]])
    result = policy.apply_trigger(margins, [True, False, False])
    assert result[0, 0] == 1.0 and np.isnan(result[0, 1])
    assert result[1].tolist() == [0.0, 0.0] and np.isnan(result[2]).all()
    with pytest.raises(ValueError, match="trigger"):
        policy.apply_trigger(margins, [True, False])


def test_headroom_curve_reports_the_share_of_high_load_values_that_ran_short():
    """With a high-load mask the curve adds the under-provisioned share among the high-load values only."""
    actual = np.array([[10.0, 10.0]])
    forecast = np.array([[9.0, 9.0]])
    margins = np.array([np.full((1, 2), 0.5), np.full((1, 2), 1.5)])
    curve = policy.headroom_curve(actual, forecast, margins, ["low", "high"], high_load=np.array([[True, False]]))
    assert curve["high-load under-provisioned share"].tolist() == [1.0, 0.0]
    none = policy.headroom_curve(actual, forecast, margins, ["low", "high"], high_load=np.zeros((1, 2), dtype=bool))
    assert none["high-load under-provisioned share"].isna().all()
    with pytest.raises(ValueError, match="high_load"):
        policy.headroom_curve(actual, forecast, margins, ["low", "high"], high_load=np.zeros((2, 2), dtype=bool))


def test_choose_option_takes_the_smallest_margin_that_meets_the_target():
    """The cheapest option that keeps high-load shortfalls within the target wins; if none does, the most protective one."""
    curve = pd.DataFrame(
        {"mean margin": [0.0, 0.1, 0.2, 0.3], "high-load under-provisioned share": [0.6, 0.08, 0.04, 0.0]},
        index=["none", "a", "b", "c"],
    )
    assert policy.choose_option(curve, target=0.05) == "b"
    assert policy.choose_option(curve, target=0.1) == "a"
    assert policy.choose_option(curve, target=-0.0) == "c"
    assert policy.choose_option(curve.assign(**{"high-load under-provisioned share": [0.6, 0.5, 0.4, 0.3]}), target=0.05) == "c"
    with pytest.raises(ValueError, match="needs"):
        policy.choose_option(curve.drop(columns="high-load under-provisioned share"))
    with pytest.raises(ValueError, match="target"):
        policy.choose_option(curve, target=1.5)


def test_capacity_plan_never_plans_below_the_forecast():
    """When the errors say the forecast was too high, the capacity equals the forecast and is not lowered."""
    errors = np.full((30, 2), -0.5)
    plan = policy.capacity_plan([10.0, 10.0], [100, 400], errors, level=0.9, probability=0.9)
    assert plan["capacity"].tolist() == [10.0, 10.0]


def test_capacity_plan_without_the_margin_plans_the_forecast_itself():
    """When the trigger is off the capacity equals the forecast while the interval is still given."""
    errors = np.random.default_rng(0).normal(size=(50, 3))
    plan = policy.capacity_plan([10.0, 10.0, 10.0], [100, 400, 700], errors, level=0.9, probability=0.9, margin_active=False)
    assert plan["capacity"].tolist() == [10.0, 10.0, 10.0] and (plan["upper"] > plan["forecast"]).all()
