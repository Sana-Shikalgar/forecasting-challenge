import numpy as np
import pytest

from cwfc.metrics import mae, normalized_mae, rmse


def test_metrics_basic():
    """The basic metrics give the expected values for a small example."""
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)
    assert normalized_mae(y, p, [1, 1, 1]) > 0


def test_perfect_forecast_has_zero_error():
    """Identical inputs give zero for every error metric."""
    assert mae([1.0, 2.0], [1.0, 2.0]) == 0.0 and rmse([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_rmse_is_at_least_mae_and_penalises_one_large_error():
    """RMSE is never below MAE, and one large miss raises it more."""
    y, p = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]
    assert mae(y, p) == 1.0 and rmse(y, p) == 2.0


def test_metrics_accept_arrays_and_lists_alike():
    """Lists, tuples and numpy arrays give the same result."""
    assert mae((1, 2), np.array([2, 4])) == mae([1, 2], [2, 4]) == 1.5


@pytest.mark.parametrize("metric", [mae, rmse])
def test_metrics_reject_shape_mismatch_empty_and_non_finite_input(metric):
    """Mismatched, empty, NaN or infinite inputs raise ValueError instead of returning NaN."""
    with pytest.raises(ValueError, match="same shape"):
        metric([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        metric([], [])
    with pytest.raises(ValueError, match="finite"):
        metric([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        metric([1.0, 2.0], [np.inf, 2.0])


def test_normalized_mae_below_one_when_better_than_the_baseline():
    """A model with half the baseline error scores 0.5."""
    assert normalized_mae([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.5)


def test_normalized_mae_with_an_exact_baseline():
    """An exact baseline gives 0 for an exact model and infinity for any error."""
    assert normalized_mae([1.0], [1.0], [1.0]) == 0.0
    assert normalized_mae([1.0], [2.0], [1.0]) == float("inf")
