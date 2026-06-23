import numpy as np

from rsw.evaluation import error_cdf, rmse


def test_rmse_known_value():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]
    # Errors are [0, 0, 2]; mean squared error is 4/3.
    assert rmse(y_true, y_pred) == np.sqrt(4 / 3)


def test_rmse_zero_for_perfect():
    assert rmse([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_error_cdf_is_sorted_and_normalized():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [3.0, 1.0, 2.0, 0.0]
    sorted_errors, cdf = error_cdf(y_true, y_pred)
    assert list(sorted_errors) == [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_allclose(cdf, [0.25, 0.5, 0.75, 1.0])
