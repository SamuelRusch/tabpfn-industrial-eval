"""Evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def error_cdf(y_true, y_pred):
    """Empirical CDF of absolute errors.

    Returns the sorted absolute errors and the cumulative probabilities, so a
    point (x, y) reads as "a fraction y of predictions have absolute error at
    most x".
    """
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(errors) + 1) / len(errors)
    return sorted_errors, cdf
