"""Comparison plots for the test-set evaluation."""

import numpy as np

from .evaluation import error_cdf


def plot_rmse_bar(names, rmse_values, path):
    """Bar chart of test RMSE per model."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.color_palette("deep", len(names))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, rmse_values, color=colors)
    plt.ylabel("RMSE (test data)")
    plt.title("Model comparison on test data")
    for bar, value in zip(bars, rmse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(rmse_values),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(results, path):
    """Predicted vs. true values for each model."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    all_true = np.concatenate([np.asarray(r["y_true"]) for r in results])
    for r in results:
        plt.scatter(r["y_true"], r["y_pred"], alpha=0.5, label=r["name"], color=r["color"])
    lo, hi = all_true.min(), all_true.max()
    plt.plot([lo, hi], [lo, hi], "k--", label="Ideal")
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.legend()
    plt.title("Predictions vs. ground truth (test data)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residual_hist(results, path):
    """Histogram of residuals (true minus predicted) per model."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for r in results:
        residuals = np.asarray(r["y_true"]) - np.asarray(r["y_pred"])
        plt.hist(residuals, bins=30, alpha=0.5, label=r["name"], color=r["color"])
    plt.xlabel("Residual (true minus predicted)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Residual distribution (test data)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_error_cdf(results, path):
    """Cumulative distribution of absolute errors per model."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for r in results:
        sorted_errors, cdf = error_cdf(r["y_true"], r["y_pred"])
        plt.plot(sorted_errors, cdf, label=r["name"], color=r["color"])
    plt.xlabel("Absolute error")
    plt.ylabel("Cumulative distribution")
    plt.title("CDF of absolute errors (test data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
