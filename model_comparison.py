import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import get_features_and_target
from tabpfn import TabPFNRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Use seaborn darkgrid style
sns.set_style("darkgrid")

# Define models
models = {
    "TabPFN": TabPFNRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# Discover all use case directories
use_cases = sorted([d for d in os.listdir("data") if d.startswith("use_case_")])
results_base = "results"
os.makedirs(results_base, exist_ok=True)

for uc in use_cases:
    # Paths for this use case
    train_path = os.path.join("data", uc, "train_data.csv")
    dev_path = os.path.join("data", uc, "dev_data.csv")

    # Load and preprocess data
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    X_train, y_train = get_features_and_target(train_df)
    X_dev, y_dev = get_features_and_target(dev_df)
    X_dev = X_dev[X_train.columns]  # align columns

    # Prepare structures for results
    results = {}
    cdf_data = {}

    # Evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)

        mse = mean_squared_error(y_dev, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_dev, preds)

        results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

        abs_errors = np.abs(preds - y_dev)
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        cdf_data[name] = (sorted_errors, cdf)

    # Create results directory
    uc_results_dir = os.path.join(results_base, uc)
    os.makedirs(uc_results_dir, exist_ok=True)

    # Save results as CSV
    results_df = pd.DataFrame(results).T.round(2)
    results_df.to_csv(os.path.join(uc_results_dir, "results.csv"))

    # Plot and save individual metric charts
    for metric in ["MSE", "RMSE", "R2"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        results_df[metric].plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per Model - {uc}")
        fig.tight_layout()
        fig.savefig(os.path.join(uc_results_dir, f"metric_{metric.lower()}.png"))
        plt.close(fig)


    # Plot and save CDF chart
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (errors, cdf) in cdf_data.items():
        ax.plot(errors, cdf, marker=".", linestyle="none", label=name)
    ax.set_xlabel("Absolute Error |ŷ - y| [N]")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of Absolute Errors - {uc}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(uc_results_dir, "cdf_errors.png"))
    plt.close(fig)

print("Model evaluation complete. Check the 'results/' folder for metrics and plots.")