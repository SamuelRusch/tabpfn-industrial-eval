import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error

# Projektpfad einbinden
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Daten laden ===
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
test_df = pd.read_csv("data/test_data.csv")

# Kombiniere Trainings- und Entwicklungsdaten
full_train_df = pd.concat([train_df, dev_df], ignore_index=True)
X_train, y_train = get_features_and_target(full_train_df)
X_test, y_test = get_features_and_target(test_df)

# === Random Forest ===
with open("model_training/random_forest/tuned/rf_model_tuned_v3_params.json", "r") as f:
    rf_params = json.load(f)
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# === XGBoost ===
with open("model_training/xgboost/tuned/xgb_model_tuned_v6_params.json", "r") as f:
    xgb_params = json.load(f)
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

# === TabPFN ===
tabpfn_model = TabPFNRegressor(random_state=42)
tabpfn_model.fit(X_train, y_train)
tabpfn_preds = tabpfn_model.predict(X_test)
tabpfn_rmse = np.sqrt(mean_squared_error(y_test, tabpfn_preds))

# === XGBoost Physics-Bias ===
train_phys_df = pd.read_csv("data/train_data_with_physical_pull.csv")
dev_phys_df = pd.read_csv("data/development_data_with_physical_pull.csv")
test_phys_df = pd.read_csv("data/test_data_with_physical_pull.csv")
full_phys_df = pd.concat([train_phys_df, dev_phys_df], ignore_index=True)
X_phys_train, y_true_phys_train = get_features_and_target(full_phys_df)
y_phys_train = full_phys_df["F_pull_physical"]
y_bias_train = y_true_phys_train - y_phys_train

X_phys_test, y_true_phys_test = get_features_and_target(test_phys_df)
y_phys_test = test_phys_df["F_pull_physical"]

with open("model_training/xgb_physics/tuned/xgb_bias_tuned_v1_params.json", "r") as f:
    xgb_phys_params = json.load(f)
xgb_phys_model = XGBRegressor(**xgb_phys_params)
xgb_phys_model.fit(X_phys_train, y_bias_train)
bias_pred_test = xgb_phys_model.predict(X_phys_test)
f_pull_corrected_test = y_phys_test + bias_pred_test
xgb_phys_rmse = np.sqrt(mean_squared_error(y_true_phys_test, f_pull_corrected_test))

# === Vergleich ausgeben ===
print("\nRMSE auf Testdaten:")
print(f"Random Forest: {rf_rmse:.2f}")
print(f"XGBoost: {xgb_rmse:.2f}")
print(f"TabPFN: {tabpfn_rmse:.2f}")
print(f"XGBoost with Physics-Bias: {xgb_phys_rmse:.2f}")

# === Plot-Folder erstellen ===
plot_dir = "model_training/model_comparison"
os.makedirs(plot_dir, exist_ok=True)

# === 1. RMSE Bar Plot ===
plt.figure(figsize=(10, 6))
model_names = ["Random Forest", "XGBoost", "TabPFN", "XGBoost Physics-Bias"]
rmse_values = [rf_rmse, xgb_rmse, tabpfn_rmse, xgb_phys_rmse]
colors = sns.color_palette("deep", len(model_names))

bars = plt.bar(model_names, rmse_values, color=colors)

plt.ylabel("RMSE (Test Data)")
plt.title("Model Comparison on Test Data")

# Annotate RMSE values on bars
for bar, rmse in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(rmse_values),
             f"{rmse:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "model_rmse_test.png"))
plt.close()

# === 2. Scatter Plot: Predictions vs. Ground Truth ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.5, label="Random Forest", color="green")
plt.scatter(y_test, xgb_preds, alpha=0.5, label="XGBoost", color="red")
plt.scatter(y_test, tabpfn_preds, alpha=0.5, label="TabPFN", color="orange")
plt.scatter(y_true_phys_test, f_pull_corrected_test, alpha=0.5, label="XGBoost Physics-Bias", color="black")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Ideal")
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.legend()
plt.title("Predictions vs. Ground Truth (Test Data)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_pred_vs_true.png"))
plt.close()

# === 3. Residual Histogram ===
plt.figure(figsize=(8, 6))
plt.hist(y_test - rf_preds, bins=30, alpha=0.5, label="Random Forest", color="green")
plt.hist(y_test - xgb_preds, bins=30, alpha=0.5, label="XGBoost", color="red")
plt.hist(y_test - tabpfn_preds, bins=30, alpha=0.5, label="TabPFN", color="orange")
plt.hist(y_true_phys_test - f_pull_corrected_test, bins=30, alpha=0.5, label="XGBoost Physics-Bias", color="black")
plt.xlabel("Residual (True - Predicted)")
plt.ylabel("Count")
plt.legend()
plt.title("Residual Distribution (Test Data)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "residual_histogram.png"))
plt.close()

# === 4. CDF of Absolute Errors ===
plt.figure(figsize=(8, 6))
for name, errors, color in [
    ("Random Forest", np.abs(y_test - rf_preds), "green"),
    ("XGBoost", np.abs(y_test - xgb_preds), "red"),
    ("TabPFN", np.abs(y_test - tabpfn_preds), "orange"),
    ("XGBoost Physics-Bias", np.abs(y_true_phys_test - f_pull_corrected_test), "black"),
]:
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(errors) + 1) / len(errors)
    plt.plot(sorted_errors, cdf, label=name, color=color)
plt.xlabel("Absolute Error")
plt.ylabel("Cumulative Distribution")
plt.title("CDF of Absolute Errors (Test Data)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cdf_absolute_errors.png"))
plt.close()