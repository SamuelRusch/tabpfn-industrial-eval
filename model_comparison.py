import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import json
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

# Projektpfad einbinden
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Dev-Daten laden ===
dev_path = os.path.abspath(os.path.join("data", "development_data.csv"))
dev_df = pd.read_csv(dev_path)
X_dev, y_dev = get_features_and_target(dev_df)

# === Dev-Daten für Physikmodell ===
dev_phys_df = pd.read_csv("data/development_data_with_physical_pull.csv")
X_dev_phys, y_true_phys = get_features_and_target(dev_phys_df)
y_phys = dev_phys_df["F_pull_physical"]

# === XGBoost Bias-Modell neu trainieren ===
# Lade Trainings- und Dev-Daten für Bias-Model
train_phys_df = pd.read_csv("data/train_data_with_physical_pull.csv")
full_phys_df = pd.concat([train_phys_df, dev_phys_df], ignore_index=True)
X_full_phys, y_true_full_phys = get_features_and_target(full_phys_df)
y_full_phys = full_phys_df["F_pull_physical"]
y_bias = y_true_full_phys - y_full_phys  # Ziel: Bias

# Lade die besten Hyperparameter
with open("model_training/xgb_physics/tuned/xgb_bias_tuned_v1_params.json", "r") as f:
    xgb_params = json.load(f)

xgb_bias_model = XGBRegressor(**xgb_params)
xgb_bias_model.fit(X_full_phys, y_bias)

# Bias-Vorhersage auf Dev-Daten
X_bias_input = X_dev_phys.drop(columns=["F_pull_physical"], errors="ignore")
bias_pred = xgb_bias_model.predict(X_dev_phys)
f_pull_corrected = y_phys + bias_pred
bias_errors = np.abs(f_pull_corrected - y_true_phys)
bias_rmse = np.sqrt(mean_squared_error(y_true_phys, f_pull_corrected))

# === Modelle & Pfade ===
models = {
    "Random Forest": {
        "path": "model_training/random_forest/final_models/rf_final_model_train_only.pkl",
        "color": "green",
        "marker": "s"
    },
    "XGBoost": {
        "path": "model_training/xgboost/final_models/xgb_final_model_train_only.pkl",
        "color": "red",
        "marker": "D"
    },
    "TabPFN": {
        "path": "model_training/tabpfn/final_models/tabpfn_model_train_only.pkl",
        "color": "orange",
        "marker": "v"
    }
}

plt.figure(figsize=(10, 6))

# === CDF für Bias-Korrektur-Modell ===
sorted_errors = np.sort(bias_errors)
cdf = np.arange(1, len(bias_errors) + 1) / len(bias_errors)
plt.plot(sorted_errors, cdf, label=f"XGBoost with Physics", 
         color="black", marker="*", markevery=1)

# === CDF für jedes Modell ===
for name, cfg in models.items():
    if name == "TabPFN":
        # Train TabPFN fresh
        train_df = pd.read_csv("data/train_data.csv")
        X_train, y_train = get_features_and_target(train_df)
        model = TabPFNRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
    else:
        model = joblib.load(cfg["path"])
        preds = model.predict(X_dev)
    errors = np.abs(preds - y_dev)
    rmse = np.sqrt(mean_squared_error(y_dev, preds))

    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(errors) + 1) / len(errors)

    plt.plot(sorted_errors, cdf, label=f"{name}",
             color=cfg["color"], marker=cfg["marker"], markevery=1)

# === Plot-Design ===
plt.xlabel("absolute error")
plt.ylabel("Cumulative distribution")
plt.title("CDF-Comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# === Plot speichern ===
plot_path = os.path.join("model_training", "model_comparison", "cdf_plot_all_models.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
print(f"\n✅ CDF-Plot gespeichert unter: {plot_path}")
plt.show()

torch_load_old = torch.load
def torch_load_cpu(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return torch_load_old(*args, **kwargs)
torch.load = torch_load_cpu

# Now joblib.load will use CPU mapping for torch models
