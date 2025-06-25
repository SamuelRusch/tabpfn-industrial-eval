import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
DATA_PATH = "data/train_data.csv"
DEV_PATH = "data/development_data.csv"
CV_FOLDS = 5
SEED = 42
MODEL_NAME = "tabpfn"

# === Daten einlesen ===
train_df = pd.read_csv(DATA_PATH)
dev_df = pd.read_csv(DEV_PATH)
full_df = pd.concat([train_df, dev_df], ignore_index=True)
X, y = get_features_and_target(full_df)

# === Cross-Validation ===
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
all_metrics = {"MSE": [], "RMSE": []}

print(f"\n🔍 Evaluating model: {MODEL_NAME}")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = TabPFNRegressor(device="cuda" if "cuda" in TabPFNRegressor().device else "cpu")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)

    all_metrics["MSE"].append(mse)
    all_metrics["RMSE"].append(rmse)

    print(f"  📂 Fold {fold + 1}: RMSE={rmse:.3f}")

# === Zusammenfassung
summary = {
    metric: {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals))
    } for metric, vals in all_metrics.items()
}

# === Ergebnisse speichern
output_dir = os.path.join("model_training", MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, f"{MODEL_NAME}_cv_results.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\n✅ Ergebnisse gespeichert unter: {json_path}")

# === Übersicht anzeigen
print("\n📊 Zusammenfassung:")
for metric, values in summary.items():
    print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")

# === Plot erstellen
metric_to_plot = "RMSE"
means = [summary[metric_to_plot]["mean"]]
stds = [summary[metric_to_plot]["std"]]

plt.figure(figsize=(6, 5))
plt.bar([MODEL_NAME], means, yerr=stds, capsize=5)
plt.ylabel(f"{metric_to_plot} (mean ± std)")
plt.title(f"CV Performance of {MODEL_NAME}")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plot_path = os.path.join(output_dir, f"{MODEL_NAME}_cv_comparison_{metric_to_plot.lower()}.png")
plt.savefig(plot_path)
print(f"🖼️ Plot gespeichert unter: {plot_path}")
plt.close()
