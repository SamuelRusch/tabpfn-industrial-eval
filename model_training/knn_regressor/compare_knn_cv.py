import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# === Zugriff auf Projekt-Root sicherstellen ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
DATA_PATH = "data/train_data.csv"
DEV_PATH = "data/development_data.csv"
CV_FOLDS = 5
SEED = 42
NEIGHBORS_LIST = [1, 2, 3, 4, 5, 6]
RESULT_DIR = "model_training/knn"
os.makedirs(RESULT_DIR, exist_ok=True)

# === Daten einlesen ===
train_df = pd.read_csv(DATA_PATH)
dev_df = pd.read_csv(DEV_PATH)
full_df = pd.concat([train_df, dev_df], ignore_index=True)
X, y = get_features_and_target(full_df)

# === Cross-Validation ===
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
results = {}

for k in NEIGHBORS_LIST:
    model_name = f"KNN_k{k}"
    print(f"\n🔍 Evaluating model: {model_name}")
    all_metrics = {"MSE": [], "RMSE": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)

        all_metrics["MSE"].append(mse)
        all_metrics["RMSE"].append(rmse)
        print(f"  📂 Fold {fold+1}: RMSE={rmse:.3f}")

    summary = {
        metric: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        } for metric, vals in all_metrics.items()
    }

    results[model_name] = summary

    # Ergebnisse speichern
    result_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_cv_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Ergebnisse gespeichert unter: {result_path}")

# === Übersicht anzeigen ===
print("\n📊 Zusammenfassung:")
for model, metrics in results.items():
    print(f"\n🔸 {model}")
    for metric, values in metrics.items():
        print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")

# === Plot erstellen ===
metric_to_plot = "RMSE"
model_names = list(results.keys())
means = [results[m][metric_to_plot]["mean"] for m in model_names]
stds = [results[m][metric_to_plot]["std"] for m in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, means, yerr=stds, capsize=5)
plt.ylabel(f"{metric_to_plot} (mean ± std)")
plt.title(f"Comparison of KNN Models ({metric_to_plot})")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# === Grafik speichern ===
plot_path = os.path.join(RESULT_DIR, f"knn_cv_comparison_{metric_to_plot.lower()}.png")
plt.savefig(plot_path)
print(f"\n🖼️ Plot gespeichert unter: {plot_path}")
plt.close()
