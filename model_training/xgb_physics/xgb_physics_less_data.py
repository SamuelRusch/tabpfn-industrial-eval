import os
import json
import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# === Pfade ===
DATA_PATH = "data/train_data_with_physical_pull.csv"
PARAM_PATH = "model_training/xgb_physics/tuned/xgb_bias_tuned_v1_params.json"
OUTPUT_DIR = "model_training/xgb_physics/bias_subsampling_experiment"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Daten laden ===
df = pd.read_csv(DATA_PATH)

# === Bias berechnen ===
df["bias"] = df["PullTest (N)"] - df["F_pull_physical"]

# === Features und Ziel definieren ===
drop_cols = ["Sample ID", "PullTest (N)", "F_pull_physical", "bias",
             "NuggetDiameter (mm)", "Category", "Comments", "Material"]
X = df.drop(columns=drop_cols, errors="ignore")
X = pd.get_dummies(X)
y = df["bias"]

# === Daten einmalig in Train/Val splitten ===
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🔎 Gesamtgröße Trainingsset: {len(X_train_full)} Samples")
print(f"🔎 Gesamtgröße Validierungsset: {len(X_val)} Samples")

# === Hyperparameter laden ===
with open(PARAM_PATH, "r") as f:
    params = json.load(f)

# === Experiment-Setups definieren ===
fractions = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]
results = []

for frac in fractions:
    # Trainingsdaten subsamplen
    if frac < 1.0:
        X_sub, _, y_sub, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=frac,
            random_state=42
        )
    else:
        X_sub, y_sub = X_train_full, y_train_full

    # Modell trainieren
    print(f"\n🚀 Training Modell mit {frac*100:.0f}% der Trainingsdaten...")
    start_time = time.time()
    model = XGBRegressor(**params)
    model.fit(X_sub, y_sub)
    elapsed = time.time() - start_time

    # Validierungs-Predictions
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"📉 RMSE (Bias) auf Validierungsset: {rmse:.2f}")
    print(f"⏱️ Trainingszeit: {elapsed:.2f} Sekunden")

    # Modell speichern
    filename = f"xgb_bias_model_frac_{int(frac*100)}.pkl"
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, path)
    print(f"✅ Modell gespeichert unter: {path}")

    results.append({
        "fraction": frac,
        "rmse": rmse,
        "time": elapsed,
        "n_samples": len(X_sub)
    })

# === Ergebnisse speichern ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "experiment_results.csv"), index=False)
print("\n📝 Ergebnisse gespeichert unter:", os.path.join(OUTPUT_DIR, "experiment_results.csv"))

# === Plotten ===
plt.figure(figsize=(8,5))
plt.plot(results_df["fraction"]*100, results_df["rmse"], marker='o', label="Validation RMSE")
plt.xlabel("Prozent Trainingsdaten (%)")
plt.ylabel("Validation RMSE (Bias)")
plt.title("XGBoost Bias Model – Performance vs. Data Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_vs_data_size.png"))
print("🖼️ Plot gespeichert unter:", os.path.join(OUTPUT_DIR, "rmse_vs_data_size.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(results_df["fraction"]*100, results_df["time"], marker='s', color='orange', label="Training Time (s)")
plt.xlabel("Prozent Trainingsdaten (%)")
plt.ylabel("Trainingszeit (Sekunden)")
plt.title("XGBoost Bias Model – Trainingszeit vs. Datenmenge")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_time_vs_data_size.png"))
print("🖼️ Plot gespeichert unter:", os.path.join(OUTPUT_DIR, "train_time_vs_data_size.png"))
plt.show()