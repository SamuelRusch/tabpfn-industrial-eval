import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from kneed import KneeLocator
import joblib

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
DATA_PATH = "data/train_data.csv"
DEV_PATH = "data/development_data.csv"
CV_FOLDS = 5
SEED = 42
PLOT_PATH = "elbow_method_knn_cv.png"
MODEL_PATH = "knn_k{optimal_k}_model.pkl"
RESULT_PATH = "knn_k{optimal_k}_cv_results.json"

# === Daten einlesen ===
train_df = pd.read_csv(DATA_PATH)
dev_df = pd.read_csv(DEV_PATH)
full_df = pd.concat([train_df, dev_df], ignore_index=True)
X, y = get_features_and_target(full_df)

# === Cross-Validator ===
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# === Elbow-Methode für optimale k-Wahl ===
k_values = range(1, 21)
cv_errors = []
for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    cv_errors.append(-scores.mean())

# === Kniepunkt finden ===
knee = KneeLocator(k_values, cv_errors, curve="convex", direction="decreasing")
optimal_k = knee.knee if knee.knee else 5
print(f"🔎 Optimale Anzahl an Nachbarn (k): {optimal_k}")

# === Plot speichern ===
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_errors, marker='o')
plt.xlabel("Anzahl Nachbarn (k)")
plt.ylabel("MSE (CV)")
plt.title("Elbow-Methode für KNN (Cross Validation)")
if knee.knee:
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Knee: k={optimal_k}")
    plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

# === Finales Modell mit optimalem k ===
best_knn = KNeighborsRegressor(n_neighbors=optimal_k)
final_scores = cross_val_score(best_knn, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
final_rmse_scores = -final_scores
final_rmse_mean = final_rmse_scores.mean()
final_rmse_std = final_rmse_scores.std()

print(f"✅ Finales Modell mit k={optimal_k}:")
print(f"   RMSE (CV): {final_rmse_mean:.3f} ± {final_rmse_std:.3f}")

# === Modell speichern ===
joblib.dump(best_knn, MODEL_PATH.format(optimal_k=optimal_k))

# === Ergebnisse speichern ===
result_dict = {
    "optimal_k": optimal_k,
    "cv_rmse_mean": final_rmse_mean,
    "cv_rmse_std": final_rmse_std,
    "cv_rmse_scores": final_rmse_scores.tolist()
}
with open(RESULT_PATH.format(optimal_k=optimal_k), "w") as f:
    json.dump(result_dict, f, indent=2)

print("📁 Ergebnisse gespeichert:")
print(f"   Modell: {MODEL_PATH.format(optimal_k=optimal_k)}")
print(f"   Plot:   {PLOT_PATH}")
print(f"   Werte:  {RESULT_PATH.format(optimal_k=optimal_k)}")
