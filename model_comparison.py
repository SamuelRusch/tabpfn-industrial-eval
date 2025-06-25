import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error

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

# === Bias-Modell laden ===
bias_model_path = "model_training/xgb_physics/xgb_bias_model.pkl"
bias_model = joblib.load(bias_model_path)
X_bias_input = X_dev_phys.drop(columns=["F_pull_physical"], errors="ignore")
bias_pred = bias_model.predict(X_bias_input)
f_pull_corrected = y_phys + bias_pred
bias_errors = np.abs(f_pull_corrected - y_true_phys)
bias_rmse = np.sqrt(mean_squared_error(y_true_phys, f_pull_corrected))

# === Modelle & Pfade ===
models = {
    "Decision Tree": {
        "path": "model_training/decision_tree/final_models/dt_final_model_train_only.pkl",
        "color": "blue",
        "marker": "o"
    },
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
    "KNN (k=2)": {
        "path": "model_training/knn_regressor/final_models/knn_final_model_train_only.pkl",
        "color": "purple",
        "marker": "^"
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
plt.plot(sorted_errors, cdf, label=f"Physics + XGBoost Bias (RMSE: {bias_rmse:.2f})", 
         color="black", marker="*", markevery=1)

# === CDF für jedes Modell ===
for name, cfg in models.items():
    model = joblib.load(cfg["path"])
    preds = model.predict(X_dev)
    errors = np.abs(preds - y_dev)
    rmse = np.sqrt(mean_squared_error(y_dev, preds))

    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(errors) + 1) / len(errors)

    plt.plot(sorted_errors, cdf, label=f"{name} (RMSE: {rmse:.2f})", 
             color=cfg["color"], marker=cfg["marker"], markevery=1)

# === Plot-Design ===
plt.xlabel("Absoluter Fehler")
plt.ylabel("Kumulative Verteilung")
plt.title("CDF-Vergleich der Modelle auf dem Dev-Datensatz")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# === Plot speichern ===
plot_path = os.path.join("model_training", "model_comparison", "cdf_plot_all_models.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
print(f"\n✅ CDF-Plot gespeichert unter: {plot_path}")
plt.show()
