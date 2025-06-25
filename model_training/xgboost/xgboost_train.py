import os
import sys
import json
import joblib
import pandas as pd
from xgboost import XGBRegressor

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
PARAM_PATH = "model_training/xgboost/tuned/xgb_model_tuned_v6_params.json"
MODEL_OUTPUT_PATH = "model_training/xgboost/final_models/xgb_final_model_train_only.pkl"
TRAIN_PATH = "data/train_data.csv"

# === Trainingsdaten laden ===
train_df = pd.read_csv(TRAIN_PATH)
X, y = get_features_and_target(train_df)

# === Beste Hyperparameter laden ===
with open(PARAM_PATH, "r") as f:
    params = json.load(f)

# === Modell trainieren ===
model = XGBRegressor(**params)
model.fit(X, y)

# === Modell speichern ===
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)

print(f"✅ XGBoost-Modell (nur mit Trainingsdaten) gespeichert unter: {MODEL_OUTPUT_PATH}")
