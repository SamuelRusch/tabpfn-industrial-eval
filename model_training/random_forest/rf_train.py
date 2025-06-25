import os
import sys
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# === Projektstruktur sicherstellen ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Pfade ===
PARAM_PATH = "model_training/random_forest/tuned/rf_model_tuned_v2_params.json"
DATA_PATH = "data/train_data.csv"
OUTPUT_PATH = "model_training/random_forest/final_models"
MODEL_NAME = "rf_model_tuned_v2.pkl"

# === Daten laden ===
df = pd.read_csv(DATA_PATH)
X, y = get_features_and_target(df)

# === Parameter laden ===
with open(PARAM_PATH, "r") as f:
    params = json.load(f)

# === Modell trainieren ===
model = RandomForestRegressor(**params, n_jobs=-1)
model.fit(X, y)

# === Modell speichern ===
os.makedirs(OUTPUT_PATH, exist_ok=True)
model_path = os.path.join(OUTPUT_PATH, MODEL_NAME)
joblib.dump(model, model_path)

print(f"✅ Modell gespeichert unter: {model_path}")

