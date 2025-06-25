import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# === Pfade ===
DATA_PATH = "data/train_data_with_physical_pull.csv"
PARAM_PATH = "model_training/xgb_physics/tuned/xgb_bias_tuned_v1_params.json"
OUTPUT_PATH = "model_training/xgb_physics/xgb_bias_model.pkl"

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

# === Daten splitten ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Hyperparameter aus JSON laden ===
with open(PARAM_PATH, "r") as f:
    params = json.load(f)

# === Modell trainieren ===
model = XGBRegressor(**params)
model.fit(X_train, y_train)

# === Evaluation ===
preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"📉 RMSE auf Validierungsdaten (Bias): {rmse:.2f}")

# === Modell speichern ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
joblib.dump(model, OUTPUT_PATH)
print(f"✅ Modell gespeichert unter: {OUTPUT_PATH}")
