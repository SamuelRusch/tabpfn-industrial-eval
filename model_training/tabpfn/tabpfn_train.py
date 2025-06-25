import os
import sys
import joblib
import pandas as pd
from tabpfn import TabPFNRegressor

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
MODEL_OUTPUT_PATH = "model_training/tabpfn/final_models/tabpfn_model_train_only.pkl"
TRAIN_PATH = "data/train_data.csv"

# === Trainingsdaten laden ===
train_df = pd.read_csv(TRAIN_PATH)
X, y = get_features_and_target(train_df)

# === Modell definieren und trainieren ===
model = TabPFNRegressor(random_state=42)
model.fit(X, y)

# === Modell speichern ===
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)

print(f"✅ TabPFN Modell (nur Trainingsdaten) gespeichert unter: {MODEL_OUTPUT_PATH}")
