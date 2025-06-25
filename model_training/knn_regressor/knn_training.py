import os
import sys
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Konfiguration ===
MODEL_OUTPUT_PATH = "model_training/knn_regressor/final_models/knn_k2_model_train_only.pkl"
TRAIN_PATH = "data/train_data.csv"

# === Trainingsdaten laden ===
train_df = pd.read_csv(TRAIN_PATH)
X, y = get_features_and_target(train_df)

# === Modell definieren und trainieren ===
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, y)

# === Modell speichern ===
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)

print(f"✅ KNN Modell (n_neighbors=2, nur Trainingsdaten) gespeichert unter: {MODEL_OUTPUT_PATH}")
