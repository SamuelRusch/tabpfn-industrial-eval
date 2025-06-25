import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# === Daten laden ===
df = pd.read_csv("data/train_data_with_physical_pull.csv")

# === Bias berechnen ===
df["bias"] = df["PullTest (N)"] - df["F_pull_physical"]

# === Features und Target ===
drop_cols = ["Sample ID", "PullTest (N)", "F_pull_physical", "bias", "NuggetDiameter (mm)", "Category", "Comments", "Material"]
X = df.drop(columns=drop_cols, errors="ignore")
X = pd.get_dummies(X)  # falls "Material" oder andere Kategorische Merkmale drin sind
y = df["bias"]

# === Modell trainieren ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
bias_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, bias_pred))
print(f"📉 RMSE für Bias-Vorhersage: {rmse:.2f}")

# === Modell speichern ===
output_dir = "model_training/xgboost_bias"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "xgb_bias_model.pkl"))
print(f"✅ Modell gespeichert unter: {output_dir}/xgb_bias_model.pkl")
