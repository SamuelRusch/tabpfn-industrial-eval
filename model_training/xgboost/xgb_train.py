import os
import sys
import argparse
import joblib
import json
import pandas as pd
from xgboost import XGBRegressor

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# CLI Argumente
parser = argparse.ArgumentParser(description="Train XGBoost model (baseline or tuned)")
parser.add_argument("--variant", type=str, choices=["baseline", "tuned"], required=True,
                    help="Model variant: 'baseline' or 'tuned'")
args = parser.parse_args()
variant = args.variant

# Modellinformationen
model_name = "xgboost"
model_prefix = "xgb_model"

# Daten laden
train_df = pd.read_csv("data/train_data.csv")
X_train, y_train = get_features_and_target(train_df)

# Modell konfigurieren
if variant == "baseline":
    params = {
        "n_estimators": 100,
        "random_state": 42
    }
    model = XGBRegressor(**params)

    param_suffix = f"_n{params['n_estimators']}"

elif variant == "tuned":
    params = {
        "n_estimators": 290,
        "max_depth": 4,
        "learning_rate": 0.04106995869128968,
        "subsample": 0.6201062258952356,
        "gamma": 1.7475309196381257,
        "colsample_bytree": 0.8260112267462354,
        "min_child_weight": 7,
        "reg_alpha": 51,
        "reg_lambda": 0.3804259114008318,
        "random_state": 42
    }
    model = XGBRegressor(**params)

    param_suffix = "".join([
        f"_n{params['n_estimators']}",
        f"_md{params['max_depth']}",
        f"_lr{params['learning_rate']}"
    ])
else:
    raise ValueError("Variant must be either 'baseline' or 'tuned'")

# Training
print(f"\n🚀 Training XGBoost ({variant})...")
model.fit(X_train, y_train)

# Modell speichern
output_dir = os.path.join("model_training", model_name, variant)
os.makedirs(output_dir, exist_ok=True)
model_filename = f"{model_prefix}_{variant}{param_suffix}.pkl"
model_path = os.path.join(output_dir, model_filename)
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Parameter als JSON speichern
param_filename = model_filename.replace(".pkl", "_params.json")
param_path = os.path.join(output_dir, param_filename)
with open(param_path, "w") as f:
    json.dump(params, f, indent=4)
print(f"📝 Params saved to {param_path}")


