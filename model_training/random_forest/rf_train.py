import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target
from preprocessing import get_features_and_target

# Modell
model_name = "random_forest"
ModelClass = RandomForestRegressor
model_args = {"n_estimators": 100, "random_state": 42}

# Alle Use Cases finden
use_cases = sorted([d for d in os.listdir("data") if d.startswith("use_case_")])
output_dir = os.path.join("model_training", model_name)
os.makedirs(output_dir, exist_ok=True)

for use_case in use_cases:
    print("Detected use cases:", use_cases)
    print(f"Training {model_name} on {use_case}...")

    train_path = os.path.join("data", use_case, "train_data.csv")
    df = pd.read_csv(train_path)
    X, y = get_features_and_target(df)

    model = ModelClass(**model_args)
    model.fit(X, y)

    model_filename = f"rf_model_{use_case}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")

print(f"Finished training {model_name} for all use cases.")

