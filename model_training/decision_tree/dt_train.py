import sys
import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor

# Zugriff auf Projekt-Root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# Modell-Definition
model_name = "decision_tree"
model_prefix = "dt_model"
model = DecisionTreeRegressor(random_state=42)

# Use Cases durchlaufen
use_cases = sorted([d for d in os.listdir("data") if d.startswith("use_case_")])
print(f"Found use cases: {use_cases}")

output_dir = os.path.join("model_training", model_name)
os.makedirs(output_dir, exist_ok=True)

for uc in use_cases:
    print(f"\nTraining {model_name} on {uc}...")

    train_path = os.path.join("data", uc, "train_data.csv")
    df = pd.read_csv(train_path)
    X, y = get_features_and_target(df)

    model.fit(X, y)

    model_path = os.path.join(output_dir, f"{model_prefix}_{uc}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

print(f"\nDone training {model_name} for all use cases.")