import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Daten laden ===
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
full_df = pd.concat([train_df, dev_df], ignore_index=True)
X, y = get_features_and_target(full_df)

# === Suchraum definieren ===
space = {
    'max_depth': hp.quniform('max_depth', 3, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'random_state': 0
}

# === Ziel-Funktion mit CV ===
def objective(params):
    converted = {
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": params["max_features"],
        "random_state": params["random_state"]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = DecisionTreeRegressor(**converted)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    avg_rmse = np.mean(rmses)
    print(f"CV RMSE: {avg_rmse:.3f}")
    return {'loss': avg_rmse, 'status': STATUS_OK}

# === Optimierung starten ===
trials = Trials()
best_hyperparams = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

# === Beste Parameter umwandeln ===
converted_params = {
    "max_depth": int(best_hyperparams["max_depth"]),
    "min_samples_split": int(best_hyperparams["min_samples_split"]),
    "min_samples_leaf": int(best_hyperparams["min_samples_leaf"]),
    "max_features": [None, 'sqrt', 'log2'][best_hyperparams["max_features"]],
    "random_state": 0
}

# === JSON speichern ===
output_dir = os.path.join("model_training", "decision_tree", "tuned")
os.makedirs(output_dir, exist_ok=True)

file_name = (
    f"dt_model_tuned_md{converted_params['max_depth']}_ms{converted_params['min_samples_split']}.json"
)
file_path = os.path.join(output_dir, file_name)

with open(file_path, "w") as f:
    json.dump(converted_params, f, indent=4)

print("\n✅ Beste Hyperparameter gespeichert unter:", file_path)