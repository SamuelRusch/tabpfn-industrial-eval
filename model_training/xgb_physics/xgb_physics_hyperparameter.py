import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

print("XGBoost Version:", xgb.__version__)

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# === Daten laden ===
train_df = pd.read_csv("data/train_data_with_physical_pull.csv")
dev_df = pd.read_csv("data/development_data_with_physical_pull.csv")
full_df = pd.concat([train_df, dev_df], ignore_index=True)

# === Bias berechnen ===
full_df["bias"] = full_df["PullTest (N)"] - full_df["F_pull_physical"]

# === Features und Ziel vorbereiten ===
drop_cols = ["Sample ID", "PullTest (N)", "F_pull_physical", "bias", "NuggetDiameter (mm)", "Category", "Comments", "Material"]
X = full_df.drop(columns=drop_cols, errors="ignore")
X = pd.get_dummies(X)
y = full_df["bias"]

# === Suchraum definieren ===
space = {
    'max_depth': hp.quniform("max_depth", 2, 5, 1),
    'learning_rate': hp.uniform('learning_rate', 0.03, 0.1),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.quniform('reg_alpha', 0, 100, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 250, 10),
    'seed': 0
}

# === Ziel-Funktion für Hyperopt mit CV ===
def objective(space):
    params = {
        "n_estimators": int(space['n_estimators']),
        "max_depth": int(space['max_depth']),
        "learning_rate": float(space['learning_rate']),
        "subsample": float(space['subsample']),
        "gamma": float(space['gamma']),
        "reg_alpha": int(space['reg_alpha']),
        "reg_lambda": float(space['reg_lambda']),
        "colsample_bytree": float(space['colsample_bytree']),
        "min_child_weight": int(space['min_child_weight']),
        "random_state": 42,
        "eval_metric": "rmse",
        "verbosity": 0
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    avg_rmse = np.mean(rmses)
    print(f"CV RMSE: {avg_rmse:.3f}")
    return {'loss': avg_rmse, 'status': STATUS_OK}

# === Optimierung starten ===
trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)

# === Beste Parameter umwandeln ===
converted_params = {
    "n_estimators": int(best_hyperparams["n_estimators"]),
    "max_depth": int(best_hyperparams["max_depth"]),
    "learning_rate": float(best_hyperparams["learning_rate"]),
    "subsample": float(best_hyperparams["subsample"]),
    "gamma": float(best_hyperparams["gamma"]),
    "colsample_bytree": float(best_hyperparams["colsample_bytree"]),
    "min_child_weight": int(best_hyperparams["min_child_weight"]),
    "reg_alpha": int(best_hyperparams["reg_alpha"]),
    "reg_lambda": float(best_hyperparams["reg_lambda"]),
    "random_state": 42
}

# === JSON speichern ===
output_dir = os.path.join("model_training", "xgboost_bias", "tuned")
os.makedirs(output_dir, exist_ok=True)

file_name = (
    f"xgb_physics_tuned_n{converted_params['n_estimators']}"
    f"md{converted_params['max_depth']}"
    f"lr{converted_params['learning_rate']:.4f}_params.json"
)
file_path = os.path.join(output_dir, file_name)

with open(file_path, "w") as f:
    json.dump(converted_params, f, indent=4)

print("\n✅ Beste Hyperparameter gespeichert unter:", file_path)