import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Zugriff auf Projekt-Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target_physical

# CLI Argumente
parser = argparse.ArgumentParser(description="Train XGBoost model (baseline or tuned)")
parser.add_argument("--variant", type=str, choices=["baseline", "tuned"], required=True,
                    help="Model variant: 'baseline' or 'tuned'")
parser.add_argument("--cv", action="store_true", help="Use cross-validation")
parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
parser.add_argument("--seed", type=int, default=42, help="Random seed for CV")
args = parser.parse_args()

variant = args.variant
use_cv = args.cv
cv_folds = args.cv_folds
random_seed = args.seed

# Modellinformationen
model_name = "xgboost"
model_prefix = "xgb_model"

# Daten laden
train_df = pd.read_csv("data/train_data_with_physical_pull.csv")
X, y = get_features_and_target_physical(train_df)

# Modell konfigurieren
if variant == "baseline":
    params = {
        "n_estimators": 100,
        "random_state": random_seed
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
        "random_state": random_seed
    }
    model = XGBRegressor(**params)
    param_suffix = "".join([
        f"_n{params['n_estimators']}",
        f"_md{params['max_depth']}",
        f"_lr{params['learning_rate']:.4f}"
    ])
else:
    raise ValueError("Variant must be either 'baseline' or 'tuned'")

# Cross-Validation
if use_cv:
    print(f"\n🔁 Performing {cv_folds}-Fold Cross-Validation with seed {random_seed}...")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    all_metrics = {"MSE": [], "RMSE": [], "R2": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"📂 Fold {fold + 1}/{cv_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_fold = XGBRegressor(**params)
        model_fold.fit(X_train, y_train)
        preds = model_fold.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)

        all_metrics["MSE"].append(mse)
        all_metrics["RMSE"].append(rmse)
        all_metrics["R2"].append(r2)

        # Optional: Modell pro Fold speichern
        # joblib.dump(model_fold, f"xgb_fold{fold+1}.pkl")

    # Durchschnitt + Std berechnen
    summary = {
        metric: {
            "mean": np.mean(vals),
            "std": np.std(vals)
        } for metric, vals in all_metrics.items()
    }

    # Ausgabe
    print("\n📊 Cross-Validation Results:")
    for metric, stats in summary.items():
        print(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    # Speichern
    output_dir = os.path.join("model_training", model_name, variant)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{model_prefix}_{variant}{param_suffix}physics_cv_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"✅ CV-Metriken gespeichert unter: {summary_path}")

# Klassisches Training (kein CV)
else:
    print(f"\n🚀 Training XGBoost ({variant})...")
    model.fit(X, y)

    # Modell speichern
    output_dir = os.path.join("model_training", model_name, variant)
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"{model_prefix}_{variant}{param_suffix}_physics.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Parameter speichern
    param_filename = model_filename.replace("physics.pkl", "physics_params.json")
    param_path = os.path.join(output_dir, param_filename)
    with open(param_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"📝 Params saved to {param_path}")
