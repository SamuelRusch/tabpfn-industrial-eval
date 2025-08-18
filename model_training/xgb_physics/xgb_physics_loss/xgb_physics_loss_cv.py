import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from preprocessing import get_features_and_target

# Physikalische Formel
tau = 219

def compute_physical_pull(t):
    d = 4 * np.sqrt(t)
    return (np.pi / 4) * d**2 * tau

# Daten laden
DATA_PATH = "data/train_data.csv"
DEV_PATH = "data/development_data.csv"

train_df = pd.read_csv(DATA_PATH)
dev_df = pd.read_csv(DEV_PATH)
full_df = pd.concat([train_df, dev_df], ignore_index=True)

df = full_df.copy()
df["F_pull_physical"] = compute_physical_pull(df["Thickness A+B (mm)"])

# Features und Targets
X, y = get_features_and_target(df, target_column="PullTest (N)")
y_phys = df["F_pull_physical"].values

# CV-Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    "max_depth": 3,
    "eta": 0.1,
}

num_boost_round = 100

val_rmse_list = []

start_time = time.time()

fold = 0

for train_idx, val_idx in kf.split(X):
    fold += 1

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_phys_train = y_phys[train_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def make_custom_obj(y_phys_train, lambda_phys=1):
        def custom_obj(preds, dtrain):
            y_data = dtrain.get_label()
            grad = (preds - y_data) + lambda_phys * (preds - y_phys_train)
            hess = np.full_like(grad, 2 + 2 * lambda_phys)
            return grad, hess
        return custom_obj

    custom_obj = make_custom_obj(y_phys_train)

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        obj=custom_obj,
        evals=[(dval, "validation")],
        verbose_eval=False
    )

    y_pred_val = bst.predict(dval)
    val_rmse = np.sqrt(np.mean((y_pred_val - y_val) ** 2))
    val_rmse_list.append(val_rmse)

    print(f"Fold {fold}: Validation RMSE = {val_rmse:.3f}")

elapsed = time.time() - start_time

print(f"\nMean Validation RMSE: {np.mean(val_rmse_list):.3f} ± {np.std(val_rmse_list):.3f}")
print(f"Total CV time: {elapsed:.2f} seconds")

# === Plot erstellen (Barplot) ===
model_name = "xgb_physics"

mean_rmse = np.mean(val_rmse_list)
std_rmse = np.std(val_rmse_list)

plt.figure(figsize=(6, 5))
plt.bar([model_name], [mean_rmse], yerr=[std_rmse], capsize=5)
plt.ylabel("Validation RMSE (mean ± std)")
plt.title(f"CV Performance of {model_name}")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Optional speichern
output_dir = os.path.join("model_training", model_name)
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, f"{model_name}_cv_comparison_rmse.png")
plt.savefig(plot_path)
print(f"🖼️ Plot gespeichert unter: {plot_path}")
plt.show()