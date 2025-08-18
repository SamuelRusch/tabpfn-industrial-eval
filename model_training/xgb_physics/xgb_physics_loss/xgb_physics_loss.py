import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from preprocessing import get_features_and_target

# Physikalische Formel
tau = 219

def compute_physical_pull(t):
    d = 4 * np.sqrt(t)
    return (np.pi / 4) * d**2 * tau

# Daten laden
df = pd.read_csv("data/train_data.csv")
df["F_pull_physical"] = compute_physical_pull(df["Thickness A+B (mm)"])

# Features und Targets
X, y = get_features_and_target(df, target_column="PullTest (N)")
y_phys = df["F_pull_physical"].values

# Train/Validation Split
X_train, X_val, y_train, y_val, y_phys_train, y_phys_val = train_test_split(
    X, y, y_phys, test_size=0.2, random_state=42
)

# Custom Objective
def make_custom_obj(y_phys_train, lambda_phys=1):
    def custom_obj(preds, dtrain):
        y_data = dtrain.get_label()
        grad = 2 * (preds - y_data) + 2 * lambda_phys * (preds - y_phys_train)
        hess = np.full_like(grad, 2 + 2 * lambda_phys)
        return grad, hess
    return custom_obj

# Training
params = {
    "max_depth": 3,
    "eta": 0.1,
}

evals_result = {}

start_time = time.time()

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

custom_obj = make_custom_obj(y_phys_train)

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    obj=custom_obj,
    evals=[(dtrain, "train"), (dval, "validation")],
    evals_result=evals_result,
    verbose_eval=10
)

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.2f} seconds.")

# Manuelle Physik-RMSE
y_pred_train = bst.predict(dtrain)
y_pred_val = bst.predict(dval)

rmse_phys_train = np.sqrt(np.mean((y_pred_train - y_phys_train) ** 2))
rmse_phys_val = np.sqrt(np.mean((y_pred_val - y_phys_val) ** 2))

print(f"Physics RMSE (train): {rmse_phys_train:.4f}")
print(f"Physics RMSE (val): {rmse_phys_val:.4f}")

# Loss-Kurve plotten
train_rmse = evals_result["train"]["rmse"]
val_rmse = evals_result["validation"]["rmse"]

plt.figure(figsize=(10,6))
plt.plot(train_rmse, label="Train RMSE")
plt.plot(val_rmse, label="Validation RMSE")
plt.axhline(y=rmse_phys_train, color='r', linestyle='--', label="Train Physics RMSE (final)")
plt.axhline(y=rmse_phys_val, color='g', linestyle='--', label="Validation Physics RMSE (final)")
plt.xlabel("Boosting Iteration")
plt.ylabel("RMSE")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()