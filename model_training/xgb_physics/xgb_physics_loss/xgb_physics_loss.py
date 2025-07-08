import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_float_info("phys_pull", y_phys_train)

dval = xgb.DMatrix(X_val, label=y_val)
dval.set_float_info("phys_pull", y_phys_val)

# Custom Objective
def custom_obj(preds, dtrain):
    y_data = dtrain.get_label()
    y_phys = dtrain.get_float_info("phys_pull")
    lambda_phys = 1
    grad = 2 * (preds - y_data) + 2 * lambda_phys * (preds - y_phys)
    hess = np.full_like(grad, 2 + 2 * lambda_phys)
    return grad, hess

# Custom Eval
def physics_rmse_eval(preds, dtrain):
    y_phys = dtrain.get_float_info("phys_pull")
    rmse_phys = np.sqrt(np.mean((preds - y_phys)**2))
    return "rmse_phys", rmse_phys

# Training
params = {
    "max_depth": 3,
    "eta": 0.1,
}

evals_result = {}

start_time = time.time()

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    obj=custom_obj,
    feval=physics_rmse_eval,
    evals=[(dtrain, "train"), (dval, "validation")],
    evals_result=evals_result,
    verbose_eval=10
)

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.2f} seconds.")

# Loss-Kurve plotten
train_rmse = evals_result["train"]["rmse"]
val_rmse = evals_result["validation"]["rmse"]
train_phys = evals_result["train"]["rmse_phys"]
val_phys = evals_result["validation"]["rmse_phys"]

plt.figure(figsize=(10,6))
plt.plot(train_rmse, label="Train RMSE")
plt.plot(val_rmse, label="Validation RMSE")
plt.plot(train_phys, label="Train Physics RMSE")
plt.plot(val_phys, label="Validation Physics RMSE")
plt.xlabel("Boosting Iteration")
plt.ylabel("RMSE")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
