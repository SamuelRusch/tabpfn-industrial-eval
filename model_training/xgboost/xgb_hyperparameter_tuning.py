import os
import sys
import pandas as pd
import numpy as np
import xgboost
import json
print("XGBoost Version:", xgboost.__version__)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost.callback import EarlyStopping


from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target


# Modellinformationen
model_name = "xgboost"
model_prefix = "xgb_model"

# Daten laden
train_df = pd.read_csv("data/train_data.csv")
X_train, y_train = get_features_and_target(train_df)
dev_df = pd.read_csv("data/development_data.csv")
X_dev, y_dev = get_features_and_target(dev_df)

space={'max_depth': hp.quniform("max_depth", 2, 4, 1),
       'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
       'subsample': hp.uniform('subsample', 0.6, 1.0),
        'gamma': hp.uniform ('gamma', 0,9),
        'reg_alpha' : hp.quniform('reg_alpha', 0,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 200, 350, 10),
        'seed': 0
    }

def objective(space):
    clf = xgb.XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'],
        eval_metric="rmse",         
        seed=0,
        early_stopping_rounds = 10 
    )
    
    evaluation = [( X_train, y_train), ( X_dev, y_dev)]
    
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_dev, y_dev)],
        verbose=False
    )
    

    pred = clf.predict(X_dev)
    mse = mean_squared_error(y_dev, pred)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    return {'loss': rmse, 'status': STATUS_OK}


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

# Pfad und Umwandlung der besten Parameter
output_dir = os.path.join("model_training", model_name, "tuned")
os.makedirs(output_dir, exist_ok=True)

# Umwandlung von Float zu int, wo nötig
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

# Dateiname anhand von Parametern generieren
file_name = f"xgb_model_tuned_n{converted_params['n_estimators']}md{converted_params['max_depth']}lr{converted_params['learning_rate']:.4f}_params.json"
file_path = os.path.join(output_dir, file_name)

# Speichern als JSON
with open(file_path, "w") as f:
    json.dump(converted_params, f, indent=4)

print(f"✅ Beste Hyperparameter gespeichert unter: {file_path}")
