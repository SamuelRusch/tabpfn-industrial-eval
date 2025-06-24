import os
import sys
import pandas as pd
import numpy as np
import xgboost
import joblib
print("XGBoost Version:", xgboost.__version__)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost.callback import EarlyStopping

cv = KFold(n_splits=5, shuffle=True, random_state=42)

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

space={'max_depth': hp.quniform("max_depth", 4, 8, 1),
       'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
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
        subsample=space['subsample'],
        learning_rate=space['learning_rate'],
        reg_lambda=space['reg_lambda'],
        eval_metric="rmse",
        seed=0,
        verbosity=0
    )
    
    # Cross Validation: Nur Trainingsdaten verwenden!
    scores = cross_val_score(
        clf, X_train, y_train,
        scoring="neg_root_mean_squared_error",  # oder "neg_mean_squared_error"
        cv=cv,  # cv = z. B. KFold(n_splits=5, shuffle=True, random_state=42)
        n_jobs=-1
    )
    rmse = -scores.mean()
    print("CV RMSE:", rmse)
    return {'loss': rmse, 'status': STATUS_OK}


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

# Parameter ggf. von float zu int umwandeln, falls nötig
best_params = best_hyperparams
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['reg_alpha'] = int(best_params['reg_alpha'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

final_model = xgb.XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],
    min_child_weight=best_params['min_child_weight'],
    colsample_bytree=best_params['colsample_bytree'],
    subsample=best_params['subsample'],
    learning_rate=best_params['learning_rate'],
    reg_lambda=best_params['reg_lambda'],
    eval_metric="rmse",
    seed=0,
    verbosity=0
)

final_model.fit(X_train, y_train)
preds = final_model.predict(X_dev)
mse = mean_squared_error(y_dev, preds)
rmse = np.sqrt(mse)
print(f"Finales Modell: RMSE auf Dev-Set: {rmse:.3f}")

print("The best hyperparameters are : ","\n")
print(best_hyperparams)
joblib.dump(final_model, "xgboost_best_model.pkl")

# Hyperparameter speichern
with open("xgboost_best_params.json", "w") as f:
    json.dump(best_hyperparams, f, indent=2)

# Ergebnisse speichern
with open("xgboost_final_results.txt", "w") as f:
    f.write(f"Finales Modell: RMSE auf Dev-Set: {rmse:.3f}\n")
    f.write("Best Hyperparameters:\n")
    f.write(json.dumps(best_hyperparams, indent=2))