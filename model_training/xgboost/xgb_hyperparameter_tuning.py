import os
import sys
import pandas as pd
import numpy as np
import xgboost
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