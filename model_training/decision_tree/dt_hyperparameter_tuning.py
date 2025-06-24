from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import joblib
import json
import numpy as np
import pandas as pd
import os
import sys

# Daten laden
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

train_df = pd.read_csv("data/train_data.csv")
X_train, y_train = get_features_and_target(train_df)
dev_df = pd.read_csv("data/development_data.csv")
X_dev, y_dev = get_features_and_target(dev_df)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Suchraum definieren
space = {
    'max_depth': hp.quniform('max_depth', 3, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'random_state': 0
}

def objective(params):
    model = DecisionTreeRegressor(
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        random_state=params['random_state']
    )
    
    scores = cross_val_score(
        model, X_train, y_train,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1
    )
    rmse = -scores.mean()
    print("CV RMSE:", rmse)
    return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
best_hyperparams = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

# Werte korrekt casten
best_params = best_hyperparams.copy()
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
best_params['max_features'] = [None, 'sqrt', 'log2'][best_params['max_features']]

# Finales Modell trainieren
final_model = DecisionTreeRegressor(**best_params, random_state=0)
final_model.fit(X_train, y_train)

# Vorhersage und Bewertung auf Dev-Set
preds = final_model.predict(X_dev)
rmse = np.sqrt(mean_squared_error(y_dev, preds))
print(f"Finales Modell: RMSE auf Dev-Set: {rmse:.3f}")

# Speichern
joblib.dump(final_model, "decision_tree_best_model.pkl")
with open("decision_tree_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
with open("decision_tree_final_results.txt", "w") as f:
    f.write(f"Finales Modell: RMSE auf Dev-Set: {rmse:.3f}\n")
    f.write("Best Hyperparameters:\n")
    f.write(json.dumps(best_params, indent=2))