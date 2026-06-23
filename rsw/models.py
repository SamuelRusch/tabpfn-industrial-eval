"""Model factory functions.

XGBoost and TabPFN are imported lazily so the package can be imported (and the
preprocessing/physics/evaluation code tested) without those heavy dependencies
installed.
"""

import json
from pathlib import Path

from . import config


def load_params(path: Path) -> dict:
    """Load a tuned-hyperparameter JSON file."""
    with open(path) as f:
        return json.load(f)


def random_forest(params: dict | None = None):
    """Random Forest regressor with the tuned hyperparameters."""
    from sklearn.ensemble import RandomForestRegressor

    if params is None:
        params = load_params(config.RF_PARAMS)
    return RandomForestRegressor(**params)


def xgboost(params: dict | None = None):
    """XGBoost regressor with the tuned hyperparameters."""
    from xgboost import XGBRegressor

    if params is None:
        params = load_params(config.XGB_PARAMS)
    return XGBRegressor(**params)


def xgboost_physics(params: dict | None = None):
    """XGBoost regressor for residual learning on the physics-guided target."""
    from xgboost import XGBRegressor

    if params is None:
        params = load_params(config.XGB_PHYSICS_PARAMS)
    return XGBRegressor(**params)


def tabpfn(random_state: int = config.RANDOM_STATE):
    """TabPFN regressor in its default configuration (no task-specific tuning)."""
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(random_state=random_state)
