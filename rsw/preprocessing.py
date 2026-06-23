"""Feature and target extraction for the welding dataset."""

import pandas as pd

from . import config


def get_features_and_target(df: pd.DataFrame, target_column: str = config.TARGET):
    """Split a DataFrame into features (X) and target (y).

    Columns that are constant, textual, or unavailable at inference time are
    dropped from the feature set.

    Returns the feature DataFrame and the target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.drop(columns=config.NON_FEATURE_COLUMNS, errors="ignore")
    return X, y


def get_features_and_target_physical(df: pd.DataFrame, target_column: str = config.TARGET):
    """Build features and the residual target for the physics-guided model.

    The target is the residual between the measured pull force and the
    analytically computed physical pull force (measured minus physical). The
    physical pull column is removed from the feature set here.

    Returns the feature DataFrame and the residual target Series.
    """
    if config.PHYSICAL_PULL_COLUMN not in df.columns:
        raise ValueError(f"Column '{config.PHYSICAL_PULL_COLUMN}' is missing.")

    drop_cols = config.NON_FEATURE_COLUMNS + [config.PHYSICAL_PULL_COLUMN]
    X = df.drop(columns=[target_column] + drop_cols, errors="ignore")
    y = df[target_column] - df[config.PHYSICAL_PULL_COLUMN]
    return X, y
