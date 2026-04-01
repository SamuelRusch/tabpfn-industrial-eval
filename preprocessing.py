# preprocessing.py

import pandas as pd

def get_features_and_target(df: pd.DataFrame, target_column: str = "PullTest (N)"):
    """
    Splits a DataFrame into features (X) and target (y),
    and removes columns that are not available at inference time.

    Parameters:
    - df: full DataFrame with features and target
    - target_column: name of the target variable (default: 'PullTest (N)')

    Returns:
    - X: preprocessed feature DataFrame
    - y: target Series
    - z: stratification Series
    """
    # Split target
    X = df.drop(columns=[target_column])
    y = df[target_column]


    # Remove columns that are either constant, textual, or unavailable at prediction time
    drop_cols = ["Sample ID", "Category",  "Material", "Comments", "NuggetDiameter (mm)","PullTest (N)"]
    X = X.drop(columns=drop_cols, errors="ignore")

    return X, y

def get_features_and_target_classification(df: pd.DataFrame, target_column: str = "PullTest (N)"):
    """
    Splits a DataFrame into features (X) and target (y),
    and removes columns that are not available at inference time.

    Parameters:
    - df: full DataFrame with features and target
    - target_column: name of the target variable (default: 'PullTest (N)')

    Returns:
    - X: preprocessed feature DataFrame
    - y: target Series
    """
    # Split target
    X = df.drop(columns=[target_column])
    y = df[["Sample ID", target_column]]

    # Remove columns that are either constant, textual, or unavailable at prediction time
    drop_cols = ["PullTest (N)", "Material", "Comments", "NuggetDiameter (mm)"]
    X = X.drop(columns=drop_cols, errors="ignore")

    return X, y
