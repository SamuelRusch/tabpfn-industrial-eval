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
    """
    # Split target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove columns that are either constant, textual, or unavailable at prediction time
    drop_cols = ["Material", "Category", "Comments", "NuggetDiameter (mm)", "Sample ID"]
    X = X.drop(columns=drop_cols, errors="ignore")

    return X, y

def get_features_and_target_physical(df: pd.DataFrame, target_column: str = "PullTest (N)"):
    """
    Entfernt alle Spalten, die beim Training nicht verwendet werden sollen.
    Ziel: PullTest - F_pull_physical
    """
    drop_cols = [
        "Material",
        "Category",
        "Comments",
        "NuggetDiameter (mm)",
        "Sample ID",
        "F_pull_physical"  
    ]

    X = df.drop(columns=[target_column] + drop_cols, errors="ignore")

    if "F_pull_physical" not in df.columns:
        raise ValueError("Spalte 'F_pull_physical' fehlt.")

    y = df[target_column] - df["F_pull_physical"]
    return X, y
