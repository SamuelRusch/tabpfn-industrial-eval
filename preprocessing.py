# preprocessing.py

import pandas as pd


def get_features_and_target(df: pd.DataFrame, target_column: str = "PullTest (N)"):
    """Split a DataFrame into features (X) and target (y).

    Columns that are constant, textual, or unavailable at inference time are
    dropped from the feature set.

    Parameters:
    - df: full DataFrame with features and target
    - target_column: name of the target variable (default: 'PullTest (N)')

    Returns:
    - X: feature DataFrame
    - y: target Series
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    drop_cols = ["Material", "Category", "Comments", "NuggetDiameter (mm)", "Sample ID"]
    X = X.drop(columns=drop_cols, errors="ignore")

    return X, y


def get_features_and_target_physical(df: pd.DataFrame, target_column: str = "PullTest (N)"):
    """Build features and the residual target for the physics-guided model.

    The target is the residual between the measured pull force and the
    analytically computed physical pull force (PullTest - F_pull_physical).
    The F_pull_physical column is removed from the feature set here.

    Parameters:
    - df: full DataFrame including the F_pull_physical column
    - target_column: name of the measured target (default: 'PullTest (N)')

    Returns:
    - X: feature DataFrame
    - y: residual target Series (measured minus physical)
    """
    drop_cols = [
        "Material",
        "Category",
        "Comments",
        "NuggetDiameter (mm)",
        "Sample ID",
        "F_pull_physical",
    ]

    X = df.drop(columns=[target_column] + drop_cols, errors="ignore")

    if "F_pull_physical" not in df.columns:
        raise ValueError("Column 'F_pull_physical' is missing.")

    y = df[target_column] - df["F_pull_physical"]
    return X, y
