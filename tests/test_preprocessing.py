import pandas as pd
import pytest

from rsw.preprocessing import get_features_and_target, get_features_and_target_physical


def make_df():
    return pd.DataFrame(
        {
            "Sample ID": [1, 2],
            "Pressure (PSI)": [40.0, 45.0],
            "Welding Time (ms)": [800, 900],
            "Material": ["AISI 1010", "AISI 1010"],
            "Comments": ["", "ok"],
            "NuggetDiameter (mm)": [5.0, 6.0],
            "Category": ["good", "explode"],
            "PullTest (N)": [3000.0, 2000.0],
            "F_pull_physical": [2500.0, 1800.0],
        }
    )


def test_get_features_and_target_drops_non_features():
    X, y = get_features_and_target(make_df())
    assert list(y) == [3000.0, 2000.0]
    dropped = [
        "Sample ID",
        "Material",
        "Comments",
        "NuggetDiameter (mm)",
        "Category",
        "PullTest (N)",
    ]
    for col in dropped:
        assert col not in X.columns
    assert "Pressure (PSI)" in X.columns
    # The standard extractor keeps the physical column when present.
    assert "F_pull_physical" in X.columns


def test_physical_residual_target():
    X, y = get_features_and_target_physical(make_df())
    assert list(y) == [500.0, 200.0]
    assert "F_pull_physical" not in X.columns
    assert "PullTest (N)" not in X.columns


def test_physical_requires_column():
    df = make_df().drop(columns=["F_pull_physical"])
    with pytest.raises(ValueError):
        get_features_and_target_physical(df)
