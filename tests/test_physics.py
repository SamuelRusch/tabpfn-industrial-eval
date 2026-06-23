import math

import numpy as np
import pandas as pd

from rsw import config
from rsw.physics import add_physical_pull, min_thickness, physical_pull_force


def test_min_thickness_takes_minimum():
    df = pd.DataFrame({"Thickness A (mm)": [1.0, 2.0], "Thickness B (mm)": [1.5, 0.8]})
    assert list(min_thickness(df)) == [1.0, 0.8]


def test_physical_pull_force_formula():
    # d = 4 * sqrt(t); F = (pi / 4) * d^2 * tau = 4 * pi * t * tau.
    t, tau = 1.0, 292.0
    expected = 4 * math.pi * t * tau
    assert physical_pull_force(t, tau) == expected


def test_add_physical_pull_columns():
    df = pd.DataFrame(
        {
            "Thickness A (mm)": [1.0, 2.0],
            "Thickness B (mm)": [1.2, 1.0],
            "PullTest (N)": [3000.0, 4000.0],
        }
    )
    out = add_physical_pull(df)
    assert list(out[config.MIN_THICKNESS_COLUMN]) == [1.0, 1.0]
    expected = 4 * np.pi * np.array([1.0, 1.0]) * config.TAU_WN
    np.testing.assert_allclose(out[config.PHYSICAL_PULL_COLUMN].to_numpy(), expected)
    # Original columns are preserved.
    assert "PullTest (N)" in out.columns
