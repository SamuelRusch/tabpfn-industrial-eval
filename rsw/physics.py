"""Analytical pull-out force estimate used by the physics-guided model.

The weld nugget diameter is approximated from the minimum sheet thickness
using the recommended minimum weld size d = 4 * sqrt(t_min). The interfacial
failure load is then estimated as F = (pi / 4) * d^2 * tau, modelling the
nugget as a cylinder of diameter d.
"""

import numpy as np
import pandas as pd

from . import config


def min_thickness(df: pd.DataFrame) -> pd.Series:
    """Return the minimum of the two sheet thickness columns per row."""
    return df[config.THICKNESS_COLUMNS].min(axis=1)


def physical_pull_force(t_min, tau: float = config.TAU_WN):
    """Analytical pull-out force from the minimum sheet thickness.

    d = 4 * sqrt(t_min); F = (pi / 4) * d^2 * tau.
    """
    d = 4 * np.sqrt(t_min)
    return (np.pi / 4) * d**2 * tau


def add_physical_pull(df: pd.DataFrame, tau: float = config.TAU_WN) -> pd.DataFrame:
    """Return a copy of df with the minimum thickness and physical pull columns added."""
    out = df.copy()
    out[config.MIN_THICKNESS_COLUMN] = min_thickness(out)
    out[config.PHYSICAL_PULL_COLUMN] = physical_pull_force(out[config.MIN_THICKNESS_COLUMN], tau)
    return out
