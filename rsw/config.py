"""Project paths and constants.

All paths are resolved relative to the project root so the package works
regardless of the current working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_TRAINING_DIR = PROJECT_ROOT / "model_training"

# Target variable and reproducibility.
TARGET = "PullTest (N)"
RANDOM_STATE = 42

# Ultimate shear strength of the weld nugget for AISI 1010 steel, in MPa
# (0.8 of the 365 MPa ultimate tensile strength).
TAU_WN = 292.0

# Columns dropped from the feature set: constant, textual, or only available
# through destructive testing (not at inference time).
NON_FEATURE_COLUMNS = [
    "Material",
    "Category",
    "Comments",
    "NuggetDiameter (mm)",
    "Sample ID",
]

# Physics feature columns.
THICKNESS_COLUMNS = ["Thickness A (mm)", "Thickness B (mm)"]
MIN_THICKNESS_COLUMN = "Thickness_min (mm)"
PHYSICAL_PULL_COLUMN = "F_pull_physical"

# Data files.
TRAIN_DATA = DATA_DIR / "train_data.csv"
DEV_DATA = DATA_DIR / "development_data.csv"
TEST_DATA = DATA_DIR / "test_data.csv"
TRAIN_DATA_PHYSICAL = DATA_DIR / "train_data_with_physical_pull.csv"
DEV_DATA_PHYSICAL = DATA_DIR / "development_data_with_physical_pull.csv"
TEST_DATA_PHYSICAL = DATA_DIR / "test_data_with_physical_pull.csv"

# Tuned hyperparameter files selected during model development.
RF_PARAMS = MODEL_TRAINING_DIR / "random_forest" / "tuned" / "rf_model_tuned_v3_params.json"
XGB_PARAMS = MODEL_TRAINING_DIR / "xgboost" / "tuned" / "xgb_model_tuned_v6_params.json"
XGB_PHYSICS_PARAMS = MODEL_TRAINING_DIR / "xgb_physics" / "tuned" / "xgb_bias_tuned_v1_params.json"

# Default output location for comparison plots.
COMPARISON_DIR = MODEL_TRAINING_DIR / "model_comparison"
