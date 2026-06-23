"""Loading the train/development/test splits."""

import pandas as pd

from . import config


def combine_train_dev(train_df: pd.DataFrame, dev_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the training and development splits into one frame.

    The development set is used for tuning and model selection; the final
    models are fit on the union of train and development data.
    """
    return pd.concat([train_df, dev_df], ignore_index=True)


def load_train_dev_test(physical: bool = False):
    """Load the three splits.

    With physical=True, the variants that include the F_pull_physical column
    are loaded instead of the plain splits.
    """
    if physical:
        paths = (config.TRAIN_DATA_PHYSICAL, config.DEV_DATA_PHYSICAL, config.TEST_DATA_PHYSICAL)
    else:
        paths = (config.TRAIN_DATA, config.DEV_DATA, config.TEST_DATA)
    return tuple(pd.read_csv(p) for p in paths)


def load_combined_and_test(physical: bool = False):
    """Return the combined train+development frame and the test frame."""
    train_df, dev_df, test_df = load_train_dev_test(physical=physical)
    return combine_train_dev(train_df, dev_df), test_df
