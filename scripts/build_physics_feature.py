"""Add the analytical pull-out force feature to each data split.

Reads the plain train/development/test splits, computes the minimum sheet
thickness and the physical pull force, and writes the *_with_physical_pull.csv
files used by the physics-guided model.
"""

import pandas as pd

from rsw import config
from rsw.physics import add_physical_pull

SPLITS = [
    (config.TRAIN_DATA, config.TRAIN_DATA_PHYSICAL),
    (config.DEV_DATA, config.DEV_DATA_PHYSICAL),
    (config.TEST_DATA, config.TEST_DATA_PHYSICAL),
]


def main():
    for source, target in SPLITS:
        df = pd.read_csv(source)
        df = add_physical_pull(df)
        df.to_csv(target, index=False)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
