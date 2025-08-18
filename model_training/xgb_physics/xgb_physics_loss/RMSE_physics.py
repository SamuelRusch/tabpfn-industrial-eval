import numpy as np
import pandas as pd
import os
import sys

# Zugriff auf Root sicherstellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from preprocessing import get_features_and_target

# Physikalische Formel
tau = 219

def compute_physical_pull(t):
    d = 4 * np.sqrt(t)
    return (np.pi / 4) * d**2 * tau

# Daten laden

DATA_PATH = "data/train_data.csv"
DEV_PATH = "data/development_data.csv"

train_df = pd.read_csv(DATA_PATH)
dev_df = pd.read_csv(DEV_PATH)
full_df = pd.concat([train_df, dev_df], ignore_index=True)

df = full_df.copy()

# Physikalische Berechnung
df["F_pull_physical"] = compute_physical_pull(df["Thickness A+B (mm)"])

# Target aus Daten
y_data = df["PullTest (N)"].values

# Physikalische Vorhersage
y_phys = df["F_pull_physical"].values

# Physik RMSE berechnen
physics_rmse = np.sqrt(np.mean((y_data - y_phys) ** 2))

print(f"Physics RMSE (full dataset): {physics_rmse:.3f} N")