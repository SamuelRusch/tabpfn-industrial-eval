import pandas as pd
import numpy as np
import os

# Load base data
train = pd.read_csv("data/train_data.csv")
dev = pd.read_csv("data/development_data.csv")

# === USE CASE 1: Remove outliers from a numeric feature ===
feature = "Thickness A+B (mm)"
lower = train[feature].quantile(0.05)
upper = train[feature].quantile(0.95)

train_uc1 = train[(train[feature] >= lower) & (train[feature] <= upper)]
dev_uc1 = dev[(dev[feature] >= lower) & (dev[feature] <= upper)]

os.makedirs("data/use_case_1", exist_ok=True)
train_uc1.to_csv("data/use_case_1/train_data.csv", index=False)
dev_uc1.to_csv("data/use_case_1/dev_data.csv", index=False)

# === USE CASE 2: Break correlation by shuffling a feature ===
train_uc2 = train.copy()
train_uc2["Current (A)"] = np.random.permutation(train_uc2["Current (A)"].values)
dev_uc2 = dev.copy()  # unchanged

os.makedirs("data/use_case_2", exist_ok=True)
train_uc2.to_csv("data/use_case_2/train_data.csv", index=False)
dev_uc2.to_csv("data/use_case_2/dev_data.csv", index=False)

# === USE CASE 3: Introduce missing values in a feature ===
train_uc3 = train.copy()
train_uc3.loc[train_uc3.sample(frac=0.1).index, "Force (N)"] = None
dev_uc3 = dev.copy()  # unchanged

os.makedirs("data/use_case_3", exist_ok=True)
train_uc3.to_csv("data/use_case_3/train_data.csv", index=False)
dev_uc3.to_csv("data/use_case_3/dev_data.csv", index=False)

# === USE CASE 4: Remove a category from train, keep it in dev ===
category_feature = "Category"
rare_value = "Explode"

train_uc4 = train[train[category_feature] != rare_value]
dev_uc4 = dev[dev[category_feature] == rare_value]

os.makedirs("data/use_case_4", exist_ok=True)
train_uc4.to_csv("data/use_case_4/train_data.csv", index=False)
dev_uc4.to_csv("data/use_case_4/dev_data.csv", index=False)

