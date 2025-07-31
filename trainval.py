import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer, FTTransformer
from preprocessing import get_features_and_target
from sklearn.preprocessing import LabelEncoder
from RMSELoss import RMSELoss

dtrain_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")

target_column = "PullTest (N)"  

x_train, y_train = get_features_and_target(train_df, target_column)
x_dev, y_dev = get_features_and_target(dev_df, target_column)

# Define the categorical features
categorical_features = ["Material"]

le = LabelEncoder()

for feature in categorical_features:
    x_train[feature] = le.fit_transform(x_train[feature])
    x_dev[feature] = le.transform(x_dev[feature])  

# Drop categorical features to get the continuous features
x_train_numerical_features = x_train.drop(categorical_features, axis=1)
x_dev_numerical_features = x_dev.drop(categorical_features, axis=1)

# Seperate the categorical features
x_train_categorical_features = x_train[categorical_features]
x_dev_categorical_features = x_dev[categorical_features]

train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float)
x_train_numer_tensor = torch.tensor(x_train_numerical_features.to_numpy(),dtype=torch.float)
x_dev_numer_tensor = torch.tensor(x_dev_numerical_features.to_numpy(),dtype=torch.float)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float)

dev_tensor = torch.tensor(x_dev.to_numpy(), dtype=torch.float)
x_train_categorical_features_tensor = torch.tensor(x_train_categorical_features.to_numpy(),dtype=torch.long)
x_dev_categorical_features_tensor = torch.tensor(x_dev_categorical_features.to_numpy(),dtype=torch.long)
y_dev_tensor = torch.tensor(y_dev.to_numpy(), dtype=torch.float)

from torch.utils.data import TensorDataset,DataLoader

train_ds = TensorDataset(
    x_train_categorical_features_tensor,
    x_train_numer_tensor,
    y_train_tensor
)
val_ds = TensorDataset(
    x_dev_categorical_features_tensor,
    x_dev_numer_tensor,
    y_dev_tensor
)
g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator= g)
val_loader   = DataLoader(val_ds,   batch_size=32)