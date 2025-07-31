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

def train_one_epoch(train_loader):
    total_loss = 0.0

    for x_cat, x_cont, y in train_loader:
        optimizer.zero_grad()

        # bring y to shape [B,1]
        y = y.unsqueeze(-1) if y.dim()==1 else y

        # forward + backward + step
        pred = model(x_cat, x_cont)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        

    # return the average loss over ALL batches
    return total_loss / len(train_loader)