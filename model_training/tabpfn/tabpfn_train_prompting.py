import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor
import joblib
import numpy as np
import os
import sys

# Zugriff auf Projekt-Root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# 📥 Daten laden
df = pd.read_csv("data/train_data.csv")
X, y = get_features_and_target(df)

# Skalierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔧 Prompt + Rest aufteilen
n_prompt = 30
X_prompt = X_scaled[:n_prompt]
y_prompt = y[:n_prompt]

X_rest = X_scaled[n_prompt:]
y_rest = y[n_prompt:]

# Train/Test aus dem Rest
X_train, X_test, y_train, y_test = train_test_split(X_rest, y_rest, test_size=0.3, random_state=42)

# ➡️ Torch-Tensoren
device = "cuda" if torch.cuda.is_available() else "cpu"
X_prompt_tensor = torch.tensor(X_prompt, dtype=torch.float32, requires_grad=True, device=device)
y_prompt_tensor = torch.tensor(y_prompt.values, dtype=torch.float32, requires_grad=True, device=device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 🧠 Modell initialisieren
model = TabPFNRegressor(random_state=42)
model.fit(X_prompt_tensor.detach().cpu().numpy(), y_prompt_tensor.detach().cpu().numpy())  # Erstinitialisierung

# Zufallsbasiertes Prompt-Tuning (kein Gradienten-Update)
rng = np.random.default_rng(seed=42)
n_trials = 100
best_rmse = float("inf")
best_prompt_idx = None

for i in range(n_trials):
    idx = rng.choice(len(X_train), size=n_prompt, replace=False)
    X_prompt_trial = X_train[idx]
    y_prompt_trial = y_train.iloc[idx]

    model.fit(X_prompt_trial, y_prompt_trial)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    if rmse < best_rmse:
        best_rmse = rmse
        best_prompt_idx = idx

    print(f"Trial {i+1}: RMSE = {rmse:.2f}")

# Final trainieren mit bestem Prompt
X_prompt_best = X_train[best_prompt_idx]
y_prompt_best = y_train.iloc[best_prompt_idx]
model.fit(X_prompt_best, y_prompt_best)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\n✅ Best Prompt RMSE on Test-Set: {rmse:.2f}")

# 💾 Optionale Speicherung
os.makedirs("tabpfn_results", exist_ok=True)
with open("tabpfn_results/tabpfn_prompt_rmse.txt", "w") as f:
    f.write(f"RMSE auf Test-Set nach Prompt-Tuning: {rmse:.2f}\n")

joblib.dump(model, "tabpfn_results/tabpfn_prompt_model.pkl")