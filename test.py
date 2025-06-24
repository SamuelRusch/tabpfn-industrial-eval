import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import get_features_and_target_physical  # Stelle sicher, dass diese Funktion korrekt ist

sns.set_style("darkgrid")

# 📁 Dateien und Modellpfade
residual_model_path = "model_training/xgboost/baseline/xgb_model_baseline_n100.pkl"
dev_path = "data/development_data_with_physical_pull.csv"
results_dir = "results_residual_model"
os.makedirs(results_dir, exist_ok=True)

# 📦 Modell und Daten laden
model = joblib.load(residual_model_path)
dev_df = pd.read_csv(dev_path)
X_dev, y_residual = get_features_and_target_physical(dev_df)  # y = PullTest - F_phys
F_phys = dev_df["F_pull_physical"].values
y_true = dev_df["PullTest (N)"].values

# 📈 Vorhersage berechnen
residual_pred = model.predict(X_dev)
pull_pred = F_phys + residual_pred

# 📊 Metriken berechnen
mse = mean_squared_error(y_true, pull_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, pull_pred)

print(f"✅ MSE: {mse:.2f}")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R²: {r2:.2f}")

# 📉 CDF der absoluten Fehler
abs_errors = np.abs(pull_pred - y_true)
sorted_errors = np.sort(abs_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

# 📊 CDF-Plot
plt.figure(figsize=(8, 5))
plt.step(sorted_errors, cdf, where="post", label="XGBoost Residual", color="#17becf")
plt.scatter(sorted_errors, cdf, marker="P", s=30, color="#17becf")
plt.xlabel("Absolute Error |ŷ - y| [N]")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Absolute Errors (XGBoost Residual Model)")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(results_dir, "cdf_residual_model.png")
plt.savefig(plot_path)
plt.close()
print(f"📁 Plot gespeichert unter: {plot_path}")