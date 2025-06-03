import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocessing import get_features_and_target  # deine Funktion

sns.set_style("darkgrid")

# Ergebnisverzeichnis
results_dir = "results_feature_importance_xgb"
os.makedirs(results_dir, exist_ok=True)

# Modell laden
xgb_model_path = "model_training/xgboost/tuned/xgb_best_model.pkl"
xgb_model = joblib.load(xgb_model_path)

# Daten laden, um Feature-Namen zu bekommen
train_df = pd.read_csv("data/train_data.csv")
X_train, _ = get_features_and_target(train_df)
features = X_train.columns

# Feature Importance berechnen
importances = xgb_model.feature_importances_

# Als DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# CSV speichern
importance_df.to_csv(os.path.join(results_dir, "xgb_feature_importance.csv"), index=False)
print("✅ Feature Importance gespeichert unter:", os.path.join(results_dir, "xgb_feature_importance.csv"))

# Plot speichern
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="flare")
plt.title("Feature Importance – XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "xgb_feature_importance.png"))
plt.close()
print("📊 Plot gespeichert unter:", os.path.join(results_dir, "xgb_feature_importance.png"))
