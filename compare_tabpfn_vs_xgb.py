import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import get_features_and_target
from tabpfn import TabPFNRegressor
from sklearn.neighbors import KNeighborsRegressor


sns.set_style("darkgrid")

# Datei laden (Pfad anpassen!)
xgb_model_path = "model_training/xgboost/tuned/xgb_best_model.pkl"  # oder z. B. models/xgb_tuned.pkl
xgb_model = joblib.load(xgb_model_path)

# Ergebnisverzeichnis
results_dir = "results_comparison"
os.makedirs(results_dir, exist_ok=True)

# Daten laden
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
X_train, y_train = get_features_and_target(train_df)
X_dev, y_dev = get_features_and_target(dev_df)
X_dev = X_dev[X_train.columns]

models = {
    "TabPFN": TabPFNRegressor(random_state=42),
    "XGBoost Tuned (pkl)": xgb_model,
    "KNN Regressor (k=5)": KNeighborsRegressor(n_neighbors=5)
}


# Ergebnisse berechnen
results = {}
cdf_data = {}

for name, model in models.items():
    model.fit(X_train, y_train) if name == "TabPFN" else None  # Nur TabPFN trainieren
    preds = model.predict(X_dev)

    mse = mean_squared_error(y_dev, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_dev, preds)

    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

    abs_errors = np.abs(preds - y_dev)
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    cdf_data[name] = (sorted_errors, cdf)

    

# Ergebnisse speichern
results_df = pd.DataFrame(results).T.round(2)
results_df.to_csv(os.path.join(results_dir, "results_comparison.csv"))

# Metriken plotten
for metric in ["MSE", "RMSE", "R2"]:
    fig, ax = plt.subplots(figsize=(6, 4))
    results_df[metric].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"metric_{metric.lower()}_comparison.png"))
    plt.close(fig)

# CDF-Plot
# CDF mit Markern und gestuftem Verlauf
fig, ax = plt.subplots(figsize=(8, 5))

# Marker und Farben zuweisen
markers = {
    "TabPFN": "^",  # Dreieck
    "XGBoost Tuned (pkl)": "s"  # Quadrat
}
colors = {
    "TabPFN": "#1f77b4",
    "XGBoost Tuned (pkl)": "#ff7f0e"
}

for name, (errors, cdf) in cdf_data.items():
    ax.step(errors, cdf, where="post", label=name, color=colors[name])
    ax.scatter(errors, cdf, marker=markers[name], s=30, color=colors[name])

ax.set_xlabel("Absolute Error |ŷ - y| [N]")
ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF of Absolute Errors")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "cdf_errors_comparison.png"))
plt.close(fig)

top_errors_dir = os.path.join(results_dir, "top_errors")
os.makedirs(top_errors_dir, exist_ok=True)

# Verzeichnis für die Fehlerauswertung
top_errors_dir = os.path.join(results_dir, "top_errors_full")
os.makedirs(top_errors_dir, exist_ok=True)

for name, model in models.items():
    preds = model.predict(X_dev)
    abs_errors = np.abs(preds - y_dev)

    # Top N Indizes mit größten Fehlern
    top_n = 5
    top_indices = np.argsort(abs_errors)[-top_n:][::-1]

    # Originalzeilen aus dem dev_df (enthält alle Features)
    top_rows = dev_df.iloc[top_indices].copy()

    # Zusätzliche Spalten für Analyse
    top_rows["Prediction"] = preds[top_indices]
    top_rows["Ground Truth"] = y_dev.iloc[top_indices].values
    top_rows["Absolute Error"] = abs_errors[top_indices]

    # Datei speichern
    filename = os.path.join(top_errors_dir, f"top_{top_n}_errors_{name.replace(' ', '_')}.csv")
    top_rows.to_csv(filename, index_label="Index")

    print(f"📄 Top {top_n} Fehler für Modell '{name}' gespeichert unter: {filename}")



print("✅ Vergleich mit geladenem XGBoost-Modell abgeschlossen.")
