import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_predict
from preprocessing import get_features_and_target
from tabpfn import TabPFNRegressor

sns.set_style("darkgrid")

# Ergebnisverzeichnis
results_dir = "results_comparison"
os.makedirs(results_dir, exist_ok=True)

# Daten für alle Modelle außer Residualmodell
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
X_train, y_train = get_features_and_target(train_df)
X_dev, y_dev = get_features_and_target(dev_df)

# Daten für Residualmodell
train_df_residual = pd.read_csv("data/train_data_with_physical_pull.csv")
dev_df_residual = pd.read_csv("data/development_data_with_physical_pull.csv")
from preprocessing import get_features_and_target_physical

# Residual-Modell:
X_dev_residual_raw, y_dev_residual = get_features_and_target_physical(dev_df_residual)  # enthält y = PullTest
F_phys_dev = dev_df_residual["F_pull_physical"].values

# Noch kein Feature-Drop → passiert erst bei model.predict
F_phys_dev = dev_df_residual["F_pull_physical"].values
# Modelle definieren

xgb_residual_model_path = "model_training/xgboost/baseline/xgb_model_baseline_n100.pkl"
xgb_residual_model = joblib.load(xgb_residual_model_path)

xgb_model_path = "model_training/xgboost/tuned/xgb_best_model_cv.pkl"
xgb_model = joblib.load(xgb_model_path)

knn_model_path = "model_training/knn_regressor/knn_k4_model.pkl"
knn_model = joblib.load(knn_model_path)

rf_model_path = "model_training/random_forest/random_forest_best_model.pkl"
rf_model = joblib.load(rf_model_path)

dt_model_path = "model_training/decision_tree/decision_tree_best_model.pkl"
dt_model = joblib.load(dt_model_path)

models = {
    "TabPFN": TabPFNRegressor(random_state=42),
    "XGBoost Residual": xgb_residual_model,
    "XGBoost Tuned": xgb_model,
    "KNN Regressor (k=4)": knn_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

# Ergebnis-Container
results = {}
cdf_data = {}
all_errors = {}

for name, model in models.items():
    print(f"📊 Trainiere/Bewerte Modell: {name}")
    
    if name == "TabPFN":
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = cross_val_predict(model, X_train, y_train, cv=cv)
        X_eval, y_eval = X_train, y_train

    elif name == "XGBoost Residual":
        # Nur die Features, wie sie beim Training verwendet wurden
        X_eval = X_dev_residual_raw.drop(columns=["Sample ID", "F_pull_physical"], errors="ignore")

        residuals = model.predict(X_eval)
        preds = F_phys_dev + residuals
        y_eval = y_dev_residual

    elif name == "XGBoost Tuned":
        preds = model.predict(X_dev)
        X_eval, y_eval = X_dev, y_dev

    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
        X_eval, y_eval = X_dev, y_dev

    mse = mean_squared_error(y_eval, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_eval, preds)

    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

    abs_errors = np.abs(preds - y_eval)
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    cdf_data[name] = (sorted_errors, cdf)
    all_errors[name] = abs_errors

# Ergebnisse speichern
results_df = pd.DataFrame(results).T.round(2)
results_df.to_csv(os.path.join(results_dir, "results_comparison.csv"))
print("✅ Metriken gespeichert.")

# Metriken plotten
for metric in ["MSE", "RMSE", "R2"]:
    fig, ax = plt.subplots(figsize=(6, 4))
    results_df[metric].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"metric_{metric.lower()}_comparison.png"))
    plt.close(fig)

# CDF-Plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = {
    "TabPFN": "#1f77b4",
    "XGBoost Tuned": "#ff7f0e",
    "XGBoost Residual": "#17becf",  # 👈 neue Farbe
    "KNN Regressor (k=4)": "#2ca02c",
    "Random Forest": "#9467bd",
    "Decision Tree": "#8c564b"
}
markers = {
    "TabPFN": "^",
    "XGBoost Tuned": "s",
    "XGBoost Residual": "P",  # 👈 neues Symbol
    "KNN Regressor (k=4)": "o",
    "Random Forest": "X",
    "Decision Tree": "D"
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

# Top-Fehler ausgeben
top_errors_dir = os.path.join(results_dir, "top_errors_full")
os.makedirs(top_errors_dir, exist_ok=True)

for name, model in models.items():
    top_n = 5
    if name == "TabPFN":
        preds = cross_val_predict(model, X_train, y_train, cv=5)
        abs_errors = np.abs(preds - y_train)
        top_rows = train_df.iloc[np.argsort(abs_errors)[-top_n:][::-1]].copy()
        top_rows["Prediction"] = preds[np.argsort(abs_errors)[-top_n:][::-1]]
        top_rows["Ground Truth"] = y_train.iloc[np.argsort(abs_errors)[-top_n:][::-1]].values
        top_rows["Absolute Error"] = abs_errors[np.argsort(abs_errors)[-top_n:][::-1]]

    elif name == "XGBoost Residual":
        residuals = model.predict(X_dev_residual_raw)
        preds = F_phys_dev + residuals
        abs_errors = np.abs(preds - y_dev_residual)
        top_rows = dev_df_residual.iloc[np.argsort(abs_errors)[-top_n:][::-1]].copy()
        top_rows["Prediction"] = preds[np.argsort(abs_errors)[-top_n:][::-1]]
        top_rows["Ground Truth"] = y_dev_residual.iloc[np.argsort(abs_errors)[-top_n:][::-1]].values
        top_rows["Absolute Error"] = abs_errors[np.argsort(abs_errors)[-top_n:][::-1]]

    else:
        preds = model.predict(X_dev)
        abs_errors = np.abs(preds - y_dev)
        top_rows = dev_df.iloc[np.argsort(abs_errors)[-top_n:][::-1]].copy()
        top_rows["Prediction"] = preds[np.argsort(abs_errors)[-top_n:][::-1]]
        top_rows["Ground Truth"] = y_dev.iloc[np.argsort(abs_errors)[-top_n:][::-1]].values
        top_rows["Absolute Error"] = abs_errors[np.argsort(abs_errors)[-top_n:][::-1]]

    filename = os.path.join(top_errors_dir, f"top_{top_n}_errors_{name.replace(' ', '_')}.csv")
    top_rows.to_csv(filename, index_label="Index")
    print(f"📄 Top {top_n} Fehler für '{name}' gespeichert unter: {filename}")

# Heatmap der Metriken
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(results_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
ax.set_title("Model Performance Heatmap")
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "heatmap_model_metrics.png"))
plt.close(fig)

# Fehlerbarplot (Mean ± Std)
error_stats = {
    name: {
        "Mean Error": np.mean(err),
        "Std Error": np.std(err)
    }
    for name, err in all_errors.items()
}
error_df = pd.DataFrame(error_stats).T
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(error_df.index, error_df["Mean Error"], yerr=error_df["Std Error"], capsize=5,
       color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"])
ax.set_ylabel("Mean Absolute Error ± Std")
ax.set_title("Error Comparison Across Models")
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "error_barplot.png"))
plt.close(fig)

print("✅ Vergleich für TabPFN, XGBoost und KNN abgeschlossen.")
