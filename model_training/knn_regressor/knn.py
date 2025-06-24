import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from kneed import KneeLocator
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing import get_features_and_target

# Daten laden
train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/development_data.csv")
X_train, y_train = get_features_and_target(train_df)
X_dev, y_dev = get_features_and_target(dev_df)
X_dev = X_dev[X_train.columns]

# CrossValidator definieren
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Wertebereich für k & CV-Fehler berechnen
k_values = range(1, 21)
cv_errors = []
for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    cv_errors.append(-scores.mean())

# Kniepunkt finden
knee = KneeLocator(k_values, cv_errors, curve="convex", direction="decreasing")
optimal_k = knee.knee if knee.knee else 5
print(f"🔎 Optimale Anzahl an Nachbarn (k): {optimal_k}")

# Plot der CV-Fehlerkurve
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_errors, marker='o')
plt.xlabel("Anzahl Nachbarn (k)")
plt.ylabel("MSE (CV, Training)")
plt.title("Elbow-Methode für KNN (Cross Validation)")
if knee.knee:
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Knee: k={optimal_k}")
    plt.legend()
plt.tight_layout()
plt.savefig("elbow_method_knn_cv.png")
plt.close()

# Endgültiges Modell auf ganzem Training trainieren & auf Dev testen
best_knn = KNeighborsRegressor(n_neighbors=optimal_k)
best_knn.fit(X_train, y_train)
preds = best_knn.predict(X_dev)
mse = mean_squared_error(y_dev, preds)
print(f"Endgültiger MSE auf Dev-Set mit k={optimal_k}: {mse:.2f}")



# Modell speichern:
import joblib
joblib.dump(best_knn, f"knn_k{optimal_k}_model.pkl") 

# Ergebnisse und Plot speichern (du kannst noch erweitern, falls du weitere Metriken willst)

print("✅ KNN-Skript abgeschlossen und Ergebnisse gespeichert.")