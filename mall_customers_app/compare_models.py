import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Charger le dataset
df = pd.read_csv("data/Mall_Customers.csv")
X = df[["Age", "Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

# Charger le scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# Charger les modèles
models = {
    "Linear": "models/linear_model.pkl",
    "Ridge": "models/ridge_model.pkl",
    "Lasso": "models/lasso_model.pkl"
}

results = []

for name, path in models.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results.append({
        "Modèle": name,
        "MSE": mse,
        "R²": r2
    })

# Sauvegarde des résultats
df_results = pd.DataFrame(results)
df_results.to_csv("models/comparison_metrics.csv", index=False)
print("✅ Fichier 'comparison_metrics.csv' généré avec succès.")
