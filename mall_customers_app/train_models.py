import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pickle
import os

# Charger le dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Sélection des features et cible
X = df[["Age", "Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entraînement
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)

# Sauvegarde
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/linear_model.pkl", "wb") as f:
    pickle.dump(linear_model, f)
with open("models/ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_model, f)
with open("models/lasso_model.pkl", "wb") as f:
    pickle.dump(lasso_model, f)

print("✅ Modèles et scaler sauvegardés dans /models/")
