import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app_linear import predict_linear
from app_ridge import predict_ridge
from app_lasso import predict_lasso
import pickle

st.set_page_config(page_title="Mall Customer Regression", layout="wide")
st.title("🛍️ Mall Customer Regression - Comparateur de modèles")

# Chargement du dataset
df = pd.read_csv("data/Mall_Customers.csv")

tab1, tab2 = st.tabs(["🎯 Prédictions", "📊 Comparaison des modèles"])

# ========== Onglet 1 : Prédictions ==========
with tab1:
    st.header("Faire une prédiction")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 30)
    with col2:
        income = st.slider("Annual Income (k$)", int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max()), 50)

    input_data = [age, income]

    if st.button("Faire la prédiction"):
        linear_pred = predict_linear(input_data)
        ridge_pred = predict_ridge(input_data)
        lasso_pred = predict_lasso(input_data)

        results = pd.DataFrame({
            "Modèle": ["Linear", "Ridge", "Lasso"],
            "Score Prévu": [linear_pred, ridge_pred, lasso_pred]
        })

        st.dataframe(results)
        fig = px.bar(results, x="Modèle", y="Score Prévu", color="Modèle", title="Comparaison des prédictions")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)",
                          color="Gender", size="Age",
                          title="Données du Mall : Revenu vs Score de dépense")
        st.plotly_chart(fig2, use_container_width=True)

# ========== Onglet 2 : Comparaison ==========
with tab2:
    st.header("📊 Performance des modèles")

    try:
        metrics = pd.read_csv("models/comparison_metrics.csv")
    except FileNotFoundError:
        st.warning("⚠️ Lancez compare_models.py pour générer les métriques.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        fig_mse = px.bar(metrics, x="Modèle", y="MSE", color="Modèle", title="Erreur quadratique moyenne (MSE)")
        st.plotly_chart(fig_mse, use_container_width=True)
    with col2:
        fig_r2 = px.bar(metrics, x="Modèle", y="R²", color="Modèle", title="Score R²")
        st.plotly_chart(fig_r2, use_container_width=True)

    with open("models/linear_model.pkl", "rb") as f:
        linear_model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X = df[["Age", "Annual Income (k$)"]]
    y = df["Spending Score (1-100)"]
    X_scaled = scaler.transform(X)
    y_pred = linear_model.predict(X_scaled)
    df_pred = pd.DataFrame({"Réel": y, "Prédit": y_pred})
    fig_pred = px.scatter(df_pred, x="Réel", y="Prédit", trendline="ols",
                          title="Prédictions vs Valeurs réelles (Linear Regression)")
    st.plotly_chart(fig_pred, use_container_width=True)
