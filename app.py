"""Streamlit dashboard for explainable house price prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from price_predictor.model import (
    FEATURE_COLUMNS,
    feature_importance_for_input,
    get_sample_data,
    predict_price,
    train_model,
)


st.set_page_config(page_title="Explainable Price Predictor", page_icon="🏠")
st.title("Real Estate Price Predictor with Explainability")
st.write("Predict a home price and see which features matter most.")

dataset = get_sample_data()
model = train_model(dataset)

st.subheader("Enter property details")
size_sqft = st.number_input("Size (sqft)", min_value=300, max_value=6000, value=1500, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
age_years = st.number_input("Property age (years)", min_value=0, max_value=150, value=10, step=1)
distance_to_city_km = st.number_input(
    "Distance to city center (km)", min_value=0, max_value=80, value=8, step=1
)

input_df = pd.DataFrame(
    [
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "distance_to_city_km": distance_to_city_km,
        }
    ],
    columns=FEATURE_COLUMNS,
)

if st.button("Predict price"):
    predicted_price = predict_price(model, input_df)
    st.success(f"Predicted price: ${predicted_price:,.0f}")

    st.subheader("Why this prediction?")
    importances = feature_importance_for_input(model, input_df).set_index("feature")
    st.bar_chart(importances["importance"])
    st.caption("Chart shows feature impact (SHAP when available; model importance fallback).")

