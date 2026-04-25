import joblib
import pandas as pd
import streamlit as st

# Load model + features
model = joblib.load("stock_classifier_model.pk1")
feature_columns = joblib.load("features.pkl")

# Load data
real_data = pd.read_csv("StockData.csv")

# UI
ticker = st.text_input("Enter ticker")

# Filter data
filtered = real_data[real_data["ticker"] == ticker]

if filtered.empty:
    st.error("Ticker not found")
else:
    # Chart
    st.line_chart(filtered["close"])

    if st.button("Predict"):
        features = filtered[feature_columns].tail(1)

        pred = model.predict(features)

        if pred[0] == 1:
            st.success("Model predicts price will go UP")
        else:
            st.error("Model predicts price will go DOWN")