# ---------------------------------------------------------
# Streamlit App - Real Estate Investment Advisor
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib

# Load Models
clf_model = joblib.load("best_classifier_model.pkl")
reg_model = joblib.load("best_regression_model.pkl")

st.title("🏡 Real Estate Investment Advisor")
st.write("Predict Investment Quality & Future Price")

# Input Fields
sqft = st.number_input("Size (SqFt)", min_value=300, max_value=10000, step=50)
bhk = st.number_input("BHK", min_value=1, max_value=10)
price = st.number_input("Current Price (Lakhs)", min_value=1, max_value=10000)

# Add other inputs if needed
# Example: city, furnishing, parking, amenities etc.

if st.button("Predict"):
    # Create dataframe for prediction
    inputs = pd.DataFrame([[sqft, bhk, price]], 
                          columns=["Size_in_SqFt", "BHK", "Price_in_Lakhs"])
    
    # Classification Prediction
    invest_pred = clf_model.predict(inputs)[0]
    result = "Good Investment" if invest_pred == 1 else "Not a Good Investment"
    
    st.subheader("Investment Decision")
    st.success(result)
    
    # Regression Prediction
    price_5y = reg_model.predict(inputs)[0]
    st.subheader("Future Price (After 5 Years)")
    st.info(f"₹ {round(price_5y, 2)} Lakhs")
