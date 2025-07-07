import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("lasso_model.pkl")

st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

st.markdown("Fill the house features below to predict its **Sale Price**.")

# Base Inputs – (We choose a subset of most influential features)
numerical_inputs = {
    'GrLivArea': st.slider("Above ground living area (sq ft)", 500, 6000, 1500),
    'OverallQual': st.slider("Overall Quality (1 - 10)", 1, 10, 5),
    'YearBuilt': st.slider("Year Built", 1870, 2024, 1990),
    'GarageCars': st.slider("Garage Capacity (cars)", 0, 4, 2),
    'TotalBsmtSF': st.slider("Total Basement Area (sq ft)", 0, 3000, 800),
    '1stFlrSF': st.slider("1st Floor Area (sq ft)", 300, 3000, 1000),
    'FullBath': st.slider("No. of Full Bathrooms", 0, 4, 2),
    'BedroomAbvGr': st.slider("No. of Bedrooms above ground", 0, 10, 3),
    'TotRmsAbvGrd': st.slider("Total Rooms Above Ground", 1, 15, 6),
    'Fireplaces': st.slider("Number of Fireplaces", 0, 3, 1),
}

# Categorical Inputs (shown as dropdowns)
cat_inputs = {
    'MSZoning': st.selectbox("MS Zoning", ['RL', 'RM', 'FV', 'RH', 'C (all)']),
    'Street': st.selectbox("Street Type", ['Pave', 'Grvl']),
    'CentralAir': st.selectbox("Central Air", ['Y', 'N']),
    'KitchenQual': st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa']),
    'GarageFinish': st.selectbox("Garage Finish", ['Fin', 'RFn', 'Unf', 'None']),
    'PavedDrive': st.selectbox("Paved Driveway", ['Y', 'N', 'P']),
}

# Combine user input into a base DataFrame
input_data = {**numerical_inputs, **cat_inputs}
input_df = pd.DataFrame([input_data])

# One-hot encode categorical variables
encoded_df = pd.get_dummies(input_df)

# Add missing columns (that model expects but user didn’t select)
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in encoded_df.columns:
        encoded_df[col] = 0

# Reorder columns exactly as model expects
final_input = encoded_df[expected_cols]

# Predict on button click
if st.button("Predict House Price"):
    prediction = model.predict(final_input)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
