import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter Customer Details")

# Numerical Inputs
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure", 0, 100, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Categorical Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

if st.button("Predict Churn"):

    # Create dictionary with all features
    input_data = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,

        'gender_Male': 1 if gender == "Male" else 0,

        'Partner_Yes': 1 if Partner == "Yes" else 0,

        'Dependents_Yes': 1 if Dependents == "Yes" else 0,

        'PhoneService_Yes': 1 if PhoneService == "Yes" else 0,

        'MultipleLines_No phone service':
            1 if MultipleLines == "No phone service" else 0,

        'MultipleLines_Yes':
            1 if MultipleLines == "Yes" else 0,

        'InternetService_Fiber optic':
            1 if InternetService == "Fiber optic" else 0,

        'InternetService_No':
            1 if InternetService == "No" else 0,

        'OnlineSecurity_No internet service':
            1 if OnlineSecurity == "No internet service" else 0,

        'OnlineSecurity_Yes':
            1 if OnlineSecurity == "Yes" else 0,

        'OnlineBackup_No internet service':
            1 if OnlineBackup == "No internet service" else 0,

        'OnlineBackup_Yes':
            1 if OnlineBackup == "Yes" else 0,

        'DeviceProtection_No internet service':
            1 if DeviceProtection == "No internet service" else 0,

        'DeviceProtection_Yes':
            1 if DeviceProtection == "Yes" else 0,

        'TechSupport_No internet service':
            1 if TechSupport == "No internet service" else 0,

        'TechSupport_Yes':
            1 if TechSupport == "Yes" else 0,

        'StreamingTV_No internet service':
            1 if StreamingTV == "No internet service" else 0,

        'StreamingTV_Yes':
            1 if StreamingTV == "Yes" else 0,

        'StreamingMovies_No internet service':
            1 if StreamingMovies == "No internet service" else 0,

        'StreamingMovies_Yes':
            1 if StreamingMovies == "Yes" else 0,

        'Contract_One year':
            1 if Contract == "One year" else 0,

        'Contract_Two year':
            1 if Contract == "Two year" else 0,

        'PaperlessBilling_Yes':
            1 if PaperlessBilling == "Yes" else 0,

        'PaymentMethod_Credit card (automatic)':
            1 if PaymentMethod == "Credit card (automatic)" else 0,

        'PaymentMethod_Electronic check':
            1 if PaymentMethod == "Electronic check" else 0,

        'PaymentMethod_Mailed check':
            1 if PaymentMethod == "Mailed check" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)

    # Output
    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is NOT likely to Churn")