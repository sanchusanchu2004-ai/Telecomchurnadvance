import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("gb_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_columns = model_data["columns"]
st.title("Telecom Customer Churn Prediction")

tenure = st.number_input('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges', 0.0, 200.0, 50.0)
total_charges = st.number_input('Total Charges', 0.0, 10000.0, 500.0)

gender = st.selectbox('Gender', ('Male', 'Female'))
senior = st.selectbox('Senior Citizen', ('Yes', 'No'))
partner = st.selectbox('Partner', ('Yes', 'No'))
dependents = st.selectbox('Dependents', ('Yes', 'No'))
phone_service = st.selectbox('Phone Service', ('Yes', 'No'))
internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))

input_dict = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'InternetService': internet_service,
    'Contract': contract,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

input_df = pd.get_dummies(input_df)

input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error("Customer Churn")
    else:
        st.success("No Customer Churn")