import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f77b4;
    text-align: center;
    font-size: 42px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #45a049;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('gb_model.pkl','rb'))

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Single Prediction",
        "Bulk Prediction",
        "About Project"
    ]
)

# ---------------- DASHBOARD PAGE ---------------- #
if menu == "Dashboard":

    st.title("📈 Telecom Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", "7043")

    with col2:
        st.metric("Churn Rate", "26.5%")

    with col3:
        st.metric("Retention Rate", "73.5%")

    with col4:
        st.metric("Monthly Revenue", "$456K")

    st.divider()

    # ---------------- PIE CHART ---------------- #
    churn_data = pd.DataFrame({
        'Category': ['Churn', 'Retention'],
        'Count': [1869, 5174]
    })

    fig1 = px.pie(
        churn_data,
        values='Count',
        names='Category',
        title='Customer Churn Distribution'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- LINE CHART ---------------- #
    chart_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Revenue': [200, 240, 260, 280, 300, 350],
        'Customers': [500, 550, 620, 700, 850, 1000]
    })

    fig2 = px.line(
        chart_data,
        x='Month',
        y=['Revenue', 'Customers'],
        title='Revenue & Customer Growth'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- SINGLE PREDICTION PAGE ---------------- #
elif menu == "Single Prediction":

    st.title("🤖 Telecom Churn Prediction")

    st.subheader("📝 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.number_input(
            'Tenure (months)',
            0,
            72,
            12
        )

        monthly_charges = st.number_input(
            'Monthly Charges',
            0.0,
            200.0,
            50.0
        )

        total_charges = st.number_input(
            'Total Charges',
            0.0,
            10000.0,
            500.0
        )

        gender = st.selectbox(
            'Gender',
            ('Male', 'Female')
        )

        senior = st.selectbox(
            'Senior Citizen',
            ('Yes', 'No')
        )

    with col2:

        partner = st.selectbox(
            'Partner',
            ('Yes', 'No')
        )

        dependents = st.selectbox(
            'Dependents',
            ('Yes', 'No')
        )

        phone_service = st.selectbox(
            'Phone Service',
            ('Yes', 'No')
        )

        internet_service = st.selectbox(
            'Internet Service',
            ('DSL', 'Fiber optic', 'No')
        )

        contract = st.selectbox(
            'Contract',
            ('Month-to-month', 'One year', 'Two year')
        )

    # ---------------- INPUT DATA ---------------- #
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

    # ---------------- ENCODING ---------------- #
    input_df = pd.get_dummies(input_df)

    # ---------------- MATCH MODEL COLUMNS ---------------- #
    input_df = input_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    st.divider()

    # ---------------- PREDICTION ---------------- #
    if st.button("🔍 Predict Churn"):

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("📌 Prediction Result")

        if prob < 0.3:
            st.success("🟢 Low Churn Risk")

        elif prob < 0.7:
            st.warning("🟡 Medium Churn Risk")

        else:
            st.error("🔴 High Churn Risk")

        st.write(f"### Churn Probability: {prob*100:.2f}%")

        st.progress(float(prob))

        # ---------------- GAUGE CHART ---------------- #
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- RETENTION SUGGESTIONS ---------------- #
        st.subheader("💡 Retention Suggestions")

        if prob > 0.7:

            st.write("""
            - Provide discount offers
            - Improve customer support
            - Offer long-term plans
            - Give loyalty rewards
            """)

        elif prob > 0.3:

            st.write("""
            - Send promotional offers
            - Improve customer engagement
            - Provide service upgrades
            """)

        else:

            st.write("Customer is likely to stay with the company.")

        # ---------------- CUSTOMER SUMMARY ---------------- #
        st.subheader("📋 Customer Summary")

        summary_df = pd.DataFrame({
            "Feature": input_dict.keys(),
            "Value": input_dict.values()
        })

        st.dataframe(summary_df)

# ---------------- BULK PREDICTION PAGE ---------------- #
elif menu == "Bulk Prediction":

    st.title("📂 Bulk Customer Churn Prediction")

    st.write("Upload a CSV file to predict churn for multiple customers.")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Dataset")

        st.dataframe(df.head())

        try:

            encoded_df = pd.get_dummies(df)

            encoded_df = encoded_df.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

            predictions = model.predict(encoded_df)
            probabilities = model.predict_proba(encoded_df)[:, 1]

            df['Prediction'] = predictions
            df['Churn Probability'] = probabilities

            df['Prediction'] = df['Prediction'].map({
                1: 'Churn',
                0: 'Stay'
            })

            st.subheader("📊 Prediction Results")

            st.dataframe(df)

            # ---------------- BAR CHART ---------------- #
            churn_count = (df['Prediction'] == 'Churn').sum()
            stay_count = (df['Prediction'] == 'Stay').sum()

            result_data = pd.DataFrame({
                'Category': ['Churn', 'Stay'],
                'Count': [churn_count, stay_count]
            })

            fig = px.bar(
                result_data,
                x='Category',
                y='Count',
                title='Bulk Prediction Analysis'
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- DOWNLOAD REPORT ---------------- #
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- ABOUT PAGE ---------------- #
elif menu == "About Project":

    st.title("📖 About Project")

    st.write("""
    ## AI Telecom Customer Churn Prediction System

    This project is developed using Machine Learning and Streamlit.

    ### 🔹 Technologies Used
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly

    ### 🔹 Features
    - Single Customer Prediction
    - Bulk Customer Analysis
    - Interactive Dashboard
    - Churn Probability Meter
    - Download Prediction Reports

    ### 🔹 Objective
    To help telecom companies identify customers who are likely to churn and improve customer retention strategies.
    """)

# ---------------- FOOTER ---------------- #
st.divider()

st.markdown(
    f"""
    <center>
        <h4>Developed using ❤️ with Streamlit & Machine Learning</h4>
        <p>{datetime.now().year} | Telecom Churn Prediction System</p>
    </center>
    """,
    unsafe_allow_html=True
)
