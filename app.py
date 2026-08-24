import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ---------------- #

with open("gb_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_columns = model_data["columns"]


# ---------------- FUNCTIONS ---------------- #

def prepare_input(input_dict):
    df = pd.DataFrame([input_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match exactly the columns used during training
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Telecom Customer Churn Prediction")


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Dashboard", "Bulk CSV"]
)


# ============================================================
# PREDICTION
# ============================================================

if page == "Prediction":

    st.header("🔮 Customer Churn Prediction")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=500.0
    )

    gender = st.selectbox(
        "Gender",
        ("Male", "Female")
    )

    senior = st.selectbox(
        "Senior Citizen",
        ("Yes", "No")
    )

    partner = st.selectbox(
        "Partner",
        ("Yes", "No")
    )

    dependents = st.selectbox(
        "Dependents",
        ("Yes", "No")
    )

    phone_service = st.selectbox(
        "Phone Service",
        ("Yes", "No")
    )

    internet_service = st.selectbox(
        "Internet Service",
        ("DSL", "Fiber optic", "No")
    )

    contract = st.selectbox(
        "Contract",
        ("Month-to-month", "One year", "Two year")
    )

    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": "Yes" if partner == "Yes" else "No",
        "Dependents": "Yes" if dependents == "Yes" else "No",
        "PhoneService": "Yes" if phone_service == "Yes" else "No",
        "InternetService": internet_service,
        "Contract": contract,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    if st.button("🔮 Predict Churn", use_container_width=True):

        input_df = prepare_input(input_dict)

        pred = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = None

        if pred == 1:
            st.error("⚠️ Customer Churn")
        else:
            st.success("✅ No Customer Churn")

        if prob is not None:
            st.metric(
                "Churn Probability",
                f"{prob * 100:.2f}%"
            )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.header("📊 Telecom Churn Dashboard")

    uploaded_file = st.file_uploader(
        "Upload your telecom CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Customers", len(df))

        with col2:
            st.metric("Total Columns", len(df.columns))

        if "Churn" in df.columns:

            churn_count = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("yes")
                .sum()
            )

            with col3:
                st.metric("Churned Customers", int(churn_count))

            st.subheader("Churn Distribution")

            churn_data = (
                df["Churn"]
                .value_counts()
                .rename_axis("Churn")
                .reset_index(name="Customers")
            )

            st.bar_chart(
                churn_data.set_index("Churn")
            )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


# ============================================================
# BULK CSV
# ============================================================

elif page == "Bulk CSV":

    st.header("📁 Bulk Customer Churn Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="bulk_csv"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        if st.button(
            "🚀 Predict All Customers",
            use_container_width=True
        ):

            try:

                predictions = []

                probabilities = []

                for _, row in df.iterrows():

                    input_dict = {
                        "gender": row.get("gender", "Male"),
                        "SeniorCitizen": row.get("SeniorCitizen", 0),
                        "Partner": row.get("Partner", "No"),
                        "Dependents": row.get("Dependents", "No"),
                        "PhoneService": row.get("PhoneService", "No"),
                        "InternetService": row.get(
                            "InternetService",
                            "No"
                        ),
                        "Contract": row.get(
                            "Contract",
                            "Month-to-month"
                        ),
                        "tenure": row.get("tenure", 0),
                        "MonthlyCharges": row.get(
                            "MonthlyCharges",
                            0
                        ),
                        "TotalCharges": row.get(
                            "TotalCharges",
                            0
                        )
                    }

                    input_df = prepare_input(input_dict)

                    prediction = model.predict(input_df)[0]

                    predictions.append(
                        "Churn" if prediction == 1
                        else "No Churn"
                    )

                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(
                            input_df
                        )[0][1]

                        probabilities.append(
                            round(probability * 100, 2)
                        )
                    else:
                        probabilities.append(None)

                result_df = df.copy()

                result_df["Prediction"] = predictions
                result_df["Churn Probability (%)"] = probabilities

                st.success(
                    f"✅ Predictions completed for {len(df)} customers."
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv = result_df.to_csv(index=False)

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Error while processing CSV: {e}"
                )