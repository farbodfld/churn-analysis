import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = r"C:\Users\farbo\OneDrive\Desktop\churn-analysis\models\churn_model.pkl"
COLS_PATH  = r"C:\Users\farbo\OneDrive\Desktop\churn-analysis\models\model_columns.json"

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📉 Telco Customer Churn Predictor")
st.write("Enter customer details to estimate churn probability.")

model = joblib.load(MODEL_PATH)

with open(COLS_PATH, "r") as f:
    columns = json.load(f)

# Simple input UI: you can expand later
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=800.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# Build a single-row dataframe with ALL required columns
input_dict = {col: None for col in columns}
# Fill known fields (names must match your dataset column names!)
input_dict["tenure"] = tenure
input_dict["monthlycharges"] = monthlycharges
input_dict["totalcharges"] = totalcharges
input_dict["contract"] = contract
input_dict["paymentmethod"] = paymentmethod
input_dict["internetservice"] = internetservice
input_dict["techsupport"] = techsupport

X_input = pd.DataFrame([input_dict])

# Any remaining None columns will be handled by the pipeline imputers
if st.button("Predict"):
    prob = model.predict_proba(X_input)[:, 1][0]
    st.metric("Churn probability", f"{prob:.2%}")

    if prob >= 0.5:
        st.warning("High risk: consider retention offer + proactive support.")
    elif prob >= 0.3:
        st.info("Medium risk: consider plan review and billing/autopay incentives.")
    else:
        st.success("Low risk: maintain engagement and satisfaction.")
