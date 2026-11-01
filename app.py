import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Risk Alert Classifier - Early Warning System",
    page_icon="🛡️",
    layout="wide"
)

# Custom header styling
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 24px; border-radius: 12px; margin-bottom: 24px; color: white;">
    <h1 style="margin: 0; font-size: 2.2rem; color: white;">🛡️ Risk Alert Classifier</h1>
    <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Digital Banking Early-Warning System for Predicting High-Risk Customer Behavior & Loan Default
    </p>
</div>
""", unsafe_allow_html=True)

# Load artifacts
MODEL_FILE = "risk_alert_model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"

@st.cache_resource
def load_model_and_preprocessor():
    if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSOR_FILE):
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        return model, preprocessor
    return None, None

model, preprocessor = load_model_and_preprocessor()

if model is None or preprocessor is None:
    st.warning("⚠️ Model or preprocessor artifacts not found. Please run the Jupyter Notebook (`Risk_Alert_Classifier.ipynb`) first to train and export `risk_alert_model.pkl` and `preprocessor.pkl`.")
    st.stop()

# Sidebar for sample profiles
st.sidebar.header("📋 Quick Test Profiles")
profile_choice = st.sidebar.selectbox(
    "Load Pre-configured Profile:",
    ["Custom Input", "Low Risk (Safe Salaried Customer)", "High Risk (Default-Prone Profile)"]
)

# Defaults based on selection
if profile_choice == "Low Risk (Safe Salaried Customer)":
    d_age, d_gender, d_region, d_emp = 38, "Female", "Central", "Salaried"
    d_income, d_debt, d_spend, d_tx_count = 85000.0, 15000.0, 22000.0, 45
    d_tenure, d_recency = 60, 5
    d_credit_score, d_util = 760.0, 0.18
    d_missed, d_late_days, d_cash_adv = 0, 0.0, 0
    d_complaints, d_failed_logins = 0, 0
elif profile_choice == "High Risk (Default-Prone Profile)":
    d_age, d_gender, d_region, d_emp = 28, "Male", "North", "Self-Employed"
    d_income, d_debt, d_spend, d_tx_count = 28000.0, 95000.0, 18000.0, 15
    d_tenure, d_recency = 12, 45
    d_credit_score, d_util = 510.0, 0.88
    d_missed, d_late_days, d_cash_adv = 4, 18.5, 3
    d_complaints, d_failed_logins = 3, 5
else:
    d_age, d_gender, d_region, d_emp = 35, "Female", "Central", "Salaried"
    d_income, d_debt, d_spend, d_tx_count = 45000.0, 35000.0, 18000.0, 40
    d_tenure, d_recency = 48, 12
    d_credit_score, d_util = 680.0, 0.35
    d_missed, d_late_days, d_cash_adv = 1, 2.0, 0
    d_complaints, d_failed_logins = 1, 1

# Input layout
with st.form("risk_assessment_form"):
    st.subheader("1. Customer Demographics & Account Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=int(d_age))
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(d_gender))
    with col2:
        region = st.selectbox("Region", ["Central", "East", "North", "South", "West"], index=["Central", "East", "North", "South", "West"].index(d_region))
        emp_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Retired", "Student", "Unemployed"], index=["Salaried", "Self-Employed", "Retired", "Student", "Unemployed"].index(d_emp))
    with col3:
        account_tenure = st.number_input("Account Tenure (Months)", min_value=1, max_value=240, value=int(d_tenure))
        recency = st.number_input("Transaction Recency (Days Ago)", min_value=0, max_value=365, value=int(d_recency))
    with col4:
        annual_income = st.number_input("Annual Income (INR)", min_value=10000.0, max_value=1000000.0, value=float(d_income), step=5000.0)
        debt_balance = st.number_input("Debt Balance (INR)", min_value=0.0, max_value=1000000.0, value=float(d_debt), step=5000.0)

    st.subheader("2. Financial Activity & Behavioral Risk Indicators")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        monthly_spend = st.number_input("Monthly Spend (INR)", min_value=0.0, max_value=500000.0, value=float(d_spend), step=1000.0)
        monthly_tx_count = st.number_input("Monthly Transactions", min_value=0, max_value=300, value=int(d_tx_count))
    with col6:
        credit_score = st.number_input("Credit Score (CIBIL/FICO)", min_value=300.0, max_value=900.0, value=float(d_credit_score), step=10.0)
        credit_util = st.slider("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=float(d_util), step=0.01)
    with col7:
        missed_payments = st.number_input("Missed Payments (Last 12M)", min_value=0, max_value=12, value=int(d_missed))
        late_days = st.number_input("Avg Late Payment Days", min_value=0.0, max_value=90.0, value=float(d_late_days), step=0.5)
    with col8:
        cash_advance = st.number_input("Cash Advances (Last 6M)", min_value=0, max_value=20, value=int(d_cash_adv))
        failed_logins = st.number_input("Failed Logins (Last 3M)", min_value=0, max_value=30, value=int(d_failed_logins))
        complaints = st.number_input("Complaints (Last 6M)", min_value=0, max_value=15, value=int(d_complaints))

    submitted = st.form_submit_button("🔍 Run Risk Alert Classification", use_container_width=True)

if submitted:
    # Prepare input DataFrame with exact matching column names
    input_data = pd.DataFrame([{
        "age": float(age),
        "gender": str(gender),
        "region": str(region),
        "employment_type": str(emp_type),
        "annual_income_inr": float(annual_income),
        "credit_score": float(credit_score),
        "credit_utilization_ratio": float(credit_util),
        "missed_payments_12m": int(missed_payments),
        "avg_late_payment_days": float(late_days),
        "monthly_transaction_count": int(monthly_tx_count),
        "monthly_spend_inr": float(monthly_spend),
        "cash_advance_count_6m": int(cash_advance),
        "complaints_last_6m": int(complaints),
        "failed_login_attempts_3m": int(failed_logins),
        "account_tenure_months": int(account_tenure),
        "debt_balance_inr": float(debt_balance),
        "transaction_recency_days": int(recency)
    }])

    # Preprocess
    input_processed = preprocessor.transform(input_data)

    # Predict
    pred = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    st.markdown("---")
    st.subheader("🎯 Risk Assessment Result")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if pred == 1:
            st.error(f"🚨 **HIGH RISK ALERT (Default / Fraud Risk Detected)**\n\nClass Label: **1 (High Risk)**")
        else:
            st.success(f"✅ **LOW RISK PROFILE (Creditworthy & Stable)**\n\nClass Label: **0 (Low Risk)**")

        st.metric(label="Predicted Probability of High Risk", value=f"{prob * 100:.2f}%")
        st.progress(float(prob))

    with res_col2:
        st.markdown("### 📊 Decision Factors & Risk Insights")
        risk_flags = []
        if credit_score < 620:
            risk_flags.append(f"⚠️ Low Credit Score: {credit_score:.0f} (Threshold < 620)")
        if missed_payments > 1:
            risk_flags.append(f"⚠️ High Missed Payments in 12M: {missed_payments}")
        if late_days > 5.0:
            risk_flags.append(f"⚠️ High Average Late Days: {late_days:.1f} days")
        if credit_util > 0.60:
            risk_flags.append(f"⚠️ Elevated Credit Utilization: {credit_util*100:.1f}%")
        if cash_advance >= 2:
            risk_flags.append(f"⚠️ Frequent Cash Advances (Liquidity Stress): {cash_advance}")
        if failed_logins >= 3:
            risk_flags.append(f"⚠️ Security Flag - Multiple Failed Logins: {failed_logins}")

        if risk_flags:
            st.write("**Identified Red Flags:**")
            for flag in risk_flags:
                st.write(flag)
        else:
            st.write("✨ **No critical credit or behavioral red flags detected.**")

        if pred == 1:
            st.markdown("""
            **Recommended Bank Action:**
            - Place temporary hold on credit limit enhancements.
            - Trigger automated secondary identity verification (2FA).
            - Flag account for manual risk officer review prior to loan disbursement.
            """)
        else:
            st.markdown("""
            **Recommended Bank Action:**
            - Standard processing; eligible for fast-track approvals and preferential credit terms.
            """)
