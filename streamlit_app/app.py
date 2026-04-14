# =========================================================
# 🧠 LOAN DEFAULT RISK SYSTEM (STREAMLIT APP)
# =========================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from src.modeling.predict import predict, load_artifacts

# =========================================================
# 🔥 PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Loan Risk Intelligence System",
    page_icon="💳",
    layout="wide",
)

# =========================================================
# 🎨 CUSTOM STYLING (ELITE UI)
# =========================================================

st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}

/* Headers */
h1, h2, h3 {
    color: #00C2FF;
}

/* Metric Cards */
.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background-color: #00C2FF;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 📦 LOAD MODEL (CACHED)
# =========================================================

@st.cache_resource
def load_system():
    model, features, threshold = load_artifacts("models")
    return model, features, threshold

model, FEATURES, THRESHOLD = load_system()

# =========================================================
# 🏠 SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("💳 Loan Risk System")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔮 Single Prediction"]
)

# =========================================================
# 🏠 OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.title("💳 Loan Default Risk Intelligence System")

    st.markdown("""
    ### 🎯 Purpose
    This system predicts the likelihood of a borrower defaulting on a loan using a production-grade machine learning pipeline.

    ### ⚙️ Model Highlights
    - Model: LightGBM
    - Validation: Time-Based Cross Validation
    - Metric: F1 Score (Optimized)
    - Threshold Optimization: Dynamic

    ### 🚀 Capabilities
    - Real-time risk prediction
    - Batch processing
    - Risk classification
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "LightGBM")

    with col2:
        st.metric("Threshold", f"{THRESHOLD:.2f}")

    with col3:
        st.metric("Status", "Production Ready")

# =========================================================
# 🔮 SINGLE PREDICTION PAGE
# =========================================================

elif page == "🔮 Single Prediction":

    st.title("🔮 Loan Risk Prediction Engine")

    st.markdown("### Input Loan Details")

    col1, col2 = st.columns(2)

    # =====================================================
    # INPUTS
    # =====================================================

    with col1:
        repayment_ratio = st.number_input("Repayment Ratio", value=1.2)
        amount_funded = st.number_input("Amount Funded by Lender", value=5000.0)
        loan_pressure = st.number_input("Loan Pressure", value=300.0)
        total_repay = st.number_input("Total Amount to Repay", value=6000.0)
        total_amount = st.number_input("Total Loan Amount", value=5000.0)
        lender_portion = st.number_input("Lender Portion Funded", value=0.8)

    with col2:
        month = st.selectbox("Month", list(range(1, 13)))
        year = st.number_input("Year", value=2023)
        duration = st.number_input("Loan Duration", value=12)

        loan_type = st.selectbox("Loan Type", ["Type_A", "Type_B", "Type_C"])
        new_repeat = st.selectbox("New vs Repeat", ["New Loan", "Repeat Loan"])

        unemployment_rate = st.number_input("Unemployment Rate", value=5.0)

    # Derived features
    is_new_customer = 1 if new_repeat == "New Loan" else 0
    new_large_loan = 1 if (new_repeat == "New Loan" and total_amount > 5000) else 0

    log_amount = np.log1p(total_amount)

    # =====================================================
    # BUILD INPUT DATAFRAME
    # =====================================================

    input_dict = {
        "repayment_ratio": repayment_ratio,
        "Amount_Funded_By_Lender": amount_funded,
        "loan_pressure": loan_pressure,
        "Total_Amount_to_Repay": total_repay,
        "Total_Amount": total_amount,
        "Lender_portion_Funded": lender_portion,
        "month": month,
        "duration": duration,
        "loan_type": loan_type,
        "log_amount": log_amount,
        "year": year,
        "New_versus_Repeat": new_repeat,
        "is_new_customer": is_new_customer,
        "new_large_loan": new_large_loan,
        "unemployment_rate": unemployment_rate,
    }

    input_df = pd.DataFrame([input_dict])

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button("🚀 Predict Risk"):

        with st.spinner("Analyzing loan risk..."):

            probs = predict(input_df, return_proba=True)
            prob = probs[0]
            pred = int(prob > THRESHOLD)

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Default Probability", f"{prob:.2%}")

        with col2:
            st.metric("Threshold", f"{THRESHOLD:.2f}")

        with col3:
            st.metric("Prediction", "Default" if pred == 1 else "Safe")

        # =================================================
        # RISK LEVEL VISUALIZATION
        # =================================================

        if prob > 0.7:
            st.error("🔴 HIGH RISK: Immediate attention required")
        elif prob > 0.4:
            st.warning("🟡 MEDIUM RISK: Monitor closely")
        else:
            st.success("🟢 LOW RISK: Acceptable")

        # =================================================
        # INTERPRETATION PANEL
        # =================================================

        st.markdown("### 🧠 Risk Interpretation")

        insights = []

        if repayment_ratio > 1.5:
            insights.append("High repayment burden detected")

        if loan_pressure > 400:
            insights.append("Loan pressure is significantly high")

        if is_new_customer:
            insights.append("New customers historically carry higher risk")

        if unemployment_rate > 7:
            insights.append("Macro-economic risk is elevated")

        if len(insights) == 0:
            insights.append("No major risk factors detected")

        for insight in insights:
            st.write(f"• {insight}")