# =========================================================
# LOAN DEFAULT RISK SYSTEM
# =========================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from utils.inference import run_single_inference, load_system

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Loan Risk Intelligence System",
    page_icon="💳",
    layout="wide",
)

# =========================================================
# UI STYLING
# =========================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #00C2FF;
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.stButton>button {
    background-color: #00C2FF;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

[data-testid="stSidebar"] {
    background-color: #161B22;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD SYSTEM
# =========================================================

@st.cache_resource
def initialize_system():
    return load_system()

try:
    _, _, THRESHOLD = initialize_system()
except Exception as e:
    st.error(f"System failed to load: {e}")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Loan Risk System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔮 Single Prediction"]
)

# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title("💳 Loan Risk Intelligence System")

    st.markdown("""
    ### 🎯 Enterprise Credit Risk Intelligence Platform

    This system delivers **production-grade machine learning predictions**
    for loan default risk, designed to simulate **real-world fintech deployment**.

    ---
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "LightGBM")
    col2.metric("Threshold", f"{THRESHOLD:.2f}")
    col3.metric("Status", "Production Ready")

    st.markdown("---")

    st.markdown("""
    ### 🚀 Capabilities
    - 🔮 Real-time risk prediction  
    - 📊 Portfolio analytics (Batch)  
    - 🧠 Explainability (SHAP)  
    - ⚙️ Automated ML pipeline  

    ### 🏗️ Architecture
    - Feature engineering pipeline  
    - Config-driven system  
    - Inference abstraction layer  
    - Docker-ready deployment  

    ---
    """)

# =========================================================
# SINGLE PREDICTION
# =========================================================

elif page == "🔮 Single Prediction":

    st.title("🔮 Loan Risk Prediction Engine")
    st.markdown("### Real-Time Credit Risk Assessment")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =========================
    # INPUTS
    # =========================

    with col1:
        total_amount = st.number_input(
            "Total Loan Amount",
            value=5000.0,
            help="Principal loan amount"
        )

        total_repay = st.number_input(
            "Total Amount to Repay",
            value=6000.0,
            help="Total repayment including interest"
        )

        amount_funded = st.number_input(
            "Amount Funded by Lender",
            value=5000.0,
            help="Capital contributed by lender"
        )

        lender_portion = st.slider(
            "Lender Portion Funded",
            0.0, 1.0, 0.8,
            help="Proportion of loan funded by lender"
        )

    with col2:
        duration = st.number_input(
            "Loan Duration (months)",
            value=12
        )

        month = st.selectbox(
            "Disbursement Month",
            list(range(1, 13))
        )

        year = st.number_input(
            "Year",
            value=2023
        )

        loan_type = st.selectbox(
            "Loan Type",
            ["Type_A", "Type_B", "Type_C"],
            help="""
            Type_A: Short-term, low-risk loans  
            Type_B: Medium-term loans  
            Type_C: High-risk / long-term loans  
            """
        )

        new_repeat = st.selectbox(
            "Customer Type",
            ["New Loan", "Repeat Loan"]
        )

    # Optional macro input
    unemployment_rate = st.slider(
        "Unemployment Rate (%)",
        0.0, 15.0, 5.0
    )

    st.markdown("---")

    # =====================================================
    # BUILD INPUT
    # =====================================================

    input_dict = {
        "Total_Amount": total_amount,
        "Total_Amount_to_Repay": total_repay,
        "Amount_Funded_By_Lender": amount_funded,
        "Lender_portion_Funded": lender_portion,
        "duration": duration,
        "month": month,
        "year": year,
        "loan_type": loan_type,
        "New_versus_Repeat": new_repeat,
        "unemployment_rate": unemployment_rate,
    }

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button("🚀 Analyze Risk"):

        with st.spinner("Running model inference..."):

            result = run_single_inference(input_dict)

            prob = result["probability"]
            pred = result["prediction"]
            risk = result["risk_level"]
            threshold = result["threshold"]

        st.markdown("---")
        st.markdown("## 📊 Risk Assessment")

        col1, col2, col3 = st.columns(3)

        col1.metric("Default Probability", f"{prob:.2%}")
        col2.metric("Threshold", f"{threshold:.2f}")
        col3.metric("Classification", "Default" if pred else "Safe")

        st.progress(float(prob))

        # =========================
        # RISK STATUS
        # =========================

        if risk == "High":
            st.error("🔴 HIGH RISK — Reject or mitigate")
        elif risk == "Medium":
            st.warning("🟡 MEDIUM RISK — Review carefully")
        else:
            st.success("🟢 LOW RISK — Acceptable")

        # =========================
        # INTERPRETATION
        # =========================

        st.markdown("## 🧠 Key Risk Drivers")

        insights = []

        if total_repay / (total_amount + 1) > 1.5:
            insights.append("High repayment burden")

        if total_amount / (duration + 1) > 400:
            insights.append("High loan pressure")

        if new_repeat == "New Loan":
            insights.append("New borrower risk")

        if unemployment_rate > 7:
            insights.append("Macroeconomic risk elevated")

        if not insights:
            insights.append("No major risk signals")

        for i in insights:
            st.write(f"• {i}")