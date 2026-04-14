# =========================================================
# 🔮 SINGLE LOAN RISK PREDICTION (ELITE SYSTEM)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from utils.inference import run_single_inference
from utils.visualization import plot_probability_gauge

# =========================================================
# 🔥 PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Single Prediction | Loan Risk System",
    layout="wide"
)

# =========================================================
# 🎨 ELITE UI STYLING
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #00C2FF;
}

.section-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 🏠 HEADER
# =========================================================

st.title("🔮 Loan Risk Prediction Engine")
st.markdown("### Real-Time Credit Risk Assessment")

with st.expander("📘 Input Guide (How to Use This Tool)"):
    st.markdown("""
    ### 💰 Financial Metrics
    - **Total Loan Amount**: Total amount borrowed
    - **Amount Funded by Lender**: Portion funded by institution
    - **Lender Portion Funded**: Percentage of funding by lender

    ### 📅 Loan Structure
    - **Loan Type**: Product category affecting risk
    - **Customer Type**: New vs repeat borrowers

    ### 🌍 Macroeconomic
    - **Unemployment Rate**: External risk factor
    """)

st.markdown("---")

# =========================================================
# 🧾 INPUT SECTION
# =========================================================

st.markdown("## 🧾 Loan Input Parameters")

col1, col2 = st.columns(2)

# =========================
# 💰 FINANCIAL
# =========================
with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Financial Metrics")

    total_amount = st.number_input(
    "Total Loan Amount",
    100.0, 100000.0, 5000.0,
    help="Total amount borrowed by the customer"
    )
    total_repay = st.number_input("Total Amount to Repay", 100.0, 150000.0, 6000.0)
    amount_funded = st.number_input(
    "Amount Funded by Lender",
    0.0, 100000.0, 5000.0,
    help="Portion of the loan funded by the institution"
    )
    lender_portion = st.slider(
    "Lender Portion Funded",
    0.0, 1.0, 0.8,
    help="Ratio of loan funded by lender (1 = fully funded)"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📅 STRUCTURE
# =========================
with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📅 Loan Structure")

    duration = st.number_input("Loan Duration (months)", 1, 60, 12)
    month = st.selectbox("Disbursement Month", list(range(1, 13)))
    year = st.number_input("Year", 2020, 2035, 2023)

    loan_type = st.selectbox(
    "Loan Type",
    ["Type_A", "Type_B", "Type_C"],
    help="Different loan products with varying risk profiles"
    )
    new_repeat = st.selectbox("Customer Type", ["New Loan", "Repeat Loan"])

    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📘 Loan Type Guide"):
    st.markdown("""
    - **Type A** → Short-term, lower risk loans (e.g. salary advances)
    - **Type B** → Medium-term loans with moderate risk
    - **Type C** → Long-term or unsecured loans (higher risk)

    These categories influence how the model evaluates default probability.
    """)
# =========================
# 🌍 MACRO (🔥 FIXED HERE)
# =========================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🌍 Macroeconomic Context")

unemployment_rate = st.number_input(
    "Unemployment Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    help="Macro-economic indicator affecting default risk"
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# 📦 BUILD INPUT DATA
# =========================================================

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
    "unemployment_rate": unemployment_rate,  # ✅ FIXED
}

# =========================================================
# 🚀 PREDICTION BUTTON
# =========================================================
if total_repay < total_amount:
    st.warning("⚠️ Repayment amount is less than loan amount — check input")

if lender_portion == 0:
    st.info("ℹ️ No lender funding — unusual scenario")

if duration > 48:
    st.info("ℹ️ Long-duration loans tend to carry higher risk")

if st.button("🚀 Analyze Risk"):

    with st.spinner("Running model inference..."):

        result = run_single_inference(input_dict)

        prob = result["probability"]
        pred = result["prediction"]
        risk = result["risk_level"]
        threshold = result["threshold"]

    st.markdown("---")
    st.markdown("## 📊 Risk Assessment Results")

    # =====================================================
    # 🎯 METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{prob:.2%}")

    with col2:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    with col3:
        st.metric("Classification", "Default" if pred else "Safe")

    # =====================================================
    # 📊 GAUGE
    # =====================================================

    st.markdown("### 🎯 Risk Gauge")

    fig = plot_probability_gauge(prob)
    st.pyplot(fig)

    # =====================================================
    # 🚦 STATUS
    # =====================================================

    st.markdown("### 🚦 Risk Status")

    st.progress(float(prob))

    if risk == "High":
        st.error("🔴 HIGH RISK — Reject or require strong mitigation")
    elif risk == "Medium":
        st.warning("🟡 MEDIUM RISK — Proceed with caution")
    else:
        st.success("🟢 LOW RISK — Acceptable borrower")

    # =====================================================
    # 🧠 INSIGHTS
    # =====================================================

    st.markdown("## 🧠 Risk Drivers")

    insights = []

    if total_repay / (total_amount + 1) > 1.5:
        insights.append("High repayment burden increases default risk")

    if total_amount / (duration + 1) > 400:
        insights.append("Loan pressure is elevated relative to duration")

    if new_repeat == "New Loan":
        insights.append("New customers historically exhibit higher default risk")

    if lender_portion < 0.5:
        insights.append("Lower lender commitment may indicate shared risk")

    if unemployment_rate > 7:
        insights.append("Macroeconomic conditions indicate elevated risk")

    if not insights:
        insights.append("No significant risk factors detected")

    for insight in insights:
        st.write(f"• {insight}")

    # =====================================================
    # 📌 DECISION
    # =====================================================

    st.markdown("## 📌 Decision Recommendation")

    if risk == "High":
        st.error("🔴 HIGH RISK")
        st.markdown("### 🚫 Recommended Action:")
        st.markdown("- Reject application OR request collateral")
    elif risk == "Medium":
        st.warning("🟡 MEDIUM RISK")
        st.markdown("### ⚠️ Recommended Action:")
        st.markdown("- Review borrower history")
        st.markdown("- Consider reducing exposure")
    else:
        st.success("🟢 LOW RISK")
        st.markdown("### ✅ Recommended Action:")
        st.markdown("- Approve under standard conditions")

    st.markdown("### 📊 Interpretation")

    st.markdown(f"""
    - The model estimates a **{prob:.2%} probability of default**
    - This is **{'above' if prob > threshold else 'below'}** the risk threshold
    - Classification: **{risk} risk borrower**
    """)