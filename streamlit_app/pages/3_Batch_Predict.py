# =========================================================
# BATCH LOAN RISK ANALYTICS DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from utils.inference import (
    run_inference,
    enrich_predictions,
    compute_portfolio_metrics
)

from utils.visualization import (
    plot_risk_distribution,
    plot_risk_segments,
    plot_portfolio_mix
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Batch Prediction | Loan Risk System",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("📊 Portfolio Risk Analytics Dashboard")
st.markdown("### Batch Loan Default Prediction & Risk Intelligence")

st.markdown("---")

# =========================================================
# USER GUIDE
# =========================================================

with st.expander("📘 How to Use This Dashboard"):
    st.markdown("""
### ✅ Supported Workflow
1. Upload a **processed dataset**
2. Run portfolio analysis
3. Explore risk insights & segmentation
4. Export results

### ⚠️ IMPORTANT NOTE
This system expects **feature-aligned data (same as training pipeline)**.

✔ Recommended:
- `test_merged.parquet`
- `train_merged.parquet`

❌ Not Supported:
- Raw datasets (missing engineered features)
""")

# =========================================================
# EXPECTED FORMAT
# =========================================================

st.markdown("### 📌 Expected Input Format")

st.info("""
Your dataset must match the processed schema used during training.

Required core columns include:
- Total_Amount
- Total_Amount_to_Repay
- duration
- loan_type
- New_versus_Repeat

Additional engineered features may be required internally.

👉 Best practice: Use the provided processed datasets.
""")

# =========================================================
# 📂FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Loan Portfolio",
    type=["csv", "parquet"]
)

# =========================================================
# LOAD DATA
# =========================================================

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_parquet(file)

# =========================================================
# PROCESS
# =========================================================

if uploaded_file:

    df = load_data(uploaded_file)

    st.markdown("### 📄 Data Preview")
    st.dataframe(df.head())

    st.markdown(f"**Dataset Shape:** {df.shape}")

    st.markdown("---")

    # =====================================================
    # SCHEMA VALIDATION (CRITICAL FIX)
    # =====================================================

    required_cols = [
        "Total_Amount",
        "Total_Amount_to_Repay",
        "duration",
        "loan_type",
        "New_versus_Repeat"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.warning("""
⚠️ This dataset does not match the expected schema.

👉 Please upload a processed dataset (e.g. test_merged.parquet)
👉 Raw datasets are not supported in this version
        """)
        st.stop()

    # =====================================================
    # RUN INFERENCE
    # =====================================================

    if st.button("🚀 Run Portfolio Analysis"):

        with st.spinner("Running full risk analytics..."):

            df_results = enrich_predictions(df)
            results = run_inference(df)

            probs = results["probabilities"]
            risk_levels = results["risk_levels"]
            summary = results["summary"]

            metrics = compute_portfolio_metrics(df_results)

        st.success("✅ Portfolio analysis complete")

        st.markdown("---")

        # =================================================
        # KPI DASHBOARD
        # =================================================

        st.markdown("## 📊 Key Portfolio Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Loans", metrics["total_loans"])
        col2.metric("Avg Risk", f"{metrics['avg_risk']:.2%}")
        col3.metric("High Risk %", f"{metrics['high_risk_ratio']:.2%}")
        col4.metric("Threshold", f"{summary['threshold']:.2f}")

        st.markdown("---")

        # =================================================
        # DISTRIBUTION
        # =================================================

        st.markdown("## 📉 Risk Distribution")

        fig = plot_risk_distribution(probs)
        st.pyplot(fig)

        st.markdown("---")

        # =================================================
        # SEGMENTATION
        # =================================================

        st.markdown("## 🧠 Risk Segmentation")

        col1, col2 = st.columns(2)

        with col1:
            fig_seg = plot_risk_segments(risk_levels)
            st.pyplot(fig_seg)

        with col2:
            fig_pie = plot_portfolio_mix(risk_levels)
            st.pyplot(fig_pie)

        # KPI breakdown
        st.markdown("### 📊 Segment Breakdown")

        col1, col2, col3 = st.columns(3)

        col1.metric("🟢 Low Risk", metrics["low_risk_count"])
        col2.metric("🟡 Medium Risk", metrics["medium_risk_count"])
        col3.metric("🔴 High Risk", metrics["high_risk_count"])

        st.markdown("---")

        # =================================================
        # HIGH RISK TABLE
        # =================================================

        st.markdown("## 🔴 High Risk Loans (Priority Review)")

        high_risk_df = df_results[df_results["Risk_Level"] == "High"]

        if len(high_risk_df) > 0:
            st.dataframe(high_risk_df.head(100))
        else:
            st.success("No high-risk loans detected")

        st.markdown("---")

        # =================================================
        # EXPORT
        # =================================================

        st.markdown("## 💾 Export Results")

        csv = df_results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full Predictions",
            data=csv,
            file_name="loan_portfolio_predictions.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # =================================================
        # INSIGHTS ENGINE
        # =================================================

        st.markdown("## 📌 Portfolio Insights")

        insights = []

        if metrics["high_risk_ratio"] > 0.4:
            insights.append("🔴 Portfolio has high default exposure")

        if metrics["avg_risk"] > 0.5:
            insights.append("⚠️ Average borrower risk is elevated")

        if metrics["high_risk_count"] > metrics["total_loans"] * 0.3:
            insights.append("⚠️ Significant proportion of loans are high risk")

        if metrics["avg_risk"] < 0.2:
            insights.append("🟢 Portfolio risk is well controlled")

        if not insights:
            insights.append("✅ Portfolio risk profile is stable")

        for i in insights:
            st.write(f"• {i}")