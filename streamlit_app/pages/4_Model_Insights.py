# =========================================================
# 🧠 MODEL INSIGHTS DASHBOARD (ELITE AI EXPLAINABILITY PLATFORM)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap

import sys, os
sys.path.append(os.path.abspath(".."))

from utils.preprocessing import preprocess_dataframe
from utils.inference import load_system, load_config
from src.modeling.predict import prepare_input

# =========================================================
# 🔥 PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Insights | Loan Risk System",
    layout="wide"
)

# =========================================================
# 🎨 UI STYLING
# =========================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #00C2FF;
}
.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 📦 LOAD SYSTEM
# =========================================================

@st.cache_resource
def load_all():
    model, features, threshold = load_system()
    config = load_config()
    return model, features, threshold, config

model, FEATURES, THRESHOLD, config = load_all()

# =========================================================
# 🏠 HEADER
# =========================================================

st.title("🧠 AI Explainability Dashboard")
st.markdown("### Deep Model Intelligence & Risk Interpretation")

st.markdown("---")

# =========================================================
# 📊 FEATURE IMPORTANCE (MODEL LEVEL)
# =========================================================

st.markdown("## 📊 Model Feature Importance")

fi_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(fi_df.set_index("Feature"))

st.markdown("---")

# =========================================================
# 📂 LOAD SAMPLE DATA
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_sample():
    try:
        path = os.path.join(BASE_DIR, "data/processed/train_merged.parquet")
        return pd.read_parquet(path).sample(300, random_state=42)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

sample_df = load_sample()

# =========================================================
# 🧠 SHAP EXPLAINABILITY ENGINE
# =========================================================

st.markdown("## 🧠 SHAP Explainability Engine")

st.markdown("""
<div class="card">
SHAP (SHapley Additive exPlanations) explains how each feature contributes to model predictions.
This dashboard provides both global and individual explanations.
</div>
""", unsafe_allow_html=True)

if sample_df is not None:

    # =====================================================
    # 🔥 PIPELINE ALIGNMENT (CRITICAL FIX)
    # =====================================================

    df_clean = preprocess_dataframe(sample_df)

    X_sample = prepare_input(df_clean, config)

    # Ensure correct feature order
    X_sample = X_sample[FEATURES]

    # =====================================================
    # 🔥 SHAP COMPUTATION
    # =====================================================

    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # =====================================================
    # 🌍 GLOBAL EXPLAINABILITY
    # =====================================================

    st.markdown("## 🌍 Global Feature Impact")

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # for binary classification

    # Now shap_values should be (n_samples, n_features)

    shap_importance = np.abs(shap_values).mean(axis=0)

    

    shap_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "SHAP Importance": shap_importance
    }).sort_values(by="SHAP Importance", ascending=False)

    st.bar_chart(shap_df.set_index("Feature"))

    st.markdown("### 🧠 Top Risk Drivers")

    top_features = shap_df.head(10)

    for _, row in top_features.iterrows():
        st.write(f"• {row['Feature']} → Impact Score: {row['SHAP Importance']:.4f}")

    st.markdown("---")

    # =====================================================
    # 🔍 LOCAL EXPLAINABILITY (INTERACTIVE)
    # =====================================================

    st.markdown("## 🔍 Individual Prediction Analysis")

    idx = st.slider("Select Loan Index", 0, len(X_sample) - 1, 0)

    selected_row = X_sample.iloc[[idx]]

    shap_val = explainer.shap_values(selected_row)[0]

    contrib_df = pd.DataFrame({
        "Feature": FEATURES,
        "Contribution": shap_val
    }).sort_values(by="Contribution", key=abs, ascending=False)

    # =====================================================
    # 📊 CONTRIBUTION TABLE
    # =====================================================

    st.markdown("### 📊 Feature Contributions")

    st.dataframe(contrib_df.head(15))

    # =====================================================
    # 📈 CONTRIBUTION BAR CHART
    # =====================================================

    st.markdown("### 📈 Contribution Visualization")

    st.bar_chart(contrib_df.set_index("Feature").head(10))

    # =====================================================
    # 🧠 INTERPRETATION ENGINE
    # =====================================================

    st.markdown("## 🧠 AI Interpretation")

    top_positive = contrib_df[contrib_df["Contribution"] > 0].head(3)
    top_negative = contrib_df[contrib_df["Contribution"] < 0].head(3)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Risk Increasing Factors")
        for _, row in top_positive.iterrows():
            st.write(f"• {row['Feature']} (+{row['Contribution']:.4f})")

    with col2:
        st.markdown("### 🟢 Risk Reducing Factors")
        for _, row in top_negative.iterrows():
            st.write(f"• {row['Feature']} ({row['Contribution']:.4f})")

else:
    st.warning("⚠️ Sample dataset not found. Upload processed data to enable SHAP.")

st.markdown("---")

# =========================================================
# 📈 MODEL SUMMARY
# =========================================================

st.markdown("## 📈 Model Summary")

st.markdown(f"""
<div class="card">

<b>Model Type:</b> LightGBM<br><br>
<b>Total Features:</b> {len(FEATURES)}<br><br>
<b>Decision Threshold:</b> {THRESHOLD:.2f}<br><br>

This model uses advanced feature engineering, optimized thresholding,
and explainability techniques to support real-world credit decisions.

</div>
""", unsafe_allow_html=True)