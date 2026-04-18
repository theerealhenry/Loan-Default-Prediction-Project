# =========================================================
# MODEL INSIGHTS DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(".."))

from utils.preprocessing import preprocess_dataframe
from utils.inference import load_system, load_config
from src.modeling.predict import prepare_input

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Insights | Loan Risk System",
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
.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD SYSTEM (SAFE CACHE)
# =========================================================

@st.cache_resource
def load_all():
    model, features, threshold = load_system()
    config = load_config()
    return model, features, threshold, config

model, FEATURES, THRESHOLD, config = load_all()

# =========================================================
# LOAD SAMPLE DATA
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_sample():
    try:
        path = os.path.join(BASE_DIR, "data", "processed", "train_merged.parquet")
        df = pd.read_parquet(path)
        return df.sample(min(200, len(df)), random_state=42)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

sample_df = load_sample()

# =========================================================
# HEADER
# =========================================================

st.title("🧠 AI Explainability Dashboard")
st.markdown("### Deep Model Intelligence & Risk Interpretation")
st.markdown("---")

# =========================================================
# MODEL FEATURE IMPORTANCE
# =========================================================

st.markdown("## 📊 Model Feature Importance")

try:
    fi_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))

except Exception:
    st.warning("Model does not expose feature_importances_")

st.markdown("---")

# =========================================================
# SHAP ENGINE
# =========================================================

st.markdown("## 🧠 SHAP Explainability Engine")

if sample_df is None:
    st.warning("⚠️ Sample dataset not found.")
    st.stop()

# =========================================================
# PREPROCESS PIPELINE
# =========================================================

df_clean = preprocess_dataframe(sample_df)
X_sample = prepare_input(df_clean, config)

# Align features safely
X_sample = X_sample.reindex(columns=FEATURES, fill_value=0)

# Reduce size for performance
X_sample = X_sample.sample(min(150, len(X_sample)), random_state=42)

# =========================================================
# SHAP COMPUTATION
# =========================================================

with st.spinner("Computing SHAP values..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

# =========================================================
# SHAP OUTPUT
# =========================================================

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.array(shap_values)

# Fix expected value
expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]

# =========================================================
# GLOBAL EXPLAINABILITY
# =========================================================

st.markdown("## 🌍 Global Feature Impact")

try:
    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": shap_importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(shap_df.set_index("Feature"))

except Exception as e:
    st.error(f"SHAP global error: {e}")

# =========================================================
# SHAP SUMMARY (BEESWARM)
# =========================================================

st.markdown("### 🔥 SHAP Summary (Beeswarm)")

try:
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)
    plt.clf()
except Exception as e:
    st.warning(f"Summary plot error: {e}")

# =========================================================
# SHAP BAR PLOT
# =========================================================

st.markdown("### 📊 SHAP Global Importance")

try:
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig2)
    plt.clf()
except Exception as e:
    st.warning(f"Bar plot error: {e}")

st.markdown("---")

# =========================================================
# LOCAL EXPLAINABILITY
# =========================================================

st.markdown("## 🔍 Individual Prediction Analysis")

idx = st.slider("Select Loan Index", 0, len(X_sample) - 1, 0)

single_row = X_sample.iloc[[idx]]
single_shap = explainer.shap_values(single_row)

if isinstance(single_shap, list):
    single_shap = single_shap[1]

single_shap = np.array(single_shap).flatten()

# =========================================================
# CONTRIBUTION TABLE
# =========================================================

contrib_df = pd.DataFrame({
    "Feature": FEATURES,
    "Contribution": single_shap
}).sort_values(by="Contribution", key=np.abs, ascending=False)

st.markdown("### 📊 Feature Contributions")
st.dataframe(contrib_df.head(15), use_container_width=True)

# =========================================================
# CONTRIBUTION BAR
# =========================================================

st.markdown("### 📈 Contribution Chart")
st.bar_chart(contrib_df.set_index("Feature").head(10))

# =========================================================
# WATERFALL PLOT 
# =========================================================

st.markdown("### 🌊 SHAP Waterfall")

try:
    fig3 = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        single_shap,
        feature_names=FEATURES
    )
    st.pyplot(fig3)
    plt.clf()
except Exception as e:
    st.warning(f"Waterfall error: {e}")

# =========================================================
# FORCE PLOT
# =========================================================

st.markdown("### ⚡ SHAP Force Plot")

try:
    force_plot = shap.force_plot(
        expected_value,
        single_shap,
        single_row,
        matplotlib=False
    )

    st.components.v1.html(shap.getjs() + force_plot.html(), height=300)

except Exception as e:
    st.warning(f"Force plot error: {e}")

# =========================================================
# INTERPRETATION ENGINE
# =========================================================

st.markdown("## 🧠 AI Interpretation")

top_pos = contrib_df[contrib_df["Contribution"] > 0].head(3)
top_neg = contrib_df[contrib_df["Contribution"] < 0].head(3)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Risk Increasing")
    for _, r in top_pos.iterrows():
        st.write(f"• {r['Feature']} (+{r['Contribution']:.4f})")

with col2:
    st.markdown("### 🟢 Risk Reducing")
    for _, r in top_neg.iterrows():
        st.write(f"• {r['Feature']} ({r['Contribution']:.4f})")

# =========================================================
# MODEL SUMMARY
# =========================================================

st.markdown("---")
st.markdown("## 📈 Model Summary")

st.markdown(f"""
<div class="card">

<b>Model:</b> LightGBM<br>
<b>Features:</b> {len(FEATURES)}<br>
<b>Threshold:</b> {THRESHOLD:.2f}<br><br>

Production-grade ML system with full explainability stack:
• Global + Local SHAP  
• Interactive force plots  
• Risk decomposition engine  

</div>
""", unsafe_allow_html=True)