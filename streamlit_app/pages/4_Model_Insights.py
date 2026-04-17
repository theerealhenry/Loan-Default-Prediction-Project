# =========================================================
# 🧠 MODEL INSIGHTS DASHBOARD (PRODUCTION-GRADE)
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
# 📂 LOAD SAMPLE DATA (ROBUST FIX)
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_sample():
    try:
        path = os.path.join(BASE_DIR, "data", "processed", "train_merged.parquet")
        df = pd.read_parquet(path)
        return df.sample(min(300, len(df)), random_state=42)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

sample_df = load_sample()

# =========================================================
# 🏠 HEADER
# =========================================================

st.title("🧠 AI Explainability Dashboard")
st.markdown("### Deep Model Intelligence & Risk Interpretation")
st.markdown("---")

# =========================================================
# 📊 MODEL FEATURE IMPORTANCE
# =========================================================

st.markdown("## 📊 Model Feature Importance")

try:
    fi_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))

except:
    st.warning("Model does not expose feature_importances_")

st.markdown("---")

# =========================================================
# 🧠 SHAP ENGINE
# =========================================================

st.markdown("## 🧠 SHAP Explainability Engine")

if sample_df is None:
    st.warning("⚠️ Sample dataset not found.")
    st.stop()

# =========================================================
# 🔧 PREPROCESS
# =========================================================

df_clean = preprocess_dataframe(sample_df)
X_sample = prepare_input(df_clean, config)

# Ensure alignment
X_sample = X_sample.reindex(columns=FEATURES, fill_value=0)

# =========================================================
# 🔥 SHAP COMPUTATION (SAFE)
# =========================================================

@st.cache_resource
def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

explainer, shap_values = compute_shap(model, X_sample)

# =========================================================
# 🔧 CLEAN SHAP FUNCTION
# =========================================================

def clean_shap_values(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.array(shap_values)

shap_values = clean_shap_values(shap_values)

# =========================================================
# 🌍 GLOBAL EXPLAINABILITY
# =========================================================

st.markdown("## 🌍 Global Feature Impact")

shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": shap_importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(shap_df.set_index("Feature"))

# =========================================================
# 🔥 SHAP SUMMARY PLOT (BEESWARM)
# =========================================================

st.markdown("### 🔥 SHAP Summary Plot")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_sample, show=False)
st.pyplot(fig)

# =========================================================
# 🔥 SHAP BAR PLOT
# =========================================================

st.markdown("### 📊 SHAP Global Importance (Bar)")

fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(fig2)

st.markdown("---")

# =========================================================
# 🔍 LOCAL EXPLAINABILITY
# =========================================================

st.markdown("## 🔍 Individual Prediction Analysis")

idx = st.slider("Select Loan Index", 0, len(X_sample) - 1, 0)

single_row = X_sample.iloc[[idx]]
single_shap = clean_shap_values(explainer.shap_values(single_row))[0].flatten()

# =========================================================
# 📊 CONTRIBUTION TABLE
# =========================================================

contrib_df = pd.DataFrame({
    "Feature": FEATURES,
    "Contribution": single_shap
}).sort_values(by="Contribution", key=np.abs, ascending=False)

st.markdown("### 📊 Feature Contributions")
st.dataframe(contrib_df.head(15))

# =========================================================
# 📈 CONTRIBUTION BAR
# =========================================================

st.markdown("### 📈 Contribution Chart")
st.bar_chart(contrib_df.set_index("Feature").head(10))

# =========================================================
# 🔥 WATERFALL PLOT
# =========================================================

st.markdown("### 🌊 SHAP Waterfall Plot")

fig3 = plt.figure()
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    single_shap,
    feature_names=FEATURES
)
st.pyplot(fig3)

# =========================================================
# 🔥 FORCE PLOT (INTERACTIVE HTML)
# =========================================================

st.markdown("### ⚡ SHAP Force Plot")

force_plot = shap.force_plot(
    explainer.expected_value,
    single_shap,
    single_row,
    matplotlib=False
)

st.components.v1.html(shap.getjs() + force_plot.html(), height=300)

# =========================================================
# 🧠 INTERPRETATION ENGINE
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

st.markdown("---")

# =========================================================
# 📈 MODEL SUMMARY
# =========================================================

st.markdown("## 📈 Model Summary")

st.markdown(f"""
<div class="card">

<b>Model:</b> LightGBM<br>
<b>Features:</b> {len(FEATURES)}<br>
<b>Threshold:</b> {THRESHOLD:.2f}

Production-grade ML system with explainability layer.

</div>
""", unsafe_allow_html=True)