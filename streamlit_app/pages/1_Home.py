# =========================================================
# HOME — LOAN RISK INTELLIGENCE DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from src.modeling.predict import load_artifacts

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Home | Loan Risk System",
    layout="wide"
)

# =========================================================
# ELITE STYLING
# =========================================================

st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Headers */
h1, h2, h3 {
    color: #00C2FF;
}

/* Cards */
.card {
    background-color: #161B22;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0, 194, 255, 0.1);
}

/* Highlight */
.highlight {
    color: #00C2FF;
    font-weight: bold;
}

/* Divider */
hr {
    border: 1px solid #2A2F36;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD SYSTEM (CACHED)
# =========================================================

@st.cache_resource
def load_system():
    model, features, threshold = load_artifacts("models")
    return model, features, threshold

model, FEATURES, THRESHOLD = load_system()

# =========================================================
# HEADER
# =========================================================

st.title("💳 Loan Default Risk Intelligence System")
st.markdown("### Enterprise-Grade Decision Support Platform")

st.markdown("---")

# =========================================================
# EXECUTIVE METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Type", "LightGBM")

with col2:
    st.metric("Features Used", len(FEATURES))

with col3:
    st.metric("Decision Threshold", f"{THRESHOLD:.2f}")

with col4:
    st.metric("System Status", "🟢 Active")

st.markdown("---")

# =========================================================
# SYSTEM OVERVIEW
# =========================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🧠 System Overview")

    st.markdown("""
    <div class="card">
    
    This platform delivers a **production-grade machine learning system** designed to assess 
    loan default risk with high precision and reliability.

    It integrates:
    
    - 🔹 Advanced feature engineering pipelines  
    - 🔹 Time-based cross-validation for realistic evaluation  
    - 🔹 Threshold-optimized classification for business alignment  
    - 🔹 Fully automated inference pipelines  

    The system is engineered to support **real-world financial decision-making**, enabling:
    
    - Risk-aware lending decisions  
    - Portfolio monitoring  
    - Scalable batch predictions  
    
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("## ⚙️ System Specs")

    st.markdown(f"""
    <div class="card">
    
    <span class="highlight">Model:</span> LightGBM<br><br>
    <span class="highlight">Validation:</span> Time-Based CV<br><br>
    <span class="highlight">Metric:</span> F1 Score<br><br>
    <span class="highlight">Threshold:</span> {THRESHOLD:.2f}<br><br>
    <span class="highlight">Pipeline:</span> End-to-End ML System
    
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# CAPABILITIES
# =========================================================

st.markdown("## 🚀 Core Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h4>🔮 Real-Time Prediction</h4>
    Instant loan risk scoring with probability-based outputs and classification.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h4>📊 Batch Processing</h4>
    Upload datasets and generate scalable predictions for entire portfolios.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h4>📈 Risk Intelligence</h4>
    Understand key drivers of risk using model insights and feature analysis.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# RISK DISTRIBUTION (SIMULATED VISUAL)
# =========================================================

st.markdown("## 📉 Risk Distribution (Illustrative)")

# Simulated distribution
sample_probs = np.random.beta(2, 5, 500)

# Create histogram manually (clean bins)
counts, bin_edges = np.histogram(sample_probs, bins=20)

# Convert to DataFrame for plotting
hist_df = pd.DataFrame({
    "Probability": np.round(bin_edges[:-1], 2),
    "Count": counts
})

# Plot 
st.bar_chart(hist_df.set_index("Probability"))

st.caption("Illustrative distribution of predicted default probabilities.")

st.markdown("---")

# =========================================================
# WORKFLOW
# =========================================================

st.markdown("## 🧭 System Workflow")

st.markdown("""
<div class="card">

1️⃣ **Data Ingestion**  
Raw loan data is loaded and validated  

2️⃣ **Feature Engineering**  
Financial, temporal, and behavioral features are constructed  

3️⃣ **Model Inference**  
Trained LightGBM model generates probability scores  

4️⃣ **Threshold Optimization**  
Predictions converted into actionable decisions  

5️⃣ **Decision Output**  
Risk classification + interpretation  

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
---
💡 **Built as a production-grade machine learning system for financial risk assessment.**

Designed to demonstrate:
- Advanced ML engineering  
- Real-world deployment readiness  
- Decision intelligence systems  
""")