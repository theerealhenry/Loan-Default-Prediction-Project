# =========================================================
# 🧾 ABOUT | LOAN RISK INTELLIGENCE PLATFORM
# =========================================================

import streamlit as st

# =========================================================
# 🔥 PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="About | Loan Risk Intelligence Platform",
    layout="wide"
)

# =========================================================
# 🎨 ELITE UI STYLING
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0E1117 0%, #0A0D12 100%);
}

h1, h2, h3 {
    color: #00C2FF;
    font-weight: 600;
}

.card {
    background: linear-gradient(145deg, #161B22, #1C2128);
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

.highlight {
    color: #00C2FF;
    font-weight: 600;
}

.small {
    font-size: 14px;
    color: #9BA3AF;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 🏠 HEADER
# =========================================================

st.title("🧠 Loan Risk Intelligence Platform")
st.markdown("### Production-Grade Machine Learning System for Credit Risk Decisioning")

st.markdown("---")

# =========================================================
# 🚀 EXECUTIVE SUMMARY
# =========================================================

st.markdown("## 🚀 Executive Overview")

st.markdown("""
<div class="card">

This platform is a <span class="highlight">full-stack machine learning system</span> designed to
simulate real-world credit risk decisioning in modern fintech environments.

It combines:

• Predictive modeling (LightGBM)  
• Explainable AI (SHAP)  
• Portfolio-level analytics  
• Real-time & batch inference pipelines  

The system is built with a strong focus on:

<span class="highlight">scalability, interpretability, and production readiness</span>.

</div>
""", unsafe_allow_html=True)

# =========================================================
# 🏗️ SYSTEM ARCHITECTURE
# =========================================================

st.markdown("## 🏗️ System Architecture")

st.markdown("""
<div class="card">

<span class="highlight">Modular ML Architecture:</span>

UI Layer → Streamlit Application  
Application Layer → utils/ (preprocessing, inference, visualization)  
ML Layer → src/ (feature engineering, modeling, prediction)  

<br>

<span class="highlight">Pipeline Flow:</span>

1. Data ingestion & validation  
2. Feature engineering & transformation  
3. Feature alignment (train vs inference consistency)  
4. Model inference (probability estimation)  
5. Risk segmentation & threshold decisioning  
6. Explainability (global & local SHAP insights)  

<br>

This architecture mirrors <span class="highlight">real-world ML systems</span> used in:

• Digital lending platforms  
• Credit scoring engines  
• Risk analytics systems  

</div>
""", unsafe_allow_html=True)

# =========================================================
# ⚙️ MODEL ENGINEERING
# =========================================================

st.markdown("## ⚙️ Model Engineering & Strategy")

st.markdown("""
<div class="card">

<span class="highlight">Model:</span> LightGBM Gradient Boosting Classifier  

<br>

<span class="highlight">Key Design Decisions:</span>

• Time-based cross-validation (prevents leakage)  
• Out-of-fold predictions for robust evaluation  
• Custom threshold optimization (business-aligned decisions)  
• Class imbalance handling  

<br>

<span class="highlight">Objective:</span>

Move beyond accuracy → optimize for <b>decision quality</b> in real-world lending scenarios.

</div>
""", unsafe_allow_html=True)

# =========================================================
# 📊 FEATURE ENGINEERING
# =========================================================

st.markdown("## 📊 Feature Engineering Intelligence")

st.markdown("""
<div class="card">

Feature engineering is designed to capture <span class="highlight">real borrower behavior</span>:

<br>

• Financial stress indicators → repayment ratios, loan pressure  
• Behavioral signals → new vs repeat borrower  
• Interaction effects → new_large_loan  
• Temporal patterns → loan timing features  
• Transformations → log scaling for stability  

<br>

All transformations are:

✔ Deterministic  
✔ Reproducible  
✔ Consistent across training & inference  

</div>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 EXPLAINABILITY PLATFORM
# =========================================================

st.markdown("## 🧠 Explainability & AI Transparency")

st.markdown("""
<div class="card">

This system integrates <span class="highlight">SHAP (Shapley Additive Explanations)</span>
to provide deep model interpretability.

<br>

<span class="highlight">Capabilities:</span>

• Global feature importance (model-level insights)  
• Local explanations (per-loan predictions)  
• Feature contribution breakdowns  
• Risk driver identification  

<br>

<span class="highlight">Why this matters:</span>

• Regulatory compliance (finance & lending)  
• Trust & transparency  
• Model debugging & validation  

</div>
""", unsafe_allow_html=True)

# =========================================================
# 📊 ANALYTICS & DECISION SUPPORT
# =========================================================

st.markdown("## 📊 Portfolio Analytics & Decision Intelligence")

st.markdown("""
<div class="card">

Beyond predictions, the platform delivers <span class="highlight">decision intelligence</span>:

<br>

• Risk segmentation (Low / Medium / High)  
• Portfolio-level risk metrics  
• High-risk loan identification  
• Downloadable analytics outputs  

<br>

This enables:

✔ Credit analysts to prioritize risk  
✔ Businesses to optimize lending strategies  
✔ Data-driven decision-making  

</div>
""", unsafe_allow_html=True)

# =========================================================
# 🚀 PRODUCTION CAPABILITIES
# =========================================================

st.markdown("## 🚀 Production-Grade Capabilities")

st.markdown("""
<div class="card">

This system is engineered like a <span class="highlight">real ML product</span>:

<br>

• Config-driven architecture (YAML-based)  
• Modular codebase (separation of concerns)  
• Reusable inference pipeline  
• Batch + real-time prediction support  
• Robust preprocessing & validation  

<br>

Designed to be easily extended into:

• REST APIs (FastAPI)  
• Cloud deployment (AWS / GCP)  
• CI/CD ML pipelines  

</div>
""", unsafe_allow_html=True)

# =========================================================
# 👤 AUTHOR (PERSONAL BRANDING)
# =========================================================

st.markdown("## 👤 Author")

col1, col2 = st.columns([1, 3])

with col1:
    st.image(
        "https://media.licdn.com/dms/image/v2/D4D03AQFmHwaRuT-ViA/profile-displayphoto-scale_400_400/B4DZ0mcuZBGoAg-/0/1774466579323?e=1778112000&v=beta&t=4UxqkdpP5Njw4gOMHGZhv0FpzMvt-5vT2cvSkkptbEI",  
        width=410
    )

with col2:
    st.markdown("""
    <div class="card">

    <h3>Henry Otsyula</h3>

    <span class="highlight">Senior Data Scientist | Machine Learning Engineer</span>

    <br><br>

    Specializing in:

    • End-to-end ML system design  
    • Predictive modeling & optimization  
    • Explainable AI systems  
    • Production-grade ML architecture  

    <br>

    This project demonstrates the ability to:

    ✔ Build scalable ML systems  
    ✔ Design real-world data pipelines  
    ✔ Translate models into business decisions  
    ✔ Implement explainability at production level  

    <br>

    <span class="small">
    Focused on building intelligent systems that are not just accurate — but usable, interpretable, and impactful.
    </span>

    <hr>

    📧 Email: henryotsyula01@gmail.com  
    🔗 GitHub: https://github.com/theerealhenry  
    💼 LinkedIn: https://www.linkedin.com/in/henry-otsyula-datascientist  

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 🔮 FUTURE ROADMAP
# =========================================================

st.markdown("## 🔮 Future Roadmap")

st.markdown("""
<div class="card">

Planned enhancements:

<br>

• API deployment (FastAPI microservice)  
• Real-time scoring pipelines  
• Advanced ensemble modeling  
• Automated retraining workflows  
• Cloud-native deployment  

<br>

Goal: evolve into a <span class="highlight">fully production-ready AI decision platform</span>

</div>
""", unsafe_allow_html=True)

# =========================================================
# 🏁 FOOTER
# =========================================================

st.markdown("---")
st.markdown("### 🚀 Built as a Real-World Machine Learning System — Not Just a Model")