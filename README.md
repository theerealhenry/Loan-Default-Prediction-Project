# 💳 Loan Risk Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> 🚀 **Production-grade machine learning platform for real-time and batch loan default prediction, featuring explainable AI, portfolio analytics, and scalable deployment.**

---

## 🌐 Live Application

🔗 **Streamlit App:**  
https://loan-default-risk-app.streamlit.app/

🔗 **GitHub Repository:**  
https://github.com/theerealhenry/Loan-Default-Prediction-Project

---

## 🎥 System Preview

### 🏠 Main Dashboard
![Dashboard](streamlit_app/assets/dashboard.png)

### 🔮 Single Loan Prediction
![Single Prediction](streamlit_app/assets/single_prediction.png)

### 📊 Portfolio Analytics
![Batch Dashboard](streamlit_app/assets/batch_dashboard.png)

### 🧠 Model Explainability (SHAP)
![SHAP Insights](streamlit_app/assets/shap_dashboard.png)

---

## 🎯 Problem Context

Financial institutions must accurately assess **loan default risk** to minimize losses and optimize lending decisions.

This challenge becomes significantly more complex in **emerging markets**, where:

- Customer behavior is heterogeneous  
- Economic conditions are dynamic  
- Data distributions vary across regions  

This system is inspired by real-world financial data from the **AI4EAC Finance Challenge (Zindi)** and is designed to:

✔ Generalize across markets (Kenya & Ghana)  
✔ Handle class imbalance and noisy features  
✔ Provide **interpretable, production-ready predictions**

---

## 💡 Solution Overview

The **Loan Risk Intelligence System** is an end-to-end machine learning platform that enables:

- 🔮 **Real-time loan risk prediction**
- 📊 **Batch portfolio risk analytics**
- 🧠 **Explainable AI (SHAP-based insights)**
- 🎯 **Risk segmentation & decision support**

The system is designed to replicate a **real-world fintech credit scoring engine**, combining predictive accuracy with transparency and usability.

---

## 🏗️ System Architecture

Raw Data → Feature Engineering → Model (LightGBM) → Inference Layer → Risk Segmentation → Streamlit UI (Real-Time + Batch + SHAP)


### Key Design Principles:
- Modular pipeline (separation of concerns)
- Config-driven architecture (YAML)
- Reproducible training & inference
- Production-ready deployment

---

## ⚙️ Key Features

### 🔮 Prediction Engine
- Single-loan real-time prediction
- Batch portfolio scoring
- Probability-based classification

### 📊 Analytics Dashboard
- Portfolio-level KPIs
- Risk distribution visualization
- Segment-level breakdown (Low / Medium / High)

### 🧠 Explainability Engine
- Global feature importance (SHAP)
- Per-loan contribution analysis
- Transparent decision reasoning

### 🎯 Decision Support System
- Risk categorization
- Actionable recommendations
- Threshold-optimized predictions

---

## 🤖 Machine Learning Pipeline

### 📦 Data Processing
- Data cleaning & validation
- Handling missing values
- Removal of leakage-prone variables

### 🧠 Feature Engineering
- Financial ratios (repayment_ratio, loan_pressure)
- Behavioral indicators (new vs repeat)
- Interaction features (new_large_loan)
- Temporal features (month, year)

### ⚡ Model
- **LightGBM Classifier**
- Optimized for tabular financial data

### 🔁 Validation Strategy
- Time-based cross-validation
- Prevents temporal leakage
- Simulates real-world deployment

### 🎯 Optimization
- Threshold tuning for F1 maximization
- Class imbalance handling (`scale_pos_weight`)

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Features | 17 |
| Best Threshold | 0.12 |
| OOF F1 Score | **0.7254** |
| CV Mean F1 | **0.8387** |

### Key Insights:
- High-quality feature selection outperformed complex ensembles  
- Financial + behavioral features dominate predictive power  
- Threshold tuning significantly improves real-world performance  
- Model demonstrates strong generalization across markets  

---

## 🧠 Explainability (SHAP)

To ensure transparency in risk-sensitive decisions, the system integrates **SHAP (SHapley Additive Explanations)**:

- 🌍 Global feature importance  
- 🔍 Individual prediction breakdown  
- 📌 Feature-level contribution analysis  

This enables:
- Regulatory compliance  
- Stakeholder trust  
- Model debugging & validation  

---

## 🚀 Deployment

### 🟢 Streamlit Cloud
- Public, interactive application  
- Instant access for stakeholders  

### 🐳 Dockerized Deployment

```bash
# Build image
docker build -t loan-default-risk-app .

# Run container
docker run -p 8501:8501 loan-default-risk-app
```
**Key Benefits:**

+ Environment consistency
+ Easy scalability
+ Production-ready packaging

---
## 📂 Project Structure

```
loan-default-prediction/
│
├── configs/                          # Configuration files
│   ├── config.yaml                   # Experiment configurations
│   └── config_final.yaml             # Final production config
│
├── data/                             # Data storage
│   ├── raw/                          # Original untouched data
│   └── processed/                    # Cleaned & feature-engineered data
│
├── docs/                             # Documentation
│   └── model_spec.md                 # Model design & specifications
│
├── models/                           # Trained model artifacts
│   ├── final_model.pkl
│   ├── final_lgb_model.pkl
│   ├── features.pkl
│   ├── final_features.pkl
│   ├── final_features_oof_v1.pkl
│   ├── final_features_advanced_v1.pkl
│   ├── threshold.json
│   └── best_threshold.json
│
├── notebooks/                        # Research & experimentation notebooks
│   ├── 00_data_validation.ipynb
│   ├── 01_advanced_eda.ipynb
│   ├── 02_pipeline_validation.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_advanced_modelling.ipynb
│   ├── 05_leakage_audit.ipynb
│   ├── 06_validation_strategy_upgrade.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_threshold_optimization.ipynb
│   ├── 09_model_interpretation.ipynb
│   ├── 10_domain_generalization.ipynb
│   ├── 11_temporal_feature_engineering.ipynb
│   ├── 12_economic_integration.ipynb
│   ├── 13_oof_model_training.ipynb
│   ├── 14_model_ensembling.ipynb
│   ├── 15_advanced_model_training.ipynb
│   ├── 16_true_stacking_v2.ipynb
│   └── 17_final_model_training.ipynb
│
├── outputs/                          # Model outputs & predictions
│   ├── final/
│   ├── oof/
│   ├── submissions/
│   ├── test/
│   └── submission.csv
│
├── reports/                          # Reports & analysis artifacts
│
├── src/                              # Core ML pipeline (modularized)
│   ├── data/
│   │   ├── data_loader.py            # Data loading & validation
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── build_features.py         # Feature construction
│   │   └── feature_engineering.py
│   │
│   ├── modeling/
│   │   ├── modeling.py               # Training logic
│   │   ├── train.py                  # Model training entry
│   │   └── predict.py                # Batch prediction logic
│   │
│   ├── preprocessing/
│   │   ├── preprocessing.py          # Data cleaning pipeline
│   │   ├── encode.py                 # Encoding logic
│   │   └── pipeline.py               # End-to-end preprocessing pipeline
│
├── streamlit_app/                    # Streamlit frontend application
│   ├── app.py                        # Main app entry point
│   │
│   ├── pages/                        # Multi-page UI
│   │   ├── 1_Home.py
│   │   ├── 2_Single_Predict.py
│   │   ├── 3_Batch_Predict.py
│   │   ├── 4_Model_Insights.py
│   │   └── 5_About.py
│   │
│   ├── utils/                        # App-specific utilities
│   │   ├── inference.py
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   │
│   └── assets/                       # UI assets (images, dashboards)
│       ├── dashboard.png
│       ├── batch_dashboard.png
│       ├── shap_dashboard.png
│       └── single_prediction.png
│
├── visuals/                          # Project visuals (EDA & results)
│
├── Dockerfile                        # Containerization setup
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore
├── train.py                          # Training entry script
└── predict.py                        # Inference entry script
```
---

## 🛠️ Tech Stack

+ Python
+ LightGBM
+ SHAP
+ Streamlit
+ Pandas / NumPy
+ Docker
+ YAML Configs

---
## 🧪 Run Locally

```bash
git clone https://github.com/theerealhenry/Loan-Default-Prediction-Project
cd Loan-Default-Prediction-Project

pip install -r requirements.txt

streamlit run streamlit_app/app.py
```
---
## 📌 Future Improvements

+ 🌐 API deployment (FastAPI)
+ ☁️ Cloud infrastructure (AWS / GCP)
+ 🔄 Automated retraining pipelines
+ 📡 Real-time data streaming
+ 🧠 Advanced ensemble models

---
## 👤 Author

### Henry Otsyula  
**Data Scientist | Machine Learning Engineer**

Building production-ready machine learning systems with a focus on **scalability, interpretability, and real-world impact**.

![GitHub](https://img.shields.io/badge/GitHub-theerealhenry-black?logo=github) 

![LinkedIn](https://img.shields.io/badge/LinkedIn-Henry%20Otsyula-blue?logo=linkedin)

📧 Email: henryotsyula01@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/henry-otsyula-datascientist  
🌐 Portfolio: https://www.datascienceportfol.io/otsyulahenry  

📍 Kenya (Open to remote opportunities)

---

💬 *Open to collaborations, internships, and full-time, remote opportunities in Data Science and Machine Learning.*

---

## 📜 License

This project is licensed under the MIT License.

---
## 🚀 Final Note

This project demonstrates the design and deployment of a production-grade machine learning system that combines:

+ Predictive modeling
+ Explainable AI
+ Scalable architecture
+ Business-oriented decision support

It reflects a strong focus on real-world applicability, robustness, and engineering excellence in financial risk modeling.
