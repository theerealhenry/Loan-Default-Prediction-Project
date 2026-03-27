# Loan Default Prediction

## 📌 Problem
Predict the likelihood of a customer defaulting on a loan using financial and macroeconomic data.

## 🎯 Objective
Build a robust and generalizable model that performs well across different countries (Kenya & Ghana).

## 🧠 Approach
- Structured feature engineering (customer, loan, lender)
- Economic data integration
- Cross-validation with leakage prevention
- Gradient boosting models (LightGBM, XGBoost, CatBoost)

## 📊 Evaluation Metric
F1 Score

## 🚀 Key Features
- Multi-level aggregation (customer, loan, lender)
- Domain generalization strategy
- Threshold optimization for F1

## 📁 Project Structure
```
loan-default-prediction/
│
├── data/
│   ├── raw/                  # Original untouched data
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final clean datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_predictions.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py            # Global config (paths, seeds)
│   ├── data_loader.py       # Data loading + validation
│   ├── preprocessing.py     # Cleaning + transformations
│   ├── feature_engineering.py
│   ├── modeling.py          # Training + evaluation
│   ├── inference.py         # Prediction pipeline
│   └── utils.py             # Helper functions
│
├── configs/
│   └── config.yaml          # Experiment configs
│
├── models/
│   └── (saved models here)
│
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── predictions/
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py (optional later)
```

## 📈 Results
(To be filled later)

## 🧠 Learnings
(To be filled later)