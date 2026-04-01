# src/feature_engineering.py

import pandas as pd
import numpy as np


# ======================
# 1. FINANCIAL FEATURES
# ======================
def create_financial_features(df):
    df = df.copy()

    df["log_amount"] = np.log1p(df["Total_Amount"])

    df["repayment_ratio"] = (
        df["Total_Amount_to_Repay"] / (df["Total_Amount"] + 1)
    )

    df["loan_pressure"] = (
        df["Total_Amount"] / (df["duration"] + 1)
    )

    return df


# ======================
# 2. TEMPORAL FEATURES
# ======================
def create_time_features(df):
    df = df.copy()

    df["year"] = df["disbursement_date"].dt.year
    df["month"] = df["disbursement_date"].dt.month

    return df


# ======================
# 3. CUSTOMER FEATURES (LEAKAGE-SAFE ✅)
# ======================
def create_customer_features(df):
    df = df.copy()

    # 🔥 CRITICAL: sort by customer + time
    df = df.sort_values(["customer_id", "disbursement_date"])

    # Number of previous loans
    df["loan_count"] = df.groupby("customer_id").cumcount()

    # Past defaults ONLY (no future leakage)
    df["past_defaults"] = (
        df.groupby("customer_id")["target"]
        .cumsum()
        .shift()
    )

    df["past_defaults"] = df["past_defaults"].fillna(0)

    # Total past loans
    df["total_loans"] = df["loan_count"]

    # 🔥 SAFE default rate (no leakage)
    df["safe_default_rate"] = (
        df["past_defaults"] /
        df["loan_count"].replace(0, np.nan)
    )

    df["safe_default_rate"] = df["safe_default_rate"].fillna(0)

    return df


# ======================
# 4. CATEGORICAL FEATURES
# ======================
def create_categorical_features(df):
    df = df.copy()

    df["is_new_customer"] = (
        df["New_versus_Repeat"] == "New Loan"
    ).astype(int)

    return df


# ======================
# 5. INTERACTION FEATURES
# ======================
def create_interactions(df):
    df = df.copy()

    df["new_large_loan"] = (
        (df["New_versus_Repeat"] == "New Loan") &
        (df["Total_Amount"] > df["Total_Amount"].median())
    ).astype(int)

    return df


# ======================
# 6. FULL FEATURE PIPELINE
# ======================
def feature_engineering(df, is_train: bool = True):
    df = df.copy()

    df = create_financial_features(df)
    df = create_time_features(df)
    df = create_categorical_features(df)
    df = create_interactions(df)

    # Apply ONLY for training data
    if is_train and "target" in df.columns:
        df = create_customer_features(df)

    return df