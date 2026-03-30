# src/customer_features.py

import pandas as pd


# ======================
# 1. BUILD CUSTOMER HISTORY FROM TRAIN
# ======================
def build_customer_history(train):
    """
    Builds customer-level historical aggregates from TRAIN ONLY.
    """

    df = train.copy()

    customer_history = df.groupby("customer_id").agg(
        total_loans=("target", "count"),
        total_defaults=("target", "sum"),
        default_rate=("target", "mean")
    ).reset_index()

    return customer_history


# ======================
# 2. APPLY HISTORY TO ANY DATASET
# ======================
def apply_customer_history(df, customer_history):
    """
    Maps historical features into a dataset (train/test)
    """

    df = df.copy()

    df = df.merge(customer_history, on="customer_id", how="left")

    # Fill missing (new customers in test)
    df["total_loans"] = df["total_loans"].fillna(0)
    df["total_defaults"] = df["total_defaults"].fillna(0)
    df["default_rate"] = df["default_rate"].fillna(0)

    return df