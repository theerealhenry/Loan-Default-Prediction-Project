# src/preprocessing.py

import pandas as pd
import numpy as np


# ======================
# 1. BASIC CLEANING
# ======================
def clean_data(df):
    df = df.copy()

    # Convert dates
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])

    # Categorical casting
    categorical_cols = ["country_id", "loan_type", "New_versus_Repeat"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


# ======================
# 2. REMOVE LEAKAGE
# ======================
def remove_leakage(df):
    df = df.copy()

    leakage_cols = [
        "Lender_portion_to_be_repaid"
    ]

    existing_cols = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=existing_cols)

    return df


# ======================
# 3. RARE CATEGORY HANDLING
# ======================
def group_rare_categories(df, col, threshold=100):
    df = df.copy()

    counts = df[col].value_counts()
    rare = counts[counts < threshold].index

    df[col] = df[col].apply(lambda x: "rare" if x in rare else x)

    return df


# ======================
# 4. APPLY ALL PREPROCESSING
# ======================
def preprocess(df):
    df = clean_data(df)
    df = remove_leakage(df)

    # Group rare loan types
    if "loan_type" in df.columns:
        df = group_rare_categories(df, "loan_type", threshold=100)

    return df