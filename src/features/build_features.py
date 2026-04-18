# =========================================================
# FEATURE BUILDING PIPELINE (PRODUCTION VERSION)
# =========================================================

import pandas as pd
import numpy as np


# =========================================================
# FINANCIAL FEATURES
# =========================================================

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


# =========================================================
# TEMPORAL FEATURES
# =========================================================

def create_time_features(df):
    df = df.copy()

    if "disbursement_date" in df.columns:
        df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])

        df["year"] = df["disbursement_date"].dt.year
        df["month"] = df["disbursement_date"].dt.month

    return df


# =========================================================
# CUSTOMER FEATURES (DISABLED IN PRODUCTION)
# =========================================================

def create_customer_features(df):
    """
    Disabled in production due to leakage risk.
    Kept for experimentation reference only.
    """
    return df


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

def create_categorical_features(df):
    df = df.copy()

    df["is_new_customer"] = (
        df["New_versus_Repeat"] == "New Loan"
    ).astype(int)

    return df


# =========================================================
# INTERACTION FEATURES
# =========================================================

def create_interactions(df):
    df = df.copy()

    df["new_large_loan"] = (
        (df["New_versus_Repeat"] == "New Loan") &
        (df["Total_Amount"] > df["Total_Amount"].median())
    ).astype(int)

    return df


# =========================================================
# FEATURE SELECTION (CONFIG-DRIVEN)
# =========================================================

def select_features(df, config):
    df = df.copy()

    drop_cols = config["features"]["drop_cols"]
    remove_low_signal = config["features"]["remove_low_signal"]
    macro_keywords = config["features"]["remove_macro_keywords"]

    # Drop known columns
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Remove macro features by keyword
    df = df[[c for c in df.columns if not any(k in c for k in macro_keywords)]]

    # Remove low-signal features
    df = df.drop(columns=[col for col in remove_low_signal if col in df.columns], errors="ignore")

    return df


# =========================================================
# FINAL FEATURE ALIGNMENT
# =========================================================

def align_features(train_df, test_df, config):
    """
    Ensure train/test have identical feature columns
    """

    final_features = config["features"]["final_features"]

    X_train = train_df[final_features].copy()
    X_test = test_df.reindex(columns=final_features, fill_value=0)

    return X_train, X_test


# =========================================================
# FULL PIPELINE
# =========================================================

def build_features(train_df, test_df, config):
    """
    Full feature pipeline (production-safe)
    """

    # -------------------------
    # Feature creation
    # -------------------------
    train_df = create_financial_features(train_df)
    test_df = create_financial_features(test_df)

    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)

    train_df = create_categorical_features(train_df)
    test_df = create_categorical_features(test_df)

    train_df = create_interactions(train_df)
    test_df = create_interactions(test_df)

    # -------------------------
    # Feature selection
    # -------------------------
    train_df = select_features(train_df, config)
    test_df = select_features(test_df, config)

    # -------------------------
    # Final alignment
    # -------------------------
    X_train, X_test = align_features(train_df, test_df, config)

    return X_train, X_test


# =========================================================
# DEBUG ENTRY POINT
# =========================================================

if __name__ == "__main__":
    from src.data.load_data import load_and_validate

    train, test, config = load_and_validate()

    X_train, X_test = build_features(train, test, config)

    print("\n✅ Feature pipeline complete")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)