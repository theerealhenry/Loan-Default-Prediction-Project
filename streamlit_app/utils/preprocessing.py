# =========================================================
# 🧹 INPUT PREPROCESSING LAYER (PRODUCTION-GRADE)
# =========================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List


# =========================================================
# 🔹 EXPECTED RAW INPUT SCHEMA
# =========================================================

EXPECTED_COLUMNS = [
    "Total_Amount",
    "Total_Amount_to_Repay",
    "duration",
    "disbursement_date",
    "New_versus_Repeat",
    "loan_type"
]


# =========================================================
# 🔹 DEFAULT VALUES (SAFE FALLBACKS)
# =========================================================

DEFAULT_VALUES = {
    "Total_Amount": 0,
    "Total_Amount_to_Repay": 0,
    "duration": 1,
    "disbursement_date": "2023-01-01",
    "New_versus_Repeat": "Repeat Loan",
    "loan_type": "Unknown"
}


# =========================================================
# 🔹 TYPE CASTING MAP
# =========================================================

TYPE_CASTS = {
    "Total_Amount": float,
    "Total_Amount_to_Repay": float,
    "duration": int,
    "New_versus_Repeat": str,
    "loan_type": str
}


# =========================================================
# 🔹 VALIDATION
# =========================================================

def validate_columns(df: pd.DataFrame) -> List[str]:
    """
    Checks for missing required columns.
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    return missing


# =========================================================
# 🔹 FILL MISSING COLUMNS
# =========================================================

def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing columns with default values.
    """
    df = df.copy()

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULT_VALUES[col]

    return df


# =========================================================
# 🔹 CLEAN TYPES
# =========================================================

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces correct data types.
    """
    df = df.copy()

    for col, dtype in TYPE_CASTS.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except:
                df[col] = DEFAULT_VALUES[col]

    return df


# =========================================================
# 🔹 HANDLE DATES
# =========================================================

def process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts date columns to proper datetime format.
    """
    df = df.copy()

    if "disbursement_date" in df.columns:
        df["disbursement_date"] = pd.to_datetime(
            df["disbursement_date"],
            errors="coerce"
        )

        df["disbursement_date"] = df["disbursement_date"].fillna(
            pd.to_datetime(DEFAULT_VALUES["disbursement_date"])
        )

    return df


# =========================================================
# 🔹 HANDLE NUMERIC ANOMALIES
# =========================================================

def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix negative or extreme values.
    """
    df = df.copy()

    numeric_cols = ["Total_Amount", "Total_Amount_to_Repay", "duration"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


# =========================================================
# 🔹 FULL PIPELINE (DATAFRAME)
# =========================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for batch inputs.
    """

    df = df.copy()

    # Step 1: Ensure required columns exist
    df = add_missing_columns(df)

    # Step 2: Type enforcement
    df = enforce_types(df)

    # Step 3: Date processing
    df = process_dates(df)

    # Step 4: Numeric cleaning
    df = clean_numeric_values(df)

    return df


# =========================================================
# 🔹 SINGLE INPUT HANDLER (STREAMLIT FORM)
# =========================================================

def preprocess_single_input(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts user input dict → clean dataframe.
    """

    df = pd.DataFrame([input_dict])

    df = preprocess_dataframe(df)

    return df


# =========================================================
# 🔹 SAFE PIPELINE WRAPPER
# =========================================================

def safe_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust preprocessing with error handling.
    """

    try:
        return preprocess_dataframe(df)

    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")

        # fallback: return minimal safe frame
        fallback = pd.DataFrame([DEFAULT_VALUES])
        return fallback


# =========================================================
# 🧪 DEBUG TOOL
# =========================================================

def debug_preprocessing(df: pd.DataFrame):
    """
    Debug preprocessing behavior.
    """

    print("\n🧪 DEBUG PREPROCESSING")
    print("=" * 50)

    print("Original shape:", df.shape)
    print("Missing columns:", validate_columns(df))

    processed = preprocess_dataframe(df)

    print("Processed shape:", processed.shape)
    print("Dtypes:\n", processed.dtypes)

    return processed