# =========================================================
# DATA LOADING MODULE (PRODUCTION VERSION)
# =========================================================

import pandas as pd
from pathlib import Path
import yaml


# =========================================================
# CONFIG LOADER
# =========================================================

def load_config(config_path="D:/AI4EAC- Loan_default_prediction/configs/config_final.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================================================
# DATA LOADING
# =========================================================

def load_processed_data(config):
    """
    Load processed train and test datasets.

    Returns:
        train (pd.DataFrame)
        test (pd.DataFrame)
    """

    train_path = Path(config["paths"]["train_data"])
    test_path = Path(config["paths"]["test_data"])

    # Safety checks
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load data
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    return train, test


# =========================================================
# DATA VALIDATION 
# =========================================================

def validate_dataframe(df, name="dataset"):
    """
    Perform strict validation checks.
    """

    print(f"\n🔍 Validating {name}...")

    # Shape
    print(f"Shape: {df.shape}")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Dtypes summary
    print("\nDtype distribution:")
    print(df.dtypes.value_counts())

    # Hard checks (fail early)
    if df.shape[0] == 0:
        raise ValueError(f"{name} is empty!")

    if duplicates > 0:
        print(f"⚠️ Warning: {duplicates} duplicate rows detected in {name}")

    return True


# =========================================================
# FULL PIPELINE LOADER
# =========================================================

def load_and_validate(config_path="configs/config_final.yaml"):
    """
    Load + validate datasets using config.
    """

    config = load_config(config_path)

    print("📦 Loading datasets...")
    train, test = load_processed_data(config)

    # Validate
    validate_dataframe(train, "Train")
    validate_dataframe(test, "Test")

    print("\n✅ Data loading complete.")

    return train, test, config


# =========================================================
# DEBUG ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train, test, config = load_and_validate()

    print("\n📊 Train preview:")
    print(train.head())

    print("\n📊 Test preview:")
    print(test.head())