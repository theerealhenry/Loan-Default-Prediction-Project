import pandas as pd
import os
from src.config import RAW_DATA_DIR


def load_data():
    train_path = os.path.join(RAW_DATA_DIR, "train.csv")
    test_path = os.path.join(RAW_DATA_DIR, "test.csv")
    econ_path = os.path.join(RAW_DATA_DIR, "economic_indicators.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    econ = pd.read_csv(econ_path)

    return train, test, econ


def validate_data(df, name="dataset"):
    print(f"\n🔍 Validating {name}...")

    print("\nShape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nPreview:")
    print(df.head())


def load_and_validate():
    train, test, econ = load_data()

    validate_data(train, "Train")
    validate_data(test, "Test")
    validate_data(econ, "Economic")

    return train, test, econ