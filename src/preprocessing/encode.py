# =========================================================
# 🔄 ENCODING MODULE (PRODUCTION VERSION)
# =========================================================

import pandas as pd


# =========================================================
# 🔹 IDENTIFY CATEGORICAL COLUMNS
# =========================================================

def get_categorical_columns(df):
    """
    Identify categorical columns automatically.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols


# =========================================================
# 🔹 CONSISTENT CATEGORY ENCODING
# =========================================================

def encode_categoricals(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

    cat_cols = get_categorical_columns(train_df)

    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        combined = combined.astype("category")
        codes = combined.cat.codes

        train_df[col] = codes.iloc[:len(train_df)].values
        test_df[col] = codes.iloc[len(train_df):].values

    # =========================
    # 🔥 FORCE NUMERIC (CRITICAL FIX)
    # =========================

    # Convert bool → int
    for col in train_df.select_dtypes(include=["bool"]).columns:
        train_df[col] = train_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    # Convert everything to numeric
    train_df = train_df.apply(pd.to_numeric, errors="coerce")
    test_df = test_df.apply(pd.to_numeric, errors="coerce")

    # Fill missing
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    return train_df, test_df


# =========================================================
# 🔹 SAFETY CHECKS
# =========================================================

def validate_encoding(train_df, test_df):
    """
    Ensure encoding consistency
    """

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and test feature mismatch after encoding!")

    if train_df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Missing values in training data after encoding")

    if test_df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Missing values in test data after encoding")

    return True


# =========================================================
# 🔹 FULL ENCODING PIPELINE
# =========================================================

def run_encoding(train_df, test_df):
    """
    Full encoding pipeline
    """

    print("🔄 Encoding categorical features...")

    train_encoded, test_encoded = encode_categoricals(train_df, test_df)

    validate_encoding(train_encoded, test_encoded)

    print("✅ Encoding complete.")

    return train_encoded, test_encoded


# =========================================================
# 🧪 DEBUG ENTRY POINT
# =========================================================

if __name__ == "__main__":
    from src.data.load_data import load_and_validate
    from src.features.build_features import build_features

    train, test, config = load_and_validate()

    X_train, X_test = build_features(train, test, config)

    X_train_enc, X_test_enc = run_encoding(X_train, X_test)

    print("\nEncoded shapes:")
    print(X_train_enc.shape, X_test_enc.shape)