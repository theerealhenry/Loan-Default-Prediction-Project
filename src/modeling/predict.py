# =========================================================
# PREDICTION PIPELINE (PRODUCTION READY)
# =========================================================

import pandas as pd
import numpy as np
import joblib
import json
import os

from src.features.build_features import build_features
from src.preprocessing.encode import encode_categoricals


# =========================================================
# LOAD ARTIFACTS
# =========================================================

def load_artifacts(model_dir="models"):
    print("Loading model artifacts...")

    model = joblib.load(os.path.join(model_dir, "final_model.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))

    with open(os.path.join(model_dir, "threshold.json"), "r") as f:
        threshold = json.load(f)["threshold"]

    print("✅ Artifacts loaded")

    return model, features, threshold


# =========================================================
# ALIGN FEATURES (CRITICAL)
# =========================================================

def align_features(df, features):
    """
    Ensures test data matches training features exactly
    """

    df = df.copy()

    # Add missing columns
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns
    df = df[features]

    return df


# =========================================================
# PREPROCESS + FEATURE PIPELINE
# =========================================================

def prepare_input(df, config):
    """
    Full preprocessing pipeline:
    - Feature engineering
    - Encoding
    - Alignment
    """

    df = df.copy()

    # Build features (use df as both train/test to reuse pipeline)
    X, _ = build_features(df, df, config)

    # Encode categoricals
    X, _ = encode_categoricals(X, X)

    return X


# =========================================================
# MAIN PREDICT FUNCTION
# =========================================================

def predict(input_df, config, model_dir="models", return_proba=False):
    """
    Main prediction function (used everywhere)

    Parameters:
    ----------
    input_df : pd.DataFrame
    model_dir : str
    return_proba : bool

    Returns:
    -------
    predictions or probabilities
    """

    print("\n🔮 Running prediction pipeline...")
    print("=" * 50)

    # =========================
    # LOAD ARTIFACTS
    # =========================

    model, features, threshold = load_artifacts(model_dir)

    # =========================
    # PREP INPUT
    # =========================

    df_processed = prepare_input(input_df, config)

    # =========================
    # ALIGN FEATURES
    # =========================

    X = align_features(df_processed, features)

    # =========================
    # PREDICT
    # =========================

    probs = model.predict_proba(X)[:, 1]

    if return_proba:
        return probs

    preds = (probs > threshold).astype(int)

    print("✅ Prediction complete")

    return preds


# =========================================================
# BATCH PREDICTION (FOR COMPETITIONS)
# =========================================================

def predict_to_submission(input_df, config, id_col="ID", model_dir="models"):
    """
    Creates submission-ready dataframe
    """

    preds = predict(input_df, config=config, model_dir=model_dir)

    submission = pd.DataFrame({
        "ID": input_df[id_col],
        "Target": preds
    })

    return submission


# =========================================================
# DEBUG MODE (VERY IMPORTANT)
# =========================================================

def debug_prediction(input_df, model_dir="models"):
    """
    Useful for debugging pipeline issues
    """

    print("\n🧪 DEBUG MODE")
    print("=" * 50)

    model, features, threshold = load_artifacts(model_dir)

    df_processed = prepare_input(input_df)

    print("\n🔍 After feature engineering:")
    print(df_processed.shape)

    X = align_features(df_processed, features)

    print("\n🔍 After alignment:")
    print(X.shape)

    missing = set(features) - set(X.columns)
    extra = set(X.columns) - set(features)

    print("\nMissing features:", missing)
    print("Extra features:", extra)

    probs = model.predict_proba(X)[:, 1]

    print("\nSample probabilities:", probs[:5])

    return probs