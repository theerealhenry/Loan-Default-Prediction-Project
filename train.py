# =========================================================
# 🚀 TRAINING CLI ENTRYPOINT (PRODUCTION READY)
# =========================================================

import argparse
import yaml
import pandas as pd
import numpy as np
import os

from src.data.load_data import load_processed_data
from src.features.build_features import build_features
from src.preprocessing.encode import encode_categoricals
from src.modeling.train import train_model


# =========================================================
# 🔹 LOAD CONFIG
# =========================================================

def load_config(config_path):
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n🚨 FULL CONFIG KEYS:")
    print(config.keys())

    print("\n🚨 TRAINING SECTION:")
    print(config.get("training"))

    return config


# =========================================================
# 🔹 FEATURE SELECTION
# =========================================================

def select_features(df, config):

    final_features = config["features"]["final_features"]

    print(f"\n📊 Final feature count: {len(final_features)}")

    return final_features


# =========================================================
# 🔹 MAIN TRAINING FUNCTION
# =========================================================

def run_training(config_path):

    print("\n🔥 STARTING TRAINING PIPELINE")
    print("=" * 60)

    # =========================
    # LOAD CONFIG
    # =========================

    config = load_config(config_path)

    # =========================
    # LOAD DATA
    # =========================

    train, test = load_processed_data(config)

    # =========================
    # FEATURE ENGINEERING
    # =========================

    X, X_test = build_features(train, test, config)

    # Target
    y = train["target"]

    # =========================
    # FEATURE SELECTION
    # =========================

    FEATURES = select_features(X, config)

    # Safety filter
    #FEATURES = [f for f in FEATURES if f in X.columns]

    # Use processed features (NOT raw train)
    X = X[FEATURES].copy()
    y = train[config["target"]].copy()

    # =========================
    # ALIGN TEST
    # =========================

    X_test = X_test[FEATURES].copy()

    # =========================
    # ENCODING
    # =========================

    X, X_test = encode_categoricals(X, X_test)

# =========================================================
# 🔥 FINAL HARD FIX — FORCE FULL NUMERIC DATA
# =========================================================

    # Convert categorical/object → numeric explicitly
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            X[col] = X[col].astype("category").cat.codes
            X_test[col] = X_test[col].astype("category").cat.codes

    # Convert booleans → int
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    # Final numeric enforcement
    X = X.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    # Fill missing
    X = X.fillna(0)
    X_test = X_test.fillna(0)

    # =========================
    # 🔍 FINAL DEBUG (MUST BE EMPTY)
    # =========================
    print("\n🚨 FINAL CHECK (ABSOLUTE):")
    print(X.dtypes.value_counts())

    non_numeric = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    print("Non-numeric columns:", non_numeric)

    # =========================
    # TRAIN MODEL
    # =========================

    results = train_model(
        X=X,
        y=y,
        df=train,
        X_test=X_test,
        config=config
    )

    print("\n🏆 FINAL RESULTS")
    print("=" * 40)
    print("F1 Score:", results["f1"])
    print("Threshold:", results["threshold"])

    print("\n✅ Training complete.")


# =========================================================
# 🔹 CLI ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Loan Default Model")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    run_training(args.config)