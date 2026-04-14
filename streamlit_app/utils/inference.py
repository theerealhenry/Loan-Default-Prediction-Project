# =========================================================
# 🧠 INFERENCE SERVICE LAYER (PRODUCTION-GRADE)
# =========================================================

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from utils.preprocessing import preprocess_dataframe
from src.modeling.predict import predict, load_artifacts

# =========================================================
# 📦 GLOBAL CACHE (MODEL + CONFIG)
# =========================================================

_MODEL = None
_FEATURES = None
_THRESHOLD = None
_CONFIG = None


# =========================================================
# 🔧 LOAD CONFIG (CACHED)
# =========================================================

def load_config(config_path: str = "configs/config_final.yaml"):
    global _CONFIG

    if _CONFIG is None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)

    return _CONFIG


# =========================================================
# 📦 LOAD SYSTEM (MODEL + FEATURES + THRESHOLD)
# =========================================================

def load_system(model_dir: str = "models"):
    global _MODEL, _FEATURES, _THRESHOLD

    if _MODEL is None:
        _MODEL, _FEATURES, _THRESHOLD = load_artifacts(model_dir)

    return _MODEL, _FEATURES, _THRESHOLD


# =========================================================
# 🔮 CORE INFERENCE ENGINE
# =========================================================

def run_inference(
    df: pd.DataFrame,
    model_dir: str = "models",
    return_proba: bool = True
) -> Dict[str, Any]:

    model, features, threshold = load_system(model_dir)
    config = load_config()

    # =========================
    # PREDICT PROBABILITIES
    # =========================
    df = preprocess_dataframe(df)
    
    probs = predict(
        input_df=df,
        config=config,
        model_dir=model_dir,
        return_proba=True
    )

    # 🔥 FORCE CLEAN NUMPY ARRAY
    probs = np.asarray(probs).astype(float).flatten()

    # 🔒 SAFETY CLIP (avoid weird model outputs)
    probs = np.clip(probs, 0, 1)

    # =========================
    # CLASS PREDICTIONS
    # =========================

    preds = (probs > threshold).astype(int)
    preds = pd.Series(preds)

    # =========================
    # RISK SEGMENTATION
    # =========================

    risk_levels = pd.Series(
        pd.cut(
            probs,
            bins=[0, 0.4, 0.7, 1],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )
    )

    # =========================
    # SUMMARY METRICS
    # =========================

    n = max(len(df), 1)  # 🔒 avoid division by zero

    summary = {
        "total_samples": int(n),
        "avg_probability": float(np.mean(probs)),
        "high_risk_ratio": float(np.mean(preds)),
        "threshold": float(threshold)
    }

    return {
        "probabilities": probs,          # numpy array
        "predictions": preds,            # pandas Series
        "risk_levels": risk_levels,      # pandas Series
        "summary": summary
    }


# =========================================================
# 🔹 SINGLE RECORD INFERENCE
# =========================================================

def run_single_inference(
    input_dict: Dict[str, Any],
    model_dir: str = "models"
) -> Dict[str, Any]:

    df = pd.DataFrame([input_dict])
    results = run_inference(df, model_dir=model_dir)

    # 🔥 BULLETPROOF EXTRACTION
    prob = float(np.asarray(results["probabilities"])[0])
    pred = int(np.asarray(results["predictions"])[0])
    risk = str(np.asarray(results["risk_levels"])[0])

    return {
        "probability": prob,
        "prediction": pred,
        "risk_level": risk,
        "threshold": float(results["summary"]["threshold"])
    }


# =========================================================
# 📊 DATAFRAME ENRICHMENT (BATCH MODE)
# =========================================================

def enrich_predictions(
    df: pd.DataFrame,
    model_dir: str = "models"
) -> pd.DataFrame:

    results = run_inference(df, model_dir=model_dir)

    df_out = df.copy()

    df_out["Default_Probability"] = results["probabilities"]
    df_out["Prediction"] = results["predictions"].values
    df_out["Risk_Level"] = results["risk_levels"].values

    return df_out


# =========================================================
# 📈 PORTFOLIO ANALYTICS
# =========================================================

def compute_portfolio_metrics(df: pd.DataFrame) -> Dict[str, Any]:

    n = max(len(df), 1)

    metrics = {
        "total_loans": int(n),
        "avg_risk": float(df["Default_Probability"].mean()),
        "high_risk_count": int((df["Risk_Level"] == "High").sum()),
        "medium_risk_count": int((df["Risk_Level"] == "Medium").sum()),
        "low_risk_count": int((df["Risk_Level"] == "Low").sum()),
    }

    metrics["high_risk_ratio"] = float(metrics["high_risk_count"] / n)

    return metrics


# =========================================================
# 🧪 DEBUG TOOLING
# =========================================================

def debug_inference(df: pd.DataFrame, model_dir="models"):

    print("\n🧪 DEBUG INFERENCE")
    print("=" * 50)

    model, features, threshold = load_system(model_dir)
    config = load_config()

    print("Loaded Features:", len(features))
    print("Threshold:", threshold)
    print("Input shape:", df.shape)

    probs = predict(
        input_df=df,
        config=config,
        model_dir=model_dir,
        return_proba=True
    )

    probs = np.asarray(probs).flatten()

    print("Sample probabilities:", probs[:5])

    return probs