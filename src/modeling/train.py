# =========================================================
# 🧠 MODEL TRAINING PIPELINE (PRODUCTION READY)
# =========================================================

import numpy as np
import pandas as pd
import joblib
import json
import os

from copy import deepcopy
from sklearn.metrics import f1_score

import lightgbm as lgb

from src.modeling.modeling import get_time_splits


# =========================================================
# 🔹 THRESHOLD OPTIMIZATION
# =========================================================

def optimize_threshold(y_true, preds, start=0.05, end=0.5, step=0.01):
    best_t, best_score = 0, 0

    for t in np.arange(start, end, step):
        score = f1_score(y_true, (preds > t).astype(int))
        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


# =========================================================
# 🔹 TIME-BASED CV ENGINE
# =========================================================

def run_cv(model, X, y, df, X_test, config):
    splits = get_time_splits(df, config["validation"]["n_splits"])

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    models = []

    print("\n🚀 Starting Time-Based CV...")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(splits):

        print(f"\n🔹 Fold {fold+1}")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_fold = deepcopy(model)

        # =========================
        # TRAINING
        # =========================

        model_fold.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(config["training"]["early_stopping"]),
                lgb.log_evaluation(0)
            ]
        )

        # =========================
        # PREDICTIONS
        # =========================

        val_preds = model_fold.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds

        test_preds += model_fold.predict_proba(X_test)[:, 1] / len(splits)

        # =========================
        # TEMP METRIC (for monitoring only)
        # =========================

        score = f1_score(y_val, (val_preds > 0.5).astype(int))
        print(f"Fold F1 (0.5): {score:.5f}")
        scores.append(score)

        models.append(model_fold)

    print("\n📊 CV Mean (0.5 threshold):", np.mean(scores))

    return oof, test_preds, models


# =========================================================
# 🔹 FINAL TRAINING FUNCTION
# =========================================================

def train_model(X, y, df, X_test, config):
    """
    Main training pipeline
    """

    print("\n🔥 TRAINING FINAL MODEL PIPELINE")
    print("=" * 60)

    # =========================
    # BUILD MODEL FROM CONFIG
    # =========================
    params = config["model"]["parameters"]
    
    # Compute proper scale_pos_weight
    if config["model"]["use_scale_pos_weight"]:
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1

    model = lgb.LGBMClassifier(
    **params,
    scale_pos_weight=scale_pos_weight,
    random_state=42
    )
    """""
    model = lgb.LGBMClassifier(
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        num_leaves=config["model"]["num_leaves"],
        max_depth=config["model"]["max_depth"],
        min_child_samples=config["model"]["min_child_samples"],
        subsample=config["model"]["subsample"],
        colsample_bytree=config["model"]["colsample_bytree"],
        reg_alpha=config["model"]["reg_alpha"],
        reg_lambda=config["model"]["reg_lambda"],
        scale_pos_weight=config["model"]["scale_pos_weight"],
        random_state=config["model"]["random_state"],
        n_jobs=-1
    )
    """

    # =========================
    # CROSS VALIDATION
    # =========================

    oof, test_preds, models = run_cv(model, X, y, df, X_test, config)

    # =========================
    # OPTIMIZE THRESHOLD
    # =========================

    best_t, best_score = optimize_threshold(
        y,
        oof,
        config["threshold"]["search_min"],
        config["threshold"]["search_max"],
        config["threshold"]["search_step"]
    )

    print("\n🎯 Optimal Threshold:", best_t)
    print("🏆 Best OOF F1:", best_score)

    # =========================
    # TRAIN FINAL MODEL ON FULL DATA
    # =========================

    print("\n🚀 Training final model on full data...")

    model.fit(X, y)

    # =========================
    # SAVE ARTIFACTS
    # =========================

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/final_model.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")

    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(best_t)}, f)

    print("\n✅ Artifacts saved:")
    print(" - model.pkl")
    print(" - features.pkl")
    print(" - threshold.json")

    # =========================
    # RETURN RESULTS
    # =========================

    return {
        "model": model,
        "oof": oof,
        "test_preds": test_preds,
        "threshold": best_t,
        "f1": best_score
    }