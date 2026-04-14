def get_time_splits(df, n_splits=5):
    """
    Create time-based splits (forward chaining)

    Ensures:
    - No future leakage
    - Increasing training window
    """

    df = df.sort_values("disbursement_date").reset_index(drop=True)

    fold_size = len(df) // n_splits
    splits = []

    for i in range(1, n_splits):
        train_end = i * fold_size
        val_end = (i + 1) * fold_size

        train_idx = list(range(0, train_end))
        val_idx = list(range(train_end, val_end))

        splits.append((train_idx, val_idx))

    return splits

def run_time_cv_oof(
    model,
    X,
    y,
    df,
    X_test,
    n_splits=5,
    model_name="model",
    early_stopping_rounds=50,
    verbose=100,
    use_proba=True
):
    """
    Advanced Time-Based CV with OOF + Test Predictions

    Parameters:
    ----------
    model : ML model (LGBM, XGB, CatBoost, sklearn)
    X : pd.DataFrame
    y : pd.Series
    df : original dataframe (for time splits)
    X_test : pd.DataFrame
    n_splits : int
    model_name : str (for logging/saving)
    early_stopping_rounds : int
    verbose : int
    use_proba : bool (True for classification probs)

    Returns:
    -------
    oof_preds : np.array
    test_preds : np.array
    scores : list
    models : list (trained models per fold)
    """

    import numpy as np
    from sklearn.metrics import f1_score
    from copy import deepcopy

    splits = get_time_splits(df, n_splits)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    models = []

    print(f"\n🚀 Running Time-Based CV for: {model_name}")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(splits):

        print(f"\n🔹 Fold {fold+1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone model to avoid leakage across folds
        model_fold = deepcopy(model)

        # =========================
        # MODEL TRAINING
        # =========================

        if "lgb" in str(type(model)).lower():
            model_fold.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[
                    # Early stopping + silent logging
                    __import__("lightgbm").early_stopping(early_stopping_rounds),
                    __import__("lightgbm").log_evaluation(0)
                ]
            )

        elif "xgb" in str(type(model)).lower():
            model_fold.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        elif "catboost" in str(type(model)).lower():
            model_fold.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
                early_stopping_rounds=early_stopping_rounds
            )

        else:
            # sklearn models
            model_fold.fit(X_train, y_train)

        # =========================
        # PREDICTIONS
        # =========================

        if use_proba and hasattr(model_fold, "predict_proba"):
            val_preds = model_fold.predict_proba(X_val)[:, 1]
            test_fold_preds = model_fold.predict_proba(X_test)[:, 1]
        else:
            val_preds = model_fold.predict(X_val)
            test_fold_preds = model_fold.predict(X_test)

        oof_preds[val_idx] = val_preds
        test_preds += test_fold_preds / n_splits

        # =========================
        # F1 EVALUATION (TEMP)
        # =========================

        val_binary = (val_preds > 0.5).astype(int)
        score = f1_score(y_val, val_binary)
        scores.append(score)

        print(f"F1 Score: {score:.5f}")

        models.append(model_fold)

    print("\n" + "=" * 50)
    print(f"📊 CV Mean F1 ({model_name}): {np.mean(scores):.5f}")
    print(f"📉 Std: {np.std(scores):.5f}")

    return oof_preds, test_preds, scores, models