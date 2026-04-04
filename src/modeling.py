# src/modeling.py

import numpy as np
from sklearn.metrics import f1_score

def run_time_cv(model, X, y, df):
    df = df.sort_values("disbursement_date")

    folds = np.array_split(df.index, 5)

    scores = []

    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        val_idx = folds[i]

        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        score = f1_score(y_val, preds)

        scores.append(score)

        print(f"Fold {i}: F1 = {score:.4f}")

    return np.mean(scores)