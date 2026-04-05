import numpy as np
from sklearn.metrics import f1_score


def get_time_splits(df, n_splits=5):
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


def run_time_cv(model, X, y, df):
    splits = get_time_splits(df)

    scores = []

    for i, (train_idx, val_idx) in enumerate(splits):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        score = f1_score(y_val, preds)

        scores.append(score)

        print(f"Fold {i+1}: F1 = {score:.4f}")

    return np.mean(scores)