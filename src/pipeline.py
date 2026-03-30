import sys, os
sys.path.append(os.path.abspath(".."))

from src.preprocessing import preprocess
from src.feature_engineering import feature_engineering
from src.customer_features import build_customer_history, apply_customer_history


def full_pipeline(train, test):
    """
    Full pipeline that handles:
    - preprocessing
    - feature engineering
    - customer history mapping
    """

    # Step 1: preprocess
    train = preprocess(train)
    test = preprocess(test)

    # Step 2: feature engineering (basic)
    train = feature_engineering(train, is_train=True)
    test = feature_engineering(test, is_train=False)

    # Step 3: build history from TRAIN ONLY
    customer_history = build_customer_history(train)

    # Step 4: apply to both
    train = apply_customer_history(train, customer_history)
    test = apply_customer_history(test, customer_history)

    return train, test