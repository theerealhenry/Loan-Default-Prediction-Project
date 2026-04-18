import sys, os
sys.path.append(os.path.abspath(".."))

from src.preprocessing.preprocessing import preprocess
from src.features.feature_engineering import feature_engineering


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

    return train, test