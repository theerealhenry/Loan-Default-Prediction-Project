import os
import random
import numpy as np

# ======================
# Paths
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ======================
# Reproducibility
# ======================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)