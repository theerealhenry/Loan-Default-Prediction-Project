# src/config.py

import os
import random
import numpy as np
import yaml


# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ======================
# REPRODUCIBILITY
# ======================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ======================
# YAML CONFIG LOADER
# ======================
def load_config(config_path=os.path.join(BASE_DIR, "configs", "config.yaml")):
    """
    Load YAML configuration file.

    Returns:
        dict: configuration dictionary
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config