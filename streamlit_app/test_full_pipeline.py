# =========================================================
# FULL PIPELINE TEST (PREPROCESS + INFERENCE)
# =========================================================

import sys
import os

# Get absolute path to project root 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(PROJECT_ROOT)


import pandas as pd
from utils.preprocessing import preprocess_dataframe
from utils.inference import run_inference

print("\n🧪 STARTING FULL PIPELINE TEST")
print("=" * 60)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet("data/processed/test_merged.parquet").head(50)

print("\n📂 Raw input shape:", df.shape)


# =========================================================
# PREPROCESSING
# =========================================================

df_clean = preprocess_dataframe(df)

print("\n🧹 After preprocessing:")
print("Shape:", df_clean.shape)
print("Columns:", df_clean.columns.tolist())


# =========================================================
# INFERENCE
# =========================================================

results = run_inference(df_clean)

print("\n🔮 Inference Results:")
print("Summary:", results["summary"])

print("\nSample Predictions:")
print(results["predictions"][:10])

print("\nSample Probabilities:")
print(results["probabilities"][:10])

print("\nSample Risk Levels:")
print(results["risk_levels"][:10])


# =========================================================
# VALIDATION CHECKS 
# =========================================================

assert len(results["predictions"]) == len(df_clean), "❌ Prediction length mismatch"
assert len(results["probabilities"]) == len(df_clean), "❌ Probability length mismatch"

assert results["summary"]["total_samples"] == len(df_clean), "❌ Summary mismatch"

print("\n✅ PASS: Output sizes correct")


# =========================================================
# EDGE CHECK — NO NaNs
# =========================================================

import numpy as np

assert not np.isnan(results["probabilities"]).any(), "❌ NaNs in probabilities"

print("✅ PASS: No NaNs in probabilities")


print("\n🏁 FULL PIPELINE TEST PASSED")
print("=" * 60)