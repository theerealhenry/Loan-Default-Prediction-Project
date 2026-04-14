import sys
import os

# ✅ Get absolute path to project root (VERY IMPORTANT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(PROJECT_ROOT)

import pandas as pd
from utils.inference import run_inference

df = pd.read_parquet(r"D:/AI4EAC- Loan_default_prediction/data/processed/test_merged.parquet").head(100)

results = run_inference(df)

print("✅ Probabilities:", results["probabilities"][:5])
print("✅ Predictions:", results["predictions"][:5])
print("✅ Risk Levels:", results["risk_levels"][:5])
print("✅ Summary:", results["summary"])