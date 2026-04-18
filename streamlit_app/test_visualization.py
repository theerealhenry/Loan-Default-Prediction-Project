# =========================================================
# VISUALIZATION ENGINE TEST
# =========================================================

import sys
import os

# Get absolute path to project root (VERY IMPORTANT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt

from utils.inference import run_inference
from utils.visualization import plot_risk_distribution

print("\n🧪 STARTING VISUALIZATION TEST")
print("=" * 60)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet("data/processed/test_merged.parquet").head(100)

print("\n📂 Loaded data shape:", df.shape)


# =========================================================
# RUN INFERENCE
# =========================================================

results = run_inference(df)

probs = results["probabilities"]

print("\n🔮 Inference complete")
print("Sample probabilities:", probs[:5])


# =========================================================
# PLOT TEST
# =========================================================

print("\n📊 Generating risk distribution plot...")

fig = plot_risk_distribution(probs)

assert fig is not None, "❌ Plot function returned None"

plt.show()

print("✅ PASS: Plot rendered successfully")


# =========================================================
# EDGE CASE TEST (EMPTY INPUT)
# =========================================================

print("\n🔹 Testing empty input...")

try:
    fig_empty = plot_risk_distribution([])
    print("⚠️ No crash on empty input")
except Exception as e:
    print("❌ Failed on empty input:", e)


# =========================================================
# EDGE CASE TEST (CONSTANT VALUES)
# =========================================================

print("\n🔹 Testing constant probabilities...")

constant_probs = [0.5] * 50

fig_const = plot_risk_distribution(constant_probs)

plt.show(block=False)
plt.pause(2)
plt.close()

print("✅ PASS: Constant values handled")


print("\n🏁 VISUALIZATION TEST COMPLETE")
print("=" * 60)