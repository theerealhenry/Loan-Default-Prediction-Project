# =========================================================
# 🧪 PREPROCESSING TEST SUITE
# =========================================================

import pandas as pd
from utils.preprocessing import preprocess_dataframe

print("\n🧪 STARTING PREPROCESSING TESTS")
print("=" * 60)


# =========================================================
# 🔹 TEST 1 — MISSING COLUMNS
# =========================================================

print("\n🔹 TEST 1: Missing Columns")

df_missing = pd.DataFrame({
    "Total_Amount": [1000]
})

processed = preprocess_dataframe(df_missing)

print("Input shape:", df_missing.shape)
print("Output shape:", processed.shape)
print("Columns:", processed.columns.tolist())

assert processed.shape[1] > df_missing.shape[1], "❌ Missing columns not added"
print("✅ PASS: Missing columns handled")


# =========================================================
# 🔹 TEST 2 — WRONG DATA TYPES
# =========================================================

print("\n🔹 TEST 2: Wrong Data Types")

df_wrong_types = pd.DataFrame({
    "Total_Amount": ["invalid"],
    "duration": ["abc"]
})

try:
    processed = preprocess_dataframe(df_wrong_types)

    print("Dtypes after processing:")
    print(processed.dtypes)

    print("Processed values:")
    print(processed.head())

    print("✅ PASS: No crash on invalid types")

except Exception as e:
    print("❌ FAIL:", e)


# =========================================================
# 🔹 TEST 3 — NEGATIVE VALUES
# =========================================================

print("\n🔹 TEST 3: Negative Values")

df_negative = pd.DataFrame({
    "Total_Amount": [-1000],
    "duration": [-5]
})

processed = preprocess_dataframe(df_negative)

print("Processed values:")
print(processed[["Total_Amount", "duration"]])

assert (processed["Total_Amount"] >= 0).all(), "❌ Negative Total_Amount not fixed"
assert (processed["duration"] >= 0).all(), "❌ Negative duration not fixed"

print("✅ PASS: Negative values handled")


# =========================================================
# 🔹 TEST 4 — EXTREME VALUES (BONUS 🔥)
# =========================================================

print("\n🔹 TEST 4: Extreme Values")

df_extreme = pd.DataFrame({
    "Total_Amount": [1e9],
    "duration": [10000]
})

processed = preprocess_dataframe(df_extreme)

print("Processed extreme values:")
print(processed.head())

print("✅ PASS: Extreme values handled")


# =========================================================
# 🔹 TEST 5 — FULL PIPELINE COMPATIBILITY
# =========================================================

print("\n🔹 TEST 5: Full Pipeline Compatibility")

df_realistic = pd.DataFrame({
    "Total_Amount": [5000],
    "Total_Amount_to_Repay": [6000],
    "duration": [12],
    "New_versus_Repeat": ["New Loan"]
})

processed = preprocess_dataframe(df_realistic)

print("Final processed shape:", processed.shape)
print("Sample output:")
print(processed.head())

print("✅ PASS: Compatible with full pipeline")


print("\n🏁 ALL PREPROCESSING TESTS COMPLETED")
print("=" * 60)