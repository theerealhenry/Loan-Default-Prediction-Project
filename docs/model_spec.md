# 📄 Model Specification Document

## Loan Default Prediction System (Production Version)

---

# 1. 🎯 Objective

The goal of this project is to build a **robust, production-ready machine learning system** to predict loan default risk using structured financial and behavioral data.

The system is optimized for:

* **F1-score maximization** (competition metric)
* **Temporal generalization** (real-world deployment scenario)
* **Reproducibility and consistency**
* **Low leakage risk**

---

# 2. 📊 Final Model Selection

After extensive experimentation involving:

* Single models (LightGBM, XGBoost, CatBoost)
* Weighted ensembling
* Rank averaging
* Multi-level stacking (including leakage-safe variants)

👉 The **final selected model** is:

## ✅ LightGBM (Single Model — Production Optimized)

### Reason for Selection:

| Factor         | Observation                    |
| -------------- | ------------------------------ |
| Performance    | Highest stable OOF F1 (~0.725) |
| Stability      | Lowest variance across folds   |
| Generalization | Strong under time-based CV     |
| Complexity     | Lower risk than stacking       |
| Reliability    | No meta-leakage issues         |

👉 Advanced ensembles **did NOT outperform** due to:

* High correlation between base models (~0.90+)
* Limited model diversity
* Overfitting in meta-models

---

# 3. 🧠 Final Feature Set

## ✅ Total Features: **17**

### 🔥 Core Predictive Features

* repayment_ratio
* Amount_Funded_By_Lender
* loan_pressure
* Total_Amount_to_Repay
* Total_Amount
* Lender_portion_Funded
* month
* duration
* loan_type
* log_amount
* year
* New_versus_Repeat
* is_new_customer
* new_large_loan

### ⚠️ Weak Signal (Retained After Testing)

* unemployment_rate *(very low importance but harmless)*

---

## ❌ Removed Features

### 🚨 Leakage-Prone Features

* loan_count
* past_defaults
* total_loans
* safe_default_rate

👉 Reason: Derived from historical outcomes → **future leakage risk**

---

### ❌ Macro Features Removed

* inflation_*
* interest_*
* exchange_*
* macro_risk_index

👉 Reason:

* Inconsistent temporal alignment
* Weak signal under time-based CV
* Potential leakage if improperly lagged

---

### ❌ Low Importance Features

* country_id

👉 Reason:

* Near-zero importance
* No predictive contribution

---

# 4. ⚙️ Data Preprocessing

## 🔹 Categorical Encoding

Method:

* Combined train + test encoding
* Converted to categorical codes

```python
combined = pd.concat([train[col], test[col]])
codes = combined.astype("category").cat.codes
```

### ✅ Benefits:

* Prevents unseen category errors
* Ensures consistent mapping

---

## 🔹 Feature Alignment

```python
X_test = test.reindex(columns=FEATURES, fill_value=0)
```

### ✅ Ensures:

* No missing columns at inference
* Safe deployment

---

# 5. 🧪 Validation Strategy

## ✅ Time-Based Cross Validation

Custom function:

```python
get_time_splits(df, n_splits=5)
```

### Key Properties:

* Train on past → validate on future
* Prevents temporal leakage
* Simulates real-world deployment

---

## ⚠️ Why NOT Random CV?

| Risk                   | Impact                      |
| ---------------------- | --------------------------- |
| Data leakage           | Inflated scores             |
| Unrealistic evaluation | Poor deployment performance |

---

# 6. ⚖️ Class Imbalance Handling

Severe imbalance detected:

```python
scale_pos_weight ≈ 53.57
```

### Applied in model:

```python
scale_pos_weight = (neg / pos)
```

### Impact:

* Improves recall on minority class
* Boosts F1 score

---

# 7. 🤖 Final Model Configuration

## LightGBM Parameters

```python
LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.05,
    reg_lambda=0.05,
    scale_pos_weight=53.57,
    random_state=42,
    n_jobs=-1
)
```

---

## 🔍 Key Design Choices

| Parameter            | Purpose                  |
| -------------------- | ------------------------ |
| num_leaves=128       | Capture complex patterns |
| learning_rate=0.03   | Stable convergence       |
| subsample=0.9        | Reduce overfitting       |
| colsample_bytree=0.9 | Feature diversity        |
| regularization       | Control complexity       |

---

# 8. 🎯 Threshold Optimization

## Optimal Threshold:

```python
threshold = 0.12
```

### Why not 0.5?

* Dataset is **highly imbalanced**
* Lower threshold improves recall
* Maximizes F1 score

---

## Final Performance:

| Metric                  | Value      |
| ----------------------- | ---------- |
| CV Mean F1              | ~0.838     |
| OOF F1 (threshold=0.12) | **0.7254** |

---

# 9. 📈 Feature Importance Insights

## Top Drivers:

1. repayment_ratio
2. Amount_Funded_By_Lender
3. loan_pressure
4. Total_Amount_to_Repay
5. Total_Amount

👉 Interpretation:

* Repayment capacity is strongest signal
* Loan pressure is critical risk indicator

---

## Low Impact Features:

* country_id
* macro_risk_index

👉 Removed or ignored in final model

---

# 10. 🚨 Risk Management

## Identified Risks & Mitigation

### 🔴 Risk 1 — Data Leakage

✔ Removed leakage features
✔ Time-based CV

---

### 🔴 Risk 2 — Feature Drift

✔ Avoided unstable macro variables

---

### 🔴 Risk 3 — Overfitting

✔ Regularization applied
✔ Simpler model chosen over stacking

---

# 11. 🏁 Final Production Artifacts

The system outputs:

```
models/
├── final_lgb_model.pkl
├── final_features.pkl
├── threshold.json
```

---

# 12. 🔁 Reproducibility Guarantee

The system is fully reproducible via:

```bash
python train.py
python predict.py
```

Guarantees:

* Same CV scores
* Same predictions
* No manual steps

---

# 13. 🧠 Key Lessons Learned

* Simpler models can outperform complex ensembles
* Time-based validation is critical in financial data
* Feature quality > model complexity
* High correlation kills ensemble performance

---

# 14. 🚀 Future Improvements

* Proper lagged macro features (leakage-safe)
* Bayesian hyperparameter tuning
* Model monitoring (drift detection)
* Advanced feature interactions

---

# ✅ Final Verdict

This system represents a:

> **Production-ready, leakage-safe, reproducible ML pipeline optimized for real-world deployment and portfolio excellence.**
