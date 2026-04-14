# =========================================================
# 📊 VISUALIZATION ENGINE (PRODUCTION-GRADE)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 🎨 GLOBAL STYLE CONFIG
# =========================================================

plt.style.use("default")


# =========================================================
# 🔹 RISK DISTRIBUTION
# =========================================================

def plot_risk_distribution(probs):
    """
    Clean histogram of predicted probabilities.
    Handles dirty inputs robustly.
    """

    # =========================
    # 🔧 CLEAN INPUT (CRITICAL FIX)
    # =========================
    probs = pd.Series(probs).astype(float).values.flatten()

    # =========================
    # 🎨 PLOT
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(probs, bins=20)
    ax.set_title("Risk Probability Distribution")
    ax.set_xlabel("Default Probability")
    ax.set_ylabel("Frequency")

    # =========================
    # 🎯 UX IMPROVEMENTS
    # =========================
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


# =========================================================
# 🔹 RISK SEGMENT BAR CHART
# =========================================================

def plot_risk_segments(risk_levels):
    """
    Count of Low / Medium / High risk.
    """

    counts = pd.Series(risk_levels).value_counts().sort_index()

    fig, ax = plt.subplots()

    counts.plot(kind="bar", ax=ax)

    ax.set_title("Risk Segment Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")

    return fig


# =========================================================
# 🔹 PORTFOLIO PIE CHART
# =========================================================

def plot_portfolio_mix(risk_levels):
    """
    Portfolio composition pie chart.
    """

    counts = pd.Series(risk_levels).value_counts()

    fig, ax = plt.subplots()

    ax.pie(counts, labels=counts.index, autopct="%1.1f%%")

    ax.set_title("Portfolio Risk Composition")

    return fig


# =========================================================
# 🔹 FEATURE IMPORTANCE (LIGHTGBM)
# =========================================================

def plot_feature_importance(model, features, top_n=15):
    """
    Feature importance plot.
    """

    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    fig, ax = plt.subplots()

    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.invert_yaxis()

    ax.set_title("Top Feature Importances")

    return fig


# =========================================================
# 🔹 SHAP GLOBAL IMPORTANCE
# =========================================================

def plot_shap_summary(model, X):
    """
    SHAP summary plot (requires shap installed).
    """

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)

    return fig


# =========================================================
# 🔹 SINGLE PREDICTION EXPLANATION
# =========================================================

def plot_shap_force(model, X_row):
    """
    SHAP force plot for a single prediction.
    """

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    return shap.force_plot(
        explainer.expected_value,
        shap_values,
        X_row
    )


# =========================================================
# 🔹 PROBABILITY GAUGE (SIMULATED)
# =========================================================

def plot_probability_gauge(prob):
    """
    Simple gauge-style visualization using matplotlib.
    """

    fig, ax = plt.subplots()

    ax.barh(["Risk"], [prob])
    ax.set_xlim(0, 1)

    ax.set_title(f"Default Probability: {prob:.2f}")

    return fig


# =========================================================
# 🔹 PORTFOLIO METRICS TABLE
# =========================================================

def create_metrics_table(metrics: dict) -> pd.DataFrame:
    """
    Converts metrics dict → display table.
    """

    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    return df


# =========================================================
# 🔹 CORRELATION HEATMAP
# =========================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Correlation matrix visualization.
    """

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()

    sns.heatmap(corr, annot=False, ax=ax)

    ax.set_title("Feature Correlation Heatmap")

    return fig


# =========================================================
# 🔹 DEBUG VISUALIZATION
# =========================================================

def debug_visuals(probs, risk_levels):
    """
    Quick debug visualization bundle.
    """

    figs = {}

    figs["distribution"] = plot_risk_distribution(probs)
    figs["segments"] = plot_risk_segments(risk_levels)
    figs["portfolio"] = plot_portfolio_mix(risk_levels)

    return figs