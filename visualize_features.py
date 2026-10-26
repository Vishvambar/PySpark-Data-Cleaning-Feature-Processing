"""
Visual Analytics and Chart Generator for PySpark Data Cleaning & Feature Processing.
Generates 300 DPI publication-quality charts for technical documentation and system verification:
  1. visualizations/1_data_cleaning_audit.png
  2. visualizations/2_feature_distributions_transformation.png
  3. visualizations/3_pyspark_feature_dashboard.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from config import VIS_DIR, RAW_DATA_PATH, CLEANED_DATA_PATH

# Set styling aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 300


def plot_data_cleaning_audit(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    """
    Generates missing value audit before vs after and deduplication donut chart.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Missing Values Comparison
    raw_missing = raw_df.isnull().sum()
    clean_missing = clean_df.isnull().sum()
    
    # Filter features that had missing values
    active_cols = [c for c in raw_missing.index if raw_missing[c] > 0]
    if not active_cols:
        active_cols = list(raw_df.columns[:7])

    audit_data = pd.DataFrame({
        "Feature": active_cols,
        "Raw Nulls": [raw_missing.get(c, 0) for c in active_cols],
        "Clean Nulls": [clean_missing.get(c, 0) for c in active_cols]
    })

    y_positions = np.arange(len(active_cols))
    height = 0.35

    bars1 = ax1.barh(y_positions - height/2, audit_data["Raw Nulls"], height, label="Raw CSV (Uncleaned)", color="#e63946", alpha=0.85)
    bars2 = ax1.barh(y_positions + height/2, audit_data["Clean Nulls"], height, label="PySpark Cleaned", color="#2a9d8f", alpha=0.9)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(active_cols, fontweight="bold")
    ax1.set_xlabel("Number of Missing / Null Values", fontweight="bold")
    ax1.set_title("Missing Value Resolution: Before vs After Imputation", fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

    for bar in bars1:
        w = bar.get_width()
        if w > 0:
            ax1.text(w + 3, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9, fontweight="bold", color="#b71c1c")
    for bar in bars2:
        w = bar.get_width()
        ax1.text(w + 3, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9, fontweight="bold", color="#1b5e20")

    # 2. Record Ingestion & Deduplication Donut Chart
    raw_count = len(raw_df)
    clean_count = len(clean_df)
    duplicates_removed = raw_count - clean_count

    donut_labels = ["Retained Clean Records", "Injected Duplicates Removed"]
    donut_vals = [clean_count, duplicates_removed]
    donut_colors = ["#1d3557", "#f4a261"]

    wedges, texts, autotexts = ax2.pie(
        donut_vals, labels=donut_labels, colors=donut_colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        explode=(0, 0.1), textprops={"fontweight": "bold"}
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax2.add_artist(centre_circle)
    ax2.text(0, 0, f"Total\n{raw_count}\nRecords", ha="center", va="center", fontweight="bold", fontsize=12)
    ax2.set_title("PySpark Deduplication Audit", fontweight="bold", pad=12)

    plt.tight_layout()
    out_file = VIS_DIR / "1_data_cleaning_audit.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file}")


def plot_feature_distributions(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    """
    Generates 4-panel comparison of raw vs cleaned feature distributions & relationships.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Annual Income Distribution (Outlier Clipping)
    sns.kdeplot(raw_df["annual_income"].dropna(), ax=axes[0, 0], color="#e63946", label="Raw Income (with Outliers)", linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(clean_df["annual_income"].dropna(), ax=axes[0, 0], color="#2a9d8f", label="Cleaned & Winsorized Income", linewidth=2.5, fill=True, alpha=0.3)
    axes[0, 0].set_title("Annual Income Distribution: Outlier Capping", fontweight="bold")
    axes[0, 0].set_xlabel("Annual Income ($)", fontweight="bold")
    axes[0, 0].set_ylabel("Density", fontweight="bold")
    axes[0, 0].set_xlim(0, 400000)
    axes[0, 0].legend(frameon=True)

    # Subplot 2: Credit Score Distribution by Credit Tier
    tier_order = ["Poor", "Fair", "Good", "Excellent"]
    palette = {"Poor": "#e63946", "Fair": "#f4a261", "Good": "#457b9d", "Excellent": "#2a9d8f"}
    sns.histplot(data=clean_df, x="credit_score", hue="credit_tier", hue_order=tier_order, palette=palette, multiple="stack", bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Credit Score Distribution by Derived Credit Tier", fontweight="bold")
    axes[0, 1].set_xlabel("Credit Score (Imputed & Standardized)", fontweight="bold")
    axes[0, 1].set_ylabel("Customer Count", fontweight="bold")

    # Subplot 3: Debt-to-Income vs Loan-to-Income by Approval Status
    sns.scatterplot(
        data=clean_df.sample(min(1000, len(clean_df)), random_state=42),
        x="debt_to_income_ratio", y="loan_to_income_ratio",
        hue="loan_approved", palette={0: "#d90429", 1: "#10b981"},
        alpha=0.6, s=40, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Debt-to-Income vs Loan-to-Income Ratio", fontweight="bold")
    axes[1, 0].set_xlabel("Debt-to-Income Ratio", fontweight="bold")
    axes[1, 0].set_ylabel("Loan-to-Income Ratio", fontweight="bold")
    axes[1, 0].set_xlim(0, 2.0)
    axes[1, 0].set_ylim(0, 1.5)

    # Subplot 4: Approval Rate by Credit Tier
    approval_rates = clean_df.groupby("credit_tier")["loan_approved"].mean().reindex(tier_order) * 100
    bars = axes[1, 1].bar(approval_rates.index, approval_rates.values, color=["#e63946", "#f4a261", "#457b9d", "#2a9d8f"], edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Loan Approval Probability by Credit Tier", fontweight="bold")
    axes[1, 1].set_xlabel("Credit Tier", fontweight="bold")
    axes[1, 1].set_ylabel("Approval Rate (%)", fontweight="bold")
    axes[1, 1].set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    out_file = VIS_DIR / "2_feature_distributions_transformation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file}")


def plot_pyspark_feature_dashboard(clean_df: pd.DataFrame):
    """
    Generates multi-metric executive dashboard for feature engineering.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Average Income and Debt by Credit Tier
    tier_summary = clean_df.groupby("credit_tier")[["annual_income", "total_debt"]].mean().reindex(["Poor", "Fair", "Good", "Excellent"])
    x = np.arange(len(tier_summary.index))
    w = 0.35
    axes[0, 0].bar(x - w/2, tier_summary["annual_income"], w, label="Avg Annual Income", color="#1d3557", alpha=0.9)
    axes[0, 0].bar(x + w/2, tier_summary["total_debt"], w, label="Avg Total Debt", color="#e76f51", alpha=0.9)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tier_summary.index, fontweight="bold")
    axes[0, 0].set_ylabel("Amount ($)", fontweight="bold")
    axes[0, 0].set_title("Cohort Economics: Income vs Debt by Credit Tier", fontweight="bold")
    axes[0, 0].legend(frameon=True)

    # 2. Correlation Heatmap of Numerical Features
    num_cols = ["age", "annual_income", "credit_score", "total_debt", "loan_amount", "account_balance", "delinquent_months", "loan_approved"]
    corr = clean_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[0, 1], cbar_kws={"label": "Pearson Correlation"}, linewidths=0.5)
    axes[0, 1].set_title("Feature Correlation Heatmap", fontweight="bold")

    # 3. Education Level Distribution vs Approval
    edu_summary = clean_df.groupby(["education_level", "loan_approved"]).size().unstack(fill_value=0)
    edu_summary.plot(kind="bar", stacked=True, color=["#d90429", "#10b981"], ax=axes[1, 0], edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Education Level Cohort Volume & Loan Decisions", fontweight="bold")
    axes[1, 0].set_xlabel("Education Level", fontweight="bold")
    axes[1, 0].set_ylabel("Customer Count", fontweight="bold")
    axes[1, 0].legend(["Rejected", "Approved"], frameon=True)
    axes[1, 0].tick_params(axis="x", rotation=30)

    # 4. Employment Status Distribution vs Median Income
    emp_income = clean_df.groupby("employment_status")["annual_income"].median().sort_values(ascending=False)
    bars = axes[1, 1].barh(emp_income.index, emp_income.values, color="#264653", alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Median Annual Income by Employment Category", fontweight="bold")
    axes[1, 1].set_xlabel("Median Annual Income ($)", fontweight="bold")
    axes[1, 1].set_ylabel("Employment Status", fontweight="bold")
    for bar in bars:
        w_val = bar.get_width()
        axes[1, 1].text(w_val + 1000, bar.get_y() + bar.get_height()/2, f"${w_val:,.0f}", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_file = VIS_DIR / "3_pyspark_feature_dashboard.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file}")


def generate_all_visualizations():
    """
    Loads raw CSV and cleaned Parquet to generate full analytics suite.
    """
    print("=" * 70)
    print("📊 GENERATING HIGH-RESOLUTION FEATURE VISUALIZATIONS (300 DPI)")
    print("=" * 70)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = pd.read_parquet(CLEANED_DATA_PATH)

    plot_data_cleaning_audit(raw_df, clean_df)
    plot_feature_distributions(raw_df, clean_df)
    plot_pyspark_feature_dashboard(clean_df)

    print("✅ All visualizations successfully generated and saved to visualizations/")


if __name__ == "__main__":
    generate_all_visualizations()
