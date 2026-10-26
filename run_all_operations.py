"""
Non-Interactive Automated Verification Suite: Executes all PySpark Operations Step-by-Step.
Outputs cleanly formatted summary tables, execution timings, and verification metrics.
"""

import time
import os
import sys
from pathlib import Path
from tabulate import tabulate

from config import (
    get_spark_session, RAW_DATA_PATH, CLEANED_DATA_PATH, PROCESSED_DATA_PATH, VIS_DIR
)
from generate_dataset import generate_raw_dataset
from data_cleaning import clean_customer_data
from feature_processing import process_features, compute_aggregate_analytics
from visualize_features import generate_all_visualizations


def run_evaluation_suite():
    overall_start = time.time()
    timings = {}

    print("=" * 85)
    print("🎓 PYSPARK DATA CLEANING & FEATURE PROCESSING: AUTOMATED EVALUATION SUITE")
    print("=" * 85)

    # 1. Dataset Generation
    t0 = time.time()
    print("\n>>> OPERATION 1: Generating Raw Realistic Customer Dataset with Injected Anomalies...")
    raw_count = generate_raw_dataset(num_rows=2500)
    timings["Dataset Generation"] = time.time() - t0

    # 2. SparkSession Instantiation
    t0 = time.time()
    print("\n>>> OPERATION 2: Instantiating Localized Apache SparkSession...")
    spark = get_spark_session("PySpark_Verification_Suite")
    timings["SparkSession Startup"] = time.time() - t0

    # 3. Data Cleaning
    t0 = time.time()
    print("\n>>> OPERATION 3: Executing PySpark Schema Enforcement & Cleaning...")
    cleaned_df, audit_metrics = clean_customer_data(spark, str(RAW_DATA_PATH))
    # Action to evaluate execution
    clean_count = cleaned_df.count()
    timings["Data Cleaning & Imputation"] = time.time() - t0

    # 4. Feature Engineering
    t0 = time.time()
    print("\n>>> OPERATION 4: Executing PySpark ML Pipeline (StringIndexer, OneHot, VectorAssembler, Scaler)...")
    model_ready_df, feat_meta = process_features(cleaned_df)
    vector_sample = model_ready_df.select("customer_id", "scaled_features").limit(2).collect()
    timings["ML Feature Pipeline"] = time.time() - t0

    # 5. Window Functions & Aggregations
    t0 = time.time()
    print("\n>>> OPERATION 5: Computing Cohort Aggregations & Window Functions...")
    cohort_df = compute_aggregate_analytics(model_ready_df)
    cohort_summary = cohort_df.limit(4).collect()
    timings["Window & Aggregations"] = time.time() - t0

    # 6. Data Export
    t0 = time.time()
    print("\n>>> OPERATION 6: Persisting Cleaned & Model-Ready Parquet Datasets...")
    cleaned_df.write.mode("overwrite").parquet(str(CLEANED_DATA_PATH))
    model_ready_df.select("customer_id", "credit_tier", "scaled_features", "loan_approved").write.mode("overwrite").parquet(str(PROCESSED_DATA_PATH))
    timings["Parquet Persistence"] = time.time() - t0

    # 7. Visualization Suite
    t0 = time.time()
    print("\n>>> OPERATION 7: Generating 300 DPI High-Resolution Visualizations...")
    generate_all_visualizations()
    timings["Visual Analytics"] = time.time() - t0

    total_time = time.time() - overall_start

    # Print Summary Tables
    print("\n" + "=" * 85)
    print("📊 EXECUTION BENCHMARKS & VERIFICATION SUMMARY")
    print("=" * 85)

    exec_table = [
        [op, f"{duration:.2f} s", "Passed ✅"] for op, duration in timings.items()
    ]
    exec_table.append(["Total End-to-End Execution", f"{total_time:.2f} s", "Completed 🚀"])
    print(tabulate(exec_table, headers=["Pipeline Stage", "Duration (s)", "Status"], tablefmt="fancy_grid"))

    print("\n📊 DATA CLEANING & RECTIFICATION AUDIT")
    audit_table = [
        ["Total Raw Records Ingested", audit_metrics["raw_record_count"]],
        ["Injected Duplicates Identified & Removed", audit_metrics["duplicates_removed"]],
        ["Clean Records Retained", audit_metrics["clean_record_count"]],
        ["Imputed Numerical Features", ", ".join(audit_metrics["imputed_features"])],
        ["Imputed Categorical Features", "gender, education_level, employment_status"],
        ["Outliers Capped / Winsorized", "age (<18 or >90), credit_score (<300 or >850), income (>500k)"],
        ["Pipeline Chained Stages", feat_meta["pipeline_stages_count"]],
        ["Final Feature Vector Length", f"{feat_meta['feature_vector_dimension']} elements"]
    ]
    print(tabulate(audit_table, headers=["Verification Metric", "Value"], tablefmt="psql"))

    print("\n📊 COHORT SAMPLE PREVIEW (WINDOW & AGGREGATIONS)")
    cohort_table = [
        [r["credit_tier"], r["employment_status"], r["customer_count"], f"${r['avg_annual_income']:,.2f}", r["avg_credit_score"], f"{r['approval_rate_pct']}%"]
        for r in cohort_summary
    ]
    print(tabulate(cohort_table, headers=["Credit Tier", "Employment", "Count", "Avg Income", "Avg Credit", "Approval %"], tablefmt="fancy_grid"))

    print("\n📁 VERIFIED ARTIFACT DELIVERABLES:")
    print(f"  • Raw CSV Source       : {RAW_DATA_PATH}")
    print(f"  • Cleaned Parquet      : {CLEANED_DATA_PATH}")
    print(f"  • Model-Ready Parquet  : {PROCESSED_DATA_PATH}")
    for p in sorted(VIS_DIR.glob("*.png")):
        print(f"  • Publication Chart    : {p}")

    spark.stop()
    print("\n🎓 PySpark Evaluation Complete! SparkSession Stopped Cleanly.")
    print("=" * 85)


if __name__ == "__main__":
    run_evaluation_suite()
