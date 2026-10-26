"""
Main Pipeline Driver: End-to-End PySpark Data Cleaning & Feature Processing.
Orchestrates:
  1. Localized SparkSession instantiation
  2. Raw CSV ingestion with explicit schema
  3. Deduplication, string cleaning, outlier clipping, and missing value imputation
  4. Feature vectorization, one-hot encoding, and standard scaling via ML Pipeline
  5. Window function analytics & aggregate cohort summaries
  6. Model-ready dataset export to Parquet format
"""

import sys
import time
from pathlib import Path
from tabulate import tabulate

from config import (
    get_spark_session, RAW_DATA_PATH, CLEANED_DATA_PATH, PROCESSED_DATA_PATH
)
from generate_dataset import generate_raw_dataset
from data_cleaning import clean_customer_data
from feature_processing import process_features, compute_aggregate_analytics


def run_full_pipeline():
    """
    Executes the comprehensive PySpark data engineering and feature processing pipeline.
    """
    start_time = time.time()
    print("=" * 80)
    print("🚀 STARTING PYSPARK DATA CLEANING & FEATURE PROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Ensure dataset exists
    if not RAW_DATA_PATH.exists():
        print("Raw dataset not found. Generating realistic customer dataset...")
        generate_raw_dataset(num_rows=2500)
    else:
        print(f"Verified source dataset at: {RAW_DATA_PATH}")

    # Step 2: Initialize SparkSession
    print("\n[Step 1/5] Initializing Localized SparkSession...")
    spark = get_spark_session("PySpark_Data_Cleaning_Pipeline")
    print(f" Spark Version: {spark.version}")
    print(f" Spark Master : {spark.conf.get('spark.master')}")
    print(f" Active App   : {spark.sparkContext.appName}")

    # Step 3: Data Cleaning
    print("\n[Step 2/5] Performing Data Quality Audit & Cleaning...")
    cleaned_df, audit_metrics = clean_customer_data(spark, str(RAW_DATA_PATH))
    
    audit_table = [
        ["Total Raw Records Ingested", audit_metrics["raw_record_count"]],
        ["Injected Duplicates Removed", audit_metrics["duplicates_removed"]],
        ["Clean Unique Records Retained", audit_metrics["clean_record_count"]],
        ["Imputed Numerical Features", ", ".join(audit_metrics["imputed_features"])]
    ]
    print(tabulate(audit_table, headers=["Data Quality Metric", "Value"], tablefmt="fancy_grid"))

    print("\n--- Missing Values Before vs After Cleaning ---")
    missing_comparison = []
    for col_name in audit_metrics["raw_missing_breakdown"]:
        raw_m = audit_metrics["raw_missing_breakdown"][col_name]
        clean_m = audit_metrics["clean_missing_breakdown"].get(col_name, 0)
        missing_comparison.append([col_name, raw_m, clean_m, "Cleaned" if clean_m == 0 else "Pending"])
    print(tabulate(missing_comparison[:8], headers=["Feature", "Raw Nulls", "Clean Nulls", "Status"], tablefmt="psql"))

    # Step 4: Feature Processing with MLlib Pipeline
    print("\n[Step 3/5] Executing PySpark ML Feature Engineering Pipeline...")
    model_ready_df, feat_meta = process_features(cleaned_df)
    
    print(f" Chained Pipeline Stages : {feat_meta['pipeline_stages_count']}")
    print(f" Assembled Feature Dim   : {feat_meta['feature_vector_dimension']} elements")

    # Step 5: Cohort Aggregation and Analytics
    print("\n[Step 4/5] Computing Cohort Analytics & Window Partitions...")
    cohort_df = compute_aggregate_analytics(model_ready_df)
    cohort_rows = cohort_df.limit(6).collect()
    cohort_table = [
        [r["credit_tier"], r["employment_status"], r["customer_count"], 
         f"${r['avg_annual_income']:,.2f}", r["avg_credit_score"], f"${r['avg_debt']:,.2f}", f"{r['approval_rate_pct']}%"]
        for r in cohort_rows
    ]
    print(tabulate(
        cohort_table, 
        headers=["Tier", "Employment", "Count", "Avg Income", "Avg Credit", "Avg Debt", "Approval %"],
        tablefmt="fancy_grid"
    ))

    # Sample Model-Ready Vectors
    print("\n--- Model-Ready Vector Features (Top 3 Records) ---")
    sample_vectors = (
        model_ready_df.select("customer_id", "credit_tier", "income_rank_in_tier", "scaled_features", "loan_approved")
        .limit(3)
        .collect()
    )
    vec_table = [
        [r["customer_id"], r["credit_tier"], r["income_rank_in_tier"], str(r["scaled_features"])[:40] + "...", r["loan_approved"]]
        for r in sample_vectors
    ]
    print(tabulate(vec_table, headers=["Customer ID", "Credit Tier", "Income Rank", "Scaled Feature Vector", "Approved"], tablefmt="psql"))

    # Step 6: Export Cleaned and Processed Data
    print("\n[Step 5/5] Exporting Parquet Datasets...")
    cleaned_df.write.mode("overwrite").parquet(str(CLEANED_DATA_PATH))
    model_ready_df.select("customer_id", "credit_tier", "scaled_features", "loan_approved").write.mode("overwrite").parquet(str(PROCESSED_DATA_PATH))
    print(f" Cleaned Dataset saved to: {CLEANED_DATA_PATH}")
    print(f" Model-Ready Dataset saved to: {PROCESSED_DATA_PATH}")

    elapsed = time.time() - start_time
    print(f"\n Pipeline Execution Completed Successfully in {elapsed:.2f} seconds!")
    print("=" * 80)

    return spark, cleaned_df, model_ready_df, audit_metrics


if __name__ == "__main__":
    spark, _, _, _ = run_full_pipeline()
    spark.stop()
