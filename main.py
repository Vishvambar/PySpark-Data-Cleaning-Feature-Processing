"""
Interactive Terminal CLI for PySpark Data Cleaning and Feature Processing.
Provides a clear menu for manual or automated execution of individual pipeline operations.
"""

import sys
import os
from pathlib import Path
from tabulate import tabulate

from config import (
    get_spark_session, RAW_DATA_PATH, CLEANED_DATA_PATH, PROCESSED_DATA_PATH, VIS_DIR
)
from generate_dataset import generate_raw_dataset
from data_cleaning import clean_customer_data
from feature_processing import process_features, compute_aggregate_analytics
from visualize_features import generate_all_visualizations
from run_all_operations import run_evaluation_suite


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║    Apache PySpark Data Cleaning & Model-Ready Feature Processing           ║
║            Distributed Data Engineering & Machine Learning CLI             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    print("\nSelect an operation:")
    print("  [1] Run Full End-to-End PySpark Pipeline")
    print("  [2] Generate / Regenerate Raw Customer Banking Dataset")
    print("  [3] Ingest CSV & Execute PySpark Data Cleaning & Imputation")
    print("  [4] Run PySpark ML Feature Pipeline (One-Hot, Assembler, Scaler)")
    print("  [5] Run Window Analytics & Cohort Aggregations")
    print("  [6] Generate Publication-Grade Visualizations (300 DPI)")
    print("  [7] Run Automated Verification Suite with Timings & Benchmarks")
    print("  [0] Exit")


def main():
    print_banner()
    spark = None

    while True:
        print_menu()
        choice = input("\nEnter choice [0-7]: ").strip()

        if choice == "0":
            if spark:
                spark.stop()
                print("SparkSession stopped.")
            print("Exiting PySpark CLI. Goodbye!")
            sys.exit(0)

        elif choice == "1":
            from pyspark_pipeline import run_full_pipeline
            s, _, _, _ = run_full_pipeline()
            s.stop()

        elif choice == "2":
            generate_raw_dataset(num_rows=2500)

        elif choice == "3":
            if not RAW_DATA_PATH.exists():
                print("Raw dataset not found. Generating...")
                generate_raw_dataset(num_rows=2500)
            if not spark:
                spark = get_spark_session("PySpark_Cleaning_CLI")
            cleaned_df, audit = clean_customer_data(spark, str(RAW_DATA_PATH))
            print(f"Cleaned {cleaned_df.count()} unique records. Dropped {audit['duplicates_removed']} duplicates.")
            cleaned_df.select("customer_id", "full_name", "credit_tier", "annual_income", "credit_score").show(5)

        elif choice == "4":
            if not spark:
                spark = get_spark_session("PySpark_ML_CLI")
            if not RAW_DATA_PATH.exists():
                generate_raw_dataset(num_rows=2500)
            cleaned_df, _ = clean_customer_data(spark, str(RAW_DATA_PATH))
            model_ready_df, meta = process_features(cleaned_df)
            print(f"Engineered {meta['feature_vector_dimension']}-dimensional feature vector across {meta['pipeline_stages_count']} stages.")
            model_ready_df.select("customer_id", "scaled_features", "loan_approved").show(5, truncate=False)

        elif choice == "5":
            if not spark:
                spark = get_spark_session("PySpark_Analytics_CLI")
            if not RAW_DATA_PATH.exists():
                generate_raw_dataset(num_rows=2500)
            cleaned_df, _ = clean_customer_data(spark, str(RAW_DATA_PATH))
            model_ready_df, _ = process_features(cleaned_df)
            cohort_df = compute_aggregate_analytics(model_ready_df)
            cohort_df.show(10)

        elif choice == "6":
            if not CLEANED_DATA_PATH.exists():
                print("Cleaned dataset not found. Running cleaning first...")
                if not spark:
                    spark = get_spark_session("PySpark_PreVis_CLI")
                cleaned_df, _ = clean_customer_data(spark, str(RAW_DATA_PATH))
                cleaned_df.write.mode("overwrite").parquet(str(CLEANED_DATA_PATH))
            generate_all_visualizations()

        elif choice == "7":
            run_evaluation_suite()

        else:
            print("Invalid selection. Please choose a valid option (0-7).")


if __name__ == "__main__":
    main()
