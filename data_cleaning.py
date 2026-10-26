"""
Data Cleaning Module using Apache PySpark.
Demonstrates:
  - Explicit Schema definition using StructType and StructFields
  - Duplicate detection and removal (.dropDuplicates)
  - String normalization (trim, lower, initcap)
  - Outlier handling (conditional filtering and Winsorization)
  - Missing value imputation using PySpark MLlib Imputer and fillna
  - Feature derivation (Debt-to-Income, Loan-to-Income, Credit Tier)
"""

from typing import Dict, Any, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.ml.feature import Imputer


def get_raw_schema() -> StructType:
    """
    Defines explicit StructType schema for the raw customer banking dataset.
    """
    return StructType([
        StructField("customer_id", StringType(), False),
        StructField("full_name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("education_level", StringType(), True),
        StructField("employment_status", StringType(), True),
        StructField("annual_income", DoubleType(), True),
        StructField("credit_score", IntegerType(), True),
        StructField("total_debt", DoubleType(), True),
        StructField("loan_amount", DoubleType(), True),
        StructField("account_balance", DoubleType(), True),
        StructField("delinquent_months", IntegerType(), True),
        StructField("loan_approved", IntegerType(), True)
    ])


def audit_missing_values(df: DataFrame) -> Dict[str, int]:
    """
    Computes missing and null value count for every column in the PySpark DataFrame.
    """
    aggregations = []
    for col_name in df.columns:
        # Check both isNull and NaN (for floating point columns)
        cond = F.col(col_name).isNull() | F.isnan(col_name) if df.schema[col_name].dataType in [DoubleType()] else F.col(col_name).isNull()
        aggregations.append(F.count(F.when(cond, col_name)).alias(col_name))
        
    audit_row = df.select(aggregations).collect()[0]
    return {col_name: audit_row[col_name] for col_name in df.columns}


def clean_customer_data(spark: SparkSession, raw_csv_path: str) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end PySpark data cleaning operations.
    
    Returns:
        Tuple[DataFrame, Dict[str, Any]]: Cleaned DataFrame and quality audit metrics.
    """
    schema = get_raw_schema()
    
    # 1. Load CSV with explicit schema
    raw_df = spark.read.csv(
        raw_csv_path,
        header=True,
        schema=schema,
        nullValue=""
    )
    raw_count = raw_df.count()
    raw_missing = audit_missing_values(raw_df)

    # 2. Duplicate Detection and Removal
    dedup_df = raw_df.dropDuplicates(["customer_id"])
    dedup_count = dedup_df.count()
    duplicates_removed = raw_count - dedup_count

    # 3. String Trimming and Normalization
    string_cleaned_df = (
        dedup_df
        .withColumn("full_name", F.initcap(F.trim(F.col("full_name"))))
        .withColumn("education_level", F.initcap(F.trim(F.col("education_level"))))
        .withColumn("employment_status", F.trim(F.col("employment_status")))
        .withColumn("gender", F.trim(F.col("gender")))
    )

    # 4. Outlier Handling / Invalidation prior to Imputation
    # Age valid range: 18 to 90
    # Credit score valid range: 300 to 850
    # Annual income extreme outlier capping (> $500,000 capped at 500,000)
    outlier_adjusted_df = (
        string_cleaned_df
        .withColumn("age", F.when((F.col("age") < 18) | (F.col("age") > 90), F.lit(None)).otherwise(F.col("age")))
        .withColumn("credit_score", F.when((F.col("credit_score") < 300) | (F.col("credit_score") > 850), F.lit(None)).otherwise(F.col("credit_score")))
        .withColumn("annual_income", F.when(F.col("annual_income") > 500000, 500000.0).otherwise(F.col("annual_income")))
    )

    # 5. Missing Value Imputation
    # A. Categorical Imputation using fillna with default 'Unknown'
    cat_filled_df = outlier_adjusted_df.fillna({
        "education_level": "Unknown",
        "employment_status": "Unknown",
        "gender": "Unknown"
    })

    # B. Numerical Imputation using PySpark ML Imputer (Median Strategy)
    num_cols = ["age", "annual_income", "credit_score", "total_debt", "account_balance"]
    # Cast integer columns to double for MLlib Imputer compatibility
    for col_name in ["age", "credit_score"]:
        cat_filled_df = cat_filled_df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    imputer = Imputer(
        inputCols=num_cols,
        outputCols=[f"{c}_imputed" for c in num_cols]
    ).setStrategy("median")

    imputer_model = imputer.fit(cat_filled_df)
    imputed_df = imputer_model.transform(cat_filled_df)

    # Drop intermediate columns and replace with imputed values
    for col_name in num_cols:
        imputed_df = imputed_df.drop(col_name).withColumnRenamed(f"{col_name}_imputed", col_name)

    # 6. Feature Engineering & Derived Ratios
    enriched_df = (
        imputed_df
        .withColumn("debt_to_income_ratio", F.round(F.col("total_debt") / (F.col("annual_income") + 1e-4), 4))
        .withColumn("loan_to_income_ratio", F.round(F.col("loan_amount") / (F.col("annual_income") + 1e-4), 4))
        .withColumn(
            "credit_tier",
            F.when(F.col("credit_score") >= 750, "Excellent")
             .when(F.col("credit_score") >= 700, "Good")
             .when(F.col("credit_score") >= 640, "Fair")
             .otherwise("Poor")
        )
    )

    cleaned_count = enriched_df.count()
    cleaned_missing = audit_missing_values(enriched_df)

    audit_metrics = {
        "raw_record_count": raw_count,
        "duplicates_removed": duplicates_removed,
        "clean_record_count": cleaned_count,
        "raw_missing_breakdown": raw_missing,
        "clean_missing_breakdown": cleaned_missing,
        "imputed_features": num_cols
    }

    return enriched_df, audit_metrics
