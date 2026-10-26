"""
Feature Processing and ML Vectorization Module using PySpark MLlib.
Demonstrates:
  - StringIndexer: Encoding string categories to numerical indices
  - OneHotEncoder: Vectorizing nominal categories into binary sparse vectors
  - VectorAssembler: Combining numerical and encoded vectors into unified feature vectors
  - StandardScaler: Scaling features to unit standard deviation
  - ML Pipeline: Chaining all feature engineering stages into an atomic Pipeline
  - PySpark Window Functions: Ranking and partition-based analytics
"""

from typing import Tuple, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)


def build_feature_engineering_pipeline() -> Pipeline:
    """
    Constructs a PySpark ML Pipeline with StringIndexer, OneHotEncoder,
    VectorAssembler, and StandardScaler stages.
    """
    categorical_cols = ["gender", "education_level", "employment_status", "credit_tier"]
    indexed_cols = [f"{col}_idx" for col in categorical_cols]
    encoded_cols = [f"{col}_vec" for col in categorical_cols]

    # Stage 1: StringIndexer
    indexers = [
        StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep")
        for col, idx_col in zip(categorical_cols, indexed_cols)
    ]

    # Stage 2: OneHotEncoder
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
        handleInvalid="keep"
    )

    # Stage 3: VectorAssembler
    continuous_cols = [
        "age", "annual_income", "credit_score", "total_debt",
        "loan_amount", "account_balance", "delinquent_months",
        "debt_to_income_ratio", "loan_to_income_ratio"
    ]
    all_feature_cols = continuous_cols + encoded_cols

    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    # Stage 4: StandardScaler (withStd=True, withMean=False for sparse vector support)
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False
    )

    stages = indexers + [encoder, assembler, scaler]
    return Pipeline(stages=stages)


def process_features(df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Fits and transforms the cleaned DataFrame using the PySpark ML Pipeline,
    and applies analytical Window functions.
    
    Returns:
        Tuple[DataFrame, Dict[str, Any]]: Processed DataFrame and metadata.
    """
    pipeline = build_feature_engineering_pipeline()
    pipeline_model = pipeline.fit(df)
    transformed_df = pipeline_model.transform(df)

    # Apply PySpark Window Function: Rank customers by annual income within each Credit Tier
    window_spec = Window.partitionBy("credit_tier").orderBy(F.desc("annual_income"))
    ranked_df = transformed_df.withColumn(
        "income_rank_in_tier",
        F.dense_rank().over(window_spec)
    )

    # Calculate partition running average debt
    running_avg_window = Window.partitionBy("credit_tier").orderBy("credit_score").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    analyzed_df = ranked_df.withColumn(
        "running_avg_debt",
        F.round(F.avg("total_debt").over(running_avg_window), 2)
    )

    # Metadata extraction
    stages_count = len(pipeline_model.stages)
    vector_size = len(analyzed_df.select("scaled_features").first()[0])

    meta = {
        "pipeline_stages_count": stages_count,
        "feature_vector_dimension": vector_size,
        "categorical_features_indexed": ["gender", "education_level", "employment_status", "credit_tier"],
        "continuous_features_scaled": [
            "age", "annual_income", "credit_score", "total_debt",
            "loan_amount", "account_balance", "delinquent_months",
            "debt_to_income_ratio", "loan_to_income_ratio"
        ]
    }

    return analyzed_df, meta


def compute_aggregate_analytics(df: DataFrame) -> DataFrame:
    """
    Computes aggregated cohort analytics using PySpark groupBy and agg.
    """
    return (
        df.groupBy("credit_tier", "employment_status")
        .agg(
            F.count("customer_id").alias("customer_count"),
            F.round(F.avg("annual_income"), 2).alias("avg_annual_income"),
            F.round(F.avg("credit_score"), 1).alias("avg_credit_score"),
            F.round(F.avg("total_debt"), 2).alias("avg_debt"),
            F.round(F.avg("loan_approved") * 100, 2).alias("approval_rate_pct")
        )
        .orderBy("credit_tier", F.desc("customer_count"))
    )
