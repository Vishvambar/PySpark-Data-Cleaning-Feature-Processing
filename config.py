"""
Configuration module for PySpark localized environment and session management.
Ensures seamless execution across diverse Linux JVM and Python environments.
"""

import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualizations"
RAW_DATA_PATH = DATA_DIR / "raw_customer_features.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_customer_features.parquet"
PROCESSED_DATA_PATH = DATA_DIR / "model_ready_features.parquet"

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Automatic JDK 17 Configuration (Resolves Java 25 compatibility in newer Linux releases)
JDK_17_PATH = os.path.expanduser("~/.local/share/jvm/jdk-17.0.12+7")
if os.path.exists(JDK_17_PATH):
    os.environ["JAVA_HOME"] = JDK_17_PATH
    os.environ["PATH"] = f"{JDK_17_PATH}/bin:" + os.environ.get("PATH", "")

# 2. Worker Python Alignment (Aligns Spark worker with Python 3.12 venv)
VENV_PYTHON = os.path.expanduser("~/.local/share/pyspark-venv/bin/python")
if os.path.exists(VENV_PYTHON):
    os.environ["PYSPARK_PYTHON"] = VENV_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = VENV_PYTHON
else:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Suppress JVM / Hadoop noisy native-code warnings
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-memory 2g "
    "--executor-memory 2g "
    "pyspark-shell"
)


def get_spark_session(app_name: str = "PySpark_Data_Cleaning_Feature_Processing"):
    """
    Instantiates and configures a localized Apache SparkSession.
    
    Parameters:
        app_name (str): The logical title of the Spark application.
        
    Returns:
        SparkSession: The active localized SparkSession.
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    
    # Set logging to WARN to eliminate verbose INFO overhead
    spark.sparkContext.setLogLevel("WARN")
    return spark
