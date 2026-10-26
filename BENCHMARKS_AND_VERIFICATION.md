# Production Verification & System Benchmark Report
## Distributed Data Cleaning & Feature Processing using Apache PySpark

---

### System Verification & Execution Metadata
- **Project**: Distributed PySpark Data Cleaning & Feature Processing Pipeline
- **Execution Target**: Apache Spark 3.5.3 (PySpark MLlib) on OpenJDK 17 & Python 3.12 LTS
- **Execution Mode**: Distributed Local Cluster (`local[*]`) with Project Tungsten Engine

---

### 1. Aim & Objectives
- **Aim**: To setup a localized PySpark environment, instantiate a distributed Spark Session, construct structured DataFrames, and perform foundational operations for engineering clean, model-ready data features.
- **Key Objectives**:
  1. Enforce explicit `StructType` schemas to bypass costly schema inference passes.
  2. Perform distributed data cleaning: remove duplicate rows, normalize dirty strings, invalidate & Winsorize statistical outliers.
  3. Implement PySpark MLlib `Imputer` with median strategy for continuous numerical columns and default categorical imputation.
  4. Chain `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, and `StandardScaler` into a unified 7-stage PySpark ML `Pipeline`.
  5. Apply partition-based Window ranking (`Window.partitionBy().orderBy()`) and cohort group aggregations.
  6. Export model-ready vectors to Snappy Parquet and generate 300 DPI publication charts.

---

### 2. Execution Benchmarks & Timing Summary

| Pipeline Operation | Subsystem / API | Execution Time (s) | Verification Status |
| :--- | :--- | :---: | :---: |
| **1. Dataset Generation** | Python Synthetic Data Engine | 0.05 s | Passed ✅ (2,620 rows generated) |
| **2. SparkSession Startup** | Master `local[*]`, UI off, shuffle=4 | 6.20 s | Passed ✅ (JVM & Gateway Active) |
| **3. Data Cleaning & Imputation** | PySpark SQL & MLlib `Imputer` | 20.97 s | Passed ✅ (100% missing values resolved) |
| **4. ML Feature Pipeline** | 7-Stage Pipeline (One-Hot, Scaler) | 7.91 s | Passed ✅ (32-dim vector produced) |
| **5. Window & Cohort Analytics** | `Window.partitionBy` & `groupBy` | 1.69 s | Passed ✅ (Dense rank calculated) |
| **6. Parquet Persistence** | Snappy Compressed Parquet Output | 3.14 s | Passed ✅ (Cleaned & model-ready files) |
| **7. Visual Analytics Suite** | Matplotlib & Seaborn (300 DPI) | 7.17 s | Passed ✅ (3 publication charts saved) |
| **Total End-to-End Execution** | **Unified Automated Runner** | **47.13 s** | **Completed Successfully 🚀** |

---

### 3. Data Quality & Cleansing Rectification Audit

| Quality Dimension | Raw Ingested Dataset | PySpark Cleaned Output | Rectification Method |
| :--- | :---: | :---: | :--- |
| **Total Record Count** | 2,620 rows | 2,500 rows | `dropDuplicates(["customer_id"])` |
| **Injected Duplicates** | 120 records (4.6%) | 0 records | Distributed partition deduplication |
| **Missing `annual_income`** | 210 nulls | 0 nulls | MLlib `Imputer` (Median: $64,708.00) |
| **Missing `credit_score`** | 176 nulls | 0 nulls | MLlib `Imputer` (Median: 680.0) |
| **Missing `age`** | 156 nulls | 0 nulls | MLlib `Imputer` (Median: 44.0) |
| **Missing `total_debt`** | 153 nulls | 0 nulls | MLlib `Imputer` (Median: $37,850.50) |
| **Missing `account_balance`** | 151 nulls | 0 nulls | MLlib `Imputer` (Median: $47,381.50) |
| **Missing Categoricals** | 477 nulls total | 0 nulls | Imputed with `"Unknown"` category |
| **Income Outliers (> $500k)**| Multi-million values | Capped at $500,000 | Winsorization via `F.when().otherwise()` |
| **Invalid Credit Scores** | Erroneous values (<300, >850)| Re-imputed to 680 | Invalidation prior to median imputation |

---

### 4. Cohort Analytics Summary (Window & GroupBy)

| Credit Tier | Employment Status | Customer Volume | Avg Annual Income | Avg Credit Score | Avg Debt | Loan Approval Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Excellent** | Employed | 79 | $74,613.15 | 793.3 | $39,191.31 | **82.28%** |
| **Excellent** | Retired | 98 | $79,843.32 | 784.0 | $36,918.46 | **77.55%** |
| **Excellent** | Unemployed | 106 | $72,196.66 | 785.5 | $38,861.17 | **75.47%** |
| **Excellent** | Self-Employed | 83 | $68,286.50 | 795.4 | $38,937.27 | **74.70%** |
| **Good** | Employed | 224 | $72,504.10 | 721.4 | $37,120.50 | **78.12%** |
| **Fair** | Employed | 258 | $73,190.20 | 669.8 | $38,450.10 | **43.80%** |
| **Poor** | Employed | 185 | $71,450.80 | 580.2 | $37,920.40 | **6.49%** |

---

### 5. Sample Model-Ready Feature Vectors (PySpark MLlib)

```
+---------------+---------------+---------------+---------------------------------------------+------------+
| Customer ID   | Credit Tier   |   Income Rank | Scaled Feature Vector                       |   Approved |
|---------------+---------------+---------------+---------------------------------------------+------------|
| CUST_1614     | Excellent     |             1 | (32,[0,1,2,3,4,5,7,8,9,14,23,30],[3.7254... |          1 |
| CUST_2676     | Excellent     |             1 | (32,[0,1,2,3,4,5,6,7,8,10,18,22,30],[2.0... |          0 |
| CUST_2712     | Excellent     |             1 | (32,[0,1,2,3,4,5,7,8,10,14,24,30],[3.352... |          1 |
+---------------+---------------+---------------+---------------------------------------------+------------+
```

---

### 6. Deliverable Artifacts & Visual Charts
All artifacts generated during execution:

| Deliverable Type | Artifact Path | Description |
| :--- | :--- | :--- |
| **Automation Launcher** | [run.sh](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/run.sh) | Shell script binding JDK 17 & Python 3.12 |
| **Source Dataset** | [raw_customer_features.csv](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/data/raw_customer_features.csv) | 2,620 raw records with injected anomalies |
| **Cleaned Dataset** | [cleaned_customer_features.parquet](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/data/cleaned_customer_features.parquet) | Cleaned, deduplicated, imputed Parquet |
| **Model-Ready Dataset**| [model_ready_features.parquet](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/data/model_ready_features.parquet) | 32-dimensional scaled vector features |
| **Audit Chart** | [1_data_cleaning_audit.png](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/visualizations/1_data_cleaning_audit.png) | 300 DPI Missing value & deduplication audit |
| **Distribution Chart** | [2_feature_distributions_transformation.png](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/visualizations/2_feature_distributions_transformation.png) | 300 DPI Outlier Winsorization & Tier analytics |
| **Executive Dashboard**| [3_pyspark_feature_dashboard.png](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/visualizations/3_pyspark_feature_dashboard.png) | 300 DPI Correlation heatmap & cohort metrics |
| **Technical Specification**| [TECHNICAL_SPECIFICATION.md](file:///home/vishvambar/Desktop/Data%20Visualisation/Data-Engineering/Data_Cleaning_Feature_Processing_PySpark/TECHNICAL_SPECIFICATION.md) | Full system architecture manual & engineering FAQ |

---

### 7. Execution & CLI Commands
To execute or verify any component:
```bash
# 1. Run full automated evaluation suite with benchmarks (47 seconds):
./run.sh test

# 2. Run end-to-end pipeline:
./run.sh pipeline

# 3. Generate high-resolution charts:
./run.sh vis

# 4. Open interactive CLI:
./run.sh
```
