# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze ingestion
# MAGIC Raw partner files -> Delta, **no business logic**. Everything is stored as STRING so that dirty rows are never
# MAGIC rejected at the door; cleaning happens in Silver. Auto Loader tracks which files were already processed
# MAGIC (checkpoint), so the notebook is safe to re-run.

# COMMAND ----------
from pyspark.sql import functions as F

# Point this to your ADLS Gen2 container. For a quick test on DBFS use e.g. "dbfs:/tmp/bkn_lakehouse".
BASE = "abfss://lakehouse@<storageaccount>.dfs.core.windows.net"
LANDING, BRONZE, CHK = f"{BASE}/landing", f"{BASE}/bronze", f"{BASE}/_checkpoints"

TXN_SCHEMA = (
    "txn_id STRING, account_id STRING, txn_ts STRING, txn_type STRING, amount STRING, "
    "currency STRING, status STRING, channel STRING, merchant_category STRING, counterparty_country STRING"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Transactions (incremental, Auto Loader)

# COMMAND ----------
(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", "true")
    .schema(TXN_SCHEMA)
    .load(f"{LANDING}/transactions/")
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("ingest_date", F.regexp_extract("_source_file", r"ingest_date=(\d{4}-\d{2}-\d{2})", 1))
    .writeStream.format("delta")
    .option("checkpointLocation", f"{CHK}/bronze_transactions")
    .partitionBy("ingest_date")
    .trigger(availableNow=True)  # process everything new, then stop (batch-like, cheap)
    .start(f"{BRONZE}/transactions")
    .awaitTermination()
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Reference tables (small, full snapshot each run)

# COMMAND ----------
for name in ["partners", "customers", "accounts", "fx_rates"]:
    (
        spark.read.option("header", "true").csv(f"{LANDING}/reference/{name}.csv")
        .withColumn("_ingest_ts", F.current_timestamp())
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .save(f"{BRONZE}/{name}")
    )

# COMMAND ----------
# Sanity check
for name in ["transactions", "partners", "customers", "accounts", "fx_rates"]:
    print(name, spark.read.format("delta").load(f"{BRONZE}/{name}").count())
