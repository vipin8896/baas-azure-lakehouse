# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold: business-ready tables
# MAGIC 1. `daily_partner_volume` - volume / failure rate per partner, country and txn type (feeds Power BI)
# MAGIC 2. `fraud_velocity_alerts` - 5+ high-value cross-border / cash-out txns within 10 minutes on one account
# MAGIC 3. `reconciliation_exceptions` - txns stuck in `pending` for more than 3 days

# COMMAND ----------
from pyspark.sql import Window, functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")

BASE = "abfss://lakehouse@<storageaccount>.dfs.core.windows.net"
LANDING, SILVER, GOLD = f"{BASE}/landing", f"{BASE}/silver", f"{BASE}/gold"

silver = spark.read.format("delta").load(f"{SILVER}/transactions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Daily partner volume

# COMMAND ----------
volume = (
    silver.groupBy(
        "txn_date",
        F.coalesce("partner_id", F.lit("UNKNOWN")).alias("partner_id"),
        F.col("account_country").alias("country"),
        "txn_type",
    )
    .agg(
        F.count("*").alias("txn_count"),
        F.countDistinct("account_id").alias("active_accounts"),
        F.sum("amount_eur").alias("gross_volume_eur"),
        F.sum(F.when(F.col("status") == "settled", F.col("amount_eur"))).alias("settled_volume_eur"),
        F.avg("amount_eur").alias("avg_ticket_eur"),
        F.avg(F.when(F.col("status") == "failed", 1).otherwise(0)).alias("failure_rate"),
    )
)
(volume.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .partitionBy("txn_date").save(f"{GOLD}/daily_partner_volume"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fraud velocity rule (window function, 10-minute range window per account)

# COMMAND ----------
w10 = Window.partitionBy("account_id").orderBy("txn_epoch").rangeBetween(-600, 0)

velocity = (
    silver.filter(F.col("status") != "failed")
    .withColumn("txn_epoch", F.col("txn_ts").cast("long"))
    .withColumn("cnt_10m", F.count("*").over(w10))
    .withColumn("sum_eur_10m", F.sum("amount_eur").over(w10))
)
alerts = (
    velocity.filter(
        (F.col("cnt_10m") >= 5)
        & (F.col("sum_eur_10m") >= 1500)
        & F.col("txn_type").isin("cross_border_transfer", "cash_out")
    )
    .select("txn_id", "account_id", "partner_id", "txn_ts", "txn_date", "txn_type",
            "amount_eur", "cnt_10m", "sum_eur_10m")
)
(alerts.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .partitionBy("txn_date").save(f"{GOLD}/fraud_velocity_alerts"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Validate the rule against the generator's ground truth (account level)

# COMMAND ----------
labels = spark.read.option("header", "true").csv(f"{LANDING}/_labels/fraud_txn_ids.csv")
true_acc = silver.join(labels, "txn_id").select("account_id").distinct()
flag_acc = alerts.select("account_id").distinct()
tp = flag_acc.join(true_acc, "account_id").count()
n_flag, n_true = flag_acc.count(), true_acc.count()
print(f"flagged accounts={n_flag:,} true fraud accounts={n_true:,} true positives={tp:,}")
print(f"precision={tp / n_flag:.2%}  recall={tp / n_true:.2%}" if n_flag and n_true else "no data")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Reconciliation exceptions

# COMMAND ----------
max_date = silver.agg(F.max("txn_date")).first()[0]
recon = (
    silver.filter((F.col("status") == "pending") & (F.col("txn_date") < F.date_sub(F.lit(max_date), 3)))
    .select("txn_id", "account_id", "partner_id", "txn_date", "txn_type", "currency", "amount", "amount_eur",
            F.datediff(F.lit(max_date), "txn_date").alias("days_pending"))
)
recon.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(f"{GOLD}/reconciliation_exceptions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Table maintenance (small files + faster point lookups by account)

# COMMAND ----------
spark.sql(f"OPTIMIZE delta.`{SILVER}/transactions` ZORDER BY (account_id)")
spark.sql(f"VACUUM delta.`{SILVER}/transactions` RETAIN 168 HOURS")
