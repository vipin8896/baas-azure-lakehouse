# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver: clean, validate, enrich, upsert
# MAGIC * Dedupe on `txn_id` (latest ingest wins)
# MAGIC * Parse two timestamp formats, normalise currency, cast amount
# MAGIC * Data-quality rules -> invalid rows go to a **quarantine** table with the reasons (nothing is silently dropped)
# MAGIC * Convert to EUR using daily FX, attach partner / country / risk tier from account + customer
# MAGIC * `MERGE` into Silver so re-runs and late/replayed files are **idempotent**

# COMMAND ----------
from delta.tables import DeltaTable
from pyspark.sql import Window, functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
# Non-ANSI mode: malformed values become NULL (and get caught by DQ rules) instead of failing the job.
spark.conf.set("spark.sql.ansi.enabled", "false")

BASE = "abfss://lakehouse@<storageaccount>.dfs.core.windows.net"
BRONZE, SILVER, CHK = f"{BASE}/bronze", f"{BASE}/silver", f"{BASE}/_checkpoints"
SILVER_TXN, QUARANTINE = f"{SILVER}/transactions", f"{SILVER}/quarantine_transactions"

VALID_CCY = ["EUR", "GBP", "USD", "EGP", "KES", "NGN", "AED"]
VALID_STATUS = ["settled", "pending", "failed", "reversed"]
VALID_TYPES = ["card_payment", "bank_transfer", "cross_border_transfer", "wallet_topup", "cash_out", "fee"]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dimensions

# COMMAND ----------
fx = (
    spark.read.format("delta").load(f"{BRONZE}/fx_rates")
    .select(
        F.to_date("date").alias("fx_date"),
        F.upper(F.trim("currency")).alias("fx_ccy"),
        F.col("rate_to_eur").cast("double").alias("rate_to_eur"),
    )
)
fx.write.format("delta").mode("overwrite").save(f"{SILVER}/fx_rates")

accounts = spark.read.format("delta").load(f"{BRONZE}/accounts").alias("a")
customers = spark.read.format("delta").load(f"{BRONZE}/customers").alias("c")
dim_account = (
    accounts.join(customers, F.col("a.customer_id") == F.col("c.customer_id"))
    .select(
        F.col("a.account_id"),
        F.col("a.customer_id"),
        F.col("a.account_type"),
        F.col("a.status").alias("account_status"),
        F.col("c.partner_id"),
        F.col("c.country").alias("account_country"),
        F.col("c.kyc_status"),
        F.col("c.risk_tier"),
    )
)
dim_account.write.format("delta").mode("overwrite").save(f"{SILVER}/dim_account")
spark.read.format("delta").load(f"{BRONZE}/partners").write.format("delta").mode("overwrite").save(f"{SILVER}/dim_partner")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cleaning + data-quality rules

# COMMAND ----------
def clean(df):
    d = (
        df.withColumn("currency", F.upper(F.trim("currency")))
        .withColumn("amount", F.col("amount").cast("decimal(18,2)"))
        .withColumn(
            "txn_ts",
            F.coalesce(
                F.to_timestamp("txn_ts", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
                F.to_timestamp("txn_ts", "dd/MM/yyyy HH:mm:ss"),
            ),
        )
        .withColumn("txn_date", F.to_date("txn_ts"))
    )
    rules = {
        "null_account_id": F.col("account_id").isNull(),
        "bad_timestamp": F.col("txn_ts").isNull(),
        "non_positive_amount": F.col("amount").isNull() | (F.col("amount") <= 0),
        "unknown_currency": ~F.col("currency").isin(VALID_CCY),
        "unknown_status": ~F.col("status").isin(VALID_STATUS),
        "unknown_txn_type": ~F.col("txn_type").isin(VALID_TYPES),
    }
    failed = F.array(*[F.when(cond, F.lit(name)) for name, cond in rules.items()])
    return d.withColumn("dq_errors", F.filter(failed, lambda x: x.isNotNull()))


def upsert(batch_df, batch_id):
    # 1) dedupe inside the micro-batch: keep the latest delivery of each txn_id
    w = Window.partitionBy("txn_id").orderBy(F.col("_ingest_ts").desc())
    dedup = batch_df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

    # 2) clean + split good / bad
    d = clean(dedup)
    bad = d.filter(F.size("dq_errors") > 0)
    good = d.filter(F.size("dq_errors") == 0).drop("dq_errors")
    bad.withColumn("batch_id", F.lit(batch_id)).write.format("delta").mode("append").save(QUARANTINE)

    # 3) enrich: FX to EUR + account/partner attributes (both are small -> broadcast)
    enriched = (
        good.join(
            F.broadcast(fx),
            (good["txn_date"] == fx["fx_date"]) & (good["currency"] == fx["fx_ccy"]),
            "left",
        )
        .join(F.broadcast(dim_account), "account_id", "left")
        .withColumn("amount_eur", F.round(F.col("amount").cast("double") * F.col("rate_to_eur"), 2))
        .withColumn("is_orphan_account", F.col("partner_id").isNull())
        .drop("fx_date", "fx_ccy")
    )

    # 4) idempotent upsert (txn_date in the condition -> partition pruning)
    if DeltaTable.isDeltaTable(spark, SILVER_TXN):
        (
            DeltaTable.forPath(spark, SILVER_TXN).alias("t")
            .merge(enriched.alias("s"), "t.txn_id = s.txn_id AND t.txn_date = s.txn_date")
            .whenMatchedUpdateAll(condition="s._ingest_ts > t._ingest_ts")
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        enriched.write.format("delta").partitionBy("txn_date").save(SILVER_TXN)


# COMMAND ----------
(
    spark.readStream.format("delta")
    .option("maxFilesPerTrigger", 50)  # keeps each micro-batch a sensible size on first (full) load
    .load(f"{BRONZE}/transactions")
    .writeStream.foreachBatch(upsert)
    .option("checkpointLocation", f"{CHK}/silver_transactions")
    .trigger(availableNow=True)
    .start()
    .awaitTermination()
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data-quality report

# COMMAND ----------
silver_cnt = spark.read.format("delta").load(SILVER_TXN).count()
quar = spark.read.format("delta").load(QUARANTINE)
print(f"silver rows: {silver_cnt:,} | quarantined rows: {quar.count():,}")
display(
    quar.select(F.explode("dq_errors").alias("rule"))
    .groupBy("rule").count().orderBy(F.desc("count"))
)
