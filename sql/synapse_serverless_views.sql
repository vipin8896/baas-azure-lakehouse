-- Synapse serverless SQL pool: expose the Gold Delta tables to Power BI / analysts.
-- Replace <storageaccount> with your ADLS Gen2 account name.
--
-- Prerequisite: give the Synapse workspace managed identity the role
-- "Storage Blob Data Reader" on the storage account (Access control (IAM) -> Add role assignment).
--
-- Run each statement ONE AT A TIME in Synapse Studio (no GO between statements).

-- 1) Connected to master:
-- CREATE DATABASE bkn_gold;

-- 2) Switch "Use database" to bkn_gold, then:
CREATE MASTER KEY ENCRYPTION BY PASSWORD = '<choose-a-strong-password>';

CREATE DATABASE SCOPED CREDENTIAL wsmi WITH IDENTITY = 'Managed Identity';

CREATE EXTERNAL DATA SOURCE gold_ds WITH (
    LOCATION = 'https://<storageaccount>.dfs.core.windows.net/lakehouse/gold',
    CREDENTIAL = wsmi);

-- 3) Views over the Gold Delta tables (use CREATE VIEW the first time, ALTER VIEW to change)
CREATE VIEW dbo.daily_partner_volume AS
SELECT * FROM OPENROWSET(BULK 'daily_partner_volume', DATA_SOURCE = 'gold_ds', FORMAT = 'DELTA') AS r;

CREATE VIEW dbo.fraud_velocity_alerts AS
SELECT * FROM OPENROWSET(BULK 'fraud_velocity_alerts', DATA_SOURCE = 'gold_ds', FORMAT = 'DELTA') AS r;

CREATE VIEW dbo.reconciliation_exceptions AS
SELECT * FROM OPENROWSET(BULK 'reconciliation_exceptions', DATA_SOURCE = 'gold_ds', FORMAT = 'DELTA') AS r;

-- ---------------------------------------------------------------------------
-- Example analytics queries (good to show SQL depth)
-- ---------------------------------------------------------------------------

-- 1) Monthly settled volume per partner with month-over-month growth
WITH monthly AS (
    SELECT partner_id,
           DATEFROMPARTS(YEAR(txn_date), MONTH(txn_date), 1) AS month_start,
           SUM(settled_volume_eur) AS settled_eur
    FROM dbo.daily_partner_volume
    GROUP BY partner_id, DATEFROMPARTS(YEAR(txn_date), MONTH(txn_date), 1)
)
SELECT partner_id, month_start, settled_eur,
       LAG(settled_eur) OVER (PARTITION BY partner_id ORDER BY month_start) AS prev_month_eur,
       ROUND(100.0 * (settled_eur - LAG(settled_eur) OVER (PARTITION BY partner_id ORDER BY month_start))
             / NULLIF(LAG(settled_eur) OVER (PARTITION BY partner_id ORDER BY month_start), 0), 2) AS mom_growth_pct
FROM monthly
ORDER BY partner_id, month_start;

-- 2) Top 3 transaction types per partner by volume
SELECT *
FROM (
    SELECT partner_id, txn_type, SUM(gross_volume_eur) AS volume_eur,
           ROW_NUMBER() OVER (PARTITION BY partner_id ORDER BY SUM(gross_volume_eur) DESC) AS rn
    FROM dbo.daily_partner_volume
    GROUP BY partner_id, txn_type
) x
WHERE rn <= 3
ORDER BY partner_id, rn;

-- 3) Partner-days with an unusually high failure rate (mean + 2 std dev)
WITH daily AS (
    SELECT partner_id, txn_date,
           SUM(failure_rate * txn_count) * 1.0 / NULLIF(SUM(txn_count), 0) AS failure_rate
    FROM dbo.daily_partner_volume
    GROUP BY partner_id, txn_date
),
stats AS (
    SELECT partner_id, AVG(failure_rate) AS mu, STDEV(failure_rate) AS sd
    FROM daily GROUP BY partner_id
)
SELECT d.partner_id, d.txn_date, ROUND(d.failure_rate, 4) AS failure_rate
FROM daily d JOIN stats s ON s.partner_id = d.partner_id
WHERE d.failure_rate > s.mu + 2 * s.sd
ORDER BY d.failure_rate DESC;

-- 4) Fraud alerts per partner per month
SELECT partner_id,
       FORMAT(txn_date, 'yyyy-MM') AS month,
       COUNT(DISTINCT account_id)  AS flagged_accounts,
       SUM(amount_eur)             AS flagged_volume_eur
FROM dbo.fraud_velocity_alerts
GROUP BY partner_id, FORMAT(txn_date, 'yyyy-MM')
ORDER BY month, flagged_volume_eur DESC;
