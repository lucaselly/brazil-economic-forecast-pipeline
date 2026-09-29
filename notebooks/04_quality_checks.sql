-- Data quality checks. Run manually in Databricks after building the layers.

-- Silver: row counts and current-month flag (expected: 1701, 1701, 1)
SELECT
  (SELECT COUNT(*) FROM workspace.economic_forecast.bronze_bcb_sgs) AS bronze_rows,
  (SELECT COUNT(*) FROM workspace.economic_forecast.silver_bcb_sgs) AS silver_rows,
  (SELECT COUNT(*) FROM workspace.economic_forecast.silver_bcb_sgs WHERE is_current_month) AS current_month_rows;

-- Silver: missing months and value ranges (n_rows must equal expected_rows)
SELECT
  series_code,
  series_name,
  COUNT(*)                                                      AS n_rows,
  CAST(MONTHS_BETWEEN(MAX(ref_date), MIN(ref_date)) AS INT) + 1 AS expected_rows,
  MIN(ref_date)                                                 AS first_date,
  MAX(ref_date)                                                 AS last_date,
  ROUND(MIN(value), 2)                                          AS min_value,
  ROUND(MAX(value), 2)                                          AS max_value
FROM workspace.economic_forecast.silver_bcb_sgs
GROUP BY series_code, series_name
ORDER BY series_code;

-- Gold: row count, date range and nulls (null_mom = 1 and null_yoy = 12 are expected)
SELECT
  COUNT(*)                              AS n_rows,
  MIN(ref_date)                         AS first_date,
  MAX(ref_date)                         AS last_date,
  COUNT(*) - COUNT(ibc_br_industry)     AS null_target,
  COUNT(*) - COUNT(ibc_br_industry_sa)  AS null_industry_sa,
  COUNT(*) - COUNT(ibc_br)              AS null_ibc_br,
  COUNT(*) - COUNT(ibc_br_sa)           AS null_ibc_br_sa,
  COUNT(*) - COUNT(ipca_monthly_pct)    AS null_ipca,
  COUNT(*) - COUNT(selic_monthly_pct)   AS null_selic,
  COUNT(*) - COUNT(industry_sa_mom_pct) AS null_mom,
  COUNT(*) - COUNT(industry_yoy_pct)    AS null_yoy
FROM workspace.economic_forecast.gold_economic_monthly;

-- Gold: latest rows, for a visual sanity check
SELECT * FROM workspace.economic_forecast.gold_economic_monthly
ORDER BY ref_date DESC
LIMIT 6;