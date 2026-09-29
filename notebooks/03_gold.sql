-- Gold layer: one row per month, one column per series, aligned to the target's last month.
CREATE OR REPLACE TABLE workspace.economic_forecast.gold_economic_monthly AS
WITH bounds AS (
  SELECT MAX(ref_date) AS last_target_date
  FROM workspace.economic_forecast.silver_bcb_sgs
  WHERE series_code = 29603
),
pivoted AS (
  SELECT
    s.ref_date,
    MAX(CASE WHEN s.series_code = 29603 THEN s.value END) AS ibc_br_industry,
    MAX(CASE WHEN s.series_code = 29604 THEN s.value END) AS ibc_br_industry_sa,
    MAX(CASE WHEN s.series_code = 24363 THEN s.value END) AS ibc_br,
    MAX(CASE WHEN s.series_code = 24364 THEN s.value END) AS ibc_br_sa,
    MAX(CASE WHEN s.series_code = 433   THEN s.value END) AS ipca_monthly_pct,
    MAX(CASE WHEN s.series_code = 4390  THEN s.value END) AS selic_monthly_pct
  FROM workspace.economic_forecast.silver_bcb_sgs s
  CROSS JOIN bounds b
  WHERE NOT s.is_current_month
    AND s.ref_date <= b.last_target_date
  GROUP BY s.ref_date
)
SELECT
  ref_date,
  YEAR(ref_date)  AS ref_year,
  MONTH(ref_date) AS ref_month,
  ibc_br_industry,
  ibc_br_industry_sa,
  ibc_br,
  ibc_br_sa,
  ipca_monthly_pct,
  selic_monthly_pct,
  ROUND((ibc_br_industry_sa / LAG(ibc_br_industry_sa, 1) OVER (ORDER BY ref_date) - 1) * 100, 2) AS industry_sa_mom_pct,
  ROUND((ibc_br_industry    / LAG(ibc_br_industry, 12)    OVER (ORDER BY ref_date) - 1) * 100, 2) AS industry_yoy_pct
FROM pivoted;