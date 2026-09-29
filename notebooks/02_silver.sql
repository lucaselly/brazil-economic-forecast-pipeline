-- Silver layer: typed columns, deduplicated, current month flagged.
CREATE OR REPLACE TABLE workspace.economic_forecast.silver_bcb_sgs AS
WITH parsed AS (
  SELECT
    series_code,
    series_name,
    TRY_TO_DATE(data, 'dd/MM/yyyy')      AS ref_date,
    TRY_CAST(valor AS DOUBLE)            AS value,
    TRY_CAST(ingested_at AS TIMESTAMP)   AS ingested_at
  FROM workspace.economic_forecast.bronze_bcb_sgs
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY series_code, ref_date
      ORDER BY ingested_at DESC
    ) AS rn
  FROM parsed
  WHERE ref_date IS NOT NULL AND value IS NOT NULL
)
SELECT
  series_code,
  series_name,
  ref_date,
  value,
  ref_date = TRUNC(CURRENT_DATE(), 'MM') AS is_current_month,
  ingested_at
FROM ranked
WHERE rn = 1;