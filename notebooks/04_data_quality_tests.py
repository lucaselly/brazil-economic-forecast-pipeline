# Data quality tests: run in a Databricks notebook after the gold layer is built.
# Every check must pass. If any check fails, the notebook raises an error at the end.
import pandas as pd

catalog, schema = "workspace", "economic_forecast"
bronze = f"{catalog}.{schema}.bronze_bcb_sgs"
silver = f"{catalog}.{schema}.silver_bcb_sgs"
gold = f"{catalog}.{schema}.gold_economic_monthly"

checks = []


def scalar(query):
    return spark.sql(query).collect()[0][0]


def check(name, passed, detail=""):
    checks.append((name, bool(passed), detail))


# ---- Bronze ----
n_bronze = scalar(f"SELECT COUNT(*) FROM {bronze}")
n_bronze_dupes = scalar(f"SELECT COUNT(*) - COUNT(DISTINCT series_code, data) FROM {bronze}")
check("bronze: no duplicate (series, month)", n_bronze_dupes == 0, f"{n_bronze_dupes} duplicates")

n_series = scalar(f"SELECT COUNT(DISTINCT series_code) FROM {bronze}")
check("bronze: 6 series present", n_series == 6, f"found {n_series}")

n_bad = scalar(f"""
    SELECT COUNT(*) FROM {bronze}
    WHERE TRY_TO_DATE(data, 'dd/MM/yyyy') IS NULL OR TRY_CAST(valor AS DOUBLE) IS NULL
""")
check("bronze: all dates and values parseable", n_bad == 0, f"{n_bad} bad rows")

# ---- Silver ----
n_silver = scalar(f"SELECT COUNT(*) FROM {silver}")
check("silver: no rows lost from bronze", n_bronze == n_silver, f"bronze {n_bronze}, silver {n_silver}")

n_gaps = scalar(f"""
    SELECT COUNT(*) FROM (
      SELECT series_code FROM {silver}
      GROUP BY series_code
      HAVING COUNT(*) <> CAST(MONTHS_BETWEEN(MAX(ref_date), MIN(ref_date)) AS INT) + 1
    )
""")
check("silver: no missing months in any series", n_gaps == 0, f"{n_gaps} series with gaps")

value_ranges = {
    433: (-2, 5),        # IPCA, monthly %
    4390: (0, 5),        # Selic, accumulated in the month, %
    24363: (40, 200),    # IBC-Br index
    24364: (40, 200),
    29603: (40, 200),    # IBC-Br Industry index (target)
    29604: (40, 200),
}
for code, (lo, hi) in value_ranges.items():
    n_out = scalar(f"""
        SELECT COUNT(*) FROM {silver}
        WHERE series_code = {code} AND (value < {lo} OR value > {hi})
    """)
    check(f"silver: series {code} within [{lo}, {hi}]", n_out == 0, f"{n_out} out of range")

# ---- Gold ----
g = spark.sql(f"""
    SELECT COUNT(*) AS n,
           COUNT(DISTINCT ref_date) AS n_distinct,
           MIN(ref_date) AS first_date,
           MAX(ref_date) AS last_date,
           CAST(MONTHS_BETWEEN(MAX(ref_date), MIN(ref_date)) AS INT) + 1 AS expected
    FROM {gold}
""").collect()[0]

check("gold: no duplicate months", g.n == g.n_distinct, f"{g.n} rows, {g.n_distinct} distinct")
check("gold: months are contiguous", g.n == g.expected, f"{g.n} rows, {g.expected} expected")
check("gold: history starts in 2003-01", str(g.first_date) == "2003-01-01", f"starts {g.first_date}")

series_cols = ["ibc_br_industry", "ibc_br_industry_sa", "ibc_br", "ibc_br_sa",
               "ipca_monthly_pct", "selic_monthly_pct"]
null_expr = " + ".join(f"COUNT(*) - COUNT({c})" for c in series_cols)
n_nulls = scalar(f"SELECT {null_expr} FROM {gold}")
check("gold: no nulls in the six series", n_nulls == 0, f"{n_nulls} nulls")

n_current = scalar(f"SELECT COUNT(*) FROM {gold} WHERE ref_date >= TRUNC(CURRENT_DATE(), 'MM')")
check("gold: current (incomplete) month excluded", n_current == 0, f"{n_current} rows")

lag = scalar(f"SELECT CAST(MONTHS_BETWEEN(TRUNC(CURRENT_DATE(), 'MM'), MAX(ref_date)) AS INT) FROM {gold}")
check("gold: data is fresh (latest month at most 4 months old)", lag <= 4, f"latest month is {lag} months old")

# ---- Report ----
report = pd.DataFrame(checks, columns=["check", "passed", "detail"])
display(report)

failed = report[~report["passed"]]
if not failed.empty:
    raise AssertionError(f"{len(failed)} data quality check(s) failed: {list(failed['check'])}")
print("All data quality checks passed.")