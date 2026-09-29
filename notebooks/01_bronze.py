# Bronze layer: run in a Databricks notebook (uses spark and display).
from pyspark.sql import functions as F

catalog, schema = "workspace", "economic_forecast"
raw_path = f"/Volumes/{catalog}/{schema}/raw"

df = (
    spark.read.parquet(f"{raw_path}/*.parquet")
    .select("*", F.col("_metadata.file_name").alias("source_file"))
)

(
    df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.bronze_bcb_sgs")
)

print(f"Rows written: {df.count()}")

display(spark.sql(f"""
    SELECT series_code, series_name,
           COUNT(*) AS n_rows,
           MIN(data) AS first_date, MAX(data) AS last_date
    FROM {catalog}.{schema}.bronze_bcb_sgs
    GROUP BY series_code, series_name
    ORDER BY series_code
"""))