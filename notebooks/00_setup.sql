-- Run once in Databricks before the pipeline.
-- Creates the schema and the volume that receive the raw Parquet files.
CREATE SCHEMA IF NOT EXISTS workspace.economic_forecast;
CREATE VOLUME IF NOT EXISTS workspace.economic_forecast.raw;