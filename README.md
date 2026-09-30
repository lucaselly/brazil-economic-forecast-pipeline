# Brazil Economic Forecast Pipeline

End-to-end data project: ingest public Brazilian economic data, model it in
Databricks (medallion architecture), forecast industrial activity with
tracked experiments, and present the results in Power BI.

> **Status:** in progress. The data pipeline, quality tests and forecasting
> models are done; monitoring and the Power BI dashboard are next.

## Business question

How does industrial activity behave over time, and can macroeconomic
indicators (inflation, interest rates, economic activity) help forecast it?
The goal is to compare forecasting approaches with proper time-series
validation and pick the one that best supports planning decisions.

## Architecture

```
BCB SGS API -> src/extract_bcb.py -> Parquet files (Databricks volume)
            -> bronze -> silver -> gold (Delta tables)
            -> models: seasonal naive, SARIMA, ridge, gradient boosting -> MLflow
            -> [planned] forecast log and monitoring -> Power BI
```

## Data sources

| Source | Data | Access |
|---|---|---|
| Banco Central do Brasil (SGS) | IPCA, Selic, IBC-Br, IBC-Br Industry (target) | Public API, no key |
| IBGE (SIDRA) | Industrial production (PIM-PF), planned as a second source | Public API, no key |

## Data layers

| Layer | Table | Description |
|---|---|---|
| Bronze | `bronze_bcb_sgs` | Raw API data as ingested (text values), with source file |
| Silver | `silver_bcb_sgs` | Typed columns, deduplicated, current month flagged |
| Gold | `gold_economic_monthly` | One row per month, one column per series, aligned to the target's last month |

Target variable: `ibc_br_industry` (IBC-Br Industry index, monthly, 2003 onward).
The last, still-incomplete month of each series is excluded from the gold layer.

## Data quality

`04_data_quality_tests` runs 17 automated checks across the three layers
(series present, no duplicates, no lost rows, no missing months, plausible
value ranges, no nulls, current month excluded, data freshness). If any check
fails, the notebook raises an error, so a scheduled job would be marked as failed.

## Exploratory findings

- The target (IBC-Br Industry) has no stable trend: growth to 2011-2013, a sharp fall to 2016, a flat regime near 100 until 2021 and a mild recovery since.
- Seasonality is strong (about ±8-10 index points): January, February and December are low; August and October are high.
- April 2020 is the lowest month in the history (77.3); the shock left no level shift and is treated as an outlier.
- Inflation and interest rates are weakly related to industrial growth. IPCA lagged 2-4 months shows a consistent negative correlation (about -0.2, robust to excluding 2020 and to rank correlation); Selic shows none.

## Validation design

- Rolling-origin evaluation on the last 120 months (Aug 2016 to Jul 2026). Every model is refit at each forecast origin using only past data.
- Horizons: 1 and 12 months after the last observed month (the IBC-Br itself is published with about a two-month lag).
- Metrics: MAE, WAPE and bias.
- Significance: paired comparison of absolute errors against the baseline, with 95% confidence intervals from a moving-block bootstrap (6-month blocks), also reported excluding Mar-Dec 2020.
- Baseline to beat: seasonal naive (same month last year), WAPE 2.71%.

## Models

- **Seasonal naive** (baseline): the value of the same month last year.
- **SARIMA(1,1,1)(0,1,1,12)**: order fixed in advance, not tuned on the test window.
- **Ridge + calendar**: models the year-over-year log growth using differences versus the same month last year in business days, Easter and Carnival, plus recent growth. Forecast = last year's value x exp(predicted growth).
- **Ridge + calendar + IPCA**: the same model with lagged inflation added.
- **Gradient boosting + calendar**: same features as the ridge, conservative settings (depth 3, 200 iterations).
- **Average of SARIMA and ridge** (exploratory): equal-weight average, chosen after seeing the individual results.

## Results (120-month rolling-origin test, Aug 2016 to Jul 2026)

| Model | h=1 MAE | h=1 WAPE % | h=12 MAE | h=12 WAPE % |
|---|---|---|---|---|
| Seasonal naive (baseline) | 2.73 | 2.71 | 2.73 | 2.71 |
| SARIMA(1,1,1)(0,1,1,12) | 2.14 | 2.12 | 2.79 | 2.76 |
| Ridge + calendar | 2.34 | 2.32 | 2.91 | 2.88 |
| Ridge + calendar + IPCA | 2.36 | 2.34 | 3.38 | 3.35 |
| Gradient boosting + calendar | 2.43 | 2.41 | 2.97 | 2.94 |
| Average of SARIMA and ridge (exploratory) | 2.03 | 2.01 | 2.76 | 2.74 |

MAE gain over the baseline at 1 month ahead (positive = better), with 95% block-bootstrap intervals:

| Model | Better in % of months | All months | Excluding Mar-Dec 2020 |
|---|---|---|---|
| Ridge + calendar | 63% | +0.39 (-0.12 to +0.79) | +0.49 (+0.05 to +0.88) |
| Ridge + calendar + IPCA | 62% | +0.37 (-0.15 to +0.77) | +0.47 (+0.04 to +0.87) |
| SARIMA | 57% | +0.59 (-0.08 to +1.50) | +0.65 (-0.01 to +1.62) |
| Gradient boosting + calendar | 56% | +0.30 (-0.14 to +0.69) | +0.31 (-0.08 to +0.68) |
| Average of SARIMA and ridge | 62% | +0.70 (+0.20 to +1.25) | +0.80 (+0.38 to +1.36) |

Findings:

- One month ahead, SARIMA, ridge and gradient boosting all beat the seasonal naive baseline, cutting MAE by about 22%, 14% and 11%.
- SARIMA has the lowest average error among single models but wins in fewer months (57%) than the ridge (63%): its gain comes from large improvements in a few months.
- Evidence is suggestive, not conclusive: for single models the intervals include zero, except the ridge variants when Mar-Dec 2020 is left out.
- The average of SARIMA and ridge reached MAE 2.03 (about 26% below the baseline) and is the only candidate whose interval excludes zero with 2020 included. Because it was chosen after seeing the individual results, it is treated as exploratory.
- The average beats the baseline in both halves of the test window: MAE 2.73 versus 3.51 from Aug 2016 to Jul 2021 (which includes 2020) and 1.32 versus 1.94 from Aug 2021 to Jul 2026.
- Twelve months ahead no model beats the seasonal naive baseline.
- IPCA, despite a weak correlation in the exploratory analysis, did not improve forecasts. Gradient boosting did not beat the linear model with about 280 observations.
- All runs are tracked in MLflow (experiment `07_models`).

## Limitations

- A single series over one period; the window includes the 2020 shock.
- The calendar treats Carnival and Corpus Christi as non-working days and ignores state and municipal holidays.
- Models use fixed settings, so accuracy may be left on the table, but the test window is not used for tuning.
- Results are retrospective; a prospective evaluation with the monitoring phase is planned.

## Tech stack

Python, pandas, PyArrow, statsmodels, scikit-learn, Databricks (Free Edition),
Delta Lake, MLflow, Power BI (planned).

## Project structure

```
src/extract_bcb.py   extraction of the BCB SGS series to Parquet
notebooks/           Databricks notebooks, exported as source files
tests/               planned: unit tests for the extraction code
docs/                planned: diagrams and model card
```

| Notebook | Purpose |
|---|---|
| `00_setup.sql` | Creates the schema and the volume |
| `01_bronze.py` | Raw Parquet files to the bronze table |
| `02_silver.sql` | Typed, deduplicated silver table |
| `03_gold.sql` | Wide monthly gold table |
| `04_data_quality_tests.py` | Automated data quality checks |
| `05_exploration.py` | Exploratory analysis of the target series |
| `06_baselines.py` | Baseline forecasts with rolling-origin evaluation |
| `07_models.py` | SARIMA, ridge, gradient boosting, average and significance tests |

## Getting started

1. Create the environment (Python 3.12) and extract the data:

```
git clone https://github.com/lucaselly/brazil-economic-forecast-pipeline.git
cd brazil-economic-forecast-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/extract_bcb.py
```

2. In Databricks Free Edition, create the schema and volume with `notebooks/00_setup.sql`, and upload the Parquet files from `data/raw/bcb/` to the volume `workspace.economic_forecast.raw`.
3. Copy each notebook file into a Databricks notebook (Python for `.py`, SQL for `.sql`) and run them in numeric order. Notebooks `05_exploration` and `07_models` need `%pip install statsmodels` in their first cell.

## Roadmap

- [x] Phase 0: repository and project structure
- [x] Phase 1: data extraction from the BCB SGS API
- [x] Phase 2: Databricks bronze / silver / gold layers
- [x] Phase 3: data quality checks
- [x] Phase 4: forecasting models and experiment tracking
- [ ] Phase 5: model monitoring
- [ ] Phase 6: Power BI dashboard
- [ ] Phase 7: scheduling, documentation and model card

## License

MIT
