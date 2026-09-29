# Brazil Economic Forecast Pipeline

End-to-end data project: ingest public Brazilian economic data, model it in
Databricks (medallion architecture), forecast industrial production with
tracked experiments, and present the results in Power BI.

> **Status:** in progress. See the roadmap below.

## Business question

How does industrial production behave over time, and can macroeconomic
indicators (inflation, interest rates, economic activity) help forecast it?
The goal is to compare forecasting approaches with proper time-series
validation and pick the one that best supports planning decisions.

## Architecture

_Diagram coming soon (docs/architecture.png)._

Sources (BCB SGS API; IBGE planned) -> Python extraction -> Databricks
(bronze / silver / gold) -> forecasting models (MLflow) -> Power BI

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

## Exploratory findings

- The target (IBC-Br Industry) has no stable trend: growth to 2011-2013, a sharp fall to 2016, a flat regime near 100 until 2021 and a mild recovery since.
- Seasonality is strong (about ±8-10 index points): January, February and December are low; August and October are high.
- April 2020 is the lowest month in the history (77.3); the shock left no level shift and is treated as an outlier.
- Inflation and interest rates are weakly related to industrial growth. IPCA lagged 2-4 months shows a consistent negative correlation (about -0.2, robust to excluding 2020 and to rank correlation); Selic shows none. Most of the predictive signal is expected to come from the series' own history.

## Validation design

- Rolling-origin evaluation on the last 60 months (Aug 2021 to Jul 2026); models never see the future.
- Horizons: 1 and 12 months after the last observed month (the IBC-Br itself is published with about a two-month lag).
- Metrics: MAE, WAPE and bias.
- Baseline to beat: seasonal naive (same month last year), WAPE about 1.9%.

## Tech stack

Python, pandas, Databricks (Free Edition), Delta Lake, MLflow,
scikit-learn / statsmodels, Power BI.

## Project structure

    src/         extraction code
    notebooks/   Databricks notebooks: pipeline layers and data quality tests
    tests/       unit tests for the extraction code (planned)
    docs/        diagrams and model card

## Getting started

_Setup instructions will be added as each phase is completed._

## Roadmap

- [x] Phase 0: repository and project structure
- [x] Phase 1: data extraction from the BCB SGS API
- [x] Phase 2: Databricks bronze / silver / gold layers
- [x] Phase 3: data quality checks
- [ ] Phase 4: forecasting models and experiment tracking
- [ ] Phase 5: model monitoring
- [ ] Phase 6: Power BI dashboard
- [ ] Phase 7: scheduling, documentation and model card

## License

MIT
