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

Sources (BCB and IBGE APIs) -> Python extraction -> Databricks
(bronze / silver / gold) -> forecasting models (MLflow) -> Power BI

## Data sources

| Source | Data | Access |
|---|---|---|
| Banco Central do Brasil (SGS) | IPCA, Selic, IBC-Br, IBC-Br Industry (target) | Public API, no key |
| IBGE (SIDRA) | Industrial production (PIM-PF), planned as a second source | Public API, no key |

## Tech stack

Python, pandas, Databricks (Free Edition), Delta Lake, MLflow,
scikit-learn / statsmodels, Power BI.

## Project structure

    src/         extraction and transformation code
    notebooks/   Databricks and exploration notebooks
    tests/       data quality and unit tests
    docs/        diagrams and model card

## Getting started

_Setup instructions will be added as each phase is completed._

## Roadmap

- [x] Phase 0: repository and project structure
- [ ] Phase 1: data extraction from BCB and IBGE APIs
- [ ] Phase 2: Databricks bronze / silver / gold layers
- [ ] Phase 3: data quality checks
- [ ] Phase 4: forecasting models and experiment tracking
- [ ] Phase 5: model monitoring
- [ ] Phase 6: Power BI dashboard
- [ ] Phase 7: scheduling, documentation and model card

## License

MIT
