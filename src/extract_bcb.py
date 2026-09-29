"""Extract monthly series from the BCB SGS open-data API into raw Parquet files."""
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
RAW_DIR = Path("data/raw/bcb")
START_YEAR = 2003
WINDOW_YEARS = 10  # BCB limits each query to 10 years

SERIES = {
    433: "ipca_monthly_pct",
    4390: "selic_accumulated_month_pct",
    24363: "ibc_br",
    24364: "ibc_br_seasonally_adjusted",
    29603: "ibc_br_industry",
    29604: "ibc_br_industry_seasonally_adjusted",
}


def year_windows(start_year: int, end_year: int, size: int):
    """Yield (start, end) date windows of at most `size` years."""
    for year in range(start_year, end_year + 1, size):
        yield date(year, 1, 1), date(min(year + size - 1, end_year), 12, 31)


def fetch_window(code: int, start: date, end: date, retries: int = 3) -> list[dict]:
    """Fetch one date window, retrying with exponential backoff."""
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(BASE_URL.format(code=code), params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    return []


def extract_series(code: int, name: str) -> pd.DataFrame:
    """Download the full history of one series as a raw DataFrame."""
    records = []
    for start, end in year_windows(START_YEAR, date.today().year, WINDOW_YEARS):
        records.extend(fetch_window(code, start, end))
        time.sleep(1)  # be polite with the API

    df = pd.DataFrame(records)
    df["series_code"] = code
    df["series_name"] = name
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return df.drop_duplicates(subset=["data"])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for code, name in SERIES.items():
        df = extract_series(code, name)
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name} ({code}): {len(df)} rows, {df['data'].iloc[0]} -> {df['data'].iloc[-1]}")


if __name__ == "__main__":
    main()