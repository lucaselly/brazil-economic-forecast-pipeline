# Baseline forecasts with rolling-origin evaluation on the last 60 months.
# Run in a Databricks notebook. No extra packages needed.
import numpy as np
import pandas as pd

TEST_MONTHS = 60
HORIZONS = [1, 12]

df = spark.table("workspace.economic_forecast.gold_economic_monthly").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
y = df.set_index("ref_date").sort_index().asfreq("MS")["ibc_br_industry"]
n = len(y)
test_idx = range(n - TEST_MONTHS, n)


# Every function forecasts month t using only data up to the origin o = t - h.
def naive(y, t, h):
    """Repeat the last observed value."""
    return y.iloc[t - h]


def seasonal_naive(y, t, h):
    """Repeat the value from the same month last year."""
    return y.iloc[t - 12]


def seasonal_avg_3y(y, t, h):
    """Average of the same month over the previous three years."""
    return np.mean([y.iloc[t - 12 * k] for k in (1, 2, 3)])


def seasonal_naive_level_adj(y, t, h):
    """Same month last year, scaled by how the last 12 months compare with the 12 before."""
    o = t - h
    recent = y.iloc[o - 11:o + 1].mean()
    previous = y.iloc[o - 23:o - 11].mean()
    return y.iloc[t - 12] * recent / previous


models = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_avg_3y": seasonal_avg_3y,
    "seasonal_naive_level_adj": seasonal_naive_level_adj,
}

rows = []
for name, fn in models.items():
    for h in HORIZONS:
        for t in test_idx:
            rows.append({"model": name, "horizon": h, "ref_date": y.index[t],
                         "actual": y.iloc[t], "forecast": fn(y, t, h)})
fc = pd.DataFrame(rows)
fc["error"] = fc["forecast"] - fc["actual"]

g = fc.assign(abs_error=fc["error"].abs()).groupby(["horizon", "model"]).agg(
    MAE=("abs_error", "mean"), sum_abs=("abs_error", "sum"),
    sum_actual=("actual", "sum"), bias=("error", "mean"))
g["WAPE_%"] = g["sum_abs"] / g["sum_actual"] * 100
results = g[["MAE", "WAPE_%", "bias"]].round(2).reset_index().sort_values(["horizon", "MAE"])
display(results)

# Visual check: 1-month-ahead forecasts against the actual series
piv = fc[fc["horizon"] == 1].pivot(index="ref_date", columns="model", values="forecast")
piv["actual"] = y.loc[piv.index]
piv[["actual", "naive", "seasonal_naive"]].plot(figsize=(11, 4), title="1-month-ahead forecasts")