# Databricks notebook: install the extra dependency in its own first cell.
# %pip install statsmodels

# Exploratory analysis of the target series (IBC-Br Industry).
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

df = spark.table("workspace.economic_forecast.gold_economic_monthly").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
df = df.set_index("ref_date").sort_index().asfreq("MS")
y = df["ibc_br_industry"]

# 1. The series over time, with 2020 highlighted
fig, ax = plt.subplots(figsize=(11, 4))
y.plot(ax=ax)
ax.axvspan("2020-03-01", "2020-12-31", alpha=0.2, color="red")
ax.set_title("IBC-Br Industry (index), monthly")
plt.show()
print("Lowest month:", y.idxmin().date(), "| value:", round(y.min(), 1))

# 2. Trend / seasonal / remainder
res = STL(y, period=12, robust=True).fit()
res.plot()
plt.show()

# 3. Average seasonal effect by calendar month
seasonal_by_month = res.seasonal.groupby(res.seasonal.index.month).mean()
seasonal_by_month.plot.bar(figsize=(8, 3), title="Average seasonal effect by month")
plt.show()

# 4. Do inflation and interest rates relate to industry growth?
#    We correlate month-over-month growth (not levels), because trending
#    levels produce spurious correlations.
predictors = ["ipca_monthly_pct", "selic_monthly_pct"]
corr = pd.DataFrame({
    lag: {c: df["industry_sa_mom_pct"].corr(df[c].shift(lag)) for c in predictors}
    for lag in range(0, 7)
}).round(2)
corr.columns = [f"lag {l}" for l in corr.columns]
display(corr.reset_index().rename(columns={"index": "predictor"}))