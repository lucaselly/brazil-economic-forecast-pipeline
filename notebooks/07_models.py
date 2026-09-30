# %pip install statsmodels

# Calendar-aware regression models vs. the seasonal naive baseline, tracked with MLflow.
# Run in a Databricks notebook (uses spark and display).
import mlflow
import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_MONTHS = 120
HORIZONS = [1, 12]

df = spark.table("workspace.economic_forecast.gold_economic_monthly").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
df = df.set_index("ref_date").sort_index().asfreq("MS")
y = df["ibc_br_industry"]
n = len(y)


# ---- Calendar features (known in advance for any month) ----
def holidays_for(year):
    e = pd.Timestamp(easter(year))
    fixed = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25)]
    if year >= 2024:
        fixed.append((11, 20))
    days = [pd.Timestamp(year, m, d) for m, d in fixed]
    days.append(e - pd.Timedelta(days=2))                           # Good Friday
    days += [e - pd.Timedelta(days=48), e - pd.Timedelta(days=47)]  # Carnival Monday and Tuesday
    days.append(e + pd.Timedelta(days=60))                          # Corpus Christi
    return days


hol = np.array([d.date() for yr in range(2001, 2028) for d in holidays_for(yr)], dtype="datetime64[D]")

months = pd.date_range("2002-01-01", "2027-12-01", freq="MS")
starts = months.values.astype("datetime64[D]")
ends = (months + pd.offsets.MonthBegin(1)).values.astype("datetime64[D]")
cal = pd.DataFrame({"bdays": np.busday_count(starts, ends, holidays=hol)}, index=months)
carnival = {(pd.Timestamp(easter(yr)) - pd.Timedelta(days=47)).to_period("M") for yr in range(2001, 2028)}
good_friday = {(pd.Timestamp(easter(yr)) - pd.Timedelta(days=2)).to_period("M") for yr in range(2001, 2028)}
cal["carnival"] = [int(m.to_period("M") in carnival) for m in months]
cal["easter"] = [int(m.to_period("M") in good_friday) for m in months]
for c in ["bdays", "carnival", "easter"]:
    cal[f"d_{c}"] = cal[c] - cal[c].shift(12)  # difference vs. the same month last year

# ---- Target and features ----
g = np.log(y).diff(12)  # year-over-year log growth: the quantity we model
feat = cal.loc[y.index, ["d_bdays", "d_carnival", "d_easter"]].copy()
feat["g_recent"] = g.rolling(3).mean()
feat["ipca_3m"] = df["ipca_monthly_pct"].rolling(3).mean()


def make_x(h, use_ipca):
    """Calendar of the target month + information available h months earlier."""
    X = feat[["d_bdays", "d_carnival", "d_easter"]].copy()
    X["g_recent"] = feat["g_recent"].shift(h)
    if use_ipca:
        X["ipca_3m"] = feat["ipca_3m"].shift(h)
    return X


def seasonal_naive_forecasts(h):
    return [(t, y.iloc[t - 12]) for t in range(n - TEST_MONTHS, n)]


def ridge_forecasts(h, use_ipca):
    X = make_x(h, use_ipca)
    out = []
    for t in range(n - TEST_MONTHS, n):
        o = t - h  # last month observed at forecast time
        Xtr, ytr = X.iloc[: o + 1], g.iloc[: o + 1]
        ok = Xtr.notna().all(axis=1) & ytr.notna()
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr[ok], ytr[ok])
        g_hat = model.predict(X.iloc[[t]])[0]
        out.append((t, y.iloc[t - 12] * np.exp(g_hat)))
    return out


def summarize(rows):
    fc = pd.DataFrame(rows, columns=["t", "forecast"])
    fc["actual"] = y.iloc[fc["t"]].values
    err = fc["forecast"] - fc["actual"]
    return {"MAE": err.abs().mean(),
            "WAPE_pct": err.abs().sum() / fc["actual"].sum() * 100,
            "bias": err.mean()}


# ---- Evaluate and track with MLflow ----
runs = {
    "seasonal_naive": seasonal_naive_forecasts,
    "ridge_calendar": lambda h: ridge_forecasts(h, use_ipca=False),
    "ridge_calendar_ipca": lambda h: ridge_forecasts(h, use_ipca=True),
}
results = []
for name, fn in runs.items():
    for h in HORIZONS:
        m = summarize(fn(h))
        with mlflow.start_run(run_name=f"{name}_h{h}_w{TEST_MONTHS}"):
            mlflow.log_params({"model": name, "horizon": h, "test_months": TEST_MONTHS})
            mlflow.log_metrics(m)
        results.append({"model": name, "horizon": h, **m})

display(pd.DataFrame(results).round(2).sort_values(["horizon", "MAE"]))

# What did the model learn? (coefficients on standardized features, horizon 1, all data)
X1 = make_x(1, True)
ok = X1.notna().all(axis=1) & g.notna()
fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X1[ok], g[ok])
print(dict(zip(X1.columns, fitted[-1].coef_.round(4))))

# Paired comparison: calendar ridge vs. seasonal naive on the same test months.
def abs_errors(fn, h):
    rows = fn(h)
    idx = [r[0] for r in rows]
    fcst = np.array([r[1] for r in rows])
    return y.index[idx], np.abs(fcst - y.iloc[idx].values)


def block_ci(x, block=6, n_boot=5000, seed=42):
    """95% CI of the mean using a moving-block bootstrap (keeps neighbouring months together)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        means.append(np.concatenate([x[s:s + block] for s in starts])[:n].mean())
    return np.percentile(means, [2.5, 97.5])


for h in HORIZONS:
    dates, e_base = abs_errors(seasonal_naive_forecasts, h)
    _, e_model = abs_errors(lambda hh: ridge_forecasts(hh, use_ipca=False), h)
    diff = e_base - e_model  # positive = the model is better
    no2020 = ~((dates >= "2020-03-01") & (dates <= "2020-12-01"))
    for label, d in [("all months", diff), ("excluding Mar-Dec 2020", diff[no2020])]:
        lo, hi = block_ci(d)
        print(f"h={h} | {label} (n={len(d)}): model better in {(d > 0).mean():.0%} of months; "
              f"mean MAE gain {d.mean():.2f} (95% block-bootstrap CI {lo:.2f} to {hi:.2f})")

# Extra models: SARIMA and gradient boosting, compared against the seasonal naive baseline.
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import HistGradientBoostingRegressor


def sarima_forecasts(h):
    """Seasonal ARIMA(1,1,1)(0,1,1,12), fixed in advance and refit at every forecast origin."""
    out = []
    for t in range(n - TEST_MONTHS, n):
        o = t - h
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y.iloc[: o + 1], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
            out.append((t, float(res.forecast(steps=h).iloc[-1])))
    return out


def gbm_forecasts(h):
    """Gradient boosting on the same features as the ridge (calendar + recent growth)."""
    X = make_x(h, use_ipca=False)
    out = []
    for t in range(n - TEST_MONTHS, n):
        o = t - h
        Xtr, ytr = X.iloc[: o + 1], g.iloc[: o + 1]
        ok = Xtr.notna().all(axis=1) & ytr.notna()
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0
        ).fit(Xtr[ok], ytr[ok])
        out.append((t, y.iloc[t - 12] * np.exp(model.predict(X.iloc[[t]])[0])))
    return out


all_models = {
    "seasonal_naive": seasonal_naive_forecasts,
    "ridge_calendar": lambda h: ridge_forecasts(h, use_ipca=False),
    "ridge_calendar_ipca": lambda h: ridge_forecasts(h, use_ipca=True),
    "sarima_111_011_12": sarima_forecasts,
    "gbm_calendar": gbm_forecasts,
}
NEW_MODELS = {"sarima_111_011_12", "gbm_calendar"}

# Compute every forecast once and reuse it below
forecasts = {(name, h): fn(h) for name, fn in all_models.items() for h in HORIZONS}

# Log only the new models to MLflow (the others were logged above)
for (name, h), rows in forecasts.items():
    if name in NEW_MODELS:
        with mlflow.start_run(run_name=f"{name}_h{h}_w{TEST_MONTHS}"):
            mlflow.log_params({"model": name, "horizon": h, "test_months": TEST_MONTHS})
            mlflow.log_metrics(summarize(rows))

table = [{"model": name, "horizon": h, **summarize(rows)} for (name, h), rows in forecasts.items()]
display(pd.DataFrame(table).round(2).sort_values(["horizon", "MAE"]))


# Paired comparison of every model against the seasonal naive baseline
def abs_errors_from(rows):
    idx = [r[0] for r in rows]
    fcst = np.array([r[1] for r in rows])
    return y.index[idx], np.abs(fcst - y.iloc[idx].values)


for h in HORIZONS:
    dates, e_base = abs_errors_from(forecasts[("seasonal_naive", h)])
    no2020 = ~((dates >= "2020-03-01") & (dates <= "2020-12-01"))
    for name in all_models:
        if name == "seasonal_naive":
            continue
        _, e_model = abs_errors_from(forecasts[(name, h)])
        diff = e_base - e_model  # positive = the model is better
        for label, d in [("all", diff), ("ex-2020", diff[no2020])]:
            lo, hi = block_ci(d)
            print(f"h={h} | {name:20s} | {label:7s}: better in {(d > 0).mean():.0%} of months, "
                  f"MAE gain {d.mean():+.2f} (95% CI {lo:+.2f} to {hi:+.2f})")

# Exploratory: equal-weight average of SARIMA and the calendar ridge (chosen after seeing the results).
ensemble = {}
for h in HORIZONS:
    a = forecasts[("sarima_111_011_12", h)]
    b = forecasts[("ridge_calendar", h)]
    ensemble[h] = [(t, (fa + fb) / 2) for (t, fa), (_, fb) in zip(a, b)]

for h in HORIZONS:
    rows = ensemble[h]
    m = summarize(rows)
    with mlflow.start_run(run_name=f"ensemble_sarima_ridge_h{h}_w{TEST_MONTHS}"):
        mlflow.log_params({"model": "ensemble_sarima_ridge", "horizon": h, "test_months": TEST_MONTHS})
        mlflow.log_metrics(m)

    dates, e_base = abs_errors_from(forecasts[("seasonal_naive", h)])
    _, e_ens = abs_errors_from(rows)
    diff = e_base - e_ens
    no2020 = ~((dates >= "2020-03-01") & (dates <= "2020-12-01"))
    print(f"h={h} | ensemble: MAE {m['MAE']:.2f}, WAPE {m['WAPE_pct']:.2f}%, bias {m['bias']:+.2f}")
    for label, d in [("all", diff), ("ex-2020", diff[no2020])]:
        lo, hi = block_ci(d)
        print(f"      {label:7s}: better in {(d > 0).mean():.0%} of months, "
              f"MAE gain {d.mean():+.2f} (95% CI {lo:+.2f} to {hi:+.2f})")

# Stability check: ensemble vs. baseline in each half of the test window (horizon 1).
dates, e_base = abs_errors_from(forecasts[("seasonal_naive", 1)])
_, e_ens = abs_errors_from(ensemble[1])
half = len(dates) // 2
for label, sl in [("first 60 months", slice(0, half)), ("last 60 months", slice(half, None))]:
    print(f"{label}: baseline MAE {e_base[sl].mean():.2f} | ensemble MAE {e_ens[sl].mean():.2f}")