# galderma_sales_forecast_streamlit_app.py
# End-to-end forecasting demo on Kaggle's "Store Sales – Time Series Forecasting"
# Assumes 'store-sales-time-series-forecasting.zip' sits next to this script.

import io
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, Dict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional libs guarded by try/except
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet  # legacy
except Exception:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None

from pandas.tseries.frequencies import to_offset

# ---------- Config ----------
CLIP_FORECASTS_NONNEG = True  # clip all forecasts & lower bands to >= 0
DEFAULT_ZIP = "store-sales-time-series-forecasting.zip"


# ---------- Utils ----------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_eps(y_true, y_pred, eps: float = 1e-6) -> float:
    """MAPE safe against zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def seasonal_period_from_freq(freq: str) -> int:
    f = (freq or "").upper()
    if f.startswith("MS") or f.startswith("M"):
        return 12
    if f.startswith("W"):
        return 52
    if f.startswith("D"):
        return 7
    return 12

def train_test_split_series(y: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    test_size = max(1, min(test_size, len(y)-1))
    return y.iloc[:-test_size], y.iloc[-test_size:]

def sanitize_exog(ex: pd.DataFrame) -> pd.DataFrame:
    """Ensure exogenous matrix has no NaNs/±inf and is numeric."""
    ex = ex.copy()
    for c in ex.columns:
        ex[c] = pd.to_numeric(ex[c], errors="coerce")
    ex = ex.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return ex


# ---------- App shell ----------
st.set_page_config(page_title="Sales Forecasting – Favorita (Kaggle)", layout="wide")
st.title("🧠 AI-Driven Sales Forecasting (Corporación Favorita – Kaggle)")
st.caption("Data → EDA → Models → Backtests → Scenarios → Exports")

main_tab, compare_tab, docs_tab = st.tabs(["📊 Single Model", "🔬 Compare Models", "📖 Docs"])

# ---------- Sidebar: load ZIP & controls ----------
with st.sidebar:
    st.header("1) Load Kaggle ZIP from local folder")
    zip_path = st.text_input("Path to Kaggle ZIP", value=DEFAULT_ZIP)

    required = {"train.csv", "holidays_events.csv", "oil.csv", "stores.csv", "transactions.csv"}
    files_map: Dict[str, pd.DataFrame] = {}

    try:
        zp = Path(zip_path)
        if not zp.exists():
            st.error(f"ZIP not found at: {zp.resolve()}")
            st.stop()
        with zipfile.ZipFile(zp) as z:
            names_in_zip = {n.split("/")[-1].lower(): n for n in z.namelist()}
            missing = [f for f in required if f not in names_in_zip]
            if missing:
                st.error("Missing in ZIP: " + ", ".join(missing))
                st.stop()
            for fname in required:
                with z.open(names_in_zip[fname]) as f:
                    files_map[fname] = pd.read_csv(f)
    except zipfile.BadZipFile:
        st.error("The file is not a valid ZIP.")
        st.stop()

    st.header("2) Filters & frequency")
    agg_level = st.selectbox("Aggregate level", ["store+family", "store", "family", "all"], index=0)
    freq_target = st.selectbox("Working frequency", ["Daily", "Weekly (Mon)", "Monthly (Start)"], index=0)
    horizon = st.number_input("Forecast horizon (periods)", 1, 180, 30)

    st.header("3) Model(s)")
    model_name = st.selectbox(
        "Single-model run",
        [
            "Naive seasonal", "ETS", "SARIMAX", "GBM (lags+calendar)",
            "LightGBM (lags+calendar)", "Prophet", "LSTM",
            "Hybrid (ETS + LightGBM residuals)"
        ]
    )
    models_to_compare = st.multiselect(
        "Models to compare (batch)",
        [
            "Naive seasonal", "ETS", "SARIMAX", "GBM (lags+calendar)",
            "LightGBM (lags+calendar)", "Prophet", "LSTM",
            "Hybrid (ETS + LightGBM residuals)"
        ],
        default=["Naive seasonal", "ETS", "SARIMAX", "LightGBM (lags+calendar)"]
    )

    st.header("4) Extras")
    show_small_multiples = st.checkbox("Small multiples (per store/family)", value=False)
    small_mult_dim = st.selectbox("Facet by", ["store_nbr", "family"], index=0)
    small_mult_k = st.slider("Max facets", 2, 12, 6)
    show_revenue = st.checkbox("Revenue panel (price × volume index)", value=True)

    st.header("5) Scenarios")
    uplift = st.slider("Promo/channel uplift (%)", -50, 50, 0, 1)
    price_change = st.slider("Price change (%)", -20, 20, 0, 1)
    elasticity = st.slider("Price elasticity", -3.0, 0.0, -0.8, 0.1)

    st.header("6) Export")
    do_pptx = st.checkbox("Generate PowerPoint (python-pptx)", value=False)


# ---------- Load & merge ----------
train = files_map["train.csv"].copy()
holidays = files_map["holidays_events.csv"].copy()
oil = files_map["oil.csv"].copy()
stores = files_map["stores.csv"].copy()
transactions = files_map["transactions.csv"].copy()

# Types
train["date"] = pd.to_datetime(train["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
oil["date"] = pd.to_datetime(oil["date"])
transactions["date"] = pd.to_datetime(transactions["date"])

# Clean oil (fill daily)
oil = oil.sort_values("date").set_index("date").asfreq("D").interpolate().reset_index()

# Merge helpers
holidays_day = holidays.assign(is_holiday=1).groupby("date", as_index=False)["is_holiday"].max()
transactions_day = transactions.groupby(["date", "store_nbr"], as_index=False)["transactions"].sum()

panel = (
    train
    .merge(stores, on="store_nbr", how="left")
    .merge(oil, on="date", how="left")
    .merge(holidays_day, on="date", how="left")
    .merge(transactions_day, on=["date", "store_nbr"], how="left")
)
panel["is_holiday"] = panel["is_holiday"].fillna(0).astype(int)
panel["transactions"] = panel["transactions"].fillna(0)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    stores_list = sorted(panel["store_nbr"].unique().tolist())
    families_list = sorted(panel["family"].unique().tolist())
    sel_store = st.multiselect("Store(s)", stores_list, default=stores_list[:1])
    sel_family = st.multiselect("Family(ies)", families_list, default=families_list[:1])

# Apply selection
f = panel.copy()
if agg_level == "store+family":
    f = f[f["store_nbr"].isin(sel_store) & f["family"].isin(sel_family)]
elif agg_level == "store":
    f = f[f["store_nbr"].isin(sel_store)]
elif agg_level == "family":
    f = f[f["family"].isin(sel_family)]

# Robust date handling
date_candidates = [c for c in f.columns if c.strip().lower() == "date"]
if not date_candidates:
    raise KeyError("No 'date' column found after merges. Check uploaded CSVs.")
DATE_COL = date_candidates[0]
f[DATE_COL] = pd.to_datetime(f[DATE_COL])

# Aggregate to chosen frequency
freq_map = {"Daily": "D", "Weekly (Mon)": "W-MON", "Monthly (Start)": "MS"}
res_freq = freq_map[freq_target]
agg_cols = {
    "sales": "sum", "onpromotion": "sum", "is_holiday": "max",
    "transactions": "sum", "dcoilwtico": "mean"
}
g = (f.set_index(DATE_COL).resample(res_freq).agg(agg_cols).reset_index())
series = g.set_index(DATE_COL).sort_index()
series["sales"] = series["sales"].astype(float)

# Modeling globals
m = seasonal_period_from_freq(res_freq)
y = series["sales"].asfreq(res_freq).interpolate()
exog_all = (
    series[["onpromotion", "is_holiday", "transactions", "dcoilwtico"]]
    .asfreq(res_freq)
    .interpolate()
)
exog_all = sanitize_exog(exog_all)

if CLIP_FORECASTS_NONNEG:
    y = y.clip(lower=0)


# ---------- EDA ----------
with main_tab:
    st.subheader("🔎 Exploratory Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series["sales"], name="Sales", mode="lines"))
        fig.update_layout(height=340, title="Sales over time")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Mean sales", f"{series['sales'].mean():.2f}")
        st.metric("Std dev", f"{series['sales'].std():.2f}")
        st.metric("Last period", f"{series['sales'].iloc[-1]:.2f}")

    if res_freq in ("MS", "W-MON") and series.shape[0] >= 24:
        tmp = series[["sales"]].copy()
        tmp["year"] = tmp.index.year
        tmp["period"] = tmp.index.month if res_freq == "MS" else tmp.index.isocalendar().week
        pv = tmp.pivot_table(index="period", columns="year", values="sales", aggfunc="sum")
        hm = px.imshow(pv, origin="lower", aspect="auto",
                       labels=dict(x="Year", y="Period", color="Sales"),
                       title="Seasonal heatmap")
        st.plotly_chart(hm, use_container_width=True)

    if show_small_multiples:
        st.markdown("**Small multiples**")
        totals = (
            f.groupby(small_mult_dim)["sales"]
            .sum().sort_values(ascending=False)
            .head(small_mult_k).index
        )
        fm = f[f[small_mult_dim].isin(totals)]
        df_sm = (
            fm.set_index(DATE_COL)
            .groupby(small_mult_dim)
            .resample(res_freq)["sales"].sum()
            .reset_index()
        )
        fig_sm = px.line(df_sm, x=DATE_COL, y="sales",
                         facet_col=small_mult_dim, facet_col_wrap=3, height=420)
        st.plotly_chart(fig_sm, use_container_width=True)


# ---------- Modeling helpers ----------
def fit_ets(train: pd.Series):
    seasonal = "add" if m >= 4 else None
    model = ExponentialSmoothing(train, trend="add", seasonal=seasonal,
                                 seasonal_periods=m, initialization_method="estimated")
    return model.fit(optimized=True)

def fit_sarimax(train: pd.Series):
    # Align exogenous to training index
    ex_tr = sanitize_exog(exog_all.reindex(train.index))
    best_aic, best = np.inf, None
    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                for P in [0, 1]:
                    for D in [0, 1]:
                        for Q in [0, 1]:
                            try:
                                seas_m = m if (P + Q + D) > 0 else 0
                                r = SARIMAX(
                                    train, exog=ex_tr,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seas_m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                ).fit(disp=False)
                                if r.aic < best_aic:
                                    best_aic, best = r.aic, r
                            except Exception:
                                pass
    if best is None:
        best = SARIMAX(
            train, exog=ex_tr,
            order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    return best

def build_lagged_frame(s: pd.Series, lags=(1, 2, 3, 6, 12)):
    df = pd.DataFrame({"y": s})
    for L in lags:
        df[f"lag_{L}"] = s.shift(L)
    for w in (3, 6, 12):
        df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek
    return df

def fit_gbm(y_series: pd.Series):
    frame = build_lagged_frame(y_series).join(exog_all, how="left").dropna()
    X, yv = frame.drop(columns=["y"]), frame["y"]
    pipe = Pipeline([("imp", SimpleImputer()), ("gbm", HistGradientBoostingRegressor(random_state=42))])
    pipe.fit(X, yv)
    return pipe, X.columns

def forecast_gbm(model, cols, history: pd.Series, steps: int) -> pd.Series:
    hist = history.copy()
    preds = []
    for _ in range(steps):
        frame = build_lagged_frame(hist).join(exog_all, how="left")
        X = frame.drop(columns=["y"]).iloc[[-1]].reindex(columns=cols, fill_value=np.nan)
        yhat = float(model.predict(X)[0])
        next_idx = hist.index[-1] + to_offset(res_freq)
        hist.loc[next_idx] = yhat
        preds.append((next_idx, yhat))
    s = pd.Series([p[1] for p in preds], index=[p[0] for p in preds])
    return s.clip(lower=0) if CLIP_FORECASTS_NONNEG else s

def fit_lgbm(y_series: pd.Series):
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    frame = build_lagged_frame(y_series).join(exog_all, how="left").dropna()
    X, yv = frame.drop(columns=["y"]), frame["y"]
    model = lgb.LGBMRegressor(random_state=42, n_estimators=400, learning_rate=0.05)
    model.fit(X, yv)
    return model, X.columns

def forecast_lgbm(model, cols, history: pd.Series, steps: int) -> pd.Series:
    hist = history.copy()
    preds = []
    for _ in range(steps):
        frame = build_lagged_frame(hist).join(exog_all, how="left")
        X = frame.drop(columns=["y"]).iloc[[-1]].reindex(columns=cols, fill_value=np.nan)
        yhat = float(model.predict(X)[0])
        next_idx = hist.index[-1] + to_offset(res_freq)
        hist.loc[next_idx] = yhat
        preds.append((next_idx, yhat))
    s = pd.Series([p[1] for p in preds], index=[p[0] for p in preds])
    return s.clip(lower=0) if CLIP_FORECASTS_NONNEG else s

def fit_prophet(train: pd.Series):
    if Prophet is None:
        raise RuntimeError("prophet/fbprophet not installed")
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    mprop = Prophet(yearly_seasonality=True, weekly_seasonality=res_freq.startswith("D"))
    for col in exog_all.columns:
        mprop.add_regressor(col)
    fit_df = dfp.join(exog_all, how="left").reset_index(drop=True)
    mprop.fit(fit_df)
    return mprop

def forecast_prophet(model, last_index: pd.Timestamp, steps: int):
    future_idx = pd.date_range(last_index + to_offset(res_freq), periods=steps, freq=res_freq)
    fut = pd.DataFrame({"ds": future_idx})
    ex_future = sanitize_exog(exog_all.reindex(future_idx)).reset_index(drop=True)
    fut = fut.join(ex_future)
    fcst = model.predict(fut)
    fc = pd.Series(fcst["yhat"].values, index=future_idx)
    lo = pd.Series(fcst["yhat_lower"].values, index=future_idx)
    hi = pd.Series(fcst["yhat_upper"].values, index=future_idx)
    if CLIP_FORECASTS_NONNEG:
        fc = fc.clip(lower=0); lo = lo.clip(lower=0)
    return fc, lo, hi

def fit_lstm(train: pd.Series):
    if tf is None:
        raise RuntimeError("tensorflow not installed")
    window = max(m, 4)
    vals = train.values.astype("float32")
    X, Y = [], []
    for i in range(window, len(vals)):
        X.append(vals[i-window:i]); Y.append(vals[i])
    X = np.array(X)[..., None]; Y = np.array(Y)
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=50, batch_size=32, verbose=0)
    return model, window

def forecast_lstm(model, history: pd.Series, steps: int, window: int):
    hist = history.values.astype("float32").tolist()
    preds = []
    for _ in range(steps):
        x = np.array(hist[-window:], dtype="float32")[None, :, None]
        yhat = float(model.predict(x, verbose=0)[0, 0])
        hist.append(yhat)
        preds.append(yhat)
    idx = pd.date_range(history.index[-1] + to_offset(res_freq), periods=steps, freq=res_freq)
    s = pd.Series(preds, index=idx)
    return s.clip(lower=0) if CLIP_FORECASTS_NONNEG else s

def hybrid_ets_lgbm(train: pd.Series):
    base = fit_ets(train)
    base_fitted = pd.Series(base.fittedvalues, index=train.index)
    resid = (train - base_fitted).dropna()
    if lgb is None:
        mdl, cols = fit_gbm(resid)
        return ("GBM", base, mdl, cols)
    frame = build_lagged_frame(resid).join(exog_all, how="left").dropna()
    X, yv = frame.drop(columns=["y"]), frame["y"]
    mdl = lgb.LGBMRegressor(random_state=42, n_estimators=400, learning_rate=0.05)
    mdl.fit(X, yv)
    return ("LGBM", base, mdl, X.columns)

def forecast_hybrid(hybrid_tuple, history: pd.Series, steps: int):
    mdl_name, base, mdl, cols = hybrid_tuple
    base_fc = base.forecast(steps)
    resid_hist = (history - pd.Series(base.fittedvalues, index=history.index)).dropna()
    hist = resid_hist.copy()
    preds = []
    for _ in range(steps):
        frame = build_lagged_frame(hist).join(exog_all, how="left")
        X = frame.drop(columns=["y"]).iloc[[-1]].reindex(columns=cols, fill_value=np.nan)
        yhat = float(mdl.predict(X)[0])
        next_idx = hist.index[-1] + to_offset(res_freq)
        hist.loc[next_idx] = yhat
        preds.append(yhat)
    resid_fc = pd.Series(preds, index=base_fc.index)
    s = (base_fc + resid_fc)
    return s.clip(lower=0) if CLIP_FORECASTS_NONNEG else s


# ---------- Single-model run ----------
with main_tab:
    st.subheader("⚙️ Train & Forecast (single model)")
    horizon = int(horizon)
    train_y, _ = train_test_split_series(y, max(1, min(int(len(y)*0.2), horizon)))

    # Fit + forecast
    if model_name == "ETS":
        res = fit_ets(train_y)
        fc = res.forecast(horizon)
        resid_std = float(np.std(res.resid))
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)
    elif model_name == "SARIMAX":
        res = fit_sarimax(train_y)
        future_idx = pd.date_range(train_y.index[-1] + to_offset(res_freq), periods=horizon, freq=res_freq)
        ex_future = sanitize_exog(exog_all.reindex(future_idx))
        fr = res.get_forecast(steps=horizon, exog=ex_future)
        fc = pd.Series(fr.predicted_mean, index=future_idx)
        ci = fr.conf_int()
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    elif model_name == "GBM (lags+calendar)":
        mdl, cols = fit_gbm(train_y)
        fc = forecast_gbm(mdl, cols, train_y, horizon)
        resid_std = float((train_y.diff() - train_y.diff().mean()).std())
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)
    elif model_name == "LightGBM (lags+calendar)":
        mdl, cols = fit_lgbm(train_y)
        fc = forecast_lgbm(mdl, cols, train_y, horizon)
        resid_std = float((train_y.diff() - train_y.diff().mean()).std())
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)
    elif model_name == "Prophet":
        mdl = fit_prophet(train_y)
        fc, lower, upper = forecast_prophet(mdl, train_y.index[-1], horizon)
    elif model_name == "LSTM":
        mdl, win = fit_lstm(train_y)
        fc = forecast_lstm(mdl, train_y, horizon, win)
        resid_std = float((train_y.diff() - train_y.diff().mean()).std())
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)
    elif model_name == "Hybrid (ETS + LightGBM residuals)":
        hyb = hybrid_ets_lgbm(train_y)
        fc = forecast_hybrid(hyb, train_y, horizon)
        resid_std = float((train_y.diff() - train_y.diff().mean()).std())
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)
    else:  # Naive seasonal
        base = train_y[-m:].values if len(train_y) > m else np.repeat(train_y.iloc[-1], m)
        vals = np.tile(base, int(np.ceil(horizon / len(base))))[:horizon]
        idx = pd.date_range(train_y.index[-1] + to_offset(res_freq), periods=horizon, freq=res_freq)
        fc = pd.Series(vals, index=idx)
        resid_std = float((train_y.diff() - train_y.diff().mean()).std())
        steps_arr = np.arange(1, horizon + 1)
        half = 1.96 * resid_std * np.sqrt(steps_arr)
        lower = pd.Series(fc.values - half, index=fc.index)
        upper = pd.Series(fc.values + half, index=fc.index)

    if CLIP_FORECASTS_NONNEG:
        fc = fc.clip(lower=0)
        lower = lower.clip(lower=0)  # never negative lower bands

    # Scenario layer (used also in revenue panel & dashboard)
    mult = (1.0 + uplift/100.0) * (1.0 + elasticity * (price_change/100.0))
    fc_scn = fc * mult

    # 2×2 dashboard for readability
    last12_start = y.index.max() - pd.DateOffset(months=12)
    actual_last12 = y[y.index >= last12_start]
    fc_ma = fc.rolling(3).mean()
    scn_ma = fc_scn.rolling(3).mean()

    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Actual (history)", "Baseline vs Scenario (next horizon)",
                        "Zoom: last 12 months", "Smoothed forecast (3-period MA)"),
        vertical_spacing=0.12, horizontal_spacing=0.08
    )

    # (1) Actual
    fig4.add_trace(go.Scatter(x=y.index, y=y.values, name="Actual", mode="lines"), row=1, col=1)

    # (2) Baseline vs Scenario + CI
    fig4.add_trace(go.Scatter(x=fc.index, y=fc.values, name=f"Forecast ({model_name})", mode="lines"), row=1, col=2)
    fig4.add_trace(go.Scatter(x=fc_scn.index, y=fc_scn.values, name="Scenario", mode="lines",
                              line=dict(dash="dash")), row=1, col=2)
    fig4.add_trace(go.Scatter(x=upper.index, y=upper.values, name="Upper 95%", mode="lines",
                              line=dict(width=0.5, dash="dot")), row=1, col=2)
    fig4.add_trace(go.Scatter(x=lower.index, y=lower.values, name="Lower 95%", mode="lines",
                              line=dict(width=0.5, dash="dot"), fill="tonexty"), row=1, col=2)

    # (3) Last 12 months
    fig4.add_trace(go.Scatter(x=actual_last12.index, y=actual_last12.values, name="Actual (L12M)", mode="lines"),
                   row=2, col=1)

    # (4) Smoothed
    fig4.add_trace(go.Scatter(x=fc_ma.index, y=fc_ma.values, name="Forecast MA(3)", mode="lines"), row=2, col=2)
    fig4.add_trace(go.Scatter(x=scn_ma.index, y=scn_ma.values, name="Scenario MA(3)", mode="lines",
                              line=dict(dash="dash")), row=2, col=2)

    fig4.update_layout(height=760, legend_orientation="h", legend_yanchor="bottom", legend_y=1.04, legend_x=0)
    st.plotly_chart(fig4, use_container_width=True)

    # Optional revenue panel
    if show_revenue:
        st.subheader("💶 Revenue panel (index)")
        # Revenue Index explanation is in the Docs tab (see below).
        price_idx = 1.0 + price_change/100.0
        rev_base = fc * 1.0                   # baseline price index = 1.0
        rev_scn  = fc * price_idx * mult      # price × (volume multiplier)
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(x=fc.index, y=rev_base.values, name="Revenue (baseline)", mode="lines"))
        fig_rev.add_trace(go.Scatter(x=fc.index, y=rev_scn.values, name="Revenue (scenario)", mode="lines",
                                     line=dict(dash="dash")))
        fig_rev.update_layout(height=360, title="Revenue index (relative)")
        st.plotly_chart(fig_rev, use_container_width=True)


# ---------- Compare Models ----------
with compare_tab:
    st.subheader("🔬 Batch comparison")
    if not models_to_compare:
        st.info("Select at least one model from the sidebar.")
    else:
        results = {}
        tscv = TimeSeriesSplit(n_splits=2)
        for mdl in models_to_compare:
            rmses, mapes = [], []
            for tr, te in tscv.split(y.values):
                y_tr = pd.Series(y.values[tr], index=y.index[tr])
                y_te = pd.Series(y.values[te], index=y.index[te])
                steps_cv = len(y_te)
                try:
                    if mdl == "ETS":
                        r = fit_ets(y_tr); yhat = r.forecast(steps_cv)
                    elif mdl == "SARIMAX":
                        r = fit_sarimax(y_tr)
                        ex_te = sanitize_exog(exog_all.reindex(y_te.index))
                        yhat = r.get_forecast(steps=steps_cv, exog=ex_te).predicted_mean
                    elif mdl == "GBM (lags+calendar)":
                        m_, c_ = fit_gbm(y_tr); yhat = forecast_gbm(m_, c_, y_tr, steps_cv)
                    elif mdl == "LightGBM (lags+calendar)" and lgb is not None:
                        m_, c_ = fit_lgbm(y_tr); yhat = forecast_lgbm(m_, c_, y_tr, steps_cv)
                    elif mdl == "Prophet" and Prophet is not None:
                        m_ = fit_prophet(y_tr); yhat, _, _ = forecast_prophet(m_, y_tr.index[-1], steps_cv)
                    elif mdl == "LSTM" and tf is not None:
                        m_, win = fit_lstm(y_tr); yhat = forecast_lstm(m_, y_tr, steps_cv, win)
                    elif mdl == "Hybrid (ETS + LightGBM residuals)":
                        hyb = hybrid_ets_lgbm(y_tr); yhat = forecast_hybrid(hyb, y_tr, steps_cv)
                    else:
                        base = y_tr[-m:].values if len(y_tr) > m else np.repeat(y_tr.iloc[-1], m)
                        vals = np.tile(base, int(np.ceil(steps_cv / len(base))))[:steps_cv]
                        yhat = pd.Series(vals, index=y_te.index)

                    if CLIP_FORECASTS_NONNEG and isinstance(yhat, pd.Series):
                        yhat = yhat.clip(lower=0)

                    rmses.append(rmse(y_te, yhat))
                    mapes.append(mape_eps(y_te, yhat))
                except Exception as e:
                    st.warning(f"{mdl} CV failed: {e}")
            results[mdl] = {"RMSE": np.mean(rmses) if rmses else np.nan,
                            "MAPE%": np.mean(mapes) if mapes else np.nan}

        res_df = pd.DataFrame(results).T.sort_values("MAPE%")
        st.dataframe(res_df.round(3))

        # Forecast overlays (limit to top 3 by MAPE)
        top = res_df.sort_values("MAPE%").index.tolist()[:3]
        overlays = {}
        for mdl in top:
            try:
                if mdl == "ETS":
                    r = fit_ets(y); overlays[mdl] = r.forecast(horizon)
                elif mdl == "SARIMAX":
                    r = fit_sarimax(y)
                    future_idx = pd.date_range(y.index[-1] + to_offset(res_freq), periods=horizon, freq=res_freq)
                    ex_future = sanitize_exog(exog_all.reindex(future_idx))
                    overlays[mdl] = r.get_forecast(steps=horizon, exog=ex_future).predicted_mean
                elif mdl == "GBM (lags+calendar)":
                    m_, c_ = fit_gbm(y); overlays[mdl] = forecast_gbm(m_, c_, y, horizon)
                elif mdl == "LightGBM (lags+calendar)" and lgb is not None:
                    m_, c_ = fit_lgbm(y); overlays[mdl] = forecast_lgbm(m_, c_, y, horizon)
                elif mdl == "Prophet" and Prophet is not None:
                    m_ = fit_prophet(y); overlays[mdl], _, _ = forecast_prophet(m_, y.index[-1], horizon)
                elif mdl == "LSTM" and tf is not None:
                    m_, win = fit_lstm(y); overlays[mdl] = forecast_lstm(m_, y, horizon, win)
                elif mdl == "Hybrid (ETS + LightGBM residuals)":
                    hyb = hybrid_ets_lgbm(y); overlays[mdl] = forecast_hybrid(hyb, y, horizon)
                else:
                    base = y[-m:].values if len(y) > m else np.repeat(y.iloc[-1], m)
                    vals = np.tile(base, int(np.ceil(horizon / len(base))))[:horizon]
                    idx = pd.date_range(y.index[-1] + to_offset(res_freq), periods=horizon, freq=res_freq)
                    overlays[mdl] = pd.Series(vals, index=idx)
                if CLIP_FORECASTS_NONNEG:
                    overlays[mdl] = overlays[mdl].clip(lower=0)
            except Exception as e:
                st.warning(f"{mdl} forecast failed: {e}")

        figc = go.Figure()
        figc.add_trace(go.Scatter(x=y.index, y=y.values, name="Actual (history)", mode="lines", opacity=0.4))
        for k, s in overlays.items():
            figc.add_trace(go.Scatter(x=s.index, y=s.values, name=k, mode="lines"))
        figc.update_layout(height=480, title="Model forecast overlays (next horizon only)",
                           legend_orientation="h", legend_yanchor="bottom", legend_y=1.02, legend_x=0)
        st.plotly_chart(figc, use_container_width=True)


# ---------- Docs / Explanations ----------
with docs_tab:
    st.subheader("📖 How this app works")
    st.markdown("""
**Dataset:** Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita).  
**Files used:** `train.csv`, `holidays_events.csv`, `oil.csv`, `stores.csv`, `transactions.csv`.  
**Aggregation:** You choose **Daily / Weekly(Mon) / Monthly(Start)**. Sales are summed; holidays are maxed to a 0/1 flag; transactions/promo are summed; oil is averaged.

**Models available**  
- **Naive seasonal** – repeats the last seasonal cycle (baseline).  
- **ETS** – Holt–Winters with trend + optional seasonality.  
- **SARIMAX** – ARIMA with seasonal terms + **exogenous drivers** (promotions, holidays, transactions, oil).  
- **GBM / LightGBM** – tree models on engineered lags/rolling stats + calendar + exogenous drivers (recursive multi-step).  
- **Prophet** – decomposable trend/seasonality with the same drivers as regressors (forward-filled).  
- **LSTM** – small recurrent net trained on the last *m* steps (where *m* = season length).  
- **Hybrid (ETS + LightGBM residuals)** – forecast = ETS baseline + ML forecast of ETS residuals.

**Validation**  
- Rolling-origin CV with RMSE and **safe MAPE** (denominator clamped with ε to avoid infinite/huge values near zero).  
- **Compare Models** tab runs a small CV and overlays the next-horizon predictions of top performers.

**Uncertainty**  
- SARIMAX provides native 95% intervals; others use residual-variance bands.  
- Lower bands are **clipped at 0** because negative sales are not meaningful (you can toggle global clipping at the top of the script).

**Scenario sliders**  
- **Promo uplift (%)** – multiplicative volume boost.  
- **Price change (%)** – modifies the **price index**.  
- **Price elasticity** – volume change per 1% price change (e.g., −0.8 ⇒ +1% price → −0.8% volume).

---

### 💶 Revenue Index (what you see in the Revenue panel)

This panel is a **relative** measure (index units) to show directionally how revenue moves under scenarios without requiring actual prices.

1. **Volume forecast**:  
   Let \\( V_{t}^{\\text{base}} \\) be the baseline volume forecast at time *t*.

2. **Scenario volume multiplier**:  
   - Promo uplift: \\( (1 + u) \\) where \\( u = \\text{uplift\\%}/100 \\).  
   - Price elasticity: if price changes by \\( p = \\text{price\\%}/100 \\) and elasticity is \\( \\epsilon \\) (usually negative), the volume multiplier is \\( (1 + \\epsilon p) \\).  
   - Combined volume multiplier: \\( M_V = (1 + u)\\,(1 + \\epsilon p) \\).

3. **Price index**:  
   - Baseline price index = **1.0** (no change).  
   - Scenario price index: \\( (1 + p) \\).

4. **Revenue Index formulas** (relative units):  
   - **Baseline**: \\( R_{t}^{\\text{base}} = V_{t}^{\\text{base}} \\times 1.0 \\).  
   - **Scenario**: \\( R_{t}^{\\text{scen}} = V_{t}^{\\text{base}} \\times (1 + p) \\times M_V \\).  
   That is exactly what the code does:  
   `rev_base = fc * 1.0` and `rev_scn = fc * (1 + price_change/100) * ((1 + uplift/100) * (1 + elasticity * (price_change/100)))`.

> Interpretation: if you raise price by +5% with elasticity −0.8 and no promo uplift, the volume multiplier is \\(1 + (-0.8)\\times0.05 = 0.96\\). The scenario revenue index is \\( V \\times 1.05 \\times 0.96 \\approx 1.008 V \\): ~0.8% revenue up.

**Small multiples**  
- Lets you scan heterogeneity across stores or families (top-*k* by total sales). Great for spotting segments that need separate models.

**Limitations**  
- Recursive ML forecasts can drift; intervals for non-SARIMAX models are approximate.  
- Exogenous drivers for Prophet/LGBM/GBM are forward-filled; better practice is proper future plans.  
- Elasticity is applied as a simple constant — good for demos, not a structural demand model.

---

**Tip for presentations:** Use the 2×2 dashboard (history, baseline vs scenario, last-12-months zoom, smoothed path) + the Revenue panel, then show the **Compare Models** table/overlay to justify your model choice.
""")

st.caption("© Demo app for interview practice. Uses Kaggle Store Sales dataset (Corporación Favorita).")
