# sample_streamlit.py
# Benchmark = Lasso; Hybrid = BoostedHybrid (Ridge + LightGBM residuals)
# Works with M5 (daily) or an uploaded CSV (date,sales).
# Python 3.7-compatible type hints (Union/Optional).

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Union, Optional
from pandas.tseries.frequencies import to_offset

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
except Exception:
    lgb = None

st.set_page_config(page_title="Benchmark & Hybrid (Lasso vs BoostedHybrid)", layout="wide")
st.title("📈 Benchmark vs Hybrid Forecasting")
st.caption("Benchmark = Lasso | Hybrid = Ridge + LightGBM residuals (BoostedHybrid). Works with M5 or your CSV.")

# ---------------------------------------------------------------------
# BoostedHybrid (Ridge base + LightGBM residuals)
# ---------------------------------------------------------------------
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1  # e.g., Ridge pipeline
        self.model_2 = model_2  # LightGBM
        self.y_columns = None
        self._fitted = False
        self._per_family_models = {}  # family -> (m1, m2)

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        if isinstance(y, pd.Series):
            y_df = y.to_frame(name="y")
        else:
            y_df = y.copy()
        self.y_columns = list(y_df.columns)

        for fam in self.y_columns:
            X1_f = X1[fam].copy()
            X2_f = X2[fam].copy()
            df_f = pd.concat([y_df[[fam]], X1_f, X2_f], axis=1).dropna()
            y_f = df_f[fam].values
            X1_f = df_f[X1_f.columns]
            X2_f = df_f[X2_f.columns]

            m1 = self._clone_estimator(self.model_1)
            m2 = self._clone_estimator(self.model_2)
            m1.fit(X1_f, y_f)
            res = y_f - m1.predict(X1_f)
            m2.fit(X2_f, res)
            self._per_family_models[fam] = (m1, m2)

        self._fitted = True
        return self

    def predict(self, X1: pd.DataFrame, X2: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Call fit before predict."
        preds = {}
        for fam in self.y_columns:
            X1_f = X1[fam].fillna(0)
            X2_f = X2[fam].fillna(0)
            m1, m2 = self._per_family_models[fam]
            preds[fam] = m1.predict(X1_f) + m2.predict(X2_f)
        return pd.DataFrame(preds, index=X1.index)

    @staticmethod
    def _clone_estimator(est):
        import copy
        return copy.deepcopy(est)

# ---------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------
with st.sidebar:
    DATASET = st.selectbox("Dataset", ["M5 (daily)", "Upload CSV (date,sales)"], index=0)

    st.header("Frequency & Horizon")
    freq_label = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    FREQ = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[freq_label]
    seasonal_m = 7 if FREQ.startswith("D") else (52 if FREQ.startswith("W") else 12)
    H = st.number_input("Horizon (steps)", min_value=7 if FREQ == "D" else 4, max_value=365, value=28, step=1)

    st.header("Models")
    use_hybrid = st.checkbox("Enable Hybrid (Ridge + LightGBM residuals)", value=True)

    st.header("Prediction Intervals")
    use_pi = st.checkbox("Show intervals", value=True)
    ci_95 = st.checkbox("95% band", value=True)
    ci_80 = st.checkbox("80% band", value=False)

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_m5(DATA_DIR: str) -> dict:
    assert os.path.exists(DATA_DIR), "Update DATA_DIR to point to the M5 data folder."
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    return {"sales": sales, "calendar": calendar, "prices": prices}

def m5_series(m5: dict, item_id: str, store_id: str, freq="D") -> pd.Series:
    sales = m5["sales"]; calendar = m5["calendar"]
    row = sales[(sales["item_id"] == item_id) & (sales["store_id"] == store_id)]
    if row.empty:
        raise ValueError("No series for the chosen item_id & store_id.")
    row = row.iloc[0]
    d_cols = [c for c in row.index if c.startswith("d_")]
    s = pd.Series(row[d_cols].values.astype(float), index=d_cols, name="sales")
    d_map = calendar.set_index("d")["date"].to_dict()
    s.index = pd.to_datetime([d_map[d] for d in s.index])
    s = s.asfreq("D").fillna(0.0)
    if freq != "D":
        s = s.resample(freq).sum()
    return s

def csv_series(uploaded_file, freq="D") -> pd.Series:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    s = df.set_index("date")["sales"].astype(float).asfreq(freq)
    return s.interpolate().fillna(0.0)

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def make_supervised(series: pd.Series,
                    lags=(1, 2, 3, 7, 14, 28),
                    m_windows=(3, 7, 14),
                    with_std=True,
                    dropna=True) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for L in lags:
        df["lag_%d" % L] = series.shift(L)
    for w in m_windows:
        df["roll_mean_%d" % w] = series.shift(1).rolling(w).mean()
        if with_std:
            df["roll_std_%d" % w] = series.shift(1).rolling(w).std()
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek
    df["dom"] = df.index.day
    if dropna:
        df = df.dropna()
    return df

def recursive_forecast(model, history: pd.Series, steps: int, freq: str) -> pd.Series:
    hist = history.copy()
    preds = []
    for _ in range(steps):
        frame = make_supervised(hist, dropna=False).iloc[[-1]]
        Xlast = frame.drop(columns=["y"]).fillna(method="ffill").fillna(0.0)
        yhat = float(model.predict(Xlast)[0])
        next_idx = hist.index[-1] + to_offset(freq)
        hist.loc[next_idx] = yhat
        preds.append(yhat)
    fc_index = pd.date_range(history.index[-1] + to_offset(freq), periods=steps, freq=freq)
    return pd.Series(preds, index=fc_index)

def gaussian_bands(resid: pd.Series, steps: int, level: float = 0.95) -> np.ndarray:
    sd = float((resid.diff() - resid.diff().mean()).std())
    z = 1.96 if level >= 0.95 else 1.28
    t = np.arange(1, steps + 1)
    return z * sd * np.sqrt(t)

# ---------------------------------------------------------------------
# Load series
# ---------------------------------------------------------------------
if DATASET.startswith("M5"):
    st.sidebar.header("M5 Settings")
    DATA_DIR = st.sidebar.text_input("DATA_DIR", "m5-forecasting-accuracy")
    st.sidebar.caption("Ensure the folder contains sales_train_validation.csv, calendar.csv, sell_prices.csv")
    if st.sidebar.button("Load M5"):
        m5 = load_m5(DATA_DIR)
        stores = sorted(m5["sales"]["store_id"].unique().tolist())
        items = sorted(m5["sales"]["item_id"].unique().tolist())
        store_id = st.selectbox("store_id", stores if stores else ["N/A"])
        item_id = st.selectbox("item_id", items if items else ["N/A"])
        if stores and items:
            s = m5_series(m5, item_id=item_id, store_id=store_id, freq=FREQ)
        else:
            st.stop()
    else:
        st.info("Set DATA_DIR and click **Load M5** to continue.")
        st.stop()
else:
    st.sidebar.header("Upload CSV")
    st.sidebar.caption("Expected columns: date,sales")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        s = csv_series(uploaded, freq=FREQ)
    else:
        st.info("Upload a CSV or preview on a synthetic series.")
        idx = pd.date_range("2018-01-01", periods=500, freq=FREQ)
        rng = np.random.RandomState(7)
        seas = 80 * np.sin(2 * np.pi * np.arange(len(idx)) / (7 if FREQ == "D" else (52 if FREQ.startswith("W") else 12))) if (FREQ == "D" or FREQ.startswith("W") or FREQ == "MS") else 0
        y = 500 + 0.4 * np.arange(len(idx)) + seas + rng.normal(0, 40, len(idx))
        s = pd.Series(y, index=idx).clip(lower=0)

# ---------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------
# ---------------------------
# Train/validation split (fixed)
# ---------------------------
st.subheader("Input Series")
st.line_chart(s.rename("sales"))

# choose a reasonable validation length
T_VAL = min(
    max(14 if FREQ == "D" else 4, H),     # at least as long as horizon (and ≥14D/≥4 steps)
    max(7, int(0.1 * len(s)))             # but not more than 10% of the data (and ≥7)
)

if len(s) > T_VAL:
    train = s.iloc[:-T_VAL]
    valid = s.iloc[-T_VAL:]
else:
    # not enough history: use all for train and keep an empty valid
    train = s.copy()
    valid = pd.Series(dtype=float)

# ---------------------------------------------------------------------
# Build train frame
# ---------------------------------------------------------------------
train_df = make_supervised(train)
X_train = train_df.drop(columns=["y"])
y_train = train_df["y"].values

# ---------------------------------------------------------------------
# Benchmark (Lasso)
# ---------------------------------------------------------------------
st.subheader("Models")
with st.spinner("Fitting Benchmark (Lasso)…"):
    bench_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lasso", Lasso(alpha=0.001, random_state=42, max_iter=10000))
    ])
    bench_pipe.fit(X_train, y_train)

bench_fc = recursive_forecast(bench_pipe, train, steps=H, freq=FREQ)
bench_fit_in = bench_pipe.predict(X_train)
bench_resid = pd.Series(y_train - bench_fit_in, index=train_df.index)

# ---------------------------------------------------------------------
# Hybrid (Ridge + LightGBM residuals)
# ---------------------------------------------------------------------
if use_hybrid:
    if lgb is None:
        st.warning("LightGBM not installed. Run: pip install lightgbm. Showing Lasso only.")
        hybrid_fc = None
        hybrid_resid = None
        hybrid = None
    else:
        with st.spinner("Fitting Hybrid (Ridge + LightGBM residuals)…"):
            # Split feature blocks: X1 (structure) and X2 (nonlinear)
            # Rebuild from train to keep the same indices/columns.
            X1_train = make_supervised(train, with_std=False).drop(columns=["y"])  # lags + means + calendar
            X2_train = make_supervised(train, with_std=True).drop(columns=["y"])[[c for c in make_supervised(train, with_std=True).drop(columns=["y"]).columns if "roll_std" in c or "lag_" in c]]

            ridge_pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("ridge", Ridge(alpha=1.0, random_state=42))
            ])
            lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)

            hybrid = BoostedHybrid(model_1=ridge_pipe, model_2=lgbm)
            # For univariate in this Streamlit app, make them 1-family frames.
            hybrid.fit(X1_train.to_frame().rename(columns={X1_train.columns[0]: "series_1"}),
                       X2_train.to_frame().rename(columns={X2_train.columns[0]: "series_1"}),
                       pd.DataFrame({"series_1": y_train}, index=X1_train.index))

            # Forecast recursively using same feature maker (one-block variant):
            # For the demo, we reuse the simpler recursive forecaster on the hybrid object by
            # passing a concatenated frame. We'll just use its internal .predict on a single-row frame.
            def _hybrid_recursive(hybrid_model, history: pd.Series, steps: int, freq: str) -> pd.Series:
                hist = history.copy()
                preds = []
                for _ in range(steps):
                    X1_row = make_supervised(hist, with_std=False, dropna=False).iloc[[-1]].drop(columns=["y"])
                    X2_row_full = make_supervised(hist, with_std=True, dropna=False).iloc[[-1]].drop(columns=["y"])
                    X2_cols = [c for c in X2_row_full.columns if ("roll_std" in c) or ("lag_" in c)]
                    X2_row = X2_row_full[X2_cols]
                    X1w = X1_row.to_frame().rename(columns={X1_row.columns[0]: "series_1"})
                    X2w = X2_row.to_frame().rename(columns={X2_row.columns[0]: "series_1"})
                    yhat = float(hybrid_model.predict(X1w, X2w).iloc[-1]["series_1"])
                    next_idx = hist.index[-1] + to_offset(freq)
                    hist.loc[next_idx] = yhat
                    preds.append(yhat)
                fc_index = pd.date_range(history.index[-1] + to_offset(freq), periods=steps, freq=freq)
                return pd.Series(preds, index=fc_index)

            hybrid_fc = _hybrid_recursive(hybrid, train, steps=H, freq=FREQ)

            # residuals on train
            yhat_in = hybrid.predict(
                X1_train.to_frame().rename(columns={X1_train.columns[0]: "series_1"}),
                X2_train.to_frame().rename(columns={X2_train.columns[0]: "series_1"})
            )["series_1"].values
            hybrid_resid = pd.Series(y_train - yhat_in, index=X1_train.index)

else:
    hybrid_fc = None
    hybrid_resid = None
    hybrid = None

# ---------------------------------------------------------------------
# Plot forecasts + intervals
# ---------------------------------------------------------------------
def plot_forecasts(actual: pd.Series,
                   bench_fc: pd.Series,
                   hybrid_fc: Optional[pd.Series]):
    fig = go.Figure()
    fig.add_scatter(x=actual.index, y=actual.values, name="Actual", mode="lines")
    fig.add_scatter(x=bench_fc.index, y=bench_fc.values, name="Benchmark (Lasso)", mode="lines")

    if use_pi and len(bench_resid) > 5:
        if ci_95:
            half = gaussian_bands(bench_resid, steps=H, level=0.95)
            fig.add_scatter(x=bench_fc.index, y=bench_fc.values + half, name="Lasso +95%", mode="lines")
            fig.add_scatter(x=bench_fc.index, y=np.maximum(0, bench_fc.values - half), name="Lasso -95%", mode="lines", fill='tonexty')
        if ci_80:
            half80 = gaussian_bands(bench_resid, steps=H, level=0.80)
            fig.add_scatter(x=bench_fc.index, y=bench_fc.values + half80, name="Lasso +80%", mode="lines")
            fig.add_scatter(x=bench_fc.index, y=np.maximum(0, bench_fc.values - half80), name="Lasso -80%", mode="lines", fill='tonexty')

    if hybrid_fc is not None:
        fig.add_scatter(x=hybrid_fc.index, y=hybrid_fc.values, name="Hybrid (Ridge + LGBM)", mode="lines")
        if use_pi and hybrid_resid is not None and len(hybrid_resid) > 5:
            if ci_95:
                half_h = gaussian_bands(hybrid_resid, steps=H, level=0.95)
                fig.add_scatter(x=hybrid_fc.index, y=hybrid_fc.values + half_h, name="Hybrid +95%", mode="lines")
                fig.add_scatter(x=hybrid_fc.index, y=np.maximum(0, hybrid_fc.values - half_h), name="Hybrid -95%", mode="lines", fill='tonexty')
            if ci_80:
                half_h80 = gaussian_bands(hybrid_resid, steps=H, level=0.80)
                fig.add_scatter(x=hybrid_fc.index, y=hybrid_fc.values + half_h80, name="Hybrid +80%", mode="lines")
                fig.add_scatter(x=hybrid_fc.index, y=np.maximum(0, hybrid_fc.values - half_h80), name="Hybrid -80%", mode="lines", fill='tonexty')

    fig.update_layout(height=460, title="Forecasts (with optional intervals)")
    return fig

st.plotly_chart(plot_forecasts(s, bench_fc, hybrid_fc), use_container_width=True)

# Diagnostics
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Benchmark (Lasso) — top coefficients**")
    lasso = bench_pipe.named_steps["lasso"]
    coefs = pd.Series(lasso.coef_, index=X_train.columns).sort_values(key=np.abs, ascending=False).head(15)
    st.dataframe(coefs.rename("coeff").round(4))
with c2:
    if hybrid is not None:
        st.markdown("**Hybrid — LightGBM feature importance**")
        imps = hybrid.model_2.feature_importances_
        cols_arr = np.array(X_train.columns)
        order = np.argsort(imps)[::-1][:20]
        st.dataframe(pd.DataFrame({"feature": cols_arr[order], "importance": imps[order]}).set_index("feature"))
    else:
        st.info("Feature importance available when Hybrid is enabled.")

st.markdown("**Residual diagnostics**")
resid_to_plot = hybrid_resid if (hybrid_resid is not None) else bench_resid
title = "Hybrid residuals (train)" if (hybrid_resid is not None) else "Lasso residuals (train)"
fig, axs = plt.subplots(1, 2, figsize=(11, 3))
axs[0].hist(resid_to_plot, bins=30); axs[0].set_title("%s — histogram" % title)
axs[1].plot(resid_to_plot.index, resid_to_plot.values); axs[1].set_title("Residuals over time")
st.pyplot(fig)

with st.expander("Notes & Tips"):
    st.write("""
    **Benchmark (Lasso)**: simple linear baseline on lag/rolling/calendar features.
    **Hybrid (BoostedHybrid)**: Ridge captures structure; LightGBM learns residual patterns.

    Intervals are quick Gaussian proxies. For calibrated coverage, consider quantile models
    or conformal prediction. To adapt to new datasets, provide a two-column time series
    (`date,sales`) or hook a loader that returns a `pd.Series` indexed by date, then set `FREQ`.
    """)
