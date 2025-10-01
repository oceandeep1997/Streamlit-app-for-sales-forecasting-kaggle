
import os, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import io

try:
    import lightgbm as lgb
except Exception:
    lgb = None

st.set_page_config(page_title="Hybrid Forecast — ETS + LGBM", layout="wide")
st.title("🔗 Hybrid Forecast (ETS + LightGBM residuals)")

st.sidebar.header("Config")
freq = st.sidebar.selectbox("Frequency", ["Weekly (Mon)", "Monthly (MS)"], index=0)
freq_map = {"Weekly (Mon)":"W-MON", "Monthly (MS)":"MS"}
FREQ = freq_map[freq]
H = st.sidebar.number_input("Horizon (steps)", 4, 156, 26)
seasonal_m = 52 if FREQ.startswith("W") else 12

st.sidebar.header("Data")
mode = st.sidebar.radio("Source", ["Synthetic sample", "Upload CSV"], index=0)
uploaded = st.sidebar.file_uploader("CSV with columns: date,sales", type=["csv"])

def build_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    ex = pd.DataFrame(index=idx)
    ex["onpromotion"] = 0.0
    ex["is_holiday"] = 0.0
    ex["transactions"] = 0.0
    ex["dcoilwtico"] = 0.0
    return ex

def build_lagged_frame(s: pd.Series, lags=(1,2,3,6,12), m_windows=(3,6,12)):
    df = pd.DataFrame({"y": s})
    for L in lags:
        df[f"lag_{L}"] = s.shift(L)
    for w in m_windows:
        df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek
    return df

def fit_ets(train: pd.Series):
    model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=seasonal_m, initialization_method="estimated")
    return model.fit(optimized=True)

def hybrid_ets_lgbm(train: pd.Series, exog_all: pd.DataFrame):
    base = fit_ets(train)
    fitted = pd.Series(base.fittedvalues, index=train.index)
    resid = (train - fitted).dropna()
    if lgb is None:
        st.error("LightGBM not installed. pip install lightgbm")
        st.stop()
    frame = build_lagged_frame(resid).join(exog_all, how="left").dropna()
    X, y = frame.drop(columns=["y"]), frame["y"]
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return base, model, X.columns

def forecast_hybrid(base, mdl, cols, history: pd.Series, exog_all: pd.DataFrame, steps: int):
    base_fc = base.forecast(steps)
    resid_hist = (history - pd.Series(base.fittedvalues, index=history.index)).dropna()
    hist = resid_hist.copy()
    preds = []
    for _ in range(steps):
        frame = build_lagged_frame(hist).join(exog_all, how="left")
        X = frame.drop(columns=["y"]).iloc[[-1]].reindex(columns=cols, fill_value=np.nan)
        yhat = float(mdl.predict(X)[0])
        next_idx = hist.index[-1] + to_offset(FREQ)
        hist.loc[next_idx] = yhat
        preds.append(yhat)
    resid_fc = pd.Series(preds, index=base_fc.index)
    return (base_fc + resid_fc).clip(lower=0), resid_hist

# Load data
if mode == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"]).set_index("date").asfreq(FREQ)
    y = df["sales"].astype(float).interpolate()
else:
    idx = pd.date_range("2017-01-02", periods=200, freq=FREQ)
    rng = np.random.RandomState(7)
    y = pd.Series(500 + 2*np.arange(len(idx)) + 80*np.sin(2*np.pi*np.arange(len(idx))/seasonal_m) + rng.normal(0, 40, len(idx)),
                  index=idx).clip(lower=0)

exog = build_exog(y.index)
base, mdl, cols = hybrid_ets_lgbm(y, exog)
fc, resid_hist = forecast_hybrid(base, mdl, cols, y, exog, H)

# Confidence bands
resid_std = float((y.diff() - y.diff().mean()).std())
steps_arr = np.arange(1, H + 1)
half = 1.96 * resid_std * np.sqrt(steps_arr)
lower = pd.Series(fc.values - half, index=fc.index).clip(lower=0)
upper = pd.Series(fc.values + half, index=fc.index)

# Plotly forecast
fig = go.Figure()
fig.add_scatter(x=y.index, y=y.values, name="Actual", mode="lines")
fig.add_scatter(x=fc.index, y=fc.values, name="Hybrid forecast", mode="lines")
fig.add_scatter(x=upper.index, y=upper.values, name="Upper 95%", mode="lines")
fig.add_scatter(x=lower.index, y=lower.values, name="Lower 95%", mode="lines", fill='tonexty')
fig.update_layout(height=420, title="Hybrid forecast with 95% interval")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ETS fitted values**")
    fig2 = go.Figure()
    fig2.add_scatter(x=y.index, y=pd.Series(base.fittedvalues, index=y.index), name="ETS fitted", mode="lines")
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("**LightGBM feature importance**")
    importances = mdl.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    fi = pd.DataFrame({"feature": np.array(cols)[order], "importance": importances[order]})
    fig3 = go.Figure()
    fig3.add_bar(y=fi["feature"][::-1], x=fi["importance"][::-1], orientation='h')
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

# Residual diagnostics (matplotlib rendered to image)
st.markdown("**Residual diagnostics (histogram, ACF, PACF)**")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axs = plt.subplots(1,3, figsize=(11,3))
axs[0].hist(resid_hist, bins=30); axs[0].set_title("Residual hist")
plot_acf(resid_hist, ax=axs[1], lags=min(40, len(resid_hist)//2)); axs[1].set_title("ACF")
plot_pacf(resid_hist, ax=axs[2], lags=min(40, len(resid_hist)//2)); axs[2].set_title("PACF")
st.pyplot(fig)
