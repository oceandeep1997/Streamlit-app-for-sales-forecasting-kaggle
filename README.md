# AI-Driven Sales Forecasting (Corporación Favorita – Kaggle)
**Streamlit dashboard — User Guide / README**

This README explains how to install, run, and make the most of the sales‑forecasting dashboard built on top of the Kaggle *Store Sales – Time Series Forecasting* dataset. It clarifies dataset fields (what “store” and “family” mean), how each chart and index is computed, what every model does, how validation works, and how to interpret scenario analysis (promo uplift, price change, elasticity) and the **Revenue Index**.

---

## 1) What this app does

- Loads the official competition data from a local ZIP: `store-sales-time-series-forecasting.zip`  
- Lets you **aggregate** the panel to Daily / Weekly(Mon) / Monthly(Start) and slice by **store(s)** and/or **family(ies)**  
- Trains multiple models: **Naïve seasonal, ETS, SARIMAX (with exogenous drivers), GBM, LightGBM, Prophet, LSTM, and a Hybrid (ETS + LGBM residuals)**  
- Provides **rolling cross‑validation**, **uncertainty bands**, **scenario analysis**, a **Revenue panel**, and **Small‑multiples** to spot heterogeneity  
- Compares models side‑by‑side and overlays the best forecasts over the next horizon

---

## 2) Quick start

### Prerequisites
- Python 3.9+ recommended

### Install
```bash
pip install streamlit pandas numpy plotly statsmodels scikit-learn lightgbm prophet tensorflow python-pptx kaleido
```

> If you don’t need Prophet, LightGBM, or LSTM, you can skip those packages to start faster.

### Files expected
Place this structure in one folder:
```
├── galderma_sales_forecast_streamlit_app.py
└── store-sales-time-series-forecasting.zip
```
The ZIP must contain (Kaggle originals):
- `train.csv`
- `holidays_events.csv`
- `oil.csv`
- `stores.csv`
- `transactions.csv`

### Run
```bash
streamlit run galderma_sales_forecast_streamlit_app.py
```
In the sidebar, the ZIP path defaults to `store-sales-time-series-forecasting.zip`.

---

## 3) Dataset primer (what “store” and “family” mean)

- **store_nbr (“store”)**: a physical supermarket location operated by Corporación Favorita (metadata in `stores.csv`: city, state, type, cluster).  
- **family (“product family”)**: a high‑level **category** of products (e.g., *BEVERAGES*, *CLEANING*, *BREAD/BAKERY*).  
- **date**: daily timestamp.  
- **sales**: unit sales (float; the competition models unit sales, not currency).  
- **onpromotion**: count of items on promotion for that (store, family, date).  
- **transactions**: number of tickets/checkouts at (store, date).  
- **dcoilwtico**: daily oil price (macro proxy).  
- **holidays_events**: calendar; the app converts this into an **is_holiday ∈ {0,1}** flag by date.

**Aggregation rules**
- **Sales**: sum  
- **onpromotion, transactions**: sum  
- **is_holiday**: max (any holiday on that period → 1)  
- **dcoilwtico**: mean  
- Frequency options: **Daily / Weekly(Mon) / Monthly(Start)**

---

## 4) App layout & controls

### Tabs
1. **📊 Single Model**  
   EDA (history, seasonal heatmap), optional **Small multiples**, model training, **2×2 forecast dashboard**, **Revenue panel**.

2. **🔬 Compare Models**  
   Rolling‑origin CV (**RMSE**, **safe MAPE**) and overlay of top models over the next horizon.

3. **📖 Docs**  
   In‑app explanations (a condensed version of this README).

### Sidebar controls
- **ZIP path**: location of the Kaggle ZIP.  
- **Aggregate level**: `store+family`, `store`, `family`, or `all`.  
- **Working frequency**: Daily / Weekly(Mon) / Monthly(Start).  
- **Forecast horizon**: number of future periods to predict.  
- **Model(s)**: pick a single model to run and a list to compare in batch.  
- **Extras**: Small multiples (choose facet and max facets); Revenue panel toggle.  
- **Scenarios**: Promo uplift (%), Price change (%), Price elasticity.  

---

## 5) Models (what they do)

- **Naïve seasonal** — repeats the last seasonal cycle (baseline).  
- **ETS (Holt–Winters)** — Level + Trend + (optional) Seasonality.  
- **SARIMAX** — Seasonal ARIMA with **X**ogenous drivers (onpromotion, is_holiday, transactions, oil). Provides **native confidence intervals**.  
- **GBM / LightGBM** — Gradient‑boosted trees on engineered **lags**, rolling stats, calendar features, and exogenous drivers (recursive multi‑step).  
- **Prophet** — Decomposable trend/seasonality with the same drivers as regressors (forward‑filled).  
- **LSTM** — Small recurrent network trained on a sliding window equal to the seasonal period.  
- **Hybrid (ETS + LightGBM residuals)** — ETS captures structure; LightGBM learns residual patterns. Result = ETS forecast + ML residual forecast.

> The app **clips negative forecasts to zero** (sales can’t be negative). SARIMAX provides native intervals; other models use residual‑based bands.

---

## 6) Metrics & validation

- **RMSE** — root mean squared error (scale‑dependent).  
- **Safe MAPE** —
  \\[
  \\text{MAPE}_\\epsilon = \\frac{1}{n}\\sum_{t} \\left|\\frac{y_t - \\hat{y}_t}{\\max(|y_t|, \\epsilon)}\\right| \\times 100\\%\\,,
  \\]
  which prevents divide‑by‑zero when actuals are near 0.

**Cross‑validation**: Rolling origin (walk‑forward). The app trains on early data and tests on a later slice, preserving time order. The Compare tab averages metrics across splits for each model.

---

## 7) Forecast dashboard — how to read it

- **Actual (history)** — entire historical series after slicing/aggregation.  
- **Baseline vs Scenario** — your model’s baseline forecast versus the scenario forecast, plus **95% bands** (lower band clipped at 0).  
- **Zoom: last 12 months** — recent dynamics.  
- **Smoothed forecast** — moving average (window=3) for readability.

---

## 8) Scenario model — exact math

Let:  
- **u** = `uplift% / 100` (e.g., +10% ⇒ 0.10)  
- **p** = `price% / 100` (e.g., +5% ⇒ 0.05; −3% ⇒ −0.03)  
- **ε** = `elasticity` (usually negative)

1. **Volume multiplier from price**: \\( M_{\\text{price}} = 1 + \\epsilon p \\)  
2. **Promo multiplier**: \\( M_{\\text{promo}} = 1 + u \\)  
3. **Combined volume multiplier**: \\( M_V = (1+u)(1+\\epsilon p) \\)  
4. **Scenario forecast**: if \\(\\hat{V}_t\\) is baseline volume forecast, then  
   \\[ \\hat{V}^{\\text{scenario}}_t = \\hat{V}_t \\times M_V. \\]

> This app forecasts **units**. The scenario applies price effects as demand multipliers (good for what‑if demos; not a full causal demand model).

---

## 9) Revenue panel (Index) — how it’s computed

The **Revenue panel** shows **relative revenue** (index units), not currency. Useful to compare directionally without confidential pricing.

- **Baseline price index** = 1.0  
- **Scenario price index** = \\( 1 + p \\)

Given baseline volume \\(\\hat{V}_t\\) and volume multiplier \\( M_V \\):  
- **Baseline Revenue Index**: \\( R^{\\text{base}}_t = \\hat{V}_t \\times 1.0 \\)  
- **Scenario Revenue Index**: \\( R^{\\text{scen}}_t = \\hat{V}_t \\times (1 + p) \\times M_V \\)

**Code**:
```python
price_idx = 1.0 + price_change/100.0
mult = (1.0 + uplift/100.0) * (1.0 + elasticity * (price_change/100.0))
rev_base = fc * 1.0               # baseline price index
rev_scn  = fc * price_idx * mult  # scenario price × scenario volume
```

**Example**  
Price +5% (p=0.05), elasticity −0.8 ⇒ volume multiplier ≈ 0.96; no promo uplift → revenue index ≈ `1.05 * 0.96 = 1.008` (~+0.8% vs baseline units).

---

## 10) Compare Models tab — how to use it

- Select models in the sidebar and open **🔬 Compare Models**.  
- The app runs **rolling CV** and reports **RMSE** and **safe MAPE** per model (lower is better).  
- It overlays the next‑horizon forecasts from the **top 3** models so you can inspect divergence and justify selection.

---

## 11) Small multiples — when to use

Turn on **Small multiples** in Extras and choose `store_nbr` or `family`. The app selects **top‑k** groups by total sales and plots each subseries in a facet grid.

Use cases:  
- Spot units with distinct seasonality/volatility (candidate for separate models)  
- Identify segments where promo/price effects look stronger

---

## 12) Confidence intervals — interpretation

- **SARIMAX**: native 95% prediction intervals from the state‑space model.  
- **ETS/GBM/LGBM/Prophet/LSTM/Hybrid**: approximate bands from residual variance (growing with horizon).  
- **Lower band is clipped at 0** (negative sales are not meaningful).

---

## 13) Troubleshooting & tips

- **“exog contains inf or nans”**: Ensure exogenous features are numeric and sanitized (`replace ±inf → NaN → ffill/bfill → 0`). The app already sanitizes, but recheck if you customize.  
- **Huge/∞ MAPE**: Use **safe MAPE** (already in the app).  
- **SARIMAX order fails**: The app falls back to `(1,1,1)` non‑seasonal; try weekly/monthly aggregation for more stability.  
- **Intervals too wide**: Aggregate to weekly/monthly or pick SARIMAX for native intervals.  
- **Performance**: LSTM/Prophet/LightGBM can be slower; for quick iterations start with ETS/SARIMAX/GBM.  
- **Negative forecasts**: Disabled by global clipping (can be turned off in the code).

---

## 14) Presenting in a business case

Suggested 10‑slide flow:
1. Objective & data scope  
2. Key EDA insights (trend/seasonality/anomalies)  
3. Candidate models & rationale  
4. Validation (CV RMSE & safe MAPE)  
5. Baseline forecast + 95% bands  
6. Scenarios (promo, price, elasticity)  
7. Revenue Index insight (trade‑off explained)  
8. Segment view (small multiples) & actions  
9. Risks & monitoring plan  
10. Recommendation & next steps

---

## 15) Extending the app

- Per‑series batch training (each store/family) with a consolidated leaderboard  
- Causal price/promo modeling (elasticity by segment, demand curves)  
- Inventory/operations tie‑ins (service levels, safety stock)  
- Experiment logging & model registry

---

## 16) Privacy & licensing

- The app reads **public Kaggle data** from your local ZIP; it does not transmit your data.  
- Third‑party libraries follow their respective licenses (scikit‑learn, statsmodels, LightGBM, Prophet, TensorFlow, Plotly, Streamlit, etc.).  
- Intended for **interview practice / education**. For production, add MLOps, monitoring, and governance.

---

## 17) Glossary

- **Store**: a physical supermarket (store_nbr).  
- **Family**: product department/category.  
- **Onpromotion**: number of items on promotion for (store, family, date).  
- **Transactions**: number of receipts per store/date.  
- **Oil (dcoilwtico)**: crude price proxy.  
- **Exogenous (exog)**: external drivers used by SARIMAX/ML models.  
- **Horizon**: number of future periods to forecast.  
- **Uplift**: expected promo/channel effect (percent).  
- **Elasticity (ε)**: % volume change for a 1% price change (usually negative).  
- **Revenue Index**: relative revenue measure (price index × volume), not currency.  
- **RMSE / MAPE**: accuracy metrics (lower is better).  
- **Rolling origin CV**: time‑respecting validation.
