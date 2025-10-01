
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Forecast + Finance + Scenarios (Rossmann & Favorita)
-----------------------------------------------------------
This script now supports both datasets directly:

ROSSMANN MODE
-------------
python rossmann_forecast_finance.py \
  --dataset rossmann \
  --train /path/to/rossmann/train.csv \
  --store /path/to/rossmann/store.csv \
  --output_dir ./outputs_rossmann \
  --horizon_weeks 52

FAVORITA MODE (Kaggle "Corporación Favorita Grocery Sales Forecasting")
-----------------------------------------------------------------------
Place the competition files (CSV or .7z) in a folder (data_dir). Required files:
  - train.csv(.7z), items.csv(.7z), stores.csv(.7z),
  - holidays_events.csv(.7z), oil.csv(.7z), transactions.csv(.7z)

Then run:
python rossmann_forecast_finance.py \
  --dataset favorita \
  --data_dir ./favorita-grocery-sales-forecasting \
  --output_dir ./outputs_favorita \
  --since 2015-01-01 \
  --horizon_weeks 52

Outputs (both modes)
--------------------
- backtest_metrics.csv
- scenario_family_week.csv
- promo_roi_by_store.csv
- safety_stock.csv
- weekly_panel_model_ready.csv
- plots/net_revenue_<top_family>.png
- plots/revenue_bridge.png
- plots/promo_roi_hist.png

Finance mapping:
- Treats target 'Sales' as Net Revenue (NR) in finance layer.
- Reconstructs Gross via GTN% and computes COGS, GM, Contribution.

Intervals:
- Uses GradientBoostingRegressor quantile models to produce lo/hi bounds (80%/95% configurable).

Author: You
"""

import os
import sys
import math
import json
import argparse
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Global Config (finance & ops)
# -------------------------
GTN_DEFAULT = 0.18         # gross-to-net deduction rate
COGS_DEFAULT = 0.45        # COGS as % of Net Revenue
PROMO_COST_RATE = 0.20     # % of incremental NR used as promo cost proxy
ELASTICITY_DEFAULT = -0.6  # directional price elasticity (placeholder where price not present)
LEAD_WEEKS = {"AESTHETICS": 8, "RX_DERM": 6, "CONSUMER": 4}
SERVICE_LEVEL_Z = 1.65     # ~95%

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.where(denom==0, 0, np.abs(y_true - y_pred) / denom)
    return np.mean(out) * 100

def mase(y_true, y_pred, y_train_hist):
    if len(y_train_hist) < 2:
        return np.nan
    denom = np.mean(np.abs(np.diff(y_train_hist)))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / denom

def abc_tiers(series: pd.Series) -> pd.Series:
    s = series.sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    out = pd.Series(index=s.index, dtype="object")
    out[cum <= 0.8] = "A"
    out[(cum > 0.8) & (cum <= 0.95)] = "B"
    out[cum > 0.95] = "C"
    return out.reindex(series.index).fillna("C")

# -------------------------
# Rossmann loaders
# -------------------------
def load_rossmann(train_path: str, store_path: str) -> pd.DataFrame:
    train = pd.read_csv(train_path, parse_dates=["Date"])
    store = pd.read_csv(store_path)
    df = train.merge(store, on="Store", how="left")
    df = df[(df["Open"]==1) & (df["Sales"]>0)].copy()
    df["channel"] = "RETAIL"
    df["region"] = df["StoreType"].fillna("U")
    return df

def add_calendar_features(df: pd.DataFrame, date_col="Date", daily=True) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    d["week"] = d[date_col].dt.isocalendar().week.astype(int)
    d["dow"] = d[date_col].dt.weekday
    d["is_month_end"] = d[date_col].dt.is_month_end.astype(int)
    d["is_q_end"] = d[date_col].dt.is_quarter_end.astype(int)
    if daily:
        d["day"] = d[date_col].dt.day
        d["is_weekend"] = d["dow"].isin([5,6]).astype(int)
    return d

def add_promo_history(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["Store","Date"]).copy()
    d["promo_lag1"] = d.groupby("Store")["Promo"].shift(1).fillna(0)
    d["promo_rolling_4w"] = d.groupby("Store")["Promo"].transform(lambda s: s.rolling(28, min_periods=1).mean())
    comp_year = d["CompetitionOpenSinceYear"].fillna(0).astype(int)
    comp_month = d["CompetitionOpenSinceMonth"].fillna(1).astype(int)
    comp_date = pd.to_datetime(comp_year.astype(str) + "-" + comp_month.astype(str) + "-01", errors="coerce")
    d["comp_days"] = (d["Date"] - comp_date).dt.days.fillna(0).clip(lower=0)
    d["Promo2Since"] = pd.to_datetime(d["Promo2SinceYear"].fillna(0).astype(int).astype(str) + "-" +
                                      d["Promo2SinceWeek"].fillna(1).astype(int).astype(str) + "-1", errors="coerce")
    d["promo2_days"] = (d["Date"] - d["Promo2Since"]).dt.days.fillna(0).clip(lower=0)
    return d

def weekly_panel_rossmann(df: pd.DataFrame) -> pd.DataFrame:
    g = (df
         .assign(week_start=df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D"))
         .groupby(["Store","week_start","region","channel","StoreType","Assortment"], as_index=False)
         .agg(Sales=("Sales","sum"),
              Customers=("Customers","sum"),
              Promo=("Promo","max"),
              SchoolHoliday=("SchoolHoliday","max"),
              StateHoliday=("StateHoliday","max"),
              comp_days=("comp_days","max"),
              promo2_days=("promo2_days","max")))
    g["price_index"] = (g.groupby("Store")["Sales"].transform(lambda s: s.rolling(8, min_periods=1).mean()) /
                        g.groupby("Store")["Sales"].transform(lambda s: s.rolling(26, min_periods=1).mean())).fillna(1.0).clip(0.5, 1.5)
    map_family = {"a":"CONSUMER","b":"RX_DERM","c":"AESTHETICS"}
    g["family"] = g["StoreType"].str.lower().map(map_family).fillna("CONSUMER")
    return g.rename(columns={"week_start":"date"})

# -------------------------
# Favorita loaders (.7z aware)
# -------------------------
def extract_7z_if_needed(path: str, out_dir: str) -> str:
    if path.lower().endswith(".7z"):
        try:
            import py7zr  # lazy import
        except ImportError:
            raise SystemExit("Favorita: install py7zr to open .7z files: pip install py7zr")
        ensure_dir(out_dir)
        with py7zr.SevenZipFile(path, mode='r') as z:
            z.extractall(path=out_dir)
        # pick first CSV
        for f in os.listdir(out_dir):
            if f.lower().endswith(".csv"):
                return os.path.join(out_dir, f)
        raise FileNotFoundError(f"Extracted but no CSV found in {out_dir}")
    return path

def favor_dtypes(file: str) -> Dict[str, str]:
    if file.endswith("train.csv"):
        return dict(store_nbr="int16", item_nbr="int32", onpromotion="boolean")
    if file.endswith("test.csv"):
        return dict(store_nbr="int16", item_nbr="int32", onpromotion="boolean")
    if file.endswith("items.csv"):
        return dict(item_nbr="int32", family="category", _class="float32", perishable="int8")
    if file.endswith("stores.csv"):
        return dict(store_nbr="int16", city="category", state="category", type="category", cluster="int16")
    if file.endswith("transactions.csv"):
        return dict(store_nbr="int16", transactions="int32")
    if file.endswith("oil.csv"):
        return dict(dcoilwtico="float32")
    return {}

def resolve_in_dir(data_dir: str, name: str) -> str:
    # prefer plain CSV, else .7z
    csv = os.path.join(data_dir, name.replace(".7z",""))
    if os.path.exists(csv):
        return csv
    seven = os.path.join(data_dir, name)
    if os.path.exists(seven):
        out_dir = os.path.join(data_dir, "_extracted", os.path.splitext(name)[0])
        return extract_7z_if_needed(seven, out_dir)
    raise FileNotFoundError(f"Missing {name}(.7z) in {data_dir}")

def load_favorita(data_dir: str, since: str = None) -> pd.DataFrame:
    train_path = resolve_in_dir(data_dir, "train.csv.7z")
    items_path = resolve_in_dir(data_dir, "items.csv.7z")
    stores_path = resolve_in_dir(data_dir, "stores.csv.7z")
    holidays_path = resolve_in_dir(data_dir, "holidays_events.csv.7z")
    oil_path = resolve_in_dir(data_dir, "oil.csv.7z")
    trans_path = resolve_in_dir(data_dir, "transactions.csv.7z")

    train = pd.read_csv(train_path, parse_dates=["date"], dtype=favor_dtypes("train.csv"))
    items = pd.read_csv(items_path).rename(columns={"class":"_class"})
    stores = pd.read_csv(stores_path, dtype=favor_dtypes("stores.csv"))
    holidays = pd.read_csv(holidays_path, parse_dates=["date"])
    oil = pd.read_csv(oil_path, parse_dates=["date"], dtype=favor_dtypes("oil.csv"))
    trans = pd.read_csv(trans_path, parse_dates=["date"], dtype=favor_dtypes("transactions.csv"))

    if since:
        dt = pd.to_datetime(since)
        train = train[train["date"] >= dt]
        trans = trans[trans["date"] >= dt]
        oil = oil[oil["date"] >= dt]
        holidays = holidays[holidays["date"] >= dt]

    # Merge item & store metadata
    df = (train.merge(items, on="item_nbr", how="left")
                .merge(stores, on="store_nbr", how="left"))
    # Weekly aggregate at (store × family)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    agg = (df.groupby(["store_nbr","family","week_start","type","cluster","city","state"], as_index=False)
             .agg(unit_sales=("unit_sales","sum"),
                  onpromo=("onpromotion","mean"),
                  perishable_rate=("perishable","mean")))

    # transactions weekly
    trans["week_start"] = trans["date"] - pd.to_timedelta(trans["date"].dt.weekday, unit="D")
    trans_w = trans.groupby(["store_nbr","week_start"], as_index=False)["transactions"].sum()
    agg = agg.merge(trans_w, on=["store_nbr","week_start"], how="left")

    # oil weekly
    oil["week_start"] = oil["date"] - pd.to_timedelta(oil["date"].dt.weekday, unit="D")
    oil_w = oil.groupby("week_start", as_index=False)["dcoilwtico"].mean()
    agg = agg.merge(oil_w, on="week_start", how="left")

    # holidays weekly binary (note: transferred nuance kept simple)
    holidays["is_holiday"] = 1
    hol_w = holidays.groupby("date", as_index=False)["is_holiday"].max()
    hol_w["week_start"] = hol_w["date"] - pd.to_timedelta(hol_w["date"].dt.weekday, unit="D")
    hol_w = hol_w.groupby("week_start", as_index=False)["is_holiday"].max()
    agg = agg.merge(hol_w, on="week_start", how="left")
    agg["is_holiday"] = agg["is_holiday"].fillna(0).astype("int8")

    # Map to pipeline schema
    panel = agg.rename(columns={
        "week_start":"date",
        "store_nbr":"Store",
        "unit_sales":"Sales",
        "onpromo":"Promo"
    })
    panel["region"] = panel["state"].astype("category")
    panel["channel"] = "RETAIL"
    panel["StoreType"] = panel["type"].astype("category")
    panel["Assortment"] = panel["cluster"].astype("int16")
    panel["price_index"] = 1.0   # ASP not provided; can be derived if needed
    panel["SchoolHoliday"] = 0
    panel["StateHoliday"] = panel["is_holiday"]

    keep = ["date","Store","family","region","channel","StoreType","Assortment",
            "Promo","Sales","price_index"]
    panel = panel[keep].sort_values(["Store","family","date"]).reset_index(drop=True)
    return panel

# -------------------------
# Feature engineering & modeling
# -------------------------
def make_lags(df: pd.DataFrame, tgt="Sales", lags=(1,2,4,8,13,26,52), group=("Store","family")) -> pd.DataFrame:
    d = df.sort_values(list(group)+["date"]).copy()
    for L in lags:
        d[f"lag_{L}"] = d.groupby(list(group))[tgt].shift(L)
    for w in (4,8,13,26,52):
        d[f"ma_{w}"] = d.groupby(list(group))[tgt].transform(lambda s: s.rolling(w, min_periods=1).mean())
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    return d

def fit_models(train_df: pd.DataFrame, feature_cols: List[str], target="Sales",
               alpha_lo=0.10, alpha_hi=0.90) -> Dict[str, GradientBoostingRegressor]:
    mdl_med = GradientBoostingRegressor(random_state=7)
    mdl_lo = GradientBoostingRegressor(loss="quantile", alpha=alpha_lo, random_state=7)
    mdl_hi = GradientBoostingRegressor(loss="quantile", alpha=alpha_hi, random_state=7)

    X = train_df[feature_cols].fillna(train_df[feature_cols].median(numeric_only=True))
    y = train_df[target].values

    mdl_med.fit(X, y)
    mdl_lo.fit(X, y)
    mdl_hi.fit(X, y)
    return {"med": mdl_med, "lo": mdl_lo, "hi": mdl_hi}

def rolling_backtest(df: pd.DataFrame, feature_cols: List[str], target="Sales",
                     cutoff_weeks=16) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, GradientBoostingRegressor], pd.DataFrame]:
    last_date = df["date"].max()
    cutoff = last_date - pd.Timedelta(weeks=cutoff_weeks)
    tr = df[df["date"] <= cutoff].copy()
    te = df[df["date"] > cutoff].copy()

    models = fit_models(tr, feature_cols, target)
    Xte = te[feature_cols].fillna(tr[feature_cols].median(numeric_only=True))
    te["pred_med"] = models["med"].predict(Xte)

    WAPE = np.sum(np.abs(te[target] - te["pred_med"])) / np.sum(np.abs(te[target])) * 100
    MAPE = np.mean(np.where(te[target]==0, np.nan, np.abs((te[target] - te["pred_med"]) / te[target]))) * 100
    RMSE = math.sqrt(mean_squared_error(te[target], te["pred_med"]))
    MAE = mean_absolute_error(te[target], te["pred_med"])
    SMA = smape(te[target].values, te["pred_med"].values)
    MASE = mase(te[target].values, te["pred_med"].values, tr[target].values)

    metrics = {"WAPE%": WAPE, "MAPE%": MAPE, "RMSE": RMSE, "MAE": MAE, "SMAPE%": SMA, "MASE": MASE}
    return te, metrics, models, tr

def forecast_future(hist_df: pd.DataFrame, models: Dict[str, GradientBoostingRegressor],
                    horizon_weeks=52, scenario="BASE") -> pd.DataFrame:
    last_date = hist_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W-SUN")

    rows = []
    for (store, fam), base in hist_df.groupby(["Store","family"]):
        base = base.sort_values("date").iloc[-1:]
        for dt in future_dates:
            rows.append({
                "Store": store,
                "family": fam,
                "date": dt,
                "region": base["region"].values[0],
                "channel": base["channel"].values[0],
                "StoreType": base["StoreType"].values[0],
                "Assortment": base["Assortment"].values[0],
                "Promo": 1 if (scenario=="PROMO" and (dt.isocalendar().week in [13,14,39,40])) else 0,
                "SchoolHoliday": 0,
                "StateHoliday": 0,
                "price_index": base["price_index"].values[0]
            })
    fut = pd.DataFrame(rows)

    if scenario == "+2% PRICE":
        fut["price_index"] = fut["price_index"] * 1.02

    hist_tail = hist_df[["Store","family","date","Sales","region","channel","StoreType","Assortment",
                         "Promo","SchoolHoliday","StateHoliday","price_index"]].copy()
    full = pd.concat([hist_tail, fut], ignore_index=True).sort_values(["Store","family","date"])

    full = make_lags(full, tgt="Sales", lags=(1,2,4,8,13,26,52), group=("Store","family"))

    feature_cols = [c for c in full.columns if c not in ["Sales","date"]]
    feature_cols = [c for c in feature_cols if full[c].dtype != "O"]

    Xf = full[full["date"].isin(fut["date"])][feature_cols].fillna(full[feature_cols].median(numeric_only=True))
    preds_med = models["med"].predict(Xf)
    preds_lo = models["lo"].predict(Xf)
    preds_hi = models["hi"].predict(Xf)

    out = fut.copy()
    out["pred_med"] = np.maximum(0, preds_med)
    out["pred_lo"] = np.maximum(0, np.minimum(preds_med, preds_lo))
    out["pred_hi"] = np.maximum(0, np.maximum(preds_med, preds_hi))

    if scenario == "SUPPLY CAP":
        mask = out["family"]=="AESTHETICS"
        out.loc[mask, ["pred_med","pred_lo","pred_hi"]] *= 0.9

    return out

# -------------------------
# Finance
# -------------------------
def finance_map(df_pred: pd.DataFrame, gtn=GTN_DEFAULT, cogs=COGS_DEFAULT) -> pd.DataFrame:
    f = df_pred.copy()
    f["net_revenue"] = f["pred_med"]
    f["gross_revenue"] = f["net_revenue"] / (1 - gtn)
    f["gtn_deduction"] = f["gross_revenue"] - f["net_revenue"]
    f["cogs"] = f["net_revenue"] * cogs
    f["gross_margin"] = f["net_revenue"] - f["cogs"]
    f["contribution"] = f["gross_margin"]
    return f

def revenue_bridge(last_q: pd.DataFrame, next_q: pd.DataFrame) -> pd.DataFrame:
    b = last_q.merge(next_q, on="family", how="outer", suffixes=("_last","_next")).fillna(0)
    b["delta"] = b["net_revenue_next"] - b["net_revenue_last"]
    return b

# -------------------------
# Main
# -------------------------
def main(args):
    ensure_dir(args.output_dir)

    # 1) Load & weekly panel
    if args.dataset == "rossmann":
        if not (args.train and args.store):
            raise SystemExit("--train and --store are required for dataset=rossmann")
        raw = load_rossmann(args.train, args.store)
        raw = add_calendar_features(raw, "Date", daily=True)
        raw = add_promo_history(raw)
        wk = weekly_panel_rossmann(raw)
    else:
        if not args.data_dir:
            raise SystemExit("--data_dir is required for dataset=favorita")
        wk = load_favorita(args.data_dir, since=args.since)

    # Persist the model-ready weekly panel
    wk.to_csv(os.path.join(args.output_dir, "weekly_panel_model_ready.csv"), index=False)

    # 2) Lags/features
    wk = make_lags(wk, tgt="Sales", lags=(1,2,4,8,13,26,52), group=("Store","family"))
    feat_cols = ["Promo","SchoolHoliday","price_index",
                 "week","month","year",
                 "lag_1","lag_2","lag_4","lag_8","lag_13","lag_26","lag_52",
                 "ma_4","ma_8","ma_13","ma_26","ma_52"]
    wk_model = wk.dropna(subset=["lag_1","lag_2","lag_4"]).copy()

    # 3) Backtest & models
    test_df, metrics, models, train_df = rolling_backtest(wk_model, feat_cols, target="Sales", cutoff_weeks=16)

    # ABC tiers (by train contribution)
    contrib = train_df.groupby(["Store","family"])["Sales"].sum().sort_values(ascending=False)
    tiers = abc_tiers(contrib)
    tiers_df = tiers.reset_index().rename(columns={0:"tier"})
    tiers_df.to_csv(os.path.join(args.output_dir, "abc_tiers.csv"), index=False)

    # 4) Forecast scenarios
    fc_base   = forecast_future(wk_model, models, horizon_weeks=args.horizon_weeks, scenario="BASE")
    fc_promo  = forecast_future(wk_model, models, horizon_weeks=args.horizon_weeks, scenario="PROMO")
    fc_price  = forecast_future(wk_model, models, horizon_weeks=args.horizon_weeks, scenario="+2% PRICE")
    fc_supply = forecast_future(wk_model, models, horizon_weeks=args.horizon_weeks, scenario="SUPPLY CAP")

    # 5) Finance mapping
    fin_base   = finance_map(fc_base, gtn=args.gtn, cogs=args.cogs)
    fin_promo  = finance_map(fc_promo, gtn=args.gtn, cogs=args.cogs)
    fin_price  = finance_map(fc_price, gtn=args.gtn, cogs=args.cogs)
    fin_supply = finance_map(fc_supply, gtn=args.gtn, cogs=args.cogs)

    # Promo ROI vs base
    cmp_cols = ["Store","family","date","net_revenue","cogs","gross_margin"]
    base_cmp = fin_base[cmp_cols].rename(columns={"net_revenue":"base_nr","cogs":"base_cogs","gross_margin":"base_gm"})
    promo_cmp = fin_promo[cmp_cols].merge(base_cmp, on=["Store","family","date"], how="left")
    promo_cmp["incr_nr"] = promo_cmp["net_revenue"] - promo_cmp["base_nr"]
    promo_cmp["incr_gp"] = (promo_cmp["gross_margin"]) - (promo_cmp["base_gm"])
    promo_cmp["promo_cost"] = np.where(promo_cmp["incr_nr"]>0, PROMO_COST_RATE*promo_cmp["incr_nr"], 0.0)
    promo_cmp["ROI"] = np.where(promo_cmp["promo_cost"]>0, promo_cmp["incr_gp"]/promo_cmp["promo_cost"], np.nan)

    # 6) Aggregations for slides
    def agg_family(fin):
        g = fin.groupby(["date","family"], as_index=False).agg(
            net_revenue=("net_revenue","sum"),
            gross_margin=("gross_margin","sum"))
        return g

    dash_all = pd.concat([
        agg_family(fin_base).assign(scenario="BASE"),
        agg_family(fin_promo).assign(scenario="PROMO"),
        agg_family(fin_price).assign(scenario="+2% PRICE"),
        agg_family(fin_supply).assign(scenario="SUPPLY CAP")
    ], ignore_index=True)

    # Revenue Bridge: last quarter actual vs next quarter base
    hist_last_q = (wk_model[wk_model["date"] > wk_model["date"].max()-pd.Timedelta(weeks=13)]
                   .groupby("family", as_index=False)["Sales"].sum()
                   .rename(columns={"Sales":"net_revenue"}))
    next_q = (fin_base.groupby("family", as_index=False)["net_revenue"].sum()
              .rename(columns={"net_revenue":"net_revenue"}))
    bridge = revenue_bridge(hist_last_q, next_q)

    # 7) Inventory (safety stock per family)
    resid = test_df["Sales"] - test_df["pred_med"]
    sigma = np.std(resid)
    safety_rows = []
    for fam, lead in LEAD_WEEKS.items():
        ss = SERVICE_LEVEL_Z * max(1.0, sigma) * math.sqrt(lead)
        safety_rows.append({"family": fam, "lead_weeks": lead, "safety_stock_units": ss})
    safety_df = pd.DataFrame(safety_rows)

    # 8) Charts
    ensure_dir(os.path.join(args.output_dir, "plots"))
    top_fam = dash_all[dash_all["scenario"]=="BASE"].groupby("family")["net_revenue"].sum().sort_values(ascending=False).index[0]
    plt.figure()
    for scen, d in dash_all[dash_all["family"]==top_fam].groupby("scenario"):
        plt.plot(d["date"], d["net_revenue"], label=scen)
    plt.legend()
    plt.title(f"Net Revenue Forecast — {top_fam}")
    plt.xlabel("Week")
    plt.ylabel("Net Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plots", f"net_revenue_{top_fam}.png"))
    plt.close()

    plt.figure()
    plt.bar(bridge["family"], bridge["net_revenue_last"], label="Last Q")
    plt.bar(bridge["family"], bridge["delta"], bottom=bridge["net_revenue_last"], label="Δ to Next Q (Base)")
    plt.legend()
    plt.title("Revenue Bridge: Last Quarter → Next Quarter (Base)")
    plt.xlabel("Family")
    plt.ylabel("Net Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plots", "revenue_bridge.png"))
    plt.close()

    plt.figure()
    plt.hist(promo_cmp["ROI"].dropna(), bins=30)
    plt.title("Promo ROI distribution (by Store×Family)")
    plt.xlabel("ROI (Gross Profit / Promo Cost)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plots", "promo_roi_hist.png"))
    plt.close()

    # 9) Save tables
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "backtest_metrics.csv"), index=False)
    dash_all.to_csv(os.path.join(args.output_dir, "scenario_family_week.csv"), index=False)
    promo_cmp.groupby(["Store","family"], as_index=False)[["ROI","incr_gp","promo_cost"]].sum().to_csv(
        os.path.join(args.output_dir, "promo_roi_by_store.csv"), index=False)
    safety_df.to_csv(os.path.join(args.output_dir, "safety_stock.csv"), index=False)
    # weekly panel already saved

    print("Backtest metrics:", json.dumps(metrics, indent=2))
    print(f"Top family for visuals: {top_fam}")
    print("Artifacts saved to:", os.path.abspath(args.output_dir))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rossmann","favorita"], required=True, help="Which dataset to load")
    # Rossmann
    p.add_argument("--train", help="Rossmann train.csv")
    p.add_argument("--store", help="Rossmann store.csv")
    # Favorita
    p.add_argument("--data_dir", help="Directory containing Favorita CSV or .7z files")
    p.add_argument("--since", default=None, help="Optional filter date YYYY-MM-DD for Favorita")
    # Common
    p.add_argument("--output_dir", default="./outputs", help="Where to save outputs")
    p.add_argument("--horizon_weeks", type=int, default=52, help="Forecast horizon (weeks)")
    p.add_argument("--gtn", type=float, default=GTN_DEFAULT, help="Gross-to-Net % (0–1)")
    p.add_argument("--cogs", type=float, default=COGS_DEFAULT, help="COGS % of Net Revenue (0–1)")
    args = p.parse_args()

    main(args)
