"""Forecasts from pretrained time-series models (Kronos now, TimesFM later) as features / standalone models.

Forecasts are made every 20th trading day (monthly) using only data up to that day; each value is carried
forward for at most 31 calendar days so it lines up with the weekly evaluation grid without reaching into
the future (a forecast is only ever used on or after the day it was made).
  kronos_xret_5 / kronos_xret_20   Kronos forecast return minus Kronos's SPY forecast (excess vs SPY)
"""
import pandas as pd

from src.config import path

KRONOS_FEATURES = ["kronos_xret_5", "kronos_xret_20"]


def kronos_features(panel_index: pd.DataFrame, benchmark: str = "SPY", cross_sectional_rank: bool = True,
                    keep_raw: bool = True) -> pd.DataFrame:
    f = path("clean") / "kronos_forecasts.parquet"
    if not f.exists():
        raise SystemExit("data/clean/kronos_forecasts.parquet missing. Run: python -m scripts.kaggle_cloud kronos ...")
    k = pd.read_parquet(f)
    k["date"] = pd.to_datetime(k["date"])
    spy = k[k["ticker"] == benchmark][["date", "kronos_ret_5", "kronos_ret_20"]].rename(
        columns={"kronos_ret_5": "s5", "kronos_ret_20": "s20"})
    k = k[k["ticker"] != benchmark].merge(spy, on="date", how="left")
    k["kronos_xret_5"] = k["kronos_ret_5"] - k["s5"]
    k["kronos_xret_20"] = k["kronos_ret_20"] - k["s20"]
    k = k.dropna(subset=["kronos_xret_20"]).sort_values("date")
    base = panel_index[["date", "ticker"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(base, k[["date", "ticker"] + KRONOS_FEATURES].rename(columns={"date": "kdate"}),
                      left_on="date", right_on="kdate", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=31)).drop(columns="kdate")
    if keep_raw:                                                # unranked copies, for the standalone model row
        m["kronos_raw_5"], m["kronos_raw_20"] = m["kronos_xret_5"], m["kronos_xret_20"]
    if cross_sectional_rank:
        m[KRONOS_FEATURES] = m.groupby("date")[KRONOS_FEATURES].rank(pct=True) - 0.5
    return m
