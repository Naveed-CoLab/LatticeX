"""Forecasts from pretrained time-series models (Kronos, TimesFM) as features / standalone models.

Forecasts are made on a grid of dates (Kronos monthly, TimesFM weekly) using only data up to that day; each
value is carried forward for at most 31 calendar days so it lines up with the weekly evaluation grid without
reaching into the future (a forecast is only ever used on or after the day it was made).
  kronos_xret_5 / kronos_xret_20     Kronos forecast return minus Kronos's SPY forecast (excess vs SPY)
  timesfm_xret_5 / timesfm_xret_20   same for TimesFM
  timesfm_unc_20                     TimesFM 80% band width of the 20-day return (its own uncertainty)
Raw (unranked) excess forecasts are kept as <model>_raw_5 / <model>_raw_20 for the standalone model rows.
"""
import pandas as pd

from src.config import path

KRONOS_FEATURES = ["kronos_xret_5", "kronos_xret_20"]
TIMESFM_FEATURES = ["timesfm_xret_5", "timesfm_xret_20", "timesfm_unc_20"]
FEATURES = {"kronos": KRONOS_FEATURES, "timesfm": TIMESFM_FEATURES}


def pretrained_features(model: str, panel_index: pd.DataFrame, benchmark: str = "SPY",
                        cross_sectional_rank: bool = True, keep_raw: bool = True) -> pd.DataFrame:
    feats = FEATURES[model]
    f = path("clean") / f"{model}_forecasts.parquet"
    if not f.exists():
        raise SystemExit(f"data/clean/{model}_forecasts.parquet missing. Run: python -m scripts.kaggle_cloud {model} ...")
    k = pd.read_parquet(f)
    k["date"] = pd.to_datetime(k["date"])
    r5, r20 = f"{model}_ret_5", f"{model}_ret_20"
    spy = k[k["ticker"] == benchmark][["date", r5, r20]].rename(columns={r5: "s5", r20: "s20"})
    k = k[k["ticker"] != benchmark].merge(spy, on="date", how="left")
    k[f"{model}_xret_5"] = k[r5] - k["s5"]
    k[f"{model}_xret_20"] = k[r20] - k["s20"]
    if model == "timesfm":
        k["timesfm_unc_20"] = k["timesfm_q90_20"] - k["timesfm_q10_20"]
    k = k.dropna(subset=[f"{model}_xret_20"]).sort_values("date")
    base = panel_index[["date", "ticker"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(base, k[["date", "ticker"] + feats].rename(columns={"date": "kdate"}),
                      left_on="date", right_on="kdate", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=31)).drop(columns="kdate")
    if keep_raw:                                                # unranked copies, for the standalone model row
        m[f"{model}_raw_5"], m[f"{model}_raw_20"] = m[f"{model}_xret_5"], m[f"{model}_xret_20"]
    if cross_sectional_rank:
        m[feats] = m.groupby("date")[feats].rank(pct=True) - 0.5
    return m


def kronos_features(panel_index: pd.DataFrame, benchmark: str = "SPY", cross_sectional_rank: bool = True,
                    keep_raw: bool = True) -> pd.DataFrame:
    return pretrained_features("kronos", panel_index, benchmark, cross_sectional_rank, keep_raw)


def timesfm_features(panel_index: pd.DataFrame, benchmark: str = "SPY", cross_sectional_rank: bool = True,
                     keep_raw: bool = True) -> pd.DataFrame:
    return pretrained_features("timesfm", panel_index, benchmark, cross_sectional_rank, keep_raw)
