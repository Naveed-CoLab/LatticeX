"""Crash risk: probability that a stock FALLS more than X% (default 10%) within the next h trading days.

    python -m scripts.run_crash_risk --data synthetic --signal volcluster     # plumbing test on fake data
    python -m scripts.run_crash_risk --data real --fundamentals --sectors      # 20-day, drop > 10%
    python -m scripts.run_crash_risk --data real --fundamentals --sectors --horizon 5 --drop 0.05

What it can and cannot do: a surprise headline can't be predicted, but the CONDITIONS that make big drops
more likely can be measured: high / rising volatility, high beta, a big run-up, crowded sector, stressed
market (VIX, credit spreads), an earnings report coming up (days since the last filing ~ quarterly cycle).

Method (same walk-forward folds, embargo and holdout lock as every other model):
  1. XGBoost classifier on price + volatility + market + (optional) fundamental + sector features,
     including the RAW (unranked) volatility, beta and return levels: absolute risk matters here.
  2. Calibration: isotonic regression on the most recent 20% of training years (after an embargo gap),
     so "20% crash risk" means about 20% of such stocks really dropped that much.
Baselines (the honest bars to beat):
  baserate    every stock gets the training-years crash frequency
  vol_only    logistic regression on recent volatility + beta: "risky stocks crash more" (hard to beat)
Scores on the test years:
  auc                 how well the score separates crashes from non-crashes (0.5 = useless, 1 = perfect)
  brier / brier_skill squared error of the probability; skill = improvement over baserate (higher better)
  top10_crash_rate    actual crash rate among the 10% highest-risk stocks each date (vs the average)
  lift_top10          top10_crash_rate / average crash rate
Output: results/<run>/crash_metrics.csv, crash_calibration.csv, crash_auc_by_year.csv, predictions.parquet
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.run_experiment import build_panel, load_prices
from src.config import load_settings, path
from src.features.price import RAW_FEATURES
from src.folds import eval_dates, make_folds

PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 50,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0}


def vol_inputs(df: pd.DataFrame) -> np.ndarray:
    v = df["vol_20_raw"].fillna(df["vol_60_raw"]).fillna(0.02).clip(lower=0.003)
    b = df["beta_60_raw"].fillna(1.0).clip(-1, 4)
    return np.column_stack([np.log(v), b])


def score(y: np.ndarray, p: np.ndarray, dates: np.ndarray, base: float) -> dict:
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    d = pd.DataFrame({"d": dates, "p": p, "y": y})
    top = d[d.groupby("d")["p"].rank(pct=True, method="first") > 0.9]
    return {"n": len(y), "crash_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "brier": brier, "brier_skill": 1 - brier / brier_base if brier_base > 0 else float("nan"),
            "top10_crash_rate": float(top["y"].mean()), "lift_top10": float(top["y"].mean() / max(y.mean(), 1e-9))}


def run(source="synthetic", signal="volcluster", horizon=None, drop=0.10, train_step=5, n_stocks=60,
        asset="stock", name=None, verbose=True, use_fund=False, use_sectors=False) -> dict:
    s = load_settings()
    h = horizon or s["main_horizon"]
    ycol, end_col = f"fret_{h}", f"target_end_{h}"
    t0 = time.time()
    prices = load_prices(source, signal, n_stocks, s["benchmark"], verbose)
    panel, feats = build_panel(prices, s, source, use_macro=True, use_fund=use_fund, use_sectors=use_sectors)
    panel = panel[panel["asset"] == asset].reset_index(drop=True)
    feats = feats + RAW_FEATURES
    thr = float(np.log(1 - drop))
    panel["crash"] = (panel[ycol] < thr).astype(float).where(panel[ycol].notna())
    holdout = pd.Timestamp(s["holdout_start"])
    grid, train_grid = set(eval_dates(panel["date"], s["eval_step"])), set(eval_dates(panel["date"], train_step))
    folds = make_folds(panel["date"], s["folds"]["first_test_year"], s["folds"]["last_test_year"],
                       s["folds"]["embargo_days"], s["holdout_start"])
    if verbose:
        print(f"Crash risk: P(drop > {drop:.0%} within {h} days); {len(panel):,} rows, {panel['ticker'].nunique()} "
              f"{asset}s, {len(feats)} features, {len(folds)} folds; overall crash rate {panel['crash'].mean():.1%}")

    out, imps = [], []
    for f in folds:
        ok = panel["crash"].notna()
        train = panel[ok & (panel["date"] <= f.train_end) & panel["date"].isin(train_grid)]
        test = panel[ok & (panel["date"] >= f.test_start) & (panel["date"] <= f.test_end)
                     & panel["date"].isin(grid) & (panel[end_col] < holdout)]
        if train.empty or test.empty or train["crash"].nunique() < 2:
            continue
        assert train[end_col].max() < f.test_start, "target leakage"
        dates = np.sort(train["date"].unique())
        cut = pd.Timestamp(dates[int(len(dates) * 0.8)])
        inner = train[train["date"] <= cut - pd.offsets.BDay(s["folds"]["embargo_days"])]
        cal = train[train["date"] >= cut]
        if inner["crash"].nunique() < 2 or cal["crash"].nunique() < 2:
            continue

        clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=42,
                                device=os.getenv("LATTICEIQ_DEVICE", "cpu"), **PARAMS)
        clf.fit(inner[feats], inner["crash"].values)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        iso.fit(clf.predict_proba(cal[feats])[:, 1], cal["crash"].values)
        lr = LogisticRegression(max_iter=1000).fit(vol_inputs(inner), inner["crash"].values)
        iso_v = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        iso_v.fit(lr.predict_proba(vol_inputs(cal))[:, 1], cal["crash"].values)

        rows = test[["date", "ticker"]].assign(fold=f.year, y_true=test["crash"].values, ret=test[ycol].values)
        rows["model_p"] = iso.predict(clf.predict_proba(test[feats])[:, 1])
        rows["vol_only_p"] = iso_v.predict(lr.predict_proba(vol_inputs(test))[:, 1])
        rows["baserate_p"] = float(train["crash"].mean())
        out.append(rows)
        imps.append(pd.Series(clf.get_booster().get_score(importance_type="gain")).rename(f.year))
        if verbose:
            a = roc_auc_score(rows["y_true"], rows["model_p"]) if rows["y_true"].nunique() > 1 else float("nan")
            print(f"  fold {f.year}: train {len(inner):,} + calibrate {len(cal):,}, test {len(test):,}, "
                  f"crash rate {rows['y_true'].mean():.1%}, AUC {a:.3f}  [{time.time() - t0:.0f}s]")

    pred = pd.concat(out, ignore_index=True)
    y, d = pred["y_true"].values, pred["date"].values
    base = float(pred["baserate_p"].mean())
    table = pd.DataFrame([{"method": m, **score(y, pred[f"{m}_p"].values, d, base)}
                          for m in ("model", "vol_only", "baserate")])
    deciles = pd.qcut(pred["model_p"].rank(method="first"), 10, labels=range(1, 11))
    calib = pred.groupby(deciles, observed=True).agg(predicted=("model_p", "mean"), actual=("y_true", "mean"),
                                                     n=("y_true", "size"))
    by_year = pred.groupby("fold").apply(lambda g: pd.Series({
        "crash_rate": g["y_true"].mean(),
        "auc_model": roc_auc_score(g["y_true"], g["model_p"]) if g["y_true"].nunique() > 1 else np.nan,
        "auc_vol_only": roc_auc_score(g["y_true"], g["vol_only_p"]) if g["y_true"].nunique() > 1 else np.nan}),
        include_groups=False)

    tag = f"{source}_{asset}_h{h}_crash{int(drop * 100)}{'_fund' if use_fund else ''}{'_sect' if use_sectors else ''}"
    outdir = path("results") / (name or f"{datetime.now():%Y%m%d_%H%M}_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "crash_metrics.csv", index=False)
    calib.to_csv(outdir / "crash_calibration.csv")
    by_year.to_csv(outdir / "crash_auc_by_year.csv")
    if imps:
        pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).to_csv(outdir / "importance.csv")
    pred.to_parquet(outdir / "predictions.parquet", index=False)
    (outdir / "config.json").write_text(json.dumps({"source": source, "signal": signal, "horizon": h, "drop": drop,
                                                    "features": feats, "fundamentals": use_fund, "sectors": use_sectors,
                                                    "params": PARAMS, "settings": s}, indent=2, default=str))
    if verbose:
        pd.set_option("display.width", 160)
        print("\n" + table.round(4).to_string(index=False))
        print(f"\nCalibration by predicted-risk decile (1 = safest, 10 = riskiest):\n{calib.round(3).to_string()}")
        print(f"\nAUC by year:\n{by_year.round(3).to_string()}")
        if imps:
            print("Top risk drivers: " + ", ".join(pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).head(8).index))
        better = table.set_index("method").loc["model", "auc"] > table.set_index("method").loc["vol_only", "auc"]
        print(f"\nModel {'BEATS' if better else 'does NOT beat'} the volatility-only baseline on AUC.")
        print(f"Saved to {outdir}  ({time.time() - t0:.0f}s)")
    return {"metrics": table, "calibration": calib, "by_year": by_year, "out": outdir}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--signal", choices=["none", "momentum", "volcluster"], default="volcluster")
    p.add_argument("--horizon", type=int)
    p.add_argument("--drop", type=float, default=0.10, help="crash = fall of more than this fraction (0.10 = 10%%)")
    p.add_argument("--asset", choices=["stock", "etf"], default="stock")
    p.add_argument("--train-step", type=int, default=5)
    p.add_argument("--n-stocks", type=int, default=60)
    p.add_argument("--fundamentals", action="store_true")
    p.add_argument("--sectors", action="store_true")
    p.add_argument("--name")
    a = p.parse_args()
    run(a.data, a.signal, a.horizon, a.drop, a.train_step, a.n_stocks, a.asset, a.name,
        use_fund=a.fundamentals, use_sectors=a.sectors)
