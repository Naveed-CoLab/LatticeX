"""Recovery after a crash: once a stock has dropped sharply, how likely is it to climb back to its
pre-drop price, and how long does that usually take?

    python -m scripts.run_recovery --data synthetic --signal volcluster        # plumbing test on fake data
    python -m scripts.run_recovery --data real --fundamentals --sectors         # P(recover within 60 days)
    python -m scripts.run_recovery --data real --fundamentals --sectors --within 20

Crash event  = the stock fell `--drop` (default 10%) or more over `--window` (default 5) trading days.
               One event per stock per 20 trading days (a long slide is not counted many times).
Recovered    = a later close is back at or above the PRE-DROP price (the close `window` days before).

Two outputs:
  A. History table (recovery_curve.csv): of all past crashes, the share recovered within 5/10/20/60/120/250
     trading days and the median days to recover, split by kind of crash: market-wide vs stock-specific,
     size of the drop, one-day shock vs gradual slide. Descriptive: what usually happened.
  B. Model: P(recover within `--within` days) for each crash, from what is known on the crash day: size and
     shape of the drop, how much of it was the whole market, volatility, beta, momentum before the drop,
     market stress (VIX, credit), fundamentals, sector. XGBoost classifier + isotonic calibration, tested
     walk-forward (same folds, embargo and holdout lock as every other model).
     Baselines: baserate (training-years recovery frequency) and drop_only (logistic on drop size + market move).
Limits: a stock that was delisted before the window ended has an unknown outcome and is left out (bankruptcies
and takeovers), so real recovery rates are a little lower than shown.
Output: results/<run>/recovery_metrics.csv, recovery_curve.csv, recovery_calibration.csv,
        recovery_auc_by_year.csv, importance.csv, predictions.parquet
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
from src.folds import make_folds

CURVE = [5, 10, 20, 60, 120, 250]
EVENT_FEATURES = ["ev_drop", "ev_market", "ev_stock_specific", "ev_worst_day", "ev_below_high"]
PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 20,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0}


def find_events(prices: pd.DataFrame, benchmark: str = "SPY", drop: float = 0.10, window: int = 5,
                cooldown: int = 20, max_days: int = 250) -> pd.DataFrame:
    """One row per crash event. Everything in ev_* is known at the close of the event day.
    days = trading days until the close is back at the pre-drop price (NaN = not within what we can see);
    avail = how many future days of prices exist for that stock (to know whether 'not recovered' is certain)."""
    w = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    V = w.values
    r = V / w.shift(window).values - 1
    ref = w.shift(window).values                                   # pre-drop price
    worst = (w / w.shift(1) - 1).rolling(window).min().values      # biggest one-day fall inside the window
    high = w.shift(window).rolling(252, min_periods=60).max().values
    b = list(w.columns).index(benchmark)
    rows = []
    for j, t in enumerate(w.columns):
        if j == b:
            continue
        last = -10 ** 9
        for i in np.flatnonzero(r[:, j] <= -drop):
            if i - last < cooldown:
                continue
            last = i
            fut = V[i + 1:i + 1 + max_days, j]
            nan = np.flatnonzero(np.isnan(fut))
            avail = int(nan[0]) if len(nan) else len(fut)
            rec = np.flatnonzero(fut[:avail] >= ref[i, j])
            rows.append((i, w.index[i], t, r[i, j], r[i, b], worst[i, j],
                         ref[i, j] / high[i, j] - 1 if high[i, j] > 0 else np.nan,
                         float(rec[0] + 1) if len(rec) else np.nan, avail))
    ev = pd.DataFrame(rows, columns=["i", "date", "ticker", "ev_drop", "ev_market", "ev_worst_day",
                                     "ev_below_high", "days", "avail"])
    ev["ev_stock_specific"] = ev["ev_drop"] - ev["ev_market"]
    return ev


def recovered_within(ev: pd.DataFrame, h: int, last_i: int) -> pd.Series:
    """1 / 0 / NaN. Known only if the whole h-day window lies before `last_i` (the holdout / end of data)
    and the stock either recovered by day h or has h days of prices showing it did not."""
    known = (ev["i"] + h < last_i) & ((ev["days"] <= h) | (ev["avail"] >= h))
    return (ev["days"] <= h).astype(float).where(known)


def curve_table(ev: pd.DataFrame, last_i: int) -> pd.DataFrame:
    seg = {
        "all crashes": ev["ev_drop"] < 0,
        "market-wide (market also fell 5%+)": ev["ev_market"] <= -0.05,
        "stock-specific (market fell < 2%)": ev["ev_market"] > -0.02,
        "drop 10-15%": ev["ev_drop"] > -0.15,
        "drop 15-25%": (ev["ev_drop"] <= -0.15) & (ev["ev_drop"] > -0.25),
        "drop over 25%": ev["ev_drop"] <= -0.25,
        "one-day shock (a single day fell 8%+)": ev["ev_worst_day"] <= -0.08,
        "gradual slide (no day worse than 5%)": ev["ev_worst_day"] > -0.05,
    }
    rows = []
    for name, m in seg.items():
        e = ev[m]
        row = {"segment": name, "events": len(e)}
        for h in CURVE:
            row[f"recovered_{h}d"] = recovered_within(e, h, last_i).mean()
        full = e[recovered_within(e, CURVE[-1], last_i).notna()]
        d = full["days"].fillna(np.inf)
        med = d.median() if len(d) else np.nan
        row["median_days"] = med if np.isfinite(med) else np.nan       # NaN = more than 250 days
        rows.append(row)
    return pd.DataFrame(rows)


def score(y: np.ndarray, p: np.ndarray, base: float) -> dict:
    brier, brier_base = float(np.mean((p - y) ** 2)), float(np.mean((base - y) ** 2))
    order = np.argsort(p)
    n5 = max(len(y) // 5, 1)
    return {"n": len(y), "recovery_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "brier": brier, "brier_skill": 1 - brier / brier_base if brier_base > 0 else float("nan"),
            "top20_rate": float(y[order[-n5:]].mean()), "bottom20_rate": float(y[order[:n5]].mean())}


def run(source="synthetic", signal="volcluster", within=60, drop=0.10, window=5, n_stocks=60, asset="stock",
        name=None, verbose=True, use_fund=False, use_sectors=False) -> dict:
    s = load_settings()
    if within > s["folds"]["embargo_days"]:
        raise SystemExit(f"--within {within} is longer than the {s['folds']['embargo_days']}-day embargo: "
                         "training outcomes could overlap the test year.")
    t0 = time.time()
    prices = load_prices(source, signal, n_stocks, s["benchmark"], verbose)
    panel, feats = build_panel(prices, s, source, use_macro=True, use_fund=use_fund, use_sectors=use_sectors)
    panel = panel[panel["asset"] == asset].reset_index(drop=True)
    feats = feats + RAW_FEATURES + EVENT_FEATURES
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    last_i = int(cal.searchsorted(pd.Timestamp(s["holdout_start"])))      # nothing from the holdout is used
    ev = find_events(prices, s["benchmark"], drop, window)
    ev = ev.merge(panel[["date", "ticker"] + [c for c in feats if c not in EVENT_FEATURES]],
                  on=["date", "ticker"], how="inner")                      # index members with features only
    ev = ev[ev["i"] < last_i].reset_index(drop=True)
    curve = curve_table(ev, last_i)
    ev["y"] = recovered_within(ev, within, last_i)
    ev["end"] = cal[np.minimum(ev["i"].values + within, len(cal) - 1)]
    m_ev = ev[ev["y"].notna()].sort_values("date").reset_index(drop=True)
    folds = make_folds(panel["date"], s["folds"]["first_test_year"], s["folds"]["last_test_year"],
                       s["folds"]["embargo_days"], s["holdout_start"])
    if verbose:
        print(f"Recovery: crash = drop of {drop:.0%}+ in {window} days; {len(ev):,} events on "
              f"{ev['ticker'].nunique()} {asset}s; model target = back to the pre-drop price within {within} days "
              f"(happened in {m_ev['y'].mean():.1%} of cases); {len(feats)} features")

    out, imps = [], []
    for f in folds:
        train = m_ev[m_ev["date"] <= f.train_end]
        test = m_ev[(m_ev["date"] >= f.test_start) & (m_ev["date"] <= f.test_end)]
        if len(train) < 300 or len(test) < 20 or train["y"].nunique() < 2:
            continue
        assert train["end"].max() < f.test_start, "target leakage: a training outcome window reaches the test year"
        dates = np.sort(train["date"].unique())
        cut = pd.Timestamp(dates[int(len(dates) * 0.8)])
        inner = train[train["date"] <= cut - pd.offsets.BDay(s["folds"]["embargo_days"])]
        calib = train[train["date"] >= cut]
        if len(inner) < 200 or inner["y"].nunique() < 2 or calib["y"].nunique() < 2:
            continue
        clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=42,
                                device=os.getenv("LATTICEIQ_DEVICE", "cpu"), **PARAMS)
        clf.fit(inner[feats], inner["y"].values)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(clf.predict_proba(calib[feats])[:, 1], calib["y"].values)
        simple = ["ev_drop", "ev_market"]
        lr = LogisticRegression(max_iter=1000).fit(inner[simple].values, inner["y"].values)
        rows = test[["date", "ticker", "ev_drop", "ev_market", "days"]].assign(fold=f.year, y_true=test["y"].values)
        rows["model_p"] = iso.predict(clf.predict_proba(test[feats])[:, 1])
        rows["drop_only_p"] = lr.predict_proba(test[simple].values)[:, 1]
        rows["baserate_p"] = float(train["y"].mean())
        out.append(rows)
        imps.append(pd.Series(clf.get_booster().get_score(importance_type="gain")).rename(f.year))
        if verbose:
            a = roc_auc_score(rows["y_true"], rows["model_p"]) if rows["y_true"].nunique() > 1 else float("nan")
            print(f"  fold {f.year}: train {len(inner):,} + calibrate {len(calib):,} events, test {len(test):,}, "
                  f"recovered {rows['y_true'].mean():.0%}, AUC {a:.3f}  [{time.time() - t0:.0f}s]")

    pred = pd.concat(out, ignore_index=True)
    y, base = pred["y_true"].values, float(pred["baserate_p"].mean())
    table = pd.DataFrame([{"method": m, **score(y, pred[f"{m}_p"].values, base)}
                          for m in ("model", "drop_only", "baserate")])
    q = pd.qcut(pred["model_p"].rank(method="first"), 5, labels=range(1, 6))
    calib_t = pred.groupby(q, observed=True).agg(predicted=("model_p", "mean"), actual=("y_true", "mean"),
                                                 n=("y_true", "size"))
    by_year = pred.groupby("fold").apply(lambda g: pd.Series({
        "events": len(g), "recovered": g["y_true"].mean(),
        "auc_model": roc_auc_score(g["y_true"], g["model_p"]) if g["y_true"].nunique() > 1 else np.nan,
        "auc_drop_only": roc_auc_score(g["y_true"], g["drop_only_p"]) if g["y_true"].nunique() > 1 else np.nan}),
        include_groups=False)

    tag = (f"{source}_{asset}_recover{within}_drop{int(drop * 100)}"
           f"{'_fund' if use_fund else ''}{'_sect' if use_sectors else ''}")
    outdir = path("results") / (name or f"{datetime.now():%Y%m%d_%H%M}_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "recovery_metrics.csv", index=False)
    curve.to_csv(outdir / "recovery_curve.csv", index=False)
    calib_t.to_csv(outdir / "recovery_calibration.csv")
    by_year.to_csv(outdir / "recovery_auc_by_year.csv")
    if imps:
        pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).to_csv(outdir / "importance.csv")
    pred.to_parquet(outdir / "predictions.parquet", index=False)
    (outdir / "config.json").write_text(json.dumps({"source": source, "within": within, "drop": drop,
                                                    "window": window, "features": feats, "params": PARAMS,
                                                    "fundamentals": use_fund, "sectors": use_sectors,
                                                    "settings": s}, indent=2, default=str))
    if verbose:
        pd.set_option("display.width", 200)
        show = curve.copy()
        for c in [c for c in show if c.startswith("recovered_")]:
            show[c] = (show[c] * 100).round(0)
        print(f"\nHISTORY: % of crashes back at the pre-drop price within N trading days (median_days blank = over 250)"
              f"\n{show.to_string(index=False)}")
        print(f"\nMODEL: P(recover within {within} days), test years only\n{table.round(4).to_string(index=False)}")
        print(f"\nCalibration (1 = least likely to recover, 5 = most likely):\n{calib_t.round(3).to_string()}")
        print(f"\nBy year:\n{by_year.round(3).to_string()}")
        if imps:
            print("Top drivers: " + ", ".join(pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).head(8).index))
        t = table.set_index("method")
        print(f"\nModel {'BEATS' if t.loc['model', 'auc'] > t.loc['drop_only', 'auc'] else 'does NOT beat'} "
              f"the drop-size-only baseline on AUC.")
        print(f"Saved to {outdir}  ({time.time() - t0:.0f}s)")
    return {"metrics": table, "curve": curve, "calibration": calib_t, "by_year": by_year, "out": outdir}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--signal", choices=["none", "momentum", "volcluster"], default="volcluster")
    p.add_argument("--within", type=int, default=60, help="model target: recovered within this many trading days")
    p.add_argument("--drop", type=float, default=0.10, help="crash = fall of at least this fraction")
    p.add_argument("--window", type=int, default=5, help="...over this many trading days")
    p.add_argument("--asset", choices=["stock", "etf"], default="stock")
    p.add_argument("--n-stocks", type=int, default=60)
    p.add_argument("--fundamentals", action="store_true")
    p.add_argument("--sectors", action="store_true")
    p.add_argument("--name")
    a = p.parse_args()
    run(a.data, a.signal, a.within, a.drop, a.window, a.n_stocks, a.asset, a.name,
        use_fund=a.fundamentals, use_sectors=a.sectors)
