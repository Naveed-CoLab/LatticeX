"""Price brackets: for each stock and date, the likely range of its own return over the next h days.

    python -m scripts.run_brackets --data synthetic --signal volcluster    # test on fake data with calm/stormy periods
    python -m scripts.run_brackets --data real                             # 20-day brackets
    python -m scripts.run_brackets --data real --horizon 5

Method (per walk-forward fold, same folds/embargo/holdout lock as run_experiment):
  1. Scale: each stock's typical h-day move = recent daily volatility x sqrt(h).
  2. Quantile model: XGBoost predicts the 5/10/25/50/75/90/95% points of (return / scale),
     from price, volatility and market (VIX etc.) features.
  3. Conformal calibration: on the most recent 20% of training years (after an embargo gap) we
     measure how often the ranges missed, and widen/narrow them so "80%" really means 80%.
  4. Probability up = where 0% falls inside the predicted distribution.
Baseline: "naive" brackets from a normal distribution with the stock's recent volatility.

Scores (on the test years, never seen in training):
  coverage_80 / coverage_50  share of outcomes inside the 80% / 50% range (target 0.80 / 0.50)
  width_80                   average width of the 80% range in % (narrower is better at equal coverage)
  pinball                    standard quantile loss across all 7 points (lower is better)
  brier_up                   accuracy of the probability-up (lower is better; 0.25 = coin flip)
  baserate row               P(up) = share of up-moves in the training years (the honest bar to beat:
                             stocks rise more often than they fall, so 0.5 is too easy a baseline)
P(up) is calibrated with isotonic regression on the calibration years, so "60%" means ~60% of the time.
Output: results/<run>/bracket_metrics.csv, coverage_by_year.csv, predictions.parquet (with $ brackets)
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from scripts.run_experiment import build_panel, load_prices
from src.config import load_settings, path
from src.features.price import RAW_FEATURES
from src.folds import eval_dates, make_folds

QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
I = {q: i for i, q in enumerate(QS)}
INTERVALS = {80: (0.10, 0.90), 50: (0.25, 0.75), 90: (0.05, 0.95)}


def scale(df: pd.DataFrame, h: int) -> np.ndarray:
    v = df["vol_20_raw"].fillna(df["vol_60_raw"]).fillna(0.02).clip(lower=0.004)
    return (v * math.sqrt(h)).values


def fit_quantiles(X: pd.DataFrame, z: np.ndarray, p: dict) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QS, tree_method="hist",
                         device=os.getenv("LATTICEIQ_DEVICE", "cpu"), n_jobs=-1, random_state=42, **p)
    m.fit(X, z)
    return m


def predict_q(m, X) -> np.ndarray:
    return np.sort(m.predict(X).reshape(len(X), len(QS)), axis=1)   # sorted = no crossing quantiles


def conformal_shift(qz: np.ndarray, z: np.ndarray, lo: float, hi: float, level: float) -> float:
    """Normalized conformalized quantile regression: how much to widen (or narrow) an interval."""
    e = np.maximum(qz[:, I[lo]] - z, z - qz[:, I[hi]])
    n = len(e)
    return float(np.quantile(e, min(1.0, level * (1 + 1 / n))))


def apply_shifts(qz: np.ndarray, shifts: dict) -> np.ndarray:
    q = qz.copy()
    for lev, (lo, hi) in INTERVALS.items():
        q[:, I[lo]] -= shifts[lev]
        q[:, I[hi]] += shifts[lev]
    return np.sort(q, axis=1)


def prob_up(qr: np.ndarray) -> np.ndarray:
    """P(return > 0) by interpolating the predicted distribution; tails beyond 5/95% extrapolated flat."""
    out = np.empty(len(qr))
    for k, row in enumerate(qr):
        if 0 <= row[0]:
            out[k] = 1 - QS[0] / 2
        elif 0 >= row[-1]:
            out[k] = (1 - QS[-1]) / 2
        else:
            out[k] = 1 - np.interp(0.0, row, QS)
    return out


def pinball(qr: np.ndarray, y: np.ndarray) -> float:
    d = y[:, None] - qr
    return float(np.mean(np.maximum(QS * d, (QS - 1) * d)))


def score(df: pd.DataFrame, prefix: str) -> dict:
    y = df["y_true"].values
    q = df[[f"{prefix}q{int(x * 100):02d}" for x in QS]].values
    r = {"method": prefix.rstrip("_"), "n": len(df)}
    for lev, (lo, hi) in INTERVALS.items():
        inside = (y >= q[:, I[lo]]) & (y <= q[:, I[hi]])
        r[f"coverage_{lev}"] = float(inside.mean())
        r[f"width_{lev}_pct"] = float(np.mean(np.exp(q[:, I[hi]]) - np.exp(q[:, I[lo]])) * 100)
    r["pinball"] = pinball(q, y)
    p_up = df[f"{prefix}p_up"].values
    r["brier_up"] = float(np.mean((p_up - (y > 0)) ** 2))
    r["hit_up"] = float(np.mean((p_up > 0.5) == (y > 0)))
    return r


def run(source="synthetic", signal="volcluster", horizon=None, train_step=5, n_stocks=60, asset="stock",
        name=None, verbose=True, use_fund=False) -> dict:
    s = load_settings()
    h = horizon or s["main_horizon"]
    ycol, end_col = f"fret_{h}", f"target_end_{h}"
    t0 = time.time()
    prices = load_prices(source, signal, n_stocks, s["benchmark"], verbose)
    panel, feats = build_panel(prices, s, source, use_macro=True, use_fund=use_fund)
    panel = panel[panel["asset"] == asset].reset_index(drop=True)
    feats = feats + RAW_FEATURES
    px = prices[["date", "ticker", "adj_close"]].rename(columns={"adj_close": "price"})
    panel = panel.merge(px, on=["date", "ticker"], how="left")
    holdout = pd.Timestamp(s["holdout_start"])
    grid, train_grid = set(eval_dates(panel["date"], s["eval_step"])), set(eval_dates(panel["date"], train_step))
    folds = make_folds(panel["date"], s["folds"]["first_test_year"], s["folds"]["last_test_year"],
                       s["folds"]["embargo_days"], s["holdout_start"])
    qp = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 50,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0}
    if verbose:
        print(f"Brackets: {len(panel):,} rows, {panel['ticker'].nunique()} {asset}s, {len(feats)} features, "
              f"{len(folds)} folds, horizon {h}d")

    out, shifts_log = [], []
    for f in folds:
        ok = panel[ycol].notna() & panel["vol_20_raw"].notna()
        train = panel[ok & (panel["date"] <= f.train_end) & panel["date"].isin(train_grid)]
        test = panel[ok & (panel["date"] >= f.test_start) & (panel["date"] <= f.test_end)
                     & panel["date"].isin(grid) & (panel[end_col] < holdout)]
        if train.empty or test.empty:
            continue
        assert train[end_col].max() < f.test_start, "target leakage"
        dates = np.sort(train["date"].unique())
        cut = pd.Timestamp(dates[int(len(dates) * 0.8)])
        inner = train[train["date"] <= cut - pd.offsets.BDay(s["folds"]["embargo_days"])]
        cal = train[train["date"] >= cut]

        m = fit_quantiles(inner[feats], inner[ycol].values / scale(inner, h), qp)
        qz_cal = predict_q(m, cal[feats])
        z_cal = cal[ycol].values / scale(cal, h)
        shifts = {lev: conformal_shift(qz_cal, z_cal, lo, hi, lev / 100) for lev, (lo, hi) in INTERVALS.items()}
        shifts_log.append({"fold": f.year, **{f"shift_{k}": v for k, v in shifts.items()}})

        # calibrate P(up) on the calibration years: raw P(up) -> observed frequency of up-moves
        q_cal_r = apply_shifts(qz_cal, shifts) * scale(cal, h)[:, None]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(prob_up(q_cal_r), (cal[ycol].values > 0).astype(float))
        base_rate = float((train[ycol] > 0).mean())

        sc = scale(test, h)
        q_model = apply_shifts(predict_q(m, test[feats]), shifts) * sc[:, None]
        q_naive = norm.ppf(QS)[None, :] * sc[:, None]
        rows = test[["date", "ticker", "price"]].assign(fold=f.year, y_true=test[ycol].values)
        for prefix, q in (("model_", q_model), ("naive_", q_naive)):
            for j, x in enumerate(QS):
                rows[f"{prefix}q{int(x * 100):02d}"] = q[:, j]
            rows[f"{prefix}p_up"] = prob_up(q)
        rows["model_p_up_raw"] = rows["model_p_up"]
        rows["model_p_up"] = iso.predict(rows["model_p_up_raw"].values)
        rows["baserate_p_up"] = base_rate
        out.append(rows)
        if verbose:
            print(f"  fold {f.year}: train {len(inner):,} + calibrate {len(cal):,} rows, test {len(test):,}, "
                  f"80% shift {shifts[80]:+.2f}  [{time.time() - t0:.0f}s]")

    pred = pd.concat(out, ignore_index=True)
    for lev, (lo, hi) in INTERVALS.items():   # $ brackets for the model
        pred[f"price_lo_{lev}"] = pred["price"] * np.exp(pred[f"model_q{int(lo * 100):02d}"])
        pred[f"price_hi_{lev}"] = pred["price"] * np.exp(pred[f"model_q{int(hi * 100):02d}"])
    table = pd.DataFrame([score(pred, "model_"), score(pred, "naive_")])
    y_up = (pred["y_true"] > 0).astype(float)
    br = {"method": "baserate", "n": len(pred),
          "brier_up": float(np.mean((pred["baserate_p_up"] - y_up) ** 2)),
          "hit_up": float(np.mean((pred["baserate_p_up"] > 0.5) == (y_up > 0)))}
    table = pd.concat([table, pd.DataFrame([br])], ignore_index=True)
    # calibration table for the report: predicted P(up) bucket vs actual share of up-moves
    bins = pd.cut(pred["model_p_up"], [0, .4, .45, .5, .55, .6, .65, 1])
    calib = pred.assign(up=y_up).groupby(bins, observed=True).agg(predicted=("model_p_up", "mean"),
                                                                    actual=("up", "mean"), n=("up", "size"))
    by_year = pred.groupby("fold").apply(
        lambda g: pd.Series({"model_cov80": score(g, "model_")["coverage_80"],
                             "naive_cov80": score(g, "naive_")["coverage_80"],
                             "model_width80": score(g, "model_")["width_80_pct"]}), include_groups=False)

    tag = f"{source}{'_' + signal if source == 'synthetic' else ''}_{asset}_h{h}{'_fund' if use_fund else ''}_brackets"
    o = path("results") / (name or f"{datetime.now():%Y%m%d_%H%M}_{tag}")
    o.mkdir(parents=True, exist_ok=True)
    table.to_csv(o / "bracket_metrics.csv", index=False)
    calib.to_csv(o / "p_up_calibration.csv")
    by_year.to_csv(o / "coverage_by_year.csv")
    pd.DataFrame(shifts_log).to_csv(o / "conformal_shifts.csv", index=False)
    pred.to_parquet(o / "predictions.parquet", index=False)
    (o / "config.json").write_text(json.dumps({"source": source, "signal": signal, "horizon": h, "asset": asset,
                                               "features": feats, "quantiles": QS.tolist(), "params": qp},
                                              indent=2, default=str))
    if verbose:
        pd.set_option("display.width", 180)
        print("\n" + table.round(4).to_string(index=False))
        print(f"\nCoverage of the 80% range by test year:\n{by_year.round(3).to_string()}")
        print(f"\nP(up) calibration (predicted vs actual share of up-moves):\n{calib.round(3).to_string()}")
        ex = pred.sort_values("date").groupby("ticker").tail(1).head(3)
        for _, r in ex.iterrows():
            print(f"Example {r['ticker']} on {r['date']:%Y-%m-%d}: price {r['price']:.2f} -> 80% range "
                  f"{r['price_lo_80']:.2f}-{r['price_hi_80']:.2f}, P(up) {r['model_p_up']:.0%}, "
                  f"actual {r['price'] * math.exp(r['y_true']):.2f}")
        print(f"\nSaved to {o}  ({time.time() - t0:.0f}s)")
    return {"metrics": table, "by_year": by_year, "out": o}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--signal", choices=["none", "momentum", "volcluster"], default="volcluster")
    p.add_argument("--horizon", type=int)
    p.add_argument("--asset", choices=["stock", "etf"], default="stock")
    p.add_argument("--train-step", type=int, default=5)
    p.add_argument("--n-stocks", type=int, default=60)
    p.add_argument("--fundamentals", action="store_true")
    p.add_argument("--name")
    a = p.parse_args()
    run(a.data, a.signal, a.horizon, a.train_step, a.n_stocks, a.asset, a.name, use_fund=a.fundamentals)
