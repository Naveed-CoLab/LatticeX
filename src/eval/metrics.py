"""Shared scoring code. EVERY model is scored with these functions and nothing else.

Input: a DataFrame with columns date, ticker, y_true (raw excess return), y_pred.

Because 20-day targets overlap from one evaluation date to the next, errors are
correlated over time. t-stats and the Diebold-Mariano test therefore use Newey-West
(HAC) standard errors with `lags` = overlap length on the evaluation grid.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats


def hit_rate(y_true, y_pred) -> float:
    """Share of correct up/down calls versus the benchmark. A prediction of exactly 0 counts as a coin flip."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true != 0
    s_t, s_p = np.sign(y_true[m]), np.sign(y_pred[m])
    return float(np.mean((s_t == s_p) + 0.5 * (s_p == 0)))


def daily_rank_ic(df: pd.DataFrame) -> pd.Series:
    """Spearman correlation between prediction and outcome across stocks, per date.
    A constant prediction (e.g. Random Walk) has no ranking skill: IC = 0."""
    def ic(g):
        if g["y_pred"].nunique() < 2 or len(g) < 5:
            return 0.0
        return g["y_pred"].rank().corr(g["y_true"].rank())
    return df.groupby("date")[["y_pred", "y_true"]].apply(ic)


def long_short(df: pd.DataFrame, q: float = 0.1) -> pd.Series:
    """Per date: mean outcome of the top 10% of predictions minus the bottom 10%."""
    def spread(g):
        if g["y_pred"].nunique() < 2:
            return 0.0
        hi, lo = g["y_pred"].quantile(1 - q), g["y_pred"].quantile(q)
        return g.loc[g["y_pred"] >= hi, "y_true"].mean() - g.loc[g["y_pred"] <= lo, "y_true"].mean()
    return df.groupby("date")[["y_pred", "y_true"]].apply(spread)


def newey_west_se(x, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return math.sqrt(max(s, 1e-18) / n)


def nw_tstat(x, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean() / newey_west_se(x, lags))


def diebold_mariano(df_a: pd.DataFrame, df_b: pd.DataFrame, lags: int) -> dict:
    """Is model A's squared error significantly lower than model B's?
    Errors are averaged across stocks per date first, giving one loss difference per date.
    Includes the Harvey-Leybourne-Newbold small-sample correction. Negative stat = A better."""
    a = df_a.assign(la=(df_a["y_true"] - df_a["y_pred"]) ** 2).groupby("date")["la"].mean()
    b = df_b.assign(lb=(df_b["y_true"] - df_b["y_pred"]) ** 2).groupby("date")["lb"].mean()
    d = (a - b).dropna().values
    n, h = len(d), lags + 1
    dm = d.mean() / newey_west_se(d, lags)
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm_stat": float(stat), "p_value": float(p), "n_dates": n}


def summarize(df: pd.DataFrame, lags: int, name: str = "") -> dict:
    """Ranking and direction (hit rate, rank IC, long-short) use the model's raw output if present
    (column y_raw): shrinkage only scales forecast size and must not erase the ranking.
    Error metrics (RMSE, MAE) use the final shrunk forecast y_pred."""
    rank_df = df.assign(y_pred=df["y_raw"]) if "y_raw" in df else df
    ic = daily_rank_ic(rank_df)
    ls = long_short(rank_df)
    err = df["y_true"] - df["y_pred"]
    return {
        "model": name,
        "n_dates": int(df["date"].nunique()),
        "n_rows": int(len(df)),
        "hit_rate": hit_rate(rank_df["y_true"], rank_df["y_pred"]),
        "rank_ic": float(ic.mean()),
        "ic_tstat_nw": nw_tstat(ic, lags) if ic.std() > 0 else 0.0,
        "ic_positive_share": float((ic > 0).mean()),
        "long_short": float(ls.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
