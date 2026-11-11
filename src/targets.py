"""Targets: future excess return of each stock versus the benchmark.

y_h on date t = log(P[t+h] / P[t]) for the stock  minus  the same for SPY.
The first return in the window is day t+1, so the target never overlaps the features.
target_end_h is the date the target is finally known (t+h); used to stop targets
leaking into later folds or the holdout.
"""
import numpy as np
import pandas as pd


def make_targets(prices: pd.DataFrame, horizons=(5, 20, 60), benchmark: str = "SPY") -> pd.DataFrame:
    df = prices[["date", "ticker", "adj_close"]].sort_values(["ticker", "date"]).reset_index(drop=True)
    df["logp"] = np.log(df["adj_close"])
    g = df.groupby("ticker", sort=False)
    cols = ["date", "ticker"]
    for h in horizons:
        df[f"fwd_{h}"] = g["logp"].shift(-h) - df["logp"]          # NaN if either end has no price
        df[f"target_end_{h}"] = g["date"].shift(-h).where(df[f"fwd_{h}"].notna())
    bench = df.loc[df["ticker"] == benchmark, ["date"] + [f"fwd_{h}" for h in horizons]]
    bench = bench.rename(columns={f"fwd_{h}": f"bfwd_{h}" for h in horizons})
    df = df.merge(bench, on="date", how="left")
    for h in horizons:
        df[f"y_{h}"] = df[f"fwd_{h}"] - df[f"bfwd_{h}"]
        df[f"beat_{h}"] = (df[f"y_{h}"] > 0).astype(float).where(df[f"y_{h}"].notna())
        cols += [f"y_{h}", f"beat_{h}", f"target_end_{h}", f"fret_{h}"]
        df[f"fret_{h}"] = df[f"fwd_{h}"]                    # FUTURE stock log return (for price brackets)
    return df[cols]


def transform_target(df: pd.DataFrame, col: str, how: str = "winsor") -> pd.Series:
    """Training-only transform. Evaluation always uses the raw excess return."""
    if how == "rank":
        return df.groupby("date")[col].rank(pct=True) - 0.5
    if how == "winsor":
        lo = df.groupby("date")[col].transform(lambda s: s.quantile(0.01))
        hi = df.groupby("date")[col].transform(lambda s: s.quantile(0.99))
        return df[col].clip(lo, hi)
    return df[col]
