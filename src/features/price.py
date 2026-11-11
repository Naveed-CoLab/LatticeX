"""Price and volume features, one row per stock per date.

RULE: every feature on date t uses only data up to and including the close of date t.
tests/test_pipeline.py checks this automatically by changing future prices and making
sure no past feature moves.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_252", "mom_12_1",
    "xret_20", "xret_60",
    "vol_20", "vol_60", "vol_ratio",
    "dist_ma50", "dist_ma200",
    "volume_z20", "rsi_14",
    "beta_60", "idio_vol_60",
]
# Unranked copies: brackets need the actual size of a stock's swings, not just its rank.
RAW_FEATURES = ["vol_20_raw", "vol_60_raw", "idio_vol_60_raw", "beta_60_raw", "ret_20_raw", "ret_60_raw"]


def _roll(df: pd.DataFrame, col: str, n: int, fn: str) -> pd.Series:
    r = df.groupby("ticker", sort=False)[col].rolling(n, min_periods=n)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def price_features(prices: pd.DataFrame, benchmark: str = "SPY",
                   cross_sectional_rank: bool = True, groups: dict | None = None) -> pd.DataFrame:
    """groups: optional {ticker: 'stock'/'etf'}; ranks are then computed within each group per date."""
    df = prices[["date", "ticker", "adj_close", "volume"]].sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)

    df["logp"] = np.log(df["adj_close"])
    df["r1"] = g["logp"].diff()
    for n in (5, 20, 60, 120, 252):
        df[f"ret_{n}"] = g["logp"].diff(n)
    df["mom_12_1"] = df["ret_252"] - g["logp"].diff(21)        # 12-month momentum skipping the last month

    df["vol_20"] = _roll(df, "r1", 20, "std")
    df["vol_60"] = _roll(df, "r1", 60, "std")
    df["vol_ratio"] = df["vol_20"] / df["vol_60"]

    df["dist_ma50"] = df["adj_close"] / _roll(df, "adj_close", 50, "mean") - 1
    df["dist_ma200"] = df["adj_close"] / _roll(df, "adj_close", 200, "mean") - 1

    df["logv"] = np.log1p(df["volume"])
    df["volume_z20"] = (df["logv"] - _roll(df, "logv", 20, "mean")) / _roll(df, "logv", 20, "std")

    df["up"] = df["r1"].clip(lower=0)
    df["dn"] = (-df["r1"]).clip(lower=0)
    up, dn = _roll(df, "up", 14, "mean"), _roll(df, "dn", 14, "mean")
    df["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # returns relative to the benchmark over the same window
    bench = df.loc[df["ticker"] == benchmark, ["date", "ret_20", "ret_60", "r1"]].rename(
        columns={"ret_20": "b20", "ret_60": "b60", "r1": "rb"})
    df = df.merge(bench, on="date", how="left").sort_values(["ticker", "date"]).reset_index(drop=True)
    df["xret_20"] = df["ret_20"] - df["b20"]
    df["xret_60"] = df["ret_60"] - df["b60"]

    # 60-day beta to the benchmark and the volatility left after removing the market move
    df["r_rb"], df["rb2"] = df["r1"] * df["rb"], df["rb"] ** 2
    m_r, m_b = _roll(df, "r1", 60, "mean"), _roll(df, "rb", 60, "mean")
    cov = _roll(df, "r_rb", 60, "mean") - m_r * m_b
    var_b = _roll(df, "rb2", 60, "mean") - m_b ** 2
    df["beta_60"] = cov / var_b.replace(0, np.nan)
    df["resid"] = df["r1"] - df["beta_60"] * df["rb"]
    df["idio_vol_60"] = _roll(df, "resid", 60, "std")

    for c in ("vol_20", "vol_60", "idio_vol_60", "beta_60", "ret_20", "ret_60"):
        df[f"{c}_raw"] = df[c]
    out = df[["date", "ticker"] + FEATURES + RAW_FEATURES].replace([np.inf, -np.inf], np.nan)
    if cross_sectional_rank:
        # Rank each feature across all stocks on the same date, centred on 0 (range -0.5..0.5).
        # Removes market-wide level shifts so the model learns "which stocks", not "which year".
        keep = out["ticker"] != benchmark
        grp = out["ticker"].map(groups).fillna("stock") if groups else pd.Series("stock", index=out.index)
        ranked = out.loc[keep].groupby([out.loc[keep, "date"], grp[keep]])[FEATURES].rank(pct=True) - 0.5
        out.loc[keep, FEATURES] = ranked
    return out
