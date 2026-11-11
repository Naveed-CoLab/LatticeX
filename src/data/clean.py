"""Put every ticker on the same trading calendar before features and targets are built.

Why: some tickers have holes in their history (trading halts, delisting periods, reused symbols).
Without this step "20 rows ahead" could mean "6 months ahead" across a hole, and targets would
leak into later years. After this step one row = one trading day for every ticker.

Rules:
  - calendar = the benchmark's (SPY) trading days
  - each ticker is placed on that calendar between its first and last real price
  - gaps of up to `max_ffill` trading days: price carried forward (holiday quirks, missing rows)
  - longer gaps: left empty, so any feature or target that spans them becomes NaN
  - one-day moves bigger than x5 or smaller than /5 are treated as bad data and blanked
"""
import numpy as np
import pandas as pd


def align_to_calendar(prices: pd.DataFrame, benchmark: str = "SPY", max_ffill: int = 5,
                      verbose: bool = False) -> pd.DataFrame:
    cal = pd.DatetimeIndex(sorted(prices.loc[prices["ticker"] == benchmark, "date"].unique()))
    if len(cal) == 0:
        raise ValueError(f"benchmark {benchmark} not in prices")
    out, n_gap_days, n_bad = [], 0, 0
    for t, g in prices.groupby("ticker", sort=False):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        g = g[g.index.isin(cal)] if t != benchmark else g
        if g.empty:
            continue
        idx = cal[(cal >= g.index.min()) & (cal <= g.index.max())]
        g = g.reindex(idx)
        r = np.log(g["adj_close"]).diff()
        bad = r.abs() > np.log(5)
        if bad.any():
            n_bad += int(bad.sum())
            g.loc[bad, "adj_close"] = np.nan
        filled = g["adj_close"].ffill(limit=max_ffill)
        n_gap_days += int(g["adj_close"].isna().sum() - filled.isna().sum())
        g["adj_close"] = filled
        g["ticker"] = t
        out.append(g.rename_axis("date").reset_index())
    df = pd.concat(out, ignore_index=True)
    if verbose:
        long_gaps = int(df["adj_close"].isna().sum())
        print(f"Calendar alignment: {len(cal):,} trading days; filled {n_gap_days:,} short gaps, "
              f"{long_gaps:,} days left empty in long gaps, {n_bad} bad one-day jumps removed")
    return df
