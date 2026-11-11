"""Point-in-time sectors from prices, plus sector-relative features.

Each month-end, every stock is assigned to the sector ETF (XLK, XLF, XLE, ...) whose returns its own
returns track most closely over the past year, after removing the market (SPY) move. Only past prices
are used, so the label is known on that date, and it adapts if a company changes business.
No external data is needed.

Features (all known on date t):
  sec_mom_20 / sec_mom_60   the assigned sector ETF's return minus SPY's (sector momentum)
  sr_<feature>              the stock's rank WITHIN its sector for key price/fundamental features
                            ("cheap / strong compared with its own industry", -0.5..0.5)
"""
import numpy as np
import pandas as pd

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
RELATIVE_OF = ["ret_20", "ret_60", "mom_12_1", "vol_60", "dist_ma200",
               "f_earnings_yield", "f_book_to_market", "f_gross_prof", "f_rev_growth", "f_sue"]
SECTOR_FEATURES = ["sec_mom_20", "sec_mom_60"] + [f"sr_{c}" for c in RELATIVE_OF]


def assign_sectors(prices: pd.DataFrame, stocks, benchmark: str = "SPY", window: int = 252,
                   min_obs: int = 150) -> pd.DataFrame:
    """Returns date, ticker, sector for every trading date (sector decided at the previous month-end)."""
    w = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    r = np.log(w).diff()
    ex = r.sub(r[benchmark], axis=0)                     # remove the market move
    etfs = [e for e in SECTOR_ETFS if e in ex]
    stk = [t for t in stocks if t in ex and t not in etfs and t != benchmark]
    month_ends = ex.index.to_series().groupby(ex.index.to_period("M")).max().values
    rows = []
    for d in month_ends:
        win = ex.loc[:d].tail(window)
        E = win[etfs]
        E = E.loc[:, E.notna().sum() >= min_obs]
        if E.shape[1] < 3:
            continue
        S = win[stk]
        S = S.loc[:, S.notna().sum() >= min_obs]
        if S.empty:
            continue
        both = E.join(S).dropna(how="any", subset=E.columns)
        Ez = (both[E.columns] - both[E.columns].mean()) / both[E.columns].std()
        Sz = (both[S.columns] - both[S.columns].mean()) / both[S.columns].std()
        corr = Sz.fillna(0).T.values @ Ez.values / max(len(both) - 1, 1)
        best = np.asarray(E.columns)[np.nanargmax(corr, axis=1)]
        rows.append(pd.DataFrame({"asof": d, "ticker": S.columns, "sector": best}))
    sec = pd.concat(rows, ignore_index=True)
    dates = pd.DataFrame({"date": w.index})
    out = []
    for t, g in sec.groupby("ticker"):
        m = pd.merge_asof(dates, g.sort_values("asof"), left_on="date", right_on="asof", direction="backward")
        out.append(m.dropna(subset=["sector"]).assign(ticker=t)[["date", "ticker", "sector"]])
    return pd.concat(out, ignore_index=True)


def sector_features(panel: pd.DataFrame, prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """panel must contain date, ticker, asset and the RELATIVE_OF features that exist."""
    stocks = panel.loc[panel["asset"] == "stock", "ticker"].unique()
    sec = assign_sectors(prices, stocks, benchmark)
    w = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    lw = np.log(w)
    mom = {}
    for n in (20, 60):
        x = (lw - lw.shift(n)).sub(lw[benchmark] - lw[benchmark].shift(n), axis=0)
        mom[n] = x[[e for e in SECTOR_ETFS if e in x]].stack().rename(f"sec_mom_{n}")
    sm = pd.concat(mom.values(), axis=1).reset_index().rename(columns={"ticker": "sector"})
    p = panel[["date", "ticker"] + [c for c in RELATIVE_OF if c in panel]].merge(sec, on=["date", "ticker"], how="inner")
    p = p.merge(sm, on=["date", "sector"], how="left")
    for c in RELATIVE_OF:
        p[f"sr_{c}"] = (p.groupby(["date", "sector"])[c].rank(pct=True) - 0.5) if c in p else np.nan
    return p[["date", "ticker", "sector"] + SECTOR_FEATURES]
