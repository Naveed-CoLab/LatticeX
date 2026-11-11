"""Fundamental features from SEC filings, strictly point-in-time.

Steps
 1. For each company and item, keep the FIRST filed value of every period (later restatements ignored:
    the market only knew the original number at the time).
 2. Quarterly values: 10-Qs report year-to-date totals (3, 6, 9 months) and the 10-K the full year, so
    each quarter = YTD(now) - YTD(previous). Q4 = full year - 9 months.
 3. TTM (trailing twelve months) = sum of the last 4 quarters (or the annual figure).
 4. Every value becomes usable the day AFTER its filing date, and expires if the company hasn't
    filed for 15 months (delisted, or the ticker now belongs to someone else).
 5. Market value = unadjusted close x shares outstanding, with shares adjusted for later splits.

Features (ranked across stocks per date, like the price features)
  value:    f_earnings_yield, f_sales_yield, f_book_to_market, f_fcf_yield
  quality:  f_roe, f_gross_prof (gross profit / assets), f_op_margin, f_accruals, f_leverage
  growth:   f_rev_growth (TTM revenue vs a year earlier), f_asset_growth
  earnings: f_sue (standardised unexpected earnings: this quarter vs same quarter last year,
            scaled by how noisy that change usually is), f_days_since_filing,
            f_filing_reaction (stock minus SPY return over the 3 trading days around the latest
            10-Q/10-K filing; usable 3 trading days after the filing; prices tend to keep drifting)
  size:     f_size (log market value)
"""
import numpy as np
import pandas as pd

from src.ingest.edgar import CONCEPTS

FUND_FEATURES = [
    "f_earnings_yield", "f_sales_yield", "f_book_to_market", "f_fcf_yield",
    "f_roe", "f_gross_prof", "f_op_margin", "f_accruals", "f_leverage",
    "f_rev_growth", "f_asset_growth", "f_sue", "f_days_since_filing", "f_size", "f_filing_reaction",
]
FLOWS = ["revenue", "net_income", "gross_profit", "operating_income", "cfo", "capex"]
STOCKS = ["assets", "equity", "liabilities", "shares"]
STALE = pd.Timedelta(days=450)


def first_filed(f: pd.DataFrame) -> pd.DataFrame:
    prio = {(item, c): i for item, cs in CONCEPTS.items() for i, c in enumerate(cs)}
    f = f.assign(prio=[prio.get((i, c), 99) for i, c in zip(f["item"], f["concept"])])
    f = f.sort_values(["ticker", "item", "start", "end", "filed", "prio"])
    return f.drop_duplicates(["ticker", "item", "start", "end"], keep="first").drop(columns="prio")


def quarters(flows: pd.DataFrame) -> pd.DataFrame:
    """Quarter values from year-to-date / 3-month facts."""
    out = []
    for (t, item, start), g in flows.groupby(["ticker", "item", "start"], sort=False):
        g = g.sort_values("end")
        prev_end, prev_val, prev_filed = start - pd.Timedelta(days=1), 0.0, None
        first = True
        for end, val, filed in zip(g["end"], g["val"], g["filed"]):
            gap = (end - prev_end).days
            if 80 <= gap <= 100 and (not first or 80 <= (end - start).days <= 100):
                avail = filed if prev_filed is None else max(filed, prev_filed)
                out.append((t, item, end, val - prev_val, avail))
            elif first:
                break
            prev_end, prev_val, prev_filed, first = end, val, filed, False
    q = pd.DataFrame(out, columns=["ticker", "item", "end", "q", "avail"])
    return q.sort_values("avail").drop_duplicates(["ticker", "item", "end"]).sort_values(["ticker", "item", "end"])


def ttm(flows: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (t, item), g in q.groupby(["ticker", "item"], sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        span = (g["end"] - g["end"].shift(3)).dt.days
        ok = span.between(250, 300)
        s = g["q"].rolling(4).sum()
        a = pd.concat([g["avail"].shift(k) for k in range(4)], axis=1).max(axis=1)   # known once all 4 are filed
        parts.append(pd.DataFrame({"ticker": t, "item": item, "end": g["end"], "ttm": s, "avail": a})[ok])
    dur = (flows["end"] - flows["start"]).dt.days
    annual = flows[dur.between(350, 380)][["ticker", "item", "end", "val", "filed"]]
    parts.append(annual.rename(columns={"val": "ttm", "filed": "avail"}))
    t_ = pd.concat(parts, ignore_index=True).dropna(subset=["ttm", "avail"])
    return t_.sort_values("avail").drop_duplicates(["ticker", "item", "end"]).sort_values(["ticker", "item", "end"])


def year_ago(df: pd.DataFrame, value: str) -> pd.Series:
    """Value of the same series for the period ending ~1 year earlier (both already public)."""
    res = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["ticker", "item"], sort=False):
        g = g.sort_values("end")
        left = g[["end"]].assign(target=g["end"] - pd.Timedelta(days=365)).reset_index()
        right = g[["end", value]].rename(columns={"end": "prev_end", value: "prev"})
        m = pd.merge_asof(left.sort_values("target"), right.sort_values("prev_end"), left_on="target",
                          right_on="prev_end", direction="nearest", tolerance=pd.Timedelta(days=25))
        res.loc[m["index"].values] = m["prev"].values
    return res


def sue(q_ni: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in q_ni.groupby("ticker", sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        yoy_ok = (g["end"] - g["end"].shift(4)).dt.days.between(350, 380)
        d = (g["q"] - g["q"].shift(4)).where(yoy_ok)
        sd = d.shift(1).rolling(8, min_periods=4).std()
        out.append(pd.DataFrame({"ticker": t, "end": g["end"], "sue": d / sd.replace(0, np.nan),
                                 "avail": g["avail"]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ticker", "end", "sue", "avail"])


def _asof(base: pd.DataFrame, series: pd.DataFrame, cols: list) -> pd.DataFrame:
    s = series.dropna(subset=["avail"]).copy()
    s["avail_eff"] = s["avail"] + pd.Timedelta(days=1)        # usable the day after filing
    s = s.sort_values(["avail_eff", "end"])
    m = pd.merge_asof(base.sort_values("date"), s[["ticker", "avail_eff"] + cols], left_on="date",
                      right_on="avail_eff", by="ticker", direction="backward", tolerance=STALE)
    return m


def filing_reaction(facts: pd.DataFrame, prices_adj: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Excess log return from the close before the filing day to 2 trading days after it.
    avail = 3rd trading day after filing (the window has fully closed)."""
    if prices_adj is None or prices_adj.empty:
        return pd.DataFrame(columns=["ticker", "end", "react", "avail"])
    w = prices_adj.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    lw = np.log(w)
    ex = lw.sub(lw[benchmark], axis=0) if benchmark in lw else lw
    cal = w.index
    fil = facts.loc[facts["form"].isin(["10-Q", "10-K"]), ["ticker", "filed"]].drop_duplicates()
    fil = fil[fil["ticker"].isin(ex.columns)]
    i = cal.searchsorted(fil["filed"].values)                   # first trading day on/after filing
    ok = (i >= 1) & (i + 3 < len(cal))
    fil, i = fil[ok], i[ok]
    col = ex.columns.get_indexer(fil["ticker"])
    arr = ex.values
    react = arr[i + 2, col] - arr[i - 1, col]
    return pd.DataFrame({"ticker": fil["ticker"].values, "end": cal[i], "react": react,
                         "avail": cal[i + 3] - pd.Timedelta(days=1)}).dropna()


def fundamental_features(facts: pd.DataFrame, prices_raw: pd.DataFrame, panel_index: pd.DataFrame,
                         cross_sectional_rank: bool = True, prices_adj: pd.DataFrame | None = None) -> pd.DataFrame:
    """facts: data/clean/fundamentals.parquet. prices_raw: date, ticker, close, split_factor.
    panel_index: the (date, ticker) rows to fill."""
    f = first_filed(facts.dropna(subset=["end", "filed"]))
    flows = f[f["item"].isin(FLOWS) & f["start"].notna()]
    stocks = f[f["item"].isin(STOCKS) & f["start"].isna()]
    q = quarters(flows)
    t = ttm(flows, q)
    t["prev"] = year_ago(t, "ttm")
    inst = stocks.rename(columns={"filed": "avail"})[["ticker", "item", "end", "val", "avail"]]
    inst = inst.sort_values("avail").drop_duplicates(["ticker", "item", "end"])
    inst["prev"] = year_ago(inst, "val")

    base = panel_index[["date", "ticker"]].drop_duplicates().reset_index(drop=True)
    out = base.copy()
    for item in FLOWS:
        m = _asof(base, t[t["item"] == item].rename(columns={"ttm": item, "prev": f"{item}_prev"}),
                  [item, f"{item}_prev"])
        out = out.merge(m[["date", "ticker", item, f"{item}_prev"]], on=["date", "ticker"], how="left")
    for item in STOCKS:
        m = _asof(base, inst[inst["item"] == item].rename(columns={"val": item, "prev": f"{item}_prev"}),
                  [item, f"{item}_prev", "avail"] if item == "shares" else [item, f"{item}_prev"])
        keep = ["date", "ticker", item, f"{item}_prev"] + (["avail"] if item == "shares" else [])
        out = out.merge(m[keep].rename(columns={"avail": "shares_avail"}), on=["date", "ticker"], how="left")
    s = sue(q[q["item"] == "net_income"])
    out = out.merge(_asof(base, s, ["sue"])[["date", "ticker", "sue"]], on=["date", "ticker"], how="left")
    last = f.groupby(["ticker", "filed"]).size().reset_index()[["ticker", "filed"]].rename(columns={"filed": "avail"})
    last["end"] = last["avail"]
    lm = _asof(base, last.assign(last_filed=last["avail"]), ["last_filed"])
    out = out.merge(lm[["date", "ticker", "last_filed"]], on=["date", "ticker"], how="left")

    # market value with split-adjusted share count
    px = prices_raw[["date", "ticker", "close", "split_factor"]].sort_values(["ticker", "date"]).copy()
    px["split_factor"] = px["split_factor"].fillna(1.0).replace(0, 1.0)
    px["S"] = px.groupby("ticker")["split_factor"].cumprod()
    out = out.merge(px[["date", "ticker", "close", "S"]], on=["date", "ticker"], how="left")
    sa = pd.merge_asof(out[["ticker", "shares_avail"]].dropna().drop_duplicates().sort_values("shares_avail"),
                       px[["date", "ticker", "S"]].rename(columns={"date": "shares_avail", "S": "S_then"})
                       .sort_values("shares_avail"), on="shares_avail", by="ticker", direction="backward")
    out = out.merge(sa, on=["ticker", "shares_avail"], how="left")
    shares_now = out["shares"] * out["S"] / out["S_then"].replace(0, np.nan)
    mcap = (out["close"] * shares_now).where(lambda x: x > 0)

    pos = lambda x: x.where(x > 0)
    r = pd.DataFrame({"date": out["date"], "ticker": out["ticker"]})
    r["f_earnings_yield"] = out["net_income"] / mcap
    r["f_sales_yield"] = out["revenue"] / mcap
    r["f_book_to_market"] = out["equity"] / mcap
    r["f_fcf_yield"] = (out["cfo"] - out["capex"].fillna(0)) / mcap
    r["f_roe"] = out["net_income"] / pos(out["equity"])
    r["f_gross_prof"] = out["gross_profit"] / pos(out["assets"])
    r["f_op_margin"] = out["operating_income"] / pos(out["revenue"])
    r["f_accruals"] = (out["net_income"] - out["cfo"]) / pos(out["assets"])
    r["f_leverage"] = out["liabilities"] / pos(out["assets"])
    r["f_rev_growth"] = out["revenue"] / pos(out["revenue_prev"]) - 1
    r["f_asset_growth"] = out["assets"] / pos(out["assets_prev"]) - 1
    r["f_sue"] = out["sue"]
    r["f_days_since_filing"] = np.log1p((out["date"] - out["last_filed"]).dt.days)
    r["f_size"] = np.log(mcap)
    fr = filing_reaction(f, prices_adj)
    if len(fr):
        fr_m = _asof(base, fr, ["react"])[["date", "ticker", "react"]]
        r = r.merge(fr_m, on=["date", "ticker"], how="left").rename(columns={"react": "f_filing_reaction"})
        r.loc[(r["date"] - out["last_filed"].values).dt.days > 120, "f_filing_reaction"] = np.nan
    else:
        r["f_filing_reaction"] = np.nan
    r = r.replace([np.inf, -np.inf], np.nan)
    if cross_sectional_rank:
        r[FUND_FEATURES] = r.groupby("date")[FUND_FEATURES].rank(pct=True) - 0.5
    return r
