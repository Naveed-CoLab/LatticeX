"""Price ranges from volatility: "where is the price likely to be in H trading days?"

The range is centred on today's price and its width follows the stock's own volatility:

    low, high = price * exp(-/+ q * daily_vol * sqrt(H))

Two things make it narrower than a plain volatility rule while keeping it honest:
  1. daily_vol blends four measures (high-low range based, 60-day, 250-day, recent-weighted). On 2012-2024
     this gave 5% narrower ranges at the same hit rate than 20-day volatility alone.
  2. q depends on whether an earnings report is expected inside the window. Stocks move about 1.7x more
     in earnings weeks; one multiplier for all weeks is too narrow there (57% hit rate instead of 80%)
     and wider than needed everywhere else.

q is not assumed (no bell curve): it is the multiplier that WOULD have contained the price the stated share
of the time on earlier data. Everything here uses only prices and filing dates known on the forecast date.
"""
import numpy as np
import pandas as pd

HORIZONS = (5, 20, 60)
LEVELS = (0.5, 0.8, 0.9)
EARN_TOL = pd.to_timedelta(3, unit="D")      # an expected date counts if it falls within 3 days of the window
YEAR = pd.to_timedelta(364, unit="D")        # same weekday, one year later


def daily_vol(px: pd.DataFrame) -> pd.Series:
    """Blended daily volatility per row. px: ticker, date, open, high, low, close, adj_close, sorted by
    ticker then date. Uses only the past (each value is known at that day's close)."""
    t = px["ticker"]
    r = np.log(px["adj_close"]).groupby(t).diff()
    hl = np.log(px["high"] / px["low"]).clip(0, 0.5)
    co = np.log(px["close"] / px["open"]).clip(-0.5, 0.5)
    gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).clip(lower=0)      # Garman-Klass, intraday part
    overnight = r - co

    def roll(s, n, minp):
        return s.groupby(t).rolling(n, min_periods=minp).mean().reset_index(level=0, drop=True)

    parts = pd.concat([
        roll(gk, 20, 15) + roll(overnight ** 2, 20, 15),                      # range-based, last month
        roll(r ** 2, 60, 40),                                                 # last quarter
        roll(r ** 2, 250, 120),                                               # last year
        (r ** 2).groupby(t).transform(lambda s: s.ewm(alpha=0.06, min_periods=20).mean()),   # recent-weighted
    ], axis=1)
    return np.sqrt(parts.mean(axis=1, skipna=True).where(parts.notna().sum(axis=1) >= 2))


def expected_earnings(events: pd.DataFrame | None) -> dict:
    """ticker -> sorted array of EXPECTED earnings dates: each past earnings filing date plus one year.
    Companies report on a steady yearly calendar, so last year's date is a fair advance estimate."""
    if events is None or events.empty:
        return {}
    e = events.loc[events["ev_earnings"].astype(bool), ["ticker", "filed"]].drop_duplicates()
    e["est"] = pd.to_datetime(e["filed"]) + YEAR
    return {t: np.sort(g["est"].to_numpy()) for t, g in e.groupby("ticker")}


def earnings_in_window(tickers, start, end, est: dict) -> np.ndarray:
    """True where an expected earnings date falls in (start - 3 days, end + 3 days]."""
    tickers = np.asarray(tickers)
    start = pd.to_datetime(pd.Series(start)).to_numpy()
    end = pd.to_datetime(pd.Series(end)).to_numpy()
    out = np.zeros(len(tickers), dtype=bool)
    tol = np.timedelta64(3, "D")
    order = np.argsort(tickers, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tickers[order][1:] != tickers[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        d = est.get(tickers[idx[0]])
        if d is None or len(d) == 0:
            continue
        pos = np.searchsorted(d, start[idx] - tol, side="left")
        ok = pos < len(d)
        nxt = d[np.minimum(pos, len(d) - 1)]
        out[idx] = ok & (nxt <= end[idx] + tol)
    return out


def next_expected(ticker: str, after, est: dict):
    d = est.get(ticker)
    if d is None or len(d) == 0:
        return None
    pos = np.searchsorted(d, np.datetime64(pd.Timestamp(after)) - np.timedelta64(3, "D"), side="left")
    return pd.Timestamp(d[pos]) if pos < len(d) else None


def panel(px: pd.DataFrame, universe: pd.DataFrame, est: dict, horizon: int) -> pd.DataFrame:
    """Weekly member rows with what is needed to fit and score ranges for one horizon:
    ticker, date, d_end, year, vol, z (= |move| / (vol * sqrt(H))), earn (expected earnings in the window)."""
    p = px[["ticker", "date", "adj_close", "vol"]].copy()
    la = np.log(p["adj_close"])
    p["fwd"] = la.groupby(p["ticker"]).shift(-horizon) - la
    p["d_end"] = p.groupby("ticker")["date"].shift(-horizon)
    m = p.merge(universe[["ticker", "start", "end"]], on="ticker")
    m = m[(m["date"] >= m["start"].fillna(pd.Timestamp("1990-01-01"))) & (m["end"].isna() | (m["date"] <= m["end"]))]
    m = m[m["date"].dt.dayofweek == 2].dropna(subset=["fwd", "vol", "d_end"])
    m = m[m["vol"] > 0].copy()
    m["year"] = m["date"].dt.year
    m["z"] = m["fwd"].abs() / (m["vol"] * np.sqrt(horizon))
    m["earn"] = earnings_in_window(m["ticker"], m["date"], m["d_end"], est)
    return m[["ticker", "date", "d_end", "year", "vol", "fwd", "z", "earn"]].reset_index(drop=True)


def fit_multipliers(m: pd.DataFrame, min_rows: int = 500) -> dict:
    """{earn flag: {level: q}} from rows m. Falls back to all rows when one group is too small."""
    out = {}
    for flag in (False, True):
        z = m.loc[m["earn"] == flag, "z"]
        if len(z) < min_rows:
            z = m["z"]
        out[flag] = {lv: float(z.quantile(lv)) for lv in LEVELS}
    return out


def track_record(m: pd.DataFrame, horizon: int, first_year: int = 2012, last_year: int = 2024) -> dict:
    """Walk-forward: for each test year the multipliers come only from windows that ENDED before that year."""
    rows = []
    for y in range(first_year, last_year + 1):
        train = m[m["d_end"] < pd.Timestamp(f"{y}-01-01")]
        test = m[m["year"] == y]
        if len(train) < 5000 or test.empty:
            continue
        q = fit_multipliers(train)
        a = test["fwd"].abs().to_numpy()
        base = test["vol"].to_numpy() * np.sqrt(horizon)
        rec = {"year": y, "n": int(len(test)), "earn": test["earn"].to_numpy()}
        for lv in LEVELS:
            half = np.where(rec["earn"], q[True][lv], q[False][lv]) * base
            rec[f"hit{int(lv * 100)}"] = a <= half
            rec[f"w{int(lv * 100)}"] = 2 * half
        rows.append(rec)
    if not rows:
        return {}
    cat = lambda k: np.concatenate([r[k] for r in rows])
    earn = cat("earn")
    out = {"horizon": horizon, "years": [first_year, last_year], "forecasts": int(len(earn)),
           "share_with_earnings": float(earn.mean()), "levels": {}, "by_year": []}
    for lv in LEVELS:
        k = int(lv * 100)
        hit, w = cat(f"hit{k}"), cat(f"w{k}")
        out["levels"][str(k)] = {
            "target": lv, "hit_rate": float(hit.mean()), "avg_width": float(w.mean()),
            "hit_rate_earnings": float(hit[earn].mean()) if earn.any() else None,
            "hit_rate_other": float(hit[~earn].mean()) if (~earn).any() else None,
            "avg_width_earnings": float(w[earn].mean()) if earn.any() else None,
            "avg_width_other": float(w[~earn].mean()) if (~earn).any() else None}
    out["by_year"] = [{"year": r["year"], "hit80": float(r["hit80"].mean()), "n": r["n"]} for r in rows]
    return out


def ranges_today(price: float, vol: float, horizon: int, q: dict, earn: bool) -> dict:
    """{'50': [low, high], '80': ..., '90': ...} plus the half-widths (as a fraction of the price)."""
    out = {"bands": {}, "half": {}}
    for lv in LEVELS:
        h = q[bool(earn)][lv] * vol * np.sqrt(horizon)
        out["bands"][str(int(lv * 100))] = [round(float(price * np.exp(-h)), 2), round(float(price * np.exp(h)), 2)]
        out["half"][str(int(lv * 100))] = round(float(np.exp(h) - 1), 4)
    return out
