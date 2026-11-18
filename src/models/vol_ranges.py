"""Price ranges from volatility: "where is the price likely to be in H trading days?"

    low  = price * exp(q_low  * vol_forecast * sqrt(H))
    high = price * exp(q_high * vol_forecast * sqrt(H))

v2 (17 Nov 2026), 2.4% / 3.7% / 5.0% narrower than v1 at 5 / 20 / 60 days with the same hit rate (docs/results.md 5d):
  1. vol_forecast: one regression predicts each stock's volatility over the window (log of realised volatility) from
     what is known today: four volatility measures (high-low range based, 60-day, 250-day, recent-weighted), one-week
     volatility, the market's mood (SPY's recent volatility against its own year), whether an earnings report is
     expected in the window, and the stock's own typical earnings move (median of its last 8). v1 used a plain
     average of the four measures (daily_vol below, still shown as the stock's volatility).
  2. q_low and q_high are fitted separately ("uneven up/down"): prices drift up, so the range sits a little above
     today's price. They are the multipliers that WOULD have held the price the stated share of the time on earlier
     windows, separately for windows with and without an expected earnings report. No bell curve is assumed.
  3. New output: the target day's likely middle, "high stays below X" and "low stays above Y" (each 9 times in 10).

Everything uses only prices and filing dates known on the forecast date. The weights and multipliers used in the
product are fitted on windows that ended by 31 Dec 2024; the 2025-26 holdout is never used.
"""
import numpy as np
import pandas as pd

HORIZONS = (5, 20, 60)
LEVELS = (0.5, 0.8, 0.9)
DAY_LEVEL = 0.9                              # "the day's high stays below X" is meant to hold 9 times in 10
EARN_TOL = pd.to_timedelta(3, unit="D")      # an expected date counts if it falls within 3 days of the window
YEAR = pd.to_timedelta(364, unit="D")        # same weekday, one year later
MEASURES = ["v_gk20", "v5", "v60", "v250", "v_ew"]


def _measures(px: pd.DataFrame) -> pd.DataFrame:
    """Per-row volatility measures known at that day's close. px sorted by ticker then date."""
    t = px["ticker"]
    r = np.log(px["adj_close"]).groupby(t).diff()
    hl = np.log(px["high"] / px["low"]).clip(0, 0.5)
    co = np.log(px["close"] / px["open"]).clip(-0.5, 0.5)
    gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).clip(lower=0)      # Garman-Klass, intraday part
    overnight = r - co

    def roll(s, n, minp):
        return s.groupby(t).rolling(n, min_periods=minp).mean().reset_index(level=0, drop=True)

    sq = pd.DataFrame({
        "v_gk20": roll(gk, 20, 15) + roll(overnight ** 2, 20, 15),          # range-based, last month
        "v5": roll(r ** 2, 5, 4),                                            # last week
        "v60": roll(r ** 2, 60, 40),                                         # last quarter
        "v250": roll(r ** 2, 250, 120),                                      # last year
        "v_ew": (r ** 2).groupby(t).transform(lambda s: s.ewm(alpha=0.06, min_periods=20).mean()),   # recent-weighted
    }, index=px.index)
    blend = sq[["v_gk20", "v60", "v250", "v_ew"]]
    out = np.sqrt(sq)
    out["vol"] = np.sqrt(blend.mean(axis=1, skipna=True).where(blend.notna().sum(axis=1) >= 2))
    out["r2"] = r ** 2
    return out


def daily_vol(px: pd.DataFrame) -> pd.Series:
    """v1's blended daily volatility per row (shown on the page as the stock's volatility; an input of v2).
    px: ticker, date, open, high, low, close, adj_close, sorted by ticker then date. Uses only the past."""
    return _measures(px)["vol"]


def features(px: pd.DataFrame, spy: pd.DataFrame | None = None) -> pd.DataFrame:
    """px (stocks, sorted by ticker then date) plus the volatility measures, the v1 blend ('vol') and the market's
    mood: SPY's month (m_short) and year (m_long) volatility on the same day. Without SPY prices the mood is unknown
    and counts as neutral."""
    out = pd.concat([px.reset_index(drop=True), _measures(px.reset_index(drop=True))], axis=1)
    if spy is not None and len(spy):
        s = spy.sort_values("date").reset_index(drop=True)
        m = _measures(s)
        mk = pd.DataFrame({"date": s["date"], "m_short": m["v_gk20"], "m_long": m["v250"]})
        out = out.merge(mk, on="date", how="left")
    else:
        out["m_short"], out["m_long"] = np.nan, np.nan
    return out


def earnings_typical(px: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """Each stock's typical earnings move as it was known over time: the median of its last 8 |2-day moves| around
    an earnings filing (close before the filing day to the close after it). Columns: ticker, known (the day the
    value became known), e_typ. Needs at least 3 past reports."""
    cols = ["ticker", "known", "e_typ"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    e = events.loc[events["ev_earnings"].astype(bool), ["ticker", "filed"]].drop_duplicates()
    e["filed"] = pd.to_datetime(e["filed"])
    rows = []
    for tk, g in e.groupby("ticker"):
        p = px[px["ticker"] == tk]
        if len(p) < 30:
            continue
        d, la = p["date"].to_numpy(), np.log(p["adj_close"].to_numpy())
        i = np.searchsorted(d, g["filed"].sort_values().to_numpy())        # first trading day on or after the filing
        i = i[(i >= 1) & (i + 1 < len(d))]
        if len(i):
            rows.append(pd.DataFrame({"ticker": tk, "known": d[i + 1], "mv": np.abs(la[i + 1] - la[i - 1])}))
    if not rows:
        return pd.DataFrame(columns=cols)
    em = pd.concat(rows).sort_values(["ticker", "known"])
    em["e_typ"] = em.groupby("ticker")["mv"].transform(lambda s: s.rolling(8, min_periods=3).median())
    return em.dropna(subset=["e_typ"])[cols].reset_index(drop=True)


def add_typical(m: pd.DataFrame, et: pd.DataFrame) -> pd.DataFrame:
    """Attach the typical earnings move known strictly before each row's date (NaN when not known yet)."""
    m = m.drop(columns=["e_typ"], errors="ignore")
    if et is None or et.empty:
        return m.assign(e_typ=np.nan)
    order = m.index
    m = pd.merge_asof(m.reset_index().sort_values("date"), et.sort_values("known"), left_on="date", right_on="known",
                      by="ticker", direction="backward", allow_exact_matches=False)
    return m.drop(columns="known").set_index("index").loc[order].rename_axis(None)


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


def panel(f: pd.DataFrame, universe: pd.DataFrame, est: dict, horizon: int, et: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly (Wednesday) member rows with everything needed to fit and score ranges for one horizon.
    f: output of features(). Outcomes: fwd (log move to the close H days later), hi_end / lo_end (how far that
    day's high and low reached beyond its close) and rv (realised daily volatility over the window)."""
    g = f.groupby("ticker")
    la = np.log(f["adj_close"])
    p = f[["ticker", "date", "vol", "m_short", "m_long"] + MEASURES].copy()
    p["fwd"] = la.groupby(f["ticker"]).shift(-horizon) - la
    p["d_end"] = g["date"].shift(-horizon)
    p["hi_end"] = np.log(f["high"] / f["close"]).groupby(f["ticker"]).shift(-horizon)
    p["lo_end"] = np.log(f["low"] / f["close"]).groupby(f["ticker"]).shift(-horizon)
    c = f["r2"].fillna(0).groupby(f["ticker"]).cumsum()
    p["rv"] = np.sqrt((c.groupby(f["ticker"]).shift(-horizon) - c).clip(lower=0) / horizon)
    m = p.merge(universe[["ticker", "start", "end"]], on="ticker")
    m = m[(m["date"] >= m["start"].fillna(pd.Timestamp("1990-01-01"))) & (m["end"].isna() | (m["date"] <= m["end"]))]
    need = ["fwd", "vol", "d_end", "v60", "v_gk20", "v_ew"] + (["m_short", "m_long"] if f["m_short"].notna().any() else [])
    m = m[m["date"].dt.dayofweek == 2].dropna(subset=need)
    m = m[(m["vol"] > 0) & (m["v60"] > 0) & (m["v_gk20"] > 0)].copy()
    m["year"] = m["date"].dt.year
    m["earn"] = earnings_in_window(m["ticker"], m["date"], m["d_end"], est)
    m = add_typical(m.drop(columns=["start", "end"]), et)
    return m.reset_index(drop=True)


def design(m: pd.DataFrame) -> np.ndarray:
    """Regression inputs, all in logs: the five measures, market mood, earnings flag, the stock's typical earnings
    move against its usual volatility (only in earnings windows), and a flag for earnings windows without one."""
    lg = lambda c: np.log(m[c].astype(float).clip(lower=1e-4))
    earn = m["earn"].astype(bool).to_numpy()
    typ = m["e_typ"] if "e_typ" in m else pd.Series(np.nan, index=m.index)
    has = typ.notna().to_numpy()
    mood = np.log(m["m_short"] / m["m_long"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return np.column_stack([
        np.ones(len(m)), lg("v_gk20"), lg("v60"), lg("v250").fillna(lg("v60")), lg("v_ew"), earn.astype(float),
        lg("v5").fillna(lg("v_gk20")), mood,
        np.where(earn & has, np.log((typ.fillna(1) / m["vol"]).clip(0.3, 30)), 0.0),
        (earn & ~has).astype(float)])


def _q(z, lo, hi):
    return [float(np.quantile(z, lo)), float(np.quantile(z, hi))]


def fit(m: pd.DataFrame, horizon: int, min_rows: int = 500) -> dict:
    """Fit on rows m (windows that have ended): regression weights, then the up and down multipliers at each level
    and for the target day, separately for windows with ("1") and without ("0") an expected earnings report."""
    tr = m[m["rv"] > 0]
    beta, *_ = np.linalg.lstsq(design(tr), np.log(tr["rv"].to_numpy()), rcond=None)
    base = np.exp(design(m) @ beta) * np.sqrt(horizon)
    z = m["fwd"].to_numpy() / base
    zh, zl = (m["fwd"] + m["hi_end"]).to_numpy() / base, (m["fwd"] + m["lo_end"]).to_numpy() / base
    ok = np.isfinite(zh) & np.isfinite(zl)
    earn = m["earn"].to_numpy().astype(bool)
    out = {"beta": [float(b) for b in beta], "q": {}, "day": {}}
    for flag in (False, True):
        k = earn == flag
        if k.sum() < min_rows:
            k = np.ones(len(m), dtype=bool)
        out["q"][str(int(flag))] = {str(int(lv * 100)): _q(z[k], (1 - lv) / 2, (1 + lv) / 2) for lv in LEVELS}
        out["day"][str(int(flag))] = {"mid": float(np.median(z[k])), "hi": float(np.quantile(zh[k & ok], DAY_LEVEL)),
                                      "lo": float(np.quantile(zl[k & ok], 1 - DAY_LEVEL))}
    return out


def forecast(m: pd.DataFrame, horizon: int, fitted: dict) -> dict:
    """Log offsets from today's price for rows m: {'50': (low, high), ..., 'mid', 'day_hi', 'day_lo', 'vol'}."""
    vol = np.exp(design(m) @ np.asarray(fitted["beta"]))
    base = vol * np.sqrt(horizon)
    e = m["earn"].to_numpy().astype(bool)
    pick = lambda f: np.where(e, f("1"), f("0"))
    out = {"vol": vol}
    for lv in LEVELS:
        k = str(int(lv * 100))
        out[k] = (pick(lambda g: fitted["q"][g][k][0]) * base, pick(lambda g: fitted["q"][g][k][1]) * base)
    for name, key in (("mid", "mid"), ("day_hi", "hi"), ("day_lo", "lo")):
        out[name] = pick(lambda g: fitted["day"][g][key]) * base
    return out


def track_record(m: pd.DataFrame, horizon: int, first_year: int = 2012, last_year: int = 2024) -> dict:
    """Walk-forward: for each test year the weights and multipliers come only from windows that ENDED before it."""
    rows = []
    for y in range(first_year, last_year + 1):
        train = m[m["d_end"] < pd.Timestamp(f"{y}-01-01")]
        test = m[m["year"] == y]
        if len(train) < 5000 or test.empty:
            continue
        fc = forecast(test, horizon, fit(train, horizon))
        a = test["fwd"].to_numpy()
        rec = {"year": y, "n": int(len(test)), "earn": test["earn"].to_numpy().astype(bool), "mid": fc["mid"], "fwd": a}
        for lv in LEVELS:
            lo, hi = fc[str(int(lv * 100))]
            rec[f"hit{int(lv * 100)}"] = (a >= lo) & (a <= hi)
            rec[f"w{int(lv * 100)}"] = np.exp(hi) - np.exp(lo)
        dh, dl = a + test["hi_end"].to_numpy(), a + test["lo_end"].to_numpy()
        ok = np.isfinite(dh) & np.isfinite(dl)
        rec["dh"], rec["dl"], rec["dok"] = (dh <= fc["day_hi"]), (dl >= fc["day_lo"]), ok
        rec["dhw"], rec["dlw"] = np.exp(fc["day_hi"]) - 1, np.exp(fc["day_lo"]) - 1
        rows.append(rec)
    if not rows:
        return {}
    cat = lambda k: np.concatenate([r[k] for r in rows])
    earn = cat("earn")
    out = {"horizon": horizon, "years": [first_year, last_year], "forecasts": int(len(earn)), "method": "v2",
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
    ok = cat("dok")
    dh, dl = cat("dh")[ok], cat("dl")[ok]
    out["day"] = {"target": DAY_LEVEL, "high_held": float(dh.mean()), "low_held": float(dl.mean()),
                  "both_held": float((dh & dl).mean()), "avg_high": float(cat("dhw")[ok].mean()),
                  "avg_low": float(cat("dlw")[ok].mean()), "avg_mid": float(np.mean(np.exp(cat("mid")) - 1)),
                  "above_mid": float(np.mean(cat("fwd") > cat("mid")))}
    out["by_year"] = [{"year": r["year"], "hit80": float(r["hit80"].mean()), "n": r["n"]} for r in rows]
    return out


def ranges_today(price: float, row: dict, horizon: int, fitted: dict) -> dict:
    """Today's ranges for one stock. row: its latest features (MEASURES, vol, m_short, m_long, e_typ) and 'earn'
    (expected earnings in this window). Returns prices: bands {'50': [low, high], ...}, half (half the width
    as a fraction of the price, per level: (high - low) / 2), day {'mid', 'hi', 'lo'} and the volatility forecast."""
    fc = forecast(pd.DataFrame([row]), horizon, fitted)
    px = lambda off: round(float(price * np.exp(off)), 2 if price >= 5 else 4)
    out = {"bands": {}, "half": {}, "vol_fc": round(float(fc["vol"][0]), 5)}
    for lv in LEVELS:
        k = str(int(lv * 100))
        lo, hi = fc[k][0][0], fc[k][1][0]
        out["bands"][k] = [px(lo), px(hi)]
        out["half"][k] = round(float((np.exp(hi) - np.exp(lo)) / 2), 4)
    out["day"] = {"mid": px(fc["mid"][0]), "hi": px(fc["day_hi"][0]), "lo": px(fc["day_lo"][0])}
    return out
