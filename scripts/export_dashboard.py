"""Build the data files the dashboard (app/) reads.

    python -m scripts.export_dashboard            all current S&P 500 members
    python -m scripts.export_dashboard --limit 40 quick test

Two price sources, on purpose:
  * data/live/prices.parquet   today's display: latest closing prices, refreshed by src/ingest/live.py.
  * data/clean/prices.parquet  the frozen research data: used to fit the range multipliers and to replay the
                               track record. Shown only as a fallback when there is no live file yet.
Also reads config/universe.csv, data/clean/sec_events.parquet and (if present) sec_events_latest.parquet.
Writes app/data/index.json (one row per stock), app/data/stocks/<TICKER>.json (candles, price ranges,
expected earnings, recent filings) and app/data/track.json (how often the ranges held in 2012-2024).

Price ranges come from src/models/vol_ranges.py (v2: volatility forecast by regression + uneven up/down, plus the
target day's likely middle, high and low). Their weights and multipliers are fitted on windows that ended by
31 Dec 2024, so the 2025-26 holdout period is never scored. app/data/ is not committed (prices are licensed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, path
from src.models import vol_ranges as vr

CALIB_END = pd.Timestamp("2024-12-31")
OUT = ROOT / "app" / "data"
# the market strip at the top of the page: index funds first, then the eleven sector funds
MARKET = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Small companies",
          "XLK": "Technology", "XLF": "Financials", "XLV": "Health care", "XLY": "Consumer cyclical",
          "XLC": "Communication", "XLI": "Industrials", "XLP": "Consumer staples", "XLE": "Energy",
          "XLU": "Utilities", "XLRE": "Real estate", "XLB": "Materials"}


SUFFIXES = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "sa", "nv", "lp", "llc", "holdings", "holding", "de", "new"}
UPPER = {"aes", "eqt", "udr", "usa", "hca", "iqvia", "nvidia", "idex", "msci", "kla", "te", "ge", "rtx", "cme", "cbre", "csx", "cvs",
         "emcor", "wec", "ibm", "ii", "iii", "uk", "adp", "amd", "att", "ups", "hp", "kkr", "eog", "aig", "apa", "dte", "ppl"}
NAMES = {"AOS": "A. O. Smith", "WRB": "W. R. Berkley", "SJM": "J.M. Smucker", "BF-B": "Brown-Forman", "T": "AT&T", "MMM": "3M",
         "JPM": "JPMorgan Chase", "GOOGL": "Alphabet (class A)", "GOOG": "Alphabet (class C)", "FOXA": "Fox (class A)", "FOX": "Fox (class B)",
         "NWSA": "News Corp (class A)", "NWS": "News Corp (class B)", "BRK-B": "Berkshire Hathaway", "PG": "Procter & Gamble",
         "KO": "Coca-Cola", "MCD": "McDonald's", "LOW": "Lowe's", "HD": "Home Depot", "DIS": "Walt Disney", "TJX": "TJX Companies",
         "TROW": "T. Rowe Price", "SCHW": "Charles Schwab", "XOM": "ExxonMobil", "PYPL": "PayPal", "GDDY": "GoDaddy",
         "SBAC": "SBA Communications", "AZO": "AutoZone", "IDXX": "IDEXX Laboratories", "FDX": "FedEx", "EBAY": "eBay",
         "LULU": "Lululemon", "CRWD": "CrowdStrike", "DASH": "DoorDash", "NOW": "ServiceNow", "APP": "AppLovin",
         "NXPI": "NXP Semiconductors", "ON": "ON Semiconductor", "TMUS": "T-Mobile US", "UNH": "UnitedHealth Group",
         "USB": "U.S. Bancorp", "UPS": "United Parcel Service", "GE": "GE Aerospace", "GEHC": "GE HealthCare", "HPQ": "HP",
         "HPE": "Hewlett Packard Enterprise", "BLK": "BlackRock", "LYB": "LyondellBasell", "COF": "Capital One",
         "WFC": "Wells Fargo", "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "CSGP": "CoStar Group", "ULTA": "Ulta Beauty",
         "DD": "DuPont", "EL": "Estée Lauder", "MDLZ": "Mondelez International", "PCG": "PG&E", "MRSH": "Marsh McLennan",
         "HONA": "Honeywell Aerospace", "FDXF": "FedEx Freight"}


def tidy_name(name: str, ticker: str = "") -> str:
    """The SEC's registrant name -> the name people use: 'Church & Dwight Co Inc /De/' -> 'Church & Dwight',
    'Cdw Corp' -> 'CDW', 'Mccormick & Co Inc' -> 'McCormick'."""
    if ticker in NAMES:
        return NAMES[ticker]
    import re
    n = re.sub(r"\s*/\s*[A-Za-z]{2,4}\s*/?\s*$", "", str(name)).replace("/", " ").replace(",", " ")
    words = n.split()
    while len(words) > 1 and words[-1].lower().strip(".&") in SUFFIXES | {""}:
        words.pop()
    out = []
    for w in words:
        k = w.lower().strip(".")
        if k in UPPER or (len(k) <= 4 and "&" in k) or (2 <= len(k) <= 3 and k.isalpha() and not set(k) & set("aeiou")):
            w = w.upper()
        w = re.sub(r"^Mc([a-z])", lambda m: "Mc" + m.group(1).upper(), w)
        if out and w in ("Of", "And", "The", "For", "In"):
            w = w.lower()
        out.append(w)
    return " ".join(out) or str(name)


def company_names() -> dict:
    """ticker -> company name from the SEC's ticker list (cached in data/raw/company_names.json), tidied."""
    return {t: tidy_name(n, t) for t, n in raw_company_names().items()}


def raw_company_names() -> dict:
    cache = path("raw") / "company_names.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        from src.ingest.edgar import TICKERS_URL, _get
        r = _get(TICKERS_URL)
        r.raise_for_status()
        names = {v["ticker"].upper().replace(".", "-"): str(v["title"]).title() for v in r.json().values()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(names), encoding="utf-8")
        return names
    except Exception as e:                                   # offline or no SEC_USER_AGENT: tickers only
        print(f"Company names not available ({e!r:.80}); showing tickers only.")
        return {}


def earnings_moves(dates: np.ndarray, adj: np.ndarray, ev: pd.DataFrame, n: int = 8) -> list:
    """The stock's move on the first trading day each past earnings report could be acted on.
    A report published after 4 pm New York counts for the next trading day."""
    out = []
    for filed, accepted in zip(ev["filed"], ev["accepted"]):
        day = np.datetime64(pd.Timestamp(filed).normalize())
        after_close = True
        if pd.notna(accepted):
            ny = pd.Timestamp(accepted).tz_convert("America/New_York")
            after_close = ny.hour >= 16
            day = np.datetime64(ny.tz_localize(None).normalize())
        i = np.searchsorted(dates, day, side="right" if after_close else "left")
        if 0 < i < len(dates):
            out.append({"date": str(pd.Timestamp(filed).date()), "move": round(float(adj[i] / adj[i - 1] - 1), 4)})
    return out[-n:]


def ranking_record() -> dict:
    """v1.0 ranking results saved in results/ by the final Kaggle run (if present)."""
    out = {}
    for h in vr.HORIZONS:
        d = sorted((ROOT / "results").glob(f"*_real_stock_h{h}_fund_sect"))
        d = [x for x in d if (x / "metrics.csv").exists() and (x / f"rank_groups_h{h}.csv").exists()]
        if not d:
            continue
        m = pd.read_csv(d[-1] / "metrics.csv").set_index("model").loc["mlp"]
        g = pd.read_csv(d[-1] / f"rank_groups_h{h}.csv").set_index("rank_group")
        y = pd.read_csv(d[-1] / "rank_ic_by_year.csv")
        out[str(h)] = {"run": d[-1].name, "rank_ic": round(float(m["rank_ic"]), 4), "t": round(float(m["ic_tstat_nw"]), 2),
                       "hit_rate": round(float(m["hit_rate"]), 4), "forecasts": int(m["n_rows"]),
                       "positive_years": int((y["mlp"] > 0).sum()), "years": int(len(y)),
                       "top20_beat": round(float(g.loc["top 20%", "share_beating_spy"]), 4),
                       "bottom20_beat": round(float(g.loc["bottom 20%", "share_beating_spy"]), 4)}
        sc = d[-1] / f"rank_scores_h{h}.csv"          # how each 1-10 score did (scripts/rank_scores.py)
        if sc.exists():
            t = pd.read_csv(sc)
            out[str(h)]["scores"] = [{"score": int(r.score), "beat": round(float(r.beat_spy), 4),
                                      "excess": round(float(r.mean_excess), 5)} for r in t.itertuples()]
            out[str(h)]["score_years"] = [str(t["first"].iloc[0])[:4], str(t["last"].iloc[0])[:4]]
    return out


def past_calls(px: pd.DataFrame, spy: pd.DataFrame, since: pd.Timestamp) -> dict:
    """The model's past calls for each stock, for the chart: {ticker: {h: [[date, 'top'|'bottom', score, result]]}}.
    A call = the stock was ranked in the top or bottom fifth for that look-ahead. Result = its return minus SPY's
    over the window, or None while the window is still open. Two sources, both made before the outcome was known:
      * the one-time 2025-26 holdout test (model trained on data to 2024): results/*_h<h>_fund_sect_HOLDOUT
      * the daily forecast log written after each close since the model went live: data/live/forecasts/*.csv
    Weekly (every 5th trading day) like the backtest, from `since` on."""
    from scripts.rank_scores import score_of
    rows = []
    for h in vr.HORIZONS:
        runs = sorted((ROOT / "results").glob(f"*_real_stock_h{h}_fund_sect_HOLDOUT"))
        if runs and (runs[-1] / "predictions.parquet").exists():
            p = pd.read_parquet(runs[-1] / "predictions.parquet", columns=["date", "ticker", "model", "y_raw", "y_true"])
            p = p[p["model"] == "mlp"].copy()
            p["pct"] = p.groupby("date")["y_raw"].rank(pct=True)
            p["result"] = np.expm1(p["y_true"])
            rows.append(p.assign(h=h)[["date", "ticker", "h", "pct", "result"]])
    log_dir = ROOT / "data" / "live" / "forecasts"
    logs = sorted(log_dir.glob("*.csv")) if log_dir.exists() else []
    if logs:
        lg = pd.concat([pd.read_csv(f, usecols=["asof", "ticker", "horizon", "pct"]) for f in logs], ignore_index=True)
        lg = lg.rename(columns={"asof": "date", "horizon": "h"})
        lg["date"] = pd.to_datetime(lg["date"])
        last_holdout = max((r["date"].max() for r in rows), default=pd.Timestamp("1900-01-01"))
        lg = lg[lg["date"] > last_holdout]
        days = sorted(lg["date"].unique())
        lg = lg[lg["date"].isin(days[::5])]                      # weekly, like the backtest
        # result from the daily prices: the stock's log return minus SPY's over the next h trading days
        cal = pd.DatetimeIndex(sorted(spy["date"].unique()))
        sp = spy.set_index("date")["adj_close"]
        wide = px.pivot_table(index="date", columns="ticker", values="adj_close")
        res = []
        for d, h, t in zip(lg["date"], lg["h"], lg["ticker"]):
            i = cal.searchsorted(d)
            if i + h >= len(cal) or t not in wide or d not in wide.index:
                res.append(np.nan)
                continue
            e = cal[i + h]
            a, b = wide.at[d, t], wide.at[e, t] if e in wide.index else np.nan
            res.append(float(np.expm1(np.log(b / a) - np.log(sp[e] / sp[d]))) if a > 0 and b > 0 else np.nan)
        lg["result"] = res
        rows.append(lg[["date", "ticker", "h", "pct", "result"]])
    if not rows:
        return {}
    c = pd.concat(rows, ignore_index=True)
    c = c[(c["date"] >= since) & ((c["pct"] >= 0.8) | (c["pct"] <= 0.2))]
    c["score"] = score_of(c["pct"])
    out = {}
    for (t, h), g in c.sort_values("date").groupby(["ticker", "h"]):
        out.setdefault(t, {})[str(h)] = [[f"{d:%Y-%m-%d}", "top" if p >= 0.8 else "bottom", int(s),
                                           None if pd.isna(r) else round(float(r), 4)]
                                          for d, p, s, r in zip(g["date"], g["pct"], g["score"], g["result"])]
    return out


def holdout_record() -> dict:
    """The one-time 2025-26 holdout results (docs/holdout-lock.md), if they have been run: shown as the
    'never seen during development' figures next to the backtest."""
    out = {}
    runs = sorted((ROOT / "results").glob("*_ranges_HOLDOUT"))
    if runs and (runs[-1] / "summary.json").exists():
        s = json.loads((runs[-1] / "summary.json").read_text(encoding="utf-8"))
        out["ranges"] = {h: {"forecasts": v["forecasts"], "first": v["first"], "last": v["last"],
                             "held": {lv: round(x["held"], 4) for lv, x in v["v2"].items()},
                             "day_high": round(v["day"]["high_held"], 4), "day_low": round(v["day"]["low_held"], 4)}
                         for h, v in s["horizons"].items()}
    return out


def important(items: str) -> str:
    """Plain words for only the serious items of a filing (severity 2 or 3), for the overview's short list."""
    from src.ingest.sec_events import ITEMS
    return "; ".join(ITEMS[x][1] for x in items.replace(" ", "").split(",") if x in ITEMS and ITEMS[x][2] >= 2)


PRICE_COLS = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]
FIT_VERSION = 2                 # raise when the range method changes, so the saved fit is rebuilt


def frozen_prices() -> pd.DataFrame:
    return pd.read_parquet(path("clean") / "prices.parquet", columns=PRICE_COLS)


# SEC industry code (SIC) -> one of 11 sectors. SIC is what every company reports to the SEC; the ranges follow
# the SEC's own division of industries, with the usual adjustments (drugs, medical devices, software and
# semiconductors, food and household goods, REITs). Approximate: the commercial GICS sectors are licensed.
SIC_SECTORS = [
    (100, 999, "Consumer staples"), (1000, 1199, "Materials"), (1200, 1399, "Energy"), (1400, 1499, "Materials"),
    (1520, 1531, "Consumer cyclical"), (1500, 1799, "Industrials"), (2000, 2199, "Consumer staples"),
    (2200, 2399, "Consumer cyclical"), (2400, 2499, "Materials"), (2500, 2599, "Consumer cyclical"),
    (2600, 2699, "Materials"), (2700, 2799, "Communication"), (2830, 2836, "Health care"),
    (2840, 2844, "Consumer staples"), (2800, 2899, "Materials"), (2900, 2999, "Energy"), (3000, 3099, "Materials"),
    (3100, 3199, "Consumer cyclical"), (3200, 3399, "Materials"), (3400, 3569, "Industrials"),
    (3570, 3579, "Technology"), (3580, 3629, "Industrials"), (3630, 3659, "Consumer cyclical"),
    (3660, 3699, "Technology"), (3710, 3716, "Consumer cyclical"), (3700, 3799, "Industrials"),
    (3840, 3851, "Health care"), (3800, 3899, "Technology"), (3900, 3999, "Consumer cyclical"),
    (4700, 4799, "Consumer cyclical"), (4000, 4799, "Industrials"), (4800, 4899, "Communication"),
    (4950, 4959, "Industrials"), (4900, 4999, "Utilities"), (5122, 5122, "Health care"),
    (5140, 5149, "Consumer staples"), (5000, 5199, "Industrials"), (5400, 5499, "Consumer staples"),
    (5331, 5331, "Consumer staples"), (5912, 5912, "Consumer staples"), (5200, 5999, "Consumer cyclical"),
    (6324, 6324, "Health care"), (6798, 6798, "Real estate"), (6500, 6599, "Real estate"), (6000, 6799, "Financials"),
    (7000, 7299, "Consumer cyclical"), (7310, 7319, "Communication"), (7370, 7379, "Technology"),
    (7300, 7399, "Industrials"), (7800, 7899, "Communication"), (7900, 7999, "Consumer cyclical"),
    (8000, 8099, "Health care"), (8700, 8799, "Industrials"),
]


# companies whose SEC code points elsewhere than where investors place them
SECTOR_FIX = {"V": "Financials", "MA": "Financials", "PYPL": "Financials", "DIS": "Communication"}


def sic_sector(sic, ticker: str | None = None) -> str | None:
    if ticker in SECTOR_FIX:
        return SECTOR_FIX[ticker]
    try:
        n = int(sic)
    except (TypeError, ValueError):
        return None
    return next((name for lo, hi, name in SIC_SECTORS if lo <= n <= hi), None)


def companies() -> dict:
    """{ticker: {"sector": ..., "industry": ...}} from data/live/companies.json, which the SEC filings check
    writes (src/ingest/sec_events.py). Empty until the first check after this was added."""
    f = ROOT / "data" / "live" / "companies.json"
    if not f.exists():
        print("No company industry codes yet; the sector filter appears after the next SEC filings check.")
        return {}
    raw = json.loads(f.read_text(encoding="utf-8"))
    return {t: {"sector": sic_sector(v.get("sic"), t), "industry": (v.get("industry") or "").strip().capitalize() or None}
            for t, v in raw.items()}


def stock_rows(px: pd.DataFrame, tickers) -> pd.DataFrame:
    px = px[px["ticker"].isin(tickers) & (px[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return px.sort_values(["ticker", "date"]).reset_index(drop=True)


def fingerprint(f: Path) -> list:
    """What identifies a research file for the saved range fit. The frozen data files never change in place, so
    their size does; text files are hashed with line endings removed, because Git gives a Windows PC CRLF and a
    Linux server LF for the same file. File times are left out. So a fit copied from the PC to a server is reused."""
    if not f.exists():
        return [f.name]
    if f.suffix in (".csv", ".txt", ".json"):
        import hashlib
        return [f.name, hashlib.sha1(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]]
    return [f.name, f.stat().st_size]


def memory_gb() -> float | None:
    """Total memory of this machine in GB (Linux and macOS), None where it cannot be read (Windows)."""
    import os
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return None


def range_fit(u: pd.DataFrame, ev: pd.DataFrame | None) -> tuple:
    """Weights and multipliers (windows that ended by CALIB_END) and the 2012-2024 walk-forward record, from the
    frozen research data. Takes a minute or two, so the result is kept until one of its input files changes."""
    files = [path("clean") / "prices.parquet", path("clean") / "sec_events.parquet", ROOT / "config" / "universe.csv"]
    key = [FIT_VERSION, f"{CALIB_END:%Y-%m-%d}"] + [fingerprint(f) for f in files]
    cache = ROOT / "data" / "live" / "range_fit.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return {int(h): v for h, v in saved["fit"].items()}, saved["ranges"]
    mem = memory_gb()
    if mem is not None and mem < 3.5:          # the fit needs about 4.5 GB; a small server would be killed half-way
        raise SystemExit(f"The saved range fit (data/live/range_fit.json) is missing or out of date, and refitting it needs about "
                         f"4.5 GB of memory; this machine has {mem:.1f} GB. Run the dashboard once on the PC, then copy the file "
                         f"here with deploy/aws/upload_data.ps1.")
    print("Fitting the ranges on the research data and replaying 2012-2024 (a minute or two, only when the data changes)", flush=True)
    allpx = frozen_prices()
    px = stock_rows(allpx, u["ticker"])
    f = vr.features(px, stock_rows(allpx, ["SPY"]))
    est, et = vr.expected_earnings(ev), vr.earnings_typical(px, ev)
    fits, ranges = {}, {}
    for h in vr.HORIZONS:
        m = vr.panel(f, u, est, h, et)
        fits[h] = vr.fit(m[m["d_end"] <= CALIB_END], h)
        ranges[str(h)] = vr.track_record(m, h)
        lv, dy = ranges[str(h)]["levels"]["80"], ranges[str(h)]["day"]
        print(f"  {h:>2}-day ranges, 2012-2024: 80% range held {lv['hit_rate']:.1%} of the time "
              f"(earnings windows {lv['hit_rate_earnings']:.1%}, others {lv['hit_rate_other']:.1%}), "
              f"average width {lv['avg_width']:.1%}; day high below X {dy['high_held']:.1%}, low above Y {dy['low_held']:.1%}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "ranges": ranges, "fit": {str(h): v for h, v in fits.items()}}), encoding="utf-8")
    return fits, ranges


def all_events(ev: pd.DataFrame | None) -> pd.DataFrame | None:
    """The frozen 8-K table plus the filings fetched since (sec_events_latest.parquet), without duplicates."""
    new = path("clean") / "sec_events_latest.parquet"
    if ev is None or not new.exists():
        return ev
    add = pd.read_parquet(new)
    add = add[[c for c in ev.columns if c in add.columns]]
    add = add[add["ticker"].isin(ev["ticker"].unique()) & ~add["accession"].isin(ev["accession"])]
    return pd.concat([ev, add], ignore_index=True) if len(add) else ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N stocks (quick test)")
    ap.add_argument("--candles", type=int, default=260, help="trading days of history per stock")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "stocks").mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["start", "end"])
    u = u[u["type"] == "stock"]
    ev_file = path("clean") / "sec_events.parquet"
    ev = pd.read_parquet(ev_file) if ev_file.exists() else None
    fits, ranges = range_fit(u, ev)
    ev = all_events(ev)
    est = vr.expected_earnings(ev)

    # prices to show: the live file if there is one, else the frozen research file (with a warning)
    from src.ingest import live
    px = live.load()
    source = "live" if px is not None and len(px) else "saved"
    if source == "saved":
        px = frozen_prices()
        print("NOTE: no updated prices yet (data/live/prices.parquet). Showing the saved research prices, which "
              "stop on the day they were downloaded. Run `python -m scripts.run_dashboard` to update.")
    funds = px[px["ticker"].isin(MARKET)].sort_values(["ticker", "date"])
    info = companies()
    px = stock_rows(px, u["ticker"])
    # One market day for everything: the latest day that 9 in 10 stocks have reached. If a download stopped
    # half-way, some stocks are days ahead of others and "today's biggest moves" would mix different days.
    last = px.groupby("ticker")["date"].max()
    recent = last[last >= last.max() - pd.to_timedelta(10, unit="D")]
    asof = recent.sort_values().iloc[int(len(recent) * 0.1)]
    if (recent > asof).any():
        print(f"Note: {int((recent > asof).sum())} stocks have prices after {asof:%Y-%m-%d} and the rest do not. "
              f"Showing every stock as of {asof:%Y-%m-%d}; run `python -m src.ingest.live --force` to bring all up to date.")
        px = px[px["date"] <= asof].reset_index(drop=True)
        funds = funds[funds["date"] <= asof]
    print("Measuring each stock's volatility", flush=True)
    px = vr.features(px, stock_rows(funds, ["SPY"]))
    typical_now = vr.earnings_typical(px, ev).groupby("ticker")["e_typ"].last() if ev is not None else pd.Series(dtype=float)
    print(f"Closing prices of {asof:%a %d %b %Y} ({'updated' if source == 'live' else 'SAVED, not updated'}), "
          f"{len(set(px['ticker']) & set(u.loc[u['end'].isna(), 'ticker']))} index members; "
          f"{'earnings calendar for ' + str(len(est)) + ' companies' if est else 'no SEC events file: ranges are not earnings-aware'}")
    # the ranking model's scores for this closing day (written by src/models/production.py), if there are any
    rank_file = ROOT / "data" / "live" / "rank.json"
    rank = json.loads(rank_file.read_text(encoding="utf-8")) if rank_file.exists() else None
    if rank and rank["asof"] != f"{asof:%Y-%m-%d}":
        print(f"Model scores on disk are for {rank['asof']}, not {asof:%Y-%m-%d}: not shown. Run: python -m scripts.score_today")
        rank = None
    if rank:
        print(f"Model ranks for {rank['n']} stocks (model trained on data to {rank['model']['trained_through']})")
    track = {"asof": f"{asof:%Y-%m-%d}", "calibrated_to": f"{CALIB_END:%Y-%m-%d}", "ranges": ranges,
             "ranking": ranking_record(), "holdout": holdout_record(),
             "model": {**rank["model"], "n": rank["n"]} if rank else None}
    (out / "track.json").write_text(json.dumps(track), encoding="utf-8")

    print("Looking up company names", flush=True)
    names = company_names()
    current = u.loc[u["end"].isna(), "ticker"].unique()
    last = px.groupby("ticker")["date"].max()
    live = sorted(t for t in current if t in last.index and last[t] >= asof - pd.to_timedelta(10, unit="D"))
    if a.limit:
        live = live[: a.limit]
    evs = {t: g.sort_values("filed") for t, g in ev.groupby("ticker")} if ev is not None else {}
    words = None
    if ev is not None:
        from src.ingest.sec_events import describe
        words = describe

    index = []
    calls = past_calls(px, stock_rows(funds, ["SPY"]), asof - pd.to_timedelta(a.candles * 7 // 5 + 7, unit="D"))
    if calls:
        print(f"Past calls of the model on the charts: {sum(len(v) for x in calls.values() for v in x.values()):,} "
              f"(2025-26 holdout test and the daily log)")
    print(f"Writing {len(live)} stock files", flush=True)
    for n, t in enumerate(live, 1):
        if n % 100 == 0:
            print(f"  {n} / {len(live)}", flush=True)
        d = px[px["ticker"] == t]
        if len(d) < 60 or not np.isfinite(d["vol"].iloc[-1]):
            continue
        # live prices are already as any public chart shows them; the saved research file holds raw traded
        # prices, which need the adjustment factor to be continuous across splits
        k = np.ones(len(d)) if source == "live" else (d["adj_close"] / d["close"]).to_numpy()
        tail = d.tail(a.candles)
        kt = k[-len(tail):]
        nd = 2 if tail["close"].iloc[-1] >= 5 else 4
        candles = [[f"{dt:%Y-%m-%d}", round(o * f, nd), round(h_ * f, nd), round(l * f, nd), round(c * f, nd), int(v)]
                   for dt, o, h_, l, c, v, f in zip(tail["date"], tail["open"], tail["high"], tail["low"],
                                                    tail["close"], tail["volume"].fillna(0), kt)]
        price, vol = float(d["close"].iloc[-1]), float(d["vol"].iloc[-1])
        own = d["date"].iloc[-1]                                    # this stock's own last trading day
        future = pd.bdate_range(own + pd.to_timedelta(1, unit="D"), periods=max(vr.HORIZONS) + 24)
        nxt = vr.next_expected(t, own, est)
        e = evs.get(t)
        # The estimate stays "expected" for 3 days after its date. If the report is already filed, say so and
        # show the following one as next (the range keeps its earnings width for those days, as it was tested).
        show, reported = nxt, None
        if e is not None and nxt is not None and nxt <= own:
            done = e.loc[e["ev_earnings"].astype(bool) & (e["filed"] >= nxt - pd.to_timedelta(10, unit="D"))
                         & (e["filed"] <= own), "filed"]
            if len(done):
                reported, show = done.max(), vr.next_expected(t, own + pd.to_timedelta(4, unit="D"), est)
        rng = {}
        today = {c: d[c].iloc[-1] for c in vr.MEASURES + ["vol", "m_short", "m_long"]}
        today["e_typ"] = typical_now.get(t, np.nan)
        for h in vr.HORIZONS:
            end = future[h - 1]
            earn = bool(nxt is not None and nxt <= end + vr.EARN_TOL)
            rng[str(h)] = {"end": f"{end:%Y-%m-%d}", "earn": earn,
                           **vr.ranges_today(price, {**today, "earn": earn}, h, fits[h])}
        hist, filings = [], []
        if e is not None:
            hist = earnings_moves(d["date"].to_numpy(), d["adj_close"].to_numpy(), e[e["ev_earnings"].astype(bool)])
            recent = e[e["filed"] >= asof - pd.to_timedelta(120, unit="D")].sort_values("filed", ascending=False).head(8)
            filings = [{"date": f"{f:%Y-%m-%d}", "what": words(str(i)), "sev": int(s), "key": important(str(i))}
                       for f, i, s in zip(recent["filed"], recent["items"], recent["severity"])]
        typical = round(float(np.median([abs(x["move"]) for x in hist])), 4) if hist else None
        adj = d["adj_close"].to_numpy()
        chg = float(adj[-1] / adj[-2] - 1)
        back = lambda n: round(float(adj[-1] / adj[-1 - n] - 1), 4) if len(adj) > n else None
        yr = d.tail(252)
        ky = k[-len(yr):]
        stats = {"open": round(float(d["open"].iloc[-1]), nd), "high": round(float(d["high"].iloc[-1]), nd),
                 "low": round(float(d["low"].iloc[-1]), nd),
                 "hi52": round(float((yr["high"].to_numpy() * ky).max()), nd), "lo52": round(float((yr["low"].to_numpy() * ky).min()), nd),
                 "avgvol": int(d["volume"].tail(20).mean()), "chg5": back(5), "chg20": back(20), "chg252": back(252)}
        flag = next(({"date": f["date"], "what": f["key"] or f["what"]} for f in filings
                     if f["sev"] >= 2 and pd.Timestamp(f["date"]) >= asof - pd.to_timedelta(45, unit="D")), None)
        stock = {"t": t, "name": names.get(t, t), "asof": f"{d['date'].iloc[-1]:%Y-%m-%d}", "price": round(price, nd),
                 "chg": round(chg, 4), "vol": round(vol, 5), "stats": stats, "candles": candles,
                 "future": [f"{x:%Y-%m-%d}" for x in future], "ranges": rng,
                 "earnings": {"next": f"{show:%Y-%m-%d}" if show is not None else None,
                              "reported": f"{reported:%Y-%m-%d}" if reported is not None else None,
                              "history": hist, "typical": typical},
                 "filings": filings, "sector": info.get(t, {}).get("sector"), "calls": calls.get(t),
                 "industry": info.get(t, {}).get("industry"),
                 "rank": ({h: {"pos": v[0], "pct": v[1], "exp": v[2]} for h, v in rank["stocks"][t].items()}
                          if rank and t in rank["stocks"] else None)}
        (out / "stocks" / f"{t}.json").write_text(json.dumps(stock, separators=(",", ":")), encoding="utf-8")
        sp = adj[-30:] / adj[-30:][0]
        index.append({"t": t, "name": stock["name"], "sec": info.get(t, {}).get("sector"), "price": stock["price"], "chg": stock["chg"],
                      "w": {h: rng[h]["half"]["80"] for h in rng}, "earn": {h: rng[h]["earn"] for h in rng},
                      "next_earn": stock["earnings"]["next"], "reported": stock["earnings"]["reported"], "chg5": stats["chg5"], "chg20": stats["chg20"],
                      "flag": flag, "spark": [round(float(x), 3) for x in sp],
                      "rk": {h: v["pct"] for h, v in stock["rank"].items()} if stock["rank"] else None})
    market = []
    for t, label in MARKET.items():
        f = funds[funds["ticker"] == t]
        if len(f) < 40:
            continue
        adj = f["adj_close"].to_numpy()
        market.append({"t": t, "name": label, "price": round(float(f["close"].iloc[-1]), 2),
                       "chg": round(float(adj[-1] / adj[-2] - 1), 4), "chg20": round(float(adj[-1] / adj[-21] - 1), 4),
                       "spark": [round(float(x), 3) for x in adj[-30:] / adj[-30]]})
    shown = {x["t"] for x in index}
    gone = sorted(t for t in current if t not in shown) if not a.limit else []
    if gone:
        print(f"Not shown, no recent prices ({len(gone)}): {' '.join(gone)}")
    (out / "index.json").write_text(json.dumps({"asof": f"{asof:%Y-%m-%d}", "source": source,
                                                 "built": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ}",
                                                 "horizons": list(vr.HORIZONS),
                                                 "market": market, "stocks": index}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(index)} stocks to {out}  ->  python -m scripts.run_dashboard")


if __name__ == "__main__":
    main()
