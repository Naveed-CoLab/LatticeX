"""Free fill-in for prices Tiingo's free plan didn't give us (the ~200 tickers after PCAR), via Yahoo Finance.

Run (on your PC, needs: pip install yfinance):
  python -m src.ingest.yahoo --check          compare Yahoo vs Tiingo on 25 tickers we already have (do this first)
  python -m src.ingest.yahoo                  download every universe ticker that has no price file yet
  python -m src.ingest.yahoo PEP XOM          just these tickers

Files are written in exactly Tiingo's raw format (data/raw/prices/<TICKER>.parquet, source="yahoo"), then
src.ingest.tiingo.combine() rebuilds data/clean/prices.parquet from everything. Tiingo files are never touched.

Yahoo's "Close" is already split-adjusted, while Tiingo's "close" is the raw traded price (the fundamentals
use raw close x shares for market cap). We rebuild the raw series by undoing later splits:
  raw_close(t) = yahoo_close(t) x product of split ratios after t
Safety: only tickers that config/universe_report.csv marks as covered are used, and each history is cut to
the date range Tiingo lists for that ticker, so a recycled symbol can't bring in another company's prices.
Yahoo data is for research/backtests; a commercial launch needs a licensed data feed.
"""
import argparse
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import ROOT, path

START = "2000-01-01"
KNOWN_RECYCLED = set()       # hand-flagged symbols (none now: build_universe drops bad spells itself)
GRACE_DAYS = 400   # real recycled cases start years later; SPGI (S&P Global) starts 1 year after our data -> kept
MIN_ROWS = 20                    # fewer rows = a stub Yahoo leaves behind for a delisted symbol, not a history


def _yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def to_tiingo_format(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance history (auto_adjust=False, actions=True) -> Tiingo raw columns."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    split = df.get("Stock Splits", pd.Series(0.0, index=df.index)).fillna(0.0).replace(0.0, 1.0)
    # product of split ratios strictly AFTER each day = how much later splits shrank this day's price
    later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out = pd.DataFrame({
        "date": df.index,
        "open": df["Open"].values * later.values,
        "high": df["High"].values * later.values,
        "low": df["Low"].values * later.values,
        "close": df["Close"].values * later.values,
        "volume": df["Volume"].values / later.values,
        "adjClose": df["Adj Close"].values,
        "adjVolume": df["Volume"].values,
        "divCash": df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0).values * later.values,
        "splitFactor": split.values,
    })
    out["ticker"], out["source"] = ticker, "yahoo"
    out["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    return out.dropna(subset=["adjClose"])


def _history(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(_yahoo_symbol(ticker)).history(start=START, auto_adjust=False, actions=True)
    return h if h is not None else pd.DataFrame()


def _ranges() -> dict:
    """Tiingo's known date range per covered ticker (guards against recycled symbols)."""
    r = pd.read_csv(ROOT / "config" / "universe_report.csv")
    r = r[r["status"].astype(str).str.startswith("covered")]
    return {t: (pd.Timestamp(a), pd.Timestamp(b)) for t, a, b in zip(r["ticker"], r["t_start"], r["t_end"])}


def download(tickers: list) -> None:
    out_dir = path("raw") / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = _ranges()
    rejected = {f.stem for f in (out_dir / "_rejected").glob("*.parquet")} if (out_dir / "_rejected").exists() else set()
    todo = [t for t in tickers if not (out_dir / f"{t}.parquet").exists() and t not in rejected]
    skipped_unknown = [t for t in todo if t not in ranges]
    todo = [t for t in todo if t in ranges]
    print(f"Yahoo: {len(todo)} tickers to download ({len(skipped_unknown)} skipped: not marked covered in "
          f"universe_report.csv)")
    failed = []
    for i, t in enumerate(todo, 1):
        h, err = pd.DataFrame(), None
        for attempt in range(3):                                 # Yahoo times out / throttles now and then
            try:
                h = _history(t)
            except Exception as e:
                err = e
            if not h.empty:
                break
            time.sleep(3 * (attempt + 1))
        if h.empty and err is not None:
            failed.append(t)
            print(f"[{i}/{len(todo)}] {t}: error {err!r:.120}")
            continue
        if h.empty:
            failed.append(t)
            print(f"[{i}/{len(todo)}] {t}: no data on Yahoo (often delisted)")
            continue
        df = to_tiingo_format(h, t)
        a, b = ranges[t]
        df = df[(df["date"] >= a) & (df["date"] <= b)]
        if df.empty:
            failed.append(t)
            print(f"[{i}/{len(todo)}] {t}: Yahoo history doesn't overlap Tiingo's range -> skipped")
            continue
        if len(df) < MIN_ROWS:
            failed.append(t)
            print(f"[{i}/{len(todo)}] {t}: only {len(df)} rows on Yahoo (delisted, history removed) -> skipped")
            continue
        df["date"] = df["date"].dt.strftime("%Y-%m-%dT00:00:00.000Z")   # same as Tiingo raw files
        df.to_parquet(out_dir / f"{t}.parquet", index=False)
        print(f"[{i}/{len(todo)}] {t}: {len(df)} rows, {df['date'].min()[:10]} to {df['date'].max()[:10]}")
        time.sleep(0.3)
    if failed:
        print(f"\nNot available from Yahoo ({len(failed)}): {' '.join(failed)}")


def recycled(first_date, spells: pd.DataFrame) -> bool:
    """True if Yahoo's history can't belong to any of the stock's membership spells: it starts well after
    the stock joined the index in every spell (e.g. 'S' = Sprint in the index until 2013, SentinelOne on
    Yahoo from 2021). A spell that is still open (current member) always counts as valid: today's symbol
    is today's member."""
    first = pd.Timestamp(first_date)
    for a, b in zip(pd.to_datetime(spells["start"]), pd.to_datetime(spells["end"])):
        joined = max(a, pd.Timestamp(START)) if pd.notna(a) else pd.Timestamp(START)
        if pd.isna(b) or first <= joined + pd.to_timedelta(GRACE_DAYS, unit="D"):
            return False
    return True


def prune(prices_dir=None, universe: pd.DataFrame | None = None) -> list:
    """Move Yahoo files that fail the recycled-symbol check to <prices>/_rejected/ (kept, just not used)."""
    d = prices_dir or (path("raw") / "prices")
    rej = d / "_rejected"
    rej.mkdir(exist_ok=True)
    u = universe if universe is not None else pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["start", "end"])
    moved = []
    for f in sorted(rej.glob("*.parquet")):                     # give earlier rejects another look (rules change)
        f.replace(d / f.name)
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty or "source" not in df or not (df["source"] == "yahoo").any():
            continue                                             # Tiingo files are never touched
        t = f.stem
        first = pd.to_datetime(df["date"]).min()
        first = first.tz_localize(None) if first.tzinfo is not None else first
        spells = u[u["ticker"] == t]
        if len(df) < MIN_ROWS:
            f.replace(rej / f.name)
            moved.append(f"{t} (only {len(df)} rows: delisted, Yahoo removed its history)")
            continue
        if t in KNOWN_RECYCLED or spells.empty or recycled(first, spells):
            f.replace(rej / f.name)
            since = f", index from {pd.Timestamp(spells['start'].min()):%Y-%m-%d}" if spells["start"].notna().any() else ""
            moved.append(f"{t} (Yahoo history from {first:%Y-%m-%d}{since})")
    print(f"Recycled-symbol / stub check: {len(moved)} Yahoo files set aside" + (":\n  " + "\n  ".join(moved) if moved else ""))
    return moved


def check(n: int = 25) -> None:
    """Download from Yahoo tickers we already have from Tiingo and compare. Prints agreement stats."""
    raw = pd.read_parquet(path("clean") / "prices.parquet")
    have = sorted(set(raw["ticker"]) - {"SPY"})
    pick = list(np.random.default_rng(0).choice(have, size=min(n, len(have)), replace=False))
    rows = []
    for t in pick:
        try:
            y = to_tiingo_format(_history(t), t)
        except Exception as e:
            print(f"{t}: error {e!r:.100}")
            continue
        if y.empty:
            continue
        a = raw[raw["ticker"] == t].set_index("date")
        b = y.set_index("date")
        j = a.join(b[["close", "adjClose", "volume"]], how="inner", rsuffix="_y")
        if len(j) < 250:
            continue
        ra, rb = np.log(j["adj_close"]).diff(), np.log(j["adjClose"]).diff()
        m = ra.notna() & rb.notna()
        rows.append({"ticker": t, "days": len(j),
                     "daily_return_corr": float(np.corrcoef(ra[m], rb[m])[0, 1]),
                     "days_return_diff_over_0.5pct": float((np.abs(ra[m] - rb[m]) > 0.005).mean()),
                     "median_raw_close_ratio": float((j["close_y"] / j["close"]).median())})
        print(f"{t}: {rows[-1]}")
    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("No comparisons made (Yahoo unreachable?).")
    out = ROOT / "results" / "data_check_yahoo_vs_tiingo.csv"
    r.to_csv(out, index=False)
    print(f"\n{len(r)} tickers compared. Median daily-return correlation {r['daily_return_corr'].median():.4f}, "
          f"median share of days differing >0.5%: {r['days_return_diff_over_0.5pct'].median():.3%}, "
          f"median raw-close ratio {r['median_raw_close_ratio'].median():.4f}  -> {out}")
    print("OK to use Yahoo for the missing tickers" if r["daily_return_corr"].median() > 0.995
          and abs(r["median_raw_close_ratio"].median() - 1) < 0.01 else
          "WARNING: Yahoo and Tiingo disagree; don't mix them yet - paste this output to Claude.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*")
    p.add_argument("--check", action="store_true")
    p.add_argument("--prune", action="store_true", help="only re-run the recycled-symbol check + rebuild prices")
    a = p.parse_args()
    if a.check:
        check()
    elif a.prune:
        prune()
        from src.ingest.tiingo import combine
        combine()
    else:
        tickers = a.tickers or pd.read_csv(ROOT / "config" / "universe.csv")["ticker"].dropna().unique().tolist()
        download(tickers)
        prune()
        from src.ingest.tiingo import combine
        combine()
