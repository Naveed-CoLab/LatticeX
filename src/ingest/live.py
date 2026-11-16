"""Daily price update for the dashboard: the latest closing prices for every current S&P 500 member.

    python -m src.ingest.live            download if a newer closing day should exist
    python -m src.ingest.live --force    download again anyway

Why this is separate from data/clean/prices.parquet: that file is the frozen research data the models were
tested on ("universe v2"). Its downloaders only fetch tickers that have no file yet, so it does not move
forward in time. This script keeps a small, fresh copy for display instead: about three years of daily prices
per stock in data/live/prices.parquet (not committed), replaced in full on every update, so splits and
dividends are always consistent.

Prices are Yahoo's: open, high, low, close as shown on any public chart (adjusted for splits only) and
adj_close (also adjusted for dividends, used for returns). A trading day that has not finished is left out
of that file; quotes() returns it separately as "today so far" for the dashboard to show next to the forecast.
Yahoo data is for research and personal use; a commercial launch needs a licensed feed.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.config import ROOT

LIVE = ROOT / "data" / "live"
FILE = LIVE / "prices.parquet"
YEARS = 3                      # enough for one-year volatility, 12-month ranges and the last 8 earnings days
CLOSE_HOUR, CLOSE_MIN = 16, 20   # New York time; US markets close at 16:00, allow 20 minutes for final prices
RECHECK_HOURS = 3              # after a check that found nothing newer (holiday), wait this long before asking again
COLS = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]


def last_close_day(now=None) -> pd.Timestamp:
    """The most recent weekday whose US trading session has finished. (Market holidays are not known here;
    on a holiday this names a day with no prices, and the update simply finds nothing newer.)"""
    ny = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York")
    if ny.tzinfo is not None:
        ny = ny.tz_convert("America/New_York").tz_localize(None)
    day = ny.normalize()
    if (ny.hour, ny.minute) < (CLOSE_HOUR, CLOSE_MIN):
        day -= pd.to_timedelta(1, unit="D")
    while day.dayofweek >= 5:
        day -= pd.to_timedelta(1, unit="D")
    return day


def tidy(h: pd.DataFrame, ticker: str, last_day: pd.Timestamp) -> pd.DataFrame:
    """One yfinance history (auto_adjust=False) -> our columns. Drops empty rows and any unfinished day."""
    if h is None or len(h) == 0:
        return pd.DataFrame(columns=COLS)
    h = h.copy()
    if isinstance(h.columns, pd.MultiIndex):                       # newer yfinance: (field, ticker) columns
        h.columns = [c[0] if c[0] in ("Open", "High", "Low", "Close", "Adj Close", "Volume") else c[1] for c in h.columns]
    idx = pd.to_datetime(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    adj = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
    out = pd.DataFrame({"ticker": ticker, "date": idx.normalize(), "open": h["Open"].to_numpy(),
                        "high": h["High"].to_numpy(), "low": h["Low"].to_numpy(), "close": h["Close"].to_numpy(),
                        "volume": h["Volume"].to_numpy(), "adj_close": adj.to_numpy()})
    out = out.dropna(subset=["open", "high", "low", "close", "adj_close"])
    out = out[(out[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1) & (out["date"] <= last_day)]
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)[COLS]


def _yahoo(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    return yf.Ticker(ticker.replace(".", "-")).history(start=start, auto_adjust=False, actions=False)


def download(tickers: list, last_day: pd.Timestamp, fetch=None, workers: int = 4, pause: float = 0.0):
    """-> (prices for every ticker that worked, list of tickers that did not). `fetch` is swappable for tests."""
    fetch = fetch or _yahoo
    start = f"{last_day - pd.to_timedelta(365 * YEARS + 10, unit='D'):%Y-%m-%d}"

    def one(t):
        for attempt in range(2):                                   # Yahoo times out or throttles now and then
            try:
                df = tidy(fetch(t, start), t, last_day)
                if len(df):
                    return df
            except Exception:
                pass
            time.sleep(pause if pause else 1.5)
        return None

    parts, failed, done = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, df in zip(tickers, pool.map(one, tickers)):
            done += 1
            if df is None:
                failed.append(t)
            else:
                parts.append(df)
            if done % 100 == 0:
                print(f"  {done} / {len(tickers)}", flush=True)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)), failed


def market_state(now=None) -> dict:
    """Is the US stock market open right now, and when does it open next? Regular hours only (9:30 to 16:00
    New York, Monday to Friday). Market holidays are not known: on one, this says open and no prices arrive."""
    ny = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York")
    if ny.tzinfo is not None:
        ny = ny.tz_convert("America/New_York").tz_localize(None)
    day = ny.normalize()
    opens, closes = day + pd.to_timedelta(570, unit="m"), day + pd.to_timedelta(960, unit="m")
    is_open = bool(day.dayofweek < 5 and opens <= ny < closes)
    nxt = opens
    while nxt <= ny or nxt.dayofweek >= 5:
        nxt += pd.to_timedelta(1, unit="D")
    utc = lambda t: t.tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"open": is_open, "day": f"{day:%Y-%m-%d}", "ny_time": f"{ny:%H:%M}",
            "next_open": None if is_open else utc(nxt), "closes": utc(closes) if is_open else None}


def quotes(tickers: list, fetch=None, workers: int = 4, pause: float = 0.0) -> dict:
    """Today's prices so far: {ticker: {date, open, high, low, last, volume, prev}} for every ticker that has
    a bar AFTER the last finished trading day (so nothing before the market opens). `prev` is the last
    finished close. These are Yahoo's delayed prices and they change until the market closes."""
    fetch = fetch or _yahoo
    last_day = last_close_day()
    start = f"{last_day - pd.to_timedelta(8, unit='D'):%Y-%m-%d}"
    far = last_day + pd.to_timedelta(30, unit="D")

    def one(t):
        for attempt in range(2):
            try:
                df = tidy(fetch(t, start), t, far)
                if len(df):
                    new, old = df[df["date"] > last_day], df[df["date"] <= last_day]
                    if new.empty or old.empty:
                        return None
                    r = new.iloc[-1]
                    return {"date": f"{r['date']:%Y-%m-%d}", "open": round(float(r["open"]), 4), "high": round(float(r["high"]), 4),
                            "low": round(float(r["low"]), 4), "last": round(float(r["close"]), 4),
                            "volume": int(r["volume"]) if pd.notna(r["volume"]) else 0,
                            "prev": round(float(old["close"].iloc[-1]), 4)}
            except Exception:
                pass
            time.sleep(pause if pause else 1.0)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {t: q for t, q in zip(tickers, pool.map(one, tickers)) if q}


def wanted() -> list:
    """Current index members plus the market and sector funds shown at the top of the dashboard."""
    from scripts.export_dashboard import MARKET
    u = pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["end"])
    stocks = u.loc[(u["type"] == "stock") & u["end"].isna(), "ticker"].dropna().unique().tolist()
    return sorted(set(stocks) | set(MARKET))


def wanted_all() -> list:
    """wanted() plus every other stock in the universe that was still trading when the research data was
    frozen. The model's price features are ranks across ALL of these, as they were in training, so daily
    scoring needs their closing prices too. They are not shown on the dashboard."""
    base = set(wanted())
    try:
        from src.config import path
        u = pd.read_csv(ROOT / "config" / "universe.csv")
        fz = pd.read_parquet(path("clean") / "prices.parquet", columns=["ticker", "date"])
        last = fz.groupby("ticker")["date"].max()
        alive = last[last >= last.max() - pd.to_timedelta(15, unit="D")].index
        base |= set(alive) & set(u.loc[u["type"] == "stock", "ticker"])
    except Exception:
        pass                                               # no research file here: current members are enough to display
    return sorted(base)


def load():
    return pd.read_parquet(FILE) if FILE.exists() else None


def status() -> dict:
    """What the live file holds: {'day': the closing day most tickers have, 'tickers': n} or {} if none."""
    px = load()
    if px is None or px.empty:
        return {}
    last = px.groupby("ticker")["date"].max()
    return {"day": last.mode().iloc[0], "tickers": int(len(last)), "newest": last.max(), "names": set(last.index)}


def update(force: bool = False, fetch=None) -> bool:
    """Bring data/live/prices.parquet up to the last finished trading day. True if the file changed."""
    want_day, have = last_close_day(), status()
    tickers = wanted_all()
    tried = LIVE / "tickers_asked.json"                     # what the last download asked for (some are never available)
    asked = set(json.loads(tried.read_text(encoding="utf-8"))) if tried.exists() else (have.get("names", set()) if have else set())
    if have and not force and len(set(tickers) - asked) > 0.05 * len(tickers):
        print(f"The saved prices cover {have['tickers']} tickers; {len(tickers)} are needed now (the model also ranks against "
              f"former index members). Downloading them all.")
        have = {}
    if have and not force and have["day"] >= want_day:
        print(f"Prices are up to date: close of {have['day']:%a %d %b %Y}, {have['tickers']} tickers.")
        return False
    if have and not force and time.time() - FILE.stat().st_mtime < RECHECK_HOURS * 3600:
        print(f"Prices: close of {have['day']:%a %d %b %Y} (checked less than {RECHECK_HOURS} hours ago, nothing newer then).")
        return False
    print(f"Downloading closing prices to {want_day:%a %d %b %Y} for {len(tickers)} tickers (a minute or two)", flush=True)
    px, failed = download(tickers, want_day, fetch)
    if px.empty or px["ticker"].nunique() < 0.5 * len(tickers):
        print(f"Price update failed for most tickers ({len(failed)} of {len(tickers)}). Is the internet connected? "
              f"Keeping the prices already on disk.")
        return False
    LIVE.mkdir(parents=True, exist_ok=True)
    px.to_parquet(FILE, index=False)
    tried.write_text(json.dumps(sorted(tickers)), encoding="utf-8")
    got = status()
    print(f"Prices updated: close of {got['day']:%a %d %b %Y} for {got['tickers']} tickers."
          + (f" Not available from Yahoo ({len(failed)}): {' '.join(failed)}" if failed else ""))
    if got["day"] < want_day:
        print(f"  (No prices for {want_day:%a %d %b}: a market holiday, or Yahoo has not published them yet.)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download even if the file looks current")
    update(force=ap.parse_args().force)
