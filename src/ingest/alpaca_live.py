"""Today's prices in real time from Alpaca's free market data (IEX exchange), and the market clock.

Used by the dashboard server when ALPACA_API_KEY and ALPACA_API_SECRET are in .env; otherwise it falls back to
Yahoo (src/ingest/live.py), which can be delayed. What the free plan gives (docs.alpaca.markets, "About Market
Data API"): real-time prices from the IEX exchange only, 200 REST calls a minute. IEX handles a small share of US
trading, so its last price is usually within cents of the full market's, and its volume is IEX's volume only:
the page labels these "IEX real-time" and does not show IEX volume as the day's volume.

  * snapshots(tickers, prev) -> {ticker: quote}  latest IEX trade plus IEX's high and low so far today, in the
    same shape as live.quotes(); `prev` is our own last close (IEX's previous close is only IEX's last trade).
  * clock() -> {is_open, next_open, next_close} from Alpaca's trading API, which knows market holidays.
  * daily_bars(tickers, day) -> {ticker: bar}  the day's consolidated bar (all exchanges), used to check closes.
Both stay under a call budget well below the plan's limit and return {} / None on any failure.
"""
import threading
import time
from collections import deque

import pandas as pd
import requests

from src.config import secret

SNAPSHOTS = "https://data.alpaca.markets/v2/stocks/snapshots"
CLOCKS = ("https://paper-api.alpaca.markets/v2/clock", "https://api.alpaca.markets/v2/clock")
PER_CALL = 100                    # symbols per snapshots request
MAX_PER_MIN = 150                 # our own ceiling; the free plan allows 200 calls a minute


def available() -> bool:
    try:
        return bool(secret("ALPACA_API_KEY") and secret("ALPACA_API_SECRET"))
    except RuntimeError:
        return False


def _headers() -> dict:
    return {"APCA-API-KEY-ID": secret("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": secret("ALPACA_API_SECRET")}


class Budget:
    """At most `n` calls in any 60 seconds, shared by every thread."""

    def __init__(self, n: int = MAX_PER_MIN):
        self.n, self.calls, self.lock = n, deque(), threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.n:
                return False
            self.calls.append(now)
            return True


BUDGET = Budget()


def _ny_day(ts: str) -> str:
    return f"{pd.Timestamp(ts).tz_convert('America/New_York'):%Y-%m-%d}"


def parse(snap: dict, ticker: str, after: str, prev: float | None) -> dict | None:
    """One snapshot -> our quote dict, or None if it has no trade after the last finished trading day `after`."""
    trade = (snap or {}).get("latestTrade") or {}
    if not trade.get("p") or not trade.get("t"):
        return None
    d = _ny_day(trade["t"])
    if d <= after:
        return None
    bar = snap.get("dailyBar") or {}
    same = bool(bar.get("t")) and _ny_day(bar["t"]) == d
    last = float(trade["p"])
    pc = prev if prev else float((snap.get("prevDailyBar") or {}).get("c") or 0) or None
    if not pc:
        return None
    return {"date": d, "open": round(float(bar["o"]), 4) if same else last,
            "high": round(max(float(bar["h"]), last), 4) if same else last,
            "low": round(min(float(bar["l"]), last), 4) if same else last,
            "last": round(last, 4), "volume": 0, "prev": round(float(pc), 4),
            "time": trade["t"], "src": "iex"}


def snapshots(tickers: list, prev: dict, after: str, get=None) -> dict:
    """{ticker: quote} for the tickers that traded on IEX after `after` (YYYY-MM-DD). `prev`: our last closes.
    `get` is swappable for tests: get(url, params, headers) -> parsed JSON."""
    get = get or (lambda url, params, headers: requests.get(url, params=params, headers=headers, timeout=15).json())
    out = {}
    for i in range(0, len(tickers), PER_CALL):
        part = tickers[i:i + PER_CALL]
        if not BUDGET.take():
            break
        try:
            j = get(SNAPSHOTS, {"symbols": ",".join(t.replace("-", ".") for t in part), "feed": "iex"}, _headers())
        except Exception:
            continue
        j = j.get("snapshots", j) if isinstance(j, dict) else {}
        for t in part:
            q = parse(j.get(t.replace("-", ".")), t, after, prev.get(t))
            if q:
                out[t] = q
    return out


BARS = "https://data.alpaca.markets/v2/stocks/bars"


def daily_bars(tickers: list, day, get=None) -> dict:
    """{ticker: {open, high, low, close, volume}} for one finished trading day, from all US exchanges (SIP feed).
    The free plan allows SIP history once it is more than 15 minutes old, so this is used after the close to check
    the closing prices that came from Yahoo. Prices are unadjusted (as traded that day). {} on any failure."""
    get = get or (lambda url, params, headers: requests.get(url, params=params, headers=headers, timeout=20).json())
    d = f"{pd.Timestamp(day):%Y-%m-%d}"
    start = pd.Timestamp(d, tz="America/New_York")
    end = min(start + pd.Timedelta(hours=23, minutes=59), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))
    span = {"start": f"{start.tz_convert('UTC'):%Y-%m-%dT%H:%M:%SZ}", "end": f"{end.tz_convert('UTC'):%Y-%m-%dT%H:%M:%SZ}"}
    alias = {t.replace("-", "."): t for t in tickers}
    syms, out = list(alias), {}
    for i in range(0, len(syms), 200):
        token = None
        while True:
            if not BUDGET.take():
                return out
            params = {"symbols": ",".join(syms[i:i + 200]), "timeframe": "1Day", **span,
                      "feed": "sip", "adjustment": "raw", "limit": 10000}
            if token:
                params["page_token"] = token
            try:
                j = get(BARS, params, _headers()) or {}
            except Exception:
                break
            for sym, rows in (j.get("bars") or {}).items():
                for b in rows or []:
                    if sym in alias and b.get("c") and b.get("t") and _ny_day(b["t"]) == d:
                        out[alias[sym]] = {"open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
                                           "close": float(b["c"]), "volume": float(b["v"])}
            token = j.get("next_page_token")
            if not token:
                break
    return out


_CLOCK = {"at": 0.0, "value": None, "url": None}


def clock(get=None) -> dict | None:
    """{'is_open', 'next_open', 'next_close', 'timestamp'} from Alpaca, cached for a minute; None if unavailable."""
    if time.time() - _CLOCK["at"] < 60:
        return _CLOCK["value"]
    get = get or (lambda url, headers: requests.get(url, headers=headers, timeout=10))
    value = None
    for url in ([_CLOCK["url"]] if _CLOCK["url"] else CLOCKS):
        try:
            r = get(url, _headers())
            if r.status_code == 200:
                value, _CLOCK["url"] = r.json(), url
                break
        except Exception:
            continue
    _CLOCK.update(at=time.time(), value=value)
    return value
