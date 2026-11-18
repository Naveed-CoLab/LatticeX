"""Open the LatticeIQ dashboard in your browser, with the latest closing prices.

    python -m scripts.run_dashboard               update prices and filings if a newer day exists, then open
    python -m scripts.run_dashboard --no-update   open with the data already on disk (no internet needed)
    python -m scripts.run_dashboard --refresh     rebuild the dashboard data even if nothing changed
    python -m scripts.run_dashboard --port 8700
    python -m scripts.run_dashboard --no-browser  on a server (see deploy/aws/README.md)

Every start checks whether a newer closing day should exist (US markets close at 4 pm New York time) and, if
so, downloads it (about a minute) and the SEC filings since the last check (a few minutes, once a day).

While it runs and the US market is open, it also fetches today's prices so far every 10 minutes (and the
stock you are looking at every 20 seconds), and after the close it rolls the forecasts forward to the new closing
day by itself. The forecasts are always made from the last finished close; today's price is shown next to them.
Opening a stock also fetches what analysts expect for it (12-month price targets and ratings), once a day.
If a trained model is saved (models/latticeiq_v1.npz, see scripts/train_final.py), every stock is ranked with it
after each close and the rank is shown on the page.
Runs on this computer only (127.0.0.1) unless --host says otherwise. Stop it with Ctrl+C.

Login: if LATTICEIQ_PASSWORD is set (in .env or the environment), every page and data request needs that
password (the browser asks for it once; any user name works). Required with --host other than 127.0.0.1. On a
server, put a web server with HTTPS in front (deploy/aws/), so the password never travels unencrypted.
"""
import argparse
import base64
import functools
import hmac
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

import pandas as pd

from src.config import ROOT, path

APP = ROOT / "app"
INDEX = APP / "data" / "index.json"
FILINGS_EVERY_HOURS = 20
TODAY_EVERY_MIN = 10            # all stocks' prices so far today, while the market is open
ONE_EVERY_SEC = 20              # the stock on screen

# today's prices so far, filled by the background thread: {"fetched": ISO time, "quotes": {ticker: {...}}}
TODAY = {"fetched": None, "quotes": {}, "live": False}
ONE = {}                        # ticker -> (time fetched, quote) for the stock on screen
LOCK = threading.Lock()


def today_payload() -> dict:
    from src.ingest import live
    with LOCK:
        return {"state": live.market_state(), "fetched": TODAY["fetched"], "quotes": TODAY["quotes"],
                "updates": TODAY["live"], "built": INDEX.stat().st_mtime_ns if INDEX.exists() else 0}


def one_quote(ticker: str) -> dict:
    """Latest price for one stock, fetched at most once a minute. {} before the market opens."""
    from src.ingest import live
    with LOCK:
        hit = ONE.get(ticker)
        if hit and time.time() - hit[0] < ONE_EVERY_SEC:
            return hit[1]
    q = live.quotes([ticker], workers=1).get(ticker, {})
    with LOCK:
        ONE[ticker] = (time.time(), q)
        if q:
            TODAY["quotes"][ticker] = q
    return q


def keep_current(tickers: list) -> None:
    """Background loop. While the market is open: today's prices for every stock, every 10 minutes.
    After the close: download the finished day once, rebuild the page data, and the page reloads itself."""
    from src.ingest import live
    last_today, last_roll = 0.0, time.time()        # prices were just checked at start; look again in 30 minutes
    while True:
        try:
            st, now = live.market_state(), time.time()
            ny = pd.Timestamp.now(tz="America/New_York")
            session = st["open"] or (ny.dayofweek < 5 and (ny.hour, ny.minute) < (live.CLOSE_HOUR, live.CLOSE_MIN) and ny.hour >= 16)
            if session and now - last_today > TODAY_EVERY_MIN * 60:
                q = live.quotes(tickers)
                with LOCK:
                    TODAY["quotes"], TODAY["fetched"] = q, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                last_today = now
                if q:
                    print(f"  {ny:%H:%M} New York: today's prices so far for {len(q)} tickers", flush=True)
            have = live.status()
            if have and have["day"] < live.last_close_day() and now - last_roll > 30 * 60:
                last_roll = now
                if live.update():
                    try:
                        update_scores()
                    except Exception as e:
                        print(f"  Model ranking did not work ({e!r:.120}); the page keeps its ranges.", flush=True)
                    subprocess.run([sys.executable, "-m", "scripts.export_dashboard"], cwd=ROOT, check=True)
                    with LOCK:
                        TODAY["quotes"], TODAY["fetched"] = {}, None
                        ONE.clear()
                    print("  Forecasts rolled forward to the new closing day.", flush=True)
        except Exception as e:
            print(f"  Background update did not work ({e!r:.120}); trying again later.", flush=True)
        time.sleep(60)


def authorized(header: str | None, password: str) -> bool:
    """True if no password is set, or the request's Basic login carries it (any user name)."""
    if not password:
        return True
    if not header or not header.startswith("Basic "):
        return False
    try:
        given = base64.b64decode(header[6:], validate=True).decode("utf-8").partition(":")[2]
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8"))


class Handler(http.server.SimpleHTTPRequestHandler):
    tickers = frozenset()
    password = ""

    def locked(self) -> bool:
        """Ask for the password and return True if this request does not carry it."""
        if authorized(self.headers.get("Authorization"), self.password):
            return False
        body = b"Login required."
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="LatticeIQ", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_HEAD(self):
        if not self.locked():
            super().do_HEAD()

    def do_GET(self):
        if self.locked():
            return
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/today":
            return self.send_json(today_payload())
        if url.path == "/api/analysts":
            t = urllib.parse.parse_qs(url.query).get("t", [""])[0].upper()
            if t not in self.tickers:
                return self.send_json({"error": "unknown ticker"}, 404)
            from src.ingest import analysts
            got = dict(analysts.get(t))
            got.pop("history", None)                           # kept on disk for later study, not needed by the page
            return self.send_json(got)
        if url.path == "/api/quote":
            t = urllib.parse.parse_qs(url.query).get("t", [""])[0].upper()
            if t not in self.tickers:
                return self.send_json({"error": "unknown ticker"}, 404)
            try:
                return self.send_json({"t": t, "quote": one_quote(t)})
            except Exception as e:
                return self.send_json({"t": t, "quote": {}, "error": f"{e!r:.120}"})
        return super().do_GET()

    def send_json(self, obj, code: int = 200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")        # always show the latest export
        super().end_headers()

    def log_message(self, *args):                             # keep the terminal quiet
        pass


def update_prices() -> bool:
    from src.ingest import live
    return live.update()


def update_scores() -> bool:
    """Rank today's stocks with the saved model, if there is one and its scores are older than the prices."""
    from src.models import production
    prices = ROOT / "data" / "live" / "prices.parquet"
    if not production.MODEL.exists() or not prices.exists():
        return False
    if production.RANK.exists() and production.RANK.stat().st_mtime >= prices.stat().st_mtime:
        # ranked since the last price update; rank again only if the model itself has been retrained since
        meta, _ = production.load()
        ranked_with = json.loads(production.RANK.read_text(encoding="utf-8")).get("model", {}).get("created")
        if meta and meta.get("created") == ranked_with:
            return False
    print("Ranking today's stocks with the saved model (under a minute)", flush=True)
    # in its own process: building the features needs about 1.8 GB for a minute, which is then handed back
    # (in this long-running process it would stay taken, and a small server would have to swap)
    return subprocess.run([sys.executable, "-m", "scripts.score_today"], cwd=ROOT, check=True).returncode == 0


def update_filings() -> bool:
    """Fetch the 8-K filings made since the saved table ends (earnings reports, important events).
    True if the file was refreshed. Skipped if it was refreshed in the last 20 hours."""
    new, old = path("clean") / "sec_events_latest.parquet", path("clean") / "sec_events.parquet"
    if not old.exists():
        return False
    codes = ROOT / "data" / "live" / "companies.json"                     # industry codes for the sector filter
    try:
        have_codes = len(json.loads(codes.read_text(encoding="utf-8"))) >= 450 if codes.exists() else False
    except ValueError:
        have_codes = False
    if new.exists() and have_codes and time.time() - new.stat().st_mtime < FILINGS_EVERY_HOURS * 3600:
        return False
    from src.ingest import sec_events
    since = pd.read_parquet(old, columns=["filed"])["filed"].max()
    days = int(min(90, max(7, (pd.Timestamp.now().normalize() - since).days + 3)))
    u = pd.read_csv(ROOT / "config" / "universe.csv")
    tickers = sorted(u.loc[(u["type"] == "stock") & u["end"].isna(), "ticker"].unique())
    print(f"Checking the SEC for filings in the last {days} days ({len(tickers)} companies, a few minutes)", flush=True)
    before = new.stat().st_mtime if new.exists() else 0
    sec_events.latest(tickers, days)
    return new.exists() and new.stat().st_mtime > before


def needs_build() -> bool:
    """True if the dashboard data is missing or older than the prices, filings or code it is built from."""
    if not INDEX.exists():
        return True
    built = INDEX.stat().st_mtime
    inputs = [ROOT / "data" / "live" / "prices.parquet", ROOT / "data" / "live" / "rank.json", ROOT / "data" / "live" / "companies.json", path("clean") / "sec_events_latest.parquet",
              ROOT / "scripts" / "export_dashboard.py", ROOT / "src" / "models" / "vol_ranges.py"]
    return any(f.exists() and f.stat().st_mtime > built for f in inputs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true", help="do not download anything; use the data on disk")
    ap.add_argument("--refresh", action="store_true", help="rebuild app/data even if nothing changed")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    Handler.password = os.getenv("LATTICEIQ_PASSWORD", "")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not Handler.password:
        raise SystemExit("Refusing to listen on other machines without a login. Set LATTICEIQ_PASSWORD in .env first.")

    if not a.no_update:
        for name, step in (("Price", update_prices), ("Filings", update_filings), ("Model ranking", update_scores)):
            try:
                step()
            except KeyboardInterrupt:
                print(f"\n{name} update skipped by you.")
            except Exception as e:                            # offline, Yahoo or the SEC not answering: still open
                print(f"{name} update did not work ({e!r:.160}). Opening with the data already on disk.")
    if a.refresh or needs_build():
        print("Building dashboard data...", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.export_dashboard"], cwd=ROOT, check=True)

    if not a.no_update:
        try:
            from src.ingest import live
            tickers = live.wanted()
            Handler.tickers = frozenset(tickers)
            TODAY["live"] = True
            threading.Thread(target=keep_current, args=(tickers,), daemon=True).start()
        except Exception as e:
            print(f"Today's prices are off ({e!r:.120}).")
    url = f"http://localhost:{a.port}/"
    server = http.server.ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=str(APP)))
    print(f"\nLatticeIQ dashboard is running: {url}" + ("  (login required)" if Handler.password else "") +
          "\nKeep this window open while you use it. Press Ctrl+C here to stop.", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
