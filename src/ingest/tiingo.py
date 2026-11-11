"""Download daily prices from Tiingo for every ticker in config/universe.csv.

Run:  python -m src.ingest.tiingo                 all tickers; waits out rate limits and resumes
      python -m src.ingest.tiingo --no-wait       stop at the first rate limit (run again later)
      python -m src.ingest.tiingo --combine-only  just rebuild data/clean/prices.parquet
      python -m src.ingest.tiingo AAPL SPY        just these tickers

Safe to stop (Ctrl+C) and re-run at any time: tickers already downloaded are skipped, and the
combined file is rebuilt from whatever has been downloaded so far.

Output: data/raw/prices/<TICKER>.parquet   one per ticker
        data/raw/prices/_not_found.txt     tickers Tiingo doesn't have (not retried)
        data/clean/prices.parquet          all tickers combined, long format
"""
import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import ROOT, load_settings, path, secret

URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


class RateLimited(Exception):
    pass


class TickerError(Exception):
    pass


LIMIT_WORDS = ("allocation", "limit", "upgrade", "exceeded", "too many")


def _get(ticker: str, start: str) -> list:
    for attempt in range(5):
        try:
            r = requests.get(URL.format(ticker=ticker),
                             params={"startDate": start, "token": secret("TIINGO_API_KEY"), "format": "json"},
                             timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            raise RateLimited(r.text[:300])
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):  # Tiingo sometimes answers 200 with an error message instead of prices
            msg = str(data.get("detail", data))
            if any(w in msg.lower() for w in LIMIT_WORDS):
                raise RateLimited(msg[:300])
            raise TickerError(msg[:300])
        return data
    raise RuntimeError(f"{ticker}: failed after 5 attempts (network or server error)")


def download(tickers: list[str], wait: bool = True, wait_minutes: int = 61) -> None:
    s = load_settings()
    out_dir = path("raw") / "prices"
    out_dir.mkdir(exist_ok=True)
    nf_file = out_dir / "_not_found.txt"
    not_found = set(nf_file.read_text().split()) if nf_file.exists() else set()
    todo = [t for t in tickers if not (out_dir / f"{t}.parquet").exists() and t not in not_found]
    print(f"{len(tickers)} tickers in universe, {len(todo)} still to download")

    i, skipped = 0, []
    while i < len(todo):
        t = todo[i]
        try:
            rows = _get(t, s["start_date"])
        except RateLimited as e:
            print(f"\nTiingo rate limit reached after {i}/{len(todo)}. Message: {e}")
            if "month" in str(e).lower():
                print("This is the MONTHLY limit: waiting an hour won't help. Stopping now.\n"
                      "Options: upgrade the Tiingo plan, or run this command again next month.")
                return
            if not wait:
                print("Stopping (--no-wait). Run the same command again later to continue.")
                return
            until = datetime.now().timestamp() + wait_minutes * 60
            print(f"Waiting {wait_minutes} min, then continuing automatically "
                  f"(resume at {datetime.fromtimestamp(until):%H:%M}). Ctrl+C is safe.")
            time.sleep(wait_minutes * 60)
            continue  # retry the same ticker
        except (TickerError, RuntimeError, ValueError) as e:
            print(f"[{i + 1}/{len(todo)}] {t}: skipped ({e})")
            skipped.append(t)
            i += 1
            continue
        if not rows:
            not_found.add(t)
            nf_file.write_text("\n".join(sorted(not_found)))
            print(f"[{i + 1}/{len(todo)}] {t}: not on Tiingo")
        else:
            df = pd.DataFrame(rows)
            df["ticker"], df["source"] = t, "tiingo"
            df["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            df.to_parquet(out_dir / f"{t}.parquet", index=False)
            print(f"[{i + 1}/{len(todo)}] {t}: {len(df)} rows, {df['date'].min()[:10]} to {df['date'].max()[:10]}")
        i += 1
        time.sleep(0.3)
    if skipped:
        print(f"Skipped because of errors ({len(skipped)}): {skipped}  <- run the command again later to retry them")
    if not_found:
        print(f"Not found on Tiingo ({len(not_found)}), listed in {nf_file}")


def combine() -> pd.DataFrame:
    """Merge per-ticker files into one clean long table with the column names the pipeline expects."""
    files = sorted((path("raw") / "prices").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.rename(columns={"adjClose": "adj_close", "adjVolume": "adj_volume",
                            "divCash": "dividend", "splitFactor": "split_factor"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[["date", "ticker", "open", "high", "low", "close", "volume", "adj_close",
             "adj_volume", "dividend", "split_factor"]]
    df = df.dropna(subset=["adj_close"])
    df = df[df["adj_close"] > 0]
    df = df.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    out = path("clean") / "prices.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows for {df['ticker'].nunique()} tickers -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*")
    p.add_argument("--no-wait", action="store_true", help="stop at the first rate limit instead of waiting")
    p.add_argument("--wait-minutes", type=int, default=61)
    p.add_argument("--combine-only", action="store_true")
    a = p.parse_args()
    if not a.combine_only:
        tickers = a.tickers or pd.read_csv(ROOT / "config" / "universe.csv")["ticker"].dropna().unique().tolist()
        try:
            download(tickers, wait=not a.no_wait, wait_minutes=a.wait_minutes)
        except KeyboardInterrupt:
            print("\nStopped by you. Saving what was downloaded so far...")
        except Exception as e:
            print(f"\nUnexpected error: {e!r}. Saving what was downloaded so far; run again to continue.")
    combine()
