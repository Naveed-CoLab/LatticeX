"""Alpaca news (Benzinga and other publishers): probe, then download the history.

    python -m src.ingest.alpaca_news --probe                   is there enough news? (saves nothing)
    python -m src.ingest.alpaca_news --start 2015-01           download month by month (about 1-2 hours; safe to stop
                                                               with Ctrl+C and run again: finished months are skipped)
    python -m src.ingest.alpaca_news --combine                 merge the months into data/clean/news_articles.parquet

Alpaca has two news interfaces. This uses the historical one (GET /v1beta1/news with start and end), which
covers both needs: years of history for testing, and "everything since yesterday" for daily use. The other
one is a live stream that pushes each article the second it is published; a model that works on daily
closing prices does not need it.

Keys: create them in your Alpaca account and add two lines to the .env file in the project folder
    ALPACA_API_KEY=...        (the "Key ID")
    ALPACA_API_SECRET=...     (the "Secret Key")
Never paste them into a chat or commit them.

The probe reads one ordinary trading day per year and the last week, and prints how many articles there
are and how many S&P 500 stocks they cover. News is worth testing only if most stocks get regular coverage
in most years. Nothing is saved.

The download keeps the headline and summary (not the article body) of every article that names at least one
stock of our universe: data/raw/news_alpaca/<YYYY-MM>.parquet. --combine writes the same columns the rest of
the pipeline already reads (id, created_utc, title, teaser, tickers, channels, url) plus the publisher, and
keeps Benzinga only: another publisher (fiscal.ai) appears in late 2026 and triples the article count, which
would make "how much is this stock in the news" mean something different at the end than in the years the
model learns from. The earlier thin file from the Benzinga plan is kept once as news_articles_benzinga_plan.parquet.

Probe result, 16 Nov 2026: 435 to 1,144 articles on the sampled days of 2015-2026, naming 65 to 168 of about
500 index members (13-33%); the last week has 2,800-3,000 a day because of fiscal.ai.
News text belongs to its publishers: it is used here to compute sentiment numbers and is never committed.
"""
import argparse
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src.config import ROOT, path, secret

URL = "https://data.alpaca.markets/v1beta1/news"
PAGE = 50                      # the most the API returns per page
MAX_PAGES = 60                 # per probed day: up to 3,000 articles
SYMBOLS_PER_REQUEST = 200     # ask only for articles naming our stocks, in groups (keeps the address short)
WORKERS = 3                    # months fetched at the same time: about 3 requests a second, under the 200-a-minute limit
KEEP_SOURCES = {"benzinga", ""}   # "" = the early years, before Alpaca recorded the publisher
TAG_RE = re.compile(r"<[^>]+>")
LIMIT = {"remaining": None, "reset": 0}


def _clean(text) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", str(text or "")))).strip()


def _get(params: dict) -> dict:
    headers = {"APCA-API-KEY-ID": secret("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": secret("ALPACA_API_SECRET"),
               "accept": "application/json"}
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, headers=headers, timeout=60)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429:                                   # per-minute limit: wait until it resets
            reset = int(r.headers.get("X-RateLimit-Reset", "0") or 0)
            time.sleep(min(65, max(2, reset - time.time() + 1)))
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"Alpaca refused the keys (HTTP {r.status_code}). Check ALPACA_API_KEY and ALPACA_API_SECRET in .env; "
                             f"use the keys of the same account type (paper or live) you created them in.")
        if r.status_code in (400, 414) and "symbols" in params:
            raise SystemExit(f"Alpaca refused the request (HTTP {r.status_code}: {r.text[:200]}). Asking for fewer stocks per "
                             f"request usually fixes it: add --group 50 to the same command. Finished months are kept.")
        if r.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        try:                                                       # stay under the per-minute limit instead of hitting it
            if int(r.headers.get("X-RateLimit-Remaining", "9")) <= 1:
                time.sleep(min(65, max(1, int(r.headers.get("X-RateLimit-Reset", "0")) - time.time() + 1)))
        except ValueError:
            pass
        return r.json()
    raise SystemExit("Alpaca did not answer after several tries. Run the same command again: finished months are kept.")


def day_articles(day: pd.Timestamp) -> tuple:
    """Every article created on one calendar day (UTC): -> (list of articles, True if the page cap was hit)."""
    params = {"start": f"{day:%Y-%m-%d}T00:00:00Z", "end": f"{day:%Y-%m-%d}T23:59:59Z", "sort": "asc", "limit": PAGE}
    out, token = [], None
    for _ in range(MAX_PAGES):
        j = _get({**params, **({"page_token": token} if token else {})})
        out += j.get("news", [])
        token = j.get("next_page_token")
        if not token:
            return out, False
    return out, True


def probe() -> None:
    u = pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["start", "end"])
    u = u[u["type"] == "stock"]
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    days = [pd.Timestamp(f"{y}-03-01") + pd.offsets.Week(weekday=2) + pd.offsets.Week(1) for y in range(2015, today.year + 1)]
    days = [d for d in days if d < today] + list(pd.bdate_range(end=today - pd.to_timedelta(1, unit="D"), periods=3))
    print("One trading day per year, then the last three trading days. 'members' = stocks in the S&P 500 on that day.\n")
    print(f"{'day':<12}{'articles':>9}{'about members':>15}{'members covered':>17}{'  publishers'}")
    rows = []
    for d in days:
        arts, capped = day_articles(d)
        members = set(u.loc[(u["start"].fillna(pd.Timestamp("1900-01-01")) <= d) & (u["end"].isna() | (u["end"] > d)), "ticker"])
        hit = [a for a in arts if members & {s.replace(".", "-") for s in a.get("symbols", [])}]
        covered = {s.replace(".", "-") for a in hit for s in a.get("symbols", [])} & members
        src = pd.Series([a.get("source", "?") for a in arts]).value_counts().head(3)
        rows.append((d, len(arts), len(hit), len(covered), len(members)))
        print(f"{d:%Y-%m-%d}  {len(arts):>8}{'+' if capped else ' '}{len(hit):>14}{len(covered):>10} of {len(members):<4}  "
              + ", ".join(f"{k} {v}" for k, v in src.items()), flush=True)
    r = pd.DataFrame(rows, columns=["day", "articles", "about", "covered", "members"])
    share = (r["covered"] / r["members"].clip(lower=1))
    good = r.loc[share >= 0.15, "day"]
    print("\nReading it: on an ordinary day a stock has news perhaps once or twice a week, so about 15-30% of members "
          "covered on one day is healthy;\nunder 5% means the feed is too thin to compare stocks (what the Benzinga plan gave: about 2 articles a day).")
    if len(good):
        print(f"Coverage of 15% or more starts in {good.min():%Y} ({len(good)} of {len(r)} probed days). "
              f"If that holds, the next step is the full download from {good.min():%Y}.")
    else:
        print("No probed day reached 15% coverage: not enough for a news feature. Paste this table back before going further.")


def parse(items: list, keep: set) -> pd.DataFrame:
    """Alpaca articles -> our columns, only those that name a ticker in `keep`."""
    rows = []
    for it in items:
        tickers = sorted({str(x).upper().replace(".", "-") for x in it.get("symbols") or []} & keep)
        if tickers:
            rows.append({"id": int(it["id"]), "created_utc": it.get("created_at"), "title": _clean(it.get("headline")),
                         "teaser": _clean(it.get("summary"))[:600], "tickers": ",".join(tickers), "channels": "",
                         "url": it.get("url") or "", "source": str(it.get("source") or "").lower()})
    df = pd.DataFrame(rows, columns=["id", "created_utc", "title", "teaser", "tickers", "channels", "url", "source"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    return df


def _month(m, groups: list, keep: set, out) -> tuple:
    """Every article in month m that names at least one of our stocks -> data/raw/news_alpaca/<m>.parquet."""
    base = {"start": f"{m.start_time:%Y-%m-%d}T00:00:00Z", "end": f"{m.end_time:%Y-%m-%d}T23:59:59Z",
            "sort": "asc", "limit": PAGE}
    parts, pages = [], 0
    for g in groups:
        token = None
        while True:
            j = _get({**base, "symbols": g, **({"page_token": token} if token else {})})
            items = j.get("news", [])
            pages += 1
            parts.append(parse(items, keep))
            token = j.get("next_page_token")
            if not token or not items:
                break
    df = pd.concat(parts, ignore_index=True).drop_duplicates("id") if parts else parse([], keep)
    df.to_parquet(out / f"{m}.parquet", index=False)
    return m, len(df), pages


def download(start: str, end: str | None = None, workers: int = WORKERS, group: int = SYMBOLS_PER_REQUEST) -> None:
    """Months already saved are skipped (the current month is always fetched again). Only articles that name
    a stock of our list are asked for: the result is the same as fetching everything and filtering, with
    about a third of the requests. Earlier months saved the other way are therefore kept as they are."""
    u = pd.read_csv(ROOT / "config" / "universe.csv")
    keep = set(u.loc[u["type"] == "stock", "ticker"].astype(str).str.upper())
    alpaca = sorted({t.replace("-", ".") for t in keep})                         # Alpaca writes BRK.B, we write BRK-B
    groups = [",".join(alpaca[i:i + group]) for i in range(0, len(alpaca), group)]
    out = path("raw") / "news_alpaca"
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    months = pd.period_range(start, end or now.strftime("%Y-%m"), freq="M")
    todo = [m for m in months if not ((out / f"{m}.parquet").exists() and m.end_time < now - pd.to_timedelta(2, unit="D"))]
    print(f"{len(months)} months, {len(months) - len(todo)} already saved, {len(todo)} to fetch, {workers} at a time. "
          f"Each finished month is saved; stop with Ctrl+C and run the same command to continue.", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = [pool.submit(_month, m, groups, keep, out) for m in todo]
        try:
            for fut in as_completed(jobs):
                m, n, pages = fut.result()
                done += 1
                left = (time.time() - t0) / done * (len(todo) - done)
                print(f"[{done}/{len(todo)}] {m}: {n:,} articles about our stocks, {pages} requests  "
                      f"[{(time.time() - t0) / 60:.0f} min, about {left / 60:.0f} min left]", flush=True)
        except KeyboardInterrupt:
            for j in jobs:
                j.cancel()
            raise
    print("Download finished. Next: python -m src.ingest.alpaca_news --combine")


def combine() -> pd.DataFrame:
    files = sorted((path("raw") / "news_alpaca").glob("*.parquet"))
    if not files:
        raise SystemExit("No Alpaca news downloaded yet. Run: python -m src.ingest.alpaca_news --start 2015-01")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("id").dropna(subset=["created_utc"]).sort_values("created_utc")
    by = df["source"].replace("", "(not recorded)").value_counts()
    kept = df[df["source"].isin(KEEP_SOURCES)]
    out = path("clean") / "news_articles.parquet"
    old = path("clean") / "news_articles_benzinga_plan.parquet"
    if out.exists() and not old.exists() and "source" not in pd.read_parquet(out).columns:
        out.replace(old)                                           # the thin file from the Benzinga plan, kept once
        print(f"The earlier news file was kept as {old.name}")
    kept.to_parquet(out, index=False)
    per = kept.assign(year=kept["created_utc"].dt.year, tk=kept["tickers"].str.split(",")).explode("tk")
    table = per.groupby("year").agg(articles=("id", "nunique"), stocks=("tk", "nunique"))
    table["per_stock"] = (table["articles"] / table["stocks"]).round(0).astype(int)
    print("Publishers found: " + ", ".join(f"{k} {v:,}" for k, v in by.items()))
    print(f"Kept {len(kept):,} Benzinga articles, {kept['created_utc'].min():%Y-%m-%d} to {kept['created_utc'].max():%Y-%m-%d} -> {out}\n")
    print(table.to_string())
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="count articles on sample days (saves nothing)")
    ap.add_argument("--start", help="first month to download, e.g. 2015-01")
    ap.add_argument("--end", help="last month (default: this month)")
    ap.add_argument("--workers", type=int, default=WORKERS, help="months fetched at the same time (default 3)")
    ap.add_argument("--group", type=int, default=SYMBOLS_PER_REQUEST, help="stocks named per request (default 200)")
    ap.add_argument("--combine", action="store_true", help="merge downloaded months into data/clean/news_articles.parquet")
    a = ap.parse_args()
    if a.probe:
        probe()
    elif a.start:
        try:
            download(a.start, a.end, a.workers, a.group)
        except KeyboardInterrupt:
            print("\nStopped by you. Finished months are saved (months being fetched finish first); run the same command to continue.")
    elif a.combine:
        combine()
    else:
        print(__doc__)
