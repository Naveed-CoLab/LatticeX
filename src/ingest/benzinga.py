"""Download Benzinga news for the universe's stocks.

Step 1, check how far back your plan goes (one quick request per year):
    python -m src.ingest.benzinga --probe

Step 2, download history (month by month, 10 tickers per request, only tickers that were in
the S&P 500 that month). Safe to stop with Ctrl+C and re-run: finished months are skipped.
    python -m src.ingest.benzinga --start 2010-01 --end 2014-12

Step 3, merge everything into one file:
    python -m src.ingest.benzinga --combine

Step 4, see how much news there is per year (decides which years can test news features):
    python -m src.ingest.benzinga --report

Output: data/raw/news/<YYYY-MM>/b###.parquet   raw downloads
        data/clean/news_articles.parquet       id, created_utc, title, teaser, tickers, channels, url

We keep the headline and teaser (first paragraph), not the full article body: that's what
FinBERT reads, and it keeps the files small.
"""
import argparse
import email.utils
import html
import re
import time

import pandas as pd
import requests

from src.config import path, secret
from src.universe import load_universe

URL = "https://api.benzinga.com/api/v2/news"
PAGE_SIZE = 100      # what we ask for; the plan may return fewer per page (seen: 25)
MAX_PAGES = 400
TAG_RE = re.compile(r"<[^>]+>")


def _clean(text) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", str(text or "")))).strip()


def _fetch(params: dict) -> list:
    params = {**params, "token": secret("BENZINGA_API_KEY")}
    for attempt in range(8):
        try:
            r = requests.get(URL, params=params, headers={"accept": "application/json"}, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"  Benzinga rate limit, waiting {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"Benzinga refused the request ({r.status_code}): {r.text[:300]}\n"
                             "Check BENZINGA_API_KEY in .env and what your plan includes.")
        r.raise_for_status()
        data = r.json() if r.text.strip() else []
        if isinstance(data, dict):
            raise RuntimeError(f"Unexpected Benzinga answer: {str(data)[:300]}")
        return data
    raise RuntimeError("Benzinga: too many failed attempts")


def parse(items: list) -> pd.DataFrame:
    rows = []
    for it in items:
        try:
            created = email.utils.parsedate_to_datetime(it["created"])
        except (TypeError, ValueError, KeyError):
            created = pd.to_datetime(it.get("created"), utc=True, errors="coerce")
        rows.append({
            "id": int(it["id"]),
            "created_utc": pd.Timestamp(created).tz_convert("UTC") if created is not None else pd.NaT,
            "title": _clean(it.get("title")),
            "teaser": _clean(it.get("teaser"))[:600],
            "tickers": ",".join(sorted({s.get("name", "").upper().strip() for s in it.get("stocks") or []} - {""})),
            "channels": "|".join(c.get("name", "") for c in it.get("channels") or []),
            "url": it.get("url", ""),
        })
    df = pd.DataFrame(rows, columns=["id", "created_utc", "title", "teaser", "tickers", "channels", "url"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df


def fetch_window(tickers: list, date_from: str, date_to: str) -> pd.DataFrame:
    """All articles for these tickers in the window. Keeps paging until a page comes back
    empty or shorter than the largest page seen, so it works whatever page size the plan allows."""
    frames, truncated, cap, seen = [], False, 0, set()
    for page in range(MAX_PAGES):
        items = _fetch({"tickers": ",".join(tickers), "dateFrom": date_from, "dateTo": date_to,
                        "pageSize": PAGE_SIZE, "page": page, "displayOutput": "abstract"})
        new = [it for it in items if it.get("id") not in seen]
        seen.update(it.get("id") for it in items)
        frames.append(parse(new))
        cap = max(cap, len(items))
        if not items or len(items) < cap or not new:   # last page (or API repeating itself)
            break
        time.sleep(0.15)
    else:
        truncated = True
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    if truncated:
        print(f"  warning: hit {MAX_PAGES} pages for {date_from}..{date_to}; some articles may be missing")
    return df


def probe() -> None:
    """How far back does the plan go? Asks for AAPL news in January of each year."""
    print("Checking Benzinga history depth with AAPL (January of each year)...")
    first = None
    for y in range(2000, pd.Timestamp.today().year + 1):
        n = len(_fetch({"tickers": "AAPL", "dateFrom": f"{y}-01-01", "dateTo": f"{y}-01-31",
                        "pageSize": PAGE_SIZE, "page": 0, "displayOutput": "headline"}))
        print(f"  {y}: {n}{'+' if n == PAGE_SIZE else ''} articles")
        if n and first is None:
            first = y
        time.sleep(0.3)
    print(f"\nEarliest year with news: {first}. Use --start {first}-01 for the download." if first
          else "\nNo history returned. Your plan may only include recent news; ask Benzinga.")


def diagnose(month: str = "2019-03") -> None:
    """Why so few articles? Compare single-ticker vs batched requests for one month,
    and count all news (no ticker filter) for one day."""
    m = pd.Period(month, "M")
    d0, d1 = m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d")
    names = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "BA", "INTC", "KO", "WMT", "GE"]
    print(f"Month {month}: articles per ticker, asked ONE ticker at a time (all pages):")
    single = {}
    for t in names:
        single[t] = fetch_window([t], d0, d1)
        print(f"  {t}: {len(single[t])}")
    batch = fetch_window(names, d0, d1)
    tk = batch["tickers"].str.split(",").explode()
    print(f"\nSame 10 tickers in ONE request: {len(batch)} articles; per ticker: "
          f"{ {t: int((tk == t).sum()) for t in names} }")
    print(f"Sum of single-ticker counts: {sum(len(v) for v in single.values())}")
    day = (m.start_time + pd.offsets.BDay(5)).strftime("%Y-%m-%d")
    allnews = []
    for page in range(40):
        items = _fetch({"dateFrom": day, "dateTo": day, "pageSize": PAGE_SIZE, "page": page,
                        "displayOutput": "headline"})
        allnews += items
        if not items:
            break
    print(f"\nAll Benzinga news on {day} (no ticker filter, up to 40 pages): {len(allnews)} articles")
    print(f"Page size returned: {max((len(_fetch({'dateFrom': day, 'dateTo': day, 'pageSize': PAGE_SIZE, 'page': 0})), ), default=0)}")


def active_tickers(universe: pd.DataFrame, month: pd.Period) -> list:
    stocks = universe[universe["type"] == "stock"]
    s = stocks["start"].fillna(pd.Timestamp("1900-01-01"))
    e = stocks["end"].fillna(pd.Timestamp("2100-01-01"))
    on = (s <= month.end_time) & (e >= month.start_time)
    return sorted(stocks.loc[on, "ticker"].unique())


def download(start: str, end: str, batch: int = 10) -> None:
    universe = load_universe()
    root = path("raw") / "news"
    months = pd.period_range(start, end, freq="M")
    t0 = time.time()
    for mi, m in enumerate(months, 1):
        tickers = active_tickers(universe, m)
        mdir = root / str(m)
        mdir.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for b in range(0, len(tickers), batch):
            f = mdir / f"b{b // batch:03d}.parquet"
            if f.exists():
                continue
            df = fetch_window(tickers[b:b + batch], m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d"))
            df.to_parquet(f, index=False)   # written even if empty, so it's marked done
            n_new += len(df)
        print(f"[{mi}/{len(months)}] {m}: {len(tickers)} tickers, {n_new:,} new articles  [{time.time() - t0:.0f}s]")


def combine() -> pd.DataFrame:
    files = sorted((path("raw") / "news").glob("*/*.parquet"))
    if not files:
        raise SystemExit("No news downloaded yet.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("id").dropna(subset=["created_utc"]).sort_values("created_utc")
    out = path("clean") / "news_articles.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} articles, {df['created_utc'].min():%Y-%m-%d} to {df['created_utc'].max():%Y-%m-%d} -> {out}")
    return df


def report() -> None:
    f = path("clean") / "news_articles.parquet"
    if not f.exists():
        raise SystemExit("No news yet.")
    a = pd.read_parquet(f, columns=["id", "created_utc", "tickers"])
    a["year"] = a["created_utc"].dt.year
    t = a.assign(tk=a["tickers"].str.split(",")).explode("tk")
    try:
        stocks = set(load_universe().query("type == 'stock'")["ticker"])
        t = t[t["tk"].isin(stocks)]
    except Exception:
        pass
    per = t.groupby("year").agg(articles=("id", "nunique"), stocks_with_news=("tk", "nunique"))
    per["articles_per_stock"] = (per["articles"] / per["stocks_with_news"]).round(1)
    top = t["tk"].value_counts()
    print(per.to_string())
    print(f"\nTotal {a['id'].nunique():,} articles. Top 5 stocks: {top.head(5).to_dict()}")
    print("Rule of thumb: use news from the first year with 300+ stocks covered and 20+ articles per stock.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--start", help="first month, e.g. 2012-01")
    p.add_argument("--end", default=(pd.Timestamp.today() - pd.offsets.MonthBegin(1)).strftime("%Y-%m"))
    p.add_argument("--combine", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--diagnose", metavar="YYYY-MM", help="compare single vs batched requests for a month")
    a = p.parse_args()
    if a.diagnose:
        diagnose(a.diagnose)
    elif a.probe:
        probe()
    elif a.combine:
        combine()
    elif a.report:
        report()
    elif a.start:
        try:
            download(a.start, a.end)
        except KeyboardInterrupt:
            print("\nStopped. Finished months are saved; run the same command to continue.")
        combine()
    else:
        p.print_help()
