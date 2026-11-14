"""Official company news from the SEC: every 8-K ("current report") a company has filed, with what it was about.

    python -m src.ingest.sec_events                 full history for all stocks in config/universe.csv (resumable)
    python -m src.ingest.sec_events AAPL NVDA       just these
    python -m src.ingest.sec_events --latest 7      LIVE: what was filed in the last 7 days (re-checks every company)
    python -m src.ingest.sec_events --combine       rebuild data/clean/sec_events.parquet

Why this source: companies MUST file an 8-K within 4 business days of any material event (earnings, a CEO
leaving, a write-down, a restated report, a bankruptcy...). It is free, official, time-stamped to the second,
allowed for commercial use, and it has 20+ years of history, so the same events can be both
  - back-tested as model features (history), and
  - shown as a live alert feed in the app (--latest).
Each 8-K lists numbered "items" that say what happened; ITEMS below turns them into plain words + a severity.

Source: https://data.sec.gov/submissions/CIK##########.json (needs SEC_USER_AGENT in .env; max 10 requests/s).
Output: data/raw/sec_events/<TICKER>.parquet and data/clean/sec_events.parquet with columns
  ticker, cik, form, filed (date), accepted (UTC time the SEC published it), items, accession,
  + one true/false column per event kind (ev_earnings, ev_restatement, ...) and severity (0-3).
A filing accepted after 4 pm New York only becomes usable on the NEXT trading day (handled in the features).
"""
import argparse
import time

import pandas as pd

from src.config import ROOT, path
from src.ingest.edgar import PREDECESSOR_CIKS, _get, cik_map, predecessor_ok, reused_sec_tickers

SUB_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
OLDER_URL = "https://data.sec.gov/submissions/{name}"
FORMS = {"8-K", "8-K/A"}

# 8-K item -> (event kind, plain words, severity 0 = routine .. 3 = severe)
ITEMS = {
    "1.01": ("agreement", "signed a material agreement", 1),
    "1.02": ("agreement_ended", "a material agreement was terminated", 2),
    "1.03": ("bankruptcy", "bankruptcy or receivership", 3),
    "1.05": ("cyber_incident", "material cybersecurity incident", 2),
    "2.01": ("acquisition", "completed an acquisition or sale of assets", 1),
    "2.02": ("earnings", "reported financial results (earnings)", 1),
    "2.04": ("debt_trigger", "an event sped up a debt obligation", 3),
    "2.05": ("restructuring", "restructuring / exit costs (layoffs, closures)", 2),
    "2.06": ("impairment", "material write-down of assets", 2),
    "3.01": ("delisting_notice", "notice of delisting or listing-rule failure", 3),
    "3.02": ("share_sale", "sold unregistered shares (dilution)", 1),
    "4.01": ("auditor_change", "changed its auditor", 2),
    "4.02": ("restatement", "earlier financial reports can no longer be relied on", 3),
    "5.02": ("management_change", "director or senior officer left or was appointed", 1),
    "5.07": ("shareholder_vote", "results of a shareholder vote", 0),
    "7.01": ("disclosure", "public disclosure (Reg FD)", 0),
    "8.01": ("other", "other event the company chose to report", 0),
}
KINDS = sorted({k for k, _, _ in ITEMS.values()})


def parse_block(block: dict, ticker: str, cik: int) -> pd.DataFrame:
    """block = the SEC's column-arrays dict (filings.recent, or an older submissions file)."""
    n = len(block.get("form", []))
    rows = []
    for i in range(n):
        form = block["form"][i]
        if form not in FORMS:
            continue
        items = [x.strip() for x in str(block.get("items", [""] * n)[i] or "").split(",") if x.strip()]
        kinds = {ITEMS[x][0] for x in items if x in ITEMS}
        rows.append({"ticker": ticker, "cik": cik, "form": form, "filed": block["filingDate"][i],
                     "accepted": block.get("acceptanceDateTime", [None] * n)[i], "items": ",".join(items),
                     "accession": block["accessionNumber"][i],
                     "severity": max([ITEMS[x][2] for x in items if x in ITEMS], default=0),
                     **{f"ev_{k}": k in kinds for k in KINDS}})
    df = pd.DataFrame(rows, columns=["ticker", "cik", "form", "filed", "accepted", "items", "accession", "severity"]
                      + [f"ev_{k}" for k in KINDS])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce", utc=True, format="ISO8601")
    return df


def describe(items: str) -> str:
    """'2.02,5.02' -> plain words, most serious first."""
    known = sorted([ITEMS[x] for x in items.split(",") if x in ITEMS], key=lambda t: -t[2])
    return "; ".join(t[1] for t in known) or "routine filing"


def fetch(ticker: str, cik: int, full: bool = True) -> pd.DataFrame:
    parts = []
    ciks = [cik] + (PREDECESSOR_CIKS.get(ticker, []) if full else [])
    for n, c in enumerate(ciks):
        r = _get(SUB_URL.format(cik=c))
        if r.status_code == 404:
            continue
        r.raise_for_status()
        j = r.json()
        if n:                                          # earlier registrant of the same business
            if not predecessor_ok(c, j.get("name", "?")):
                continue
            print(f"      + earlier registrant CIK {c}: {j.get('name', '?')}")
        parts.append(parse_block(j.get("filings", {}).get("recent", {}), ticker, c))
        if full:                                       # older filings live in extra files
            for f in j.get("filings", {}).get("files", []):
                time.sleep(0.12)
                rr = _get(OLDER_URL.format(name=f["name"]))
                if rr.status_code == 200:
                    parts.append(parse_block(rr.json(), ticker, c))
        time.sleep(0.12)
    if not parts:
        return pd.DataFrame()
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else parts[0]
    return df.drop_duplicates("accession").sort_values("filed")


def download(tickers: list, force: bool = False) -> None:
    out = path("raw") / "sec_events"
    out.mkdir(parents=True, exist_ok=True)
    nf_file = out / "_not_found.txt"
    not_found = set(nf_file.read_text().split()) if nf_file.exists() else set()
    todo = [t for t in tickers if force or (not (out / f"{t}.parquet").exists() and t not in not_found)]
    print(f"{len(tickers)} stocks, {len(todo)} to download")
    if not todo:
        return
    ciks = cik_map()
    for i, t in enumerate(todo, 1):
        cik = ciks.get(t)
        if cik is None:
            not_found.add(t)
            print(f"[{i}/{len(todo)}] {t}: no current SEC ticker match")
            continue
        df = fetch(t, cik)
        df.to_parquet(out / f"{t}.parquet", index=False)
        span = f"{df['filed'].min():%Y} to {df['filed'].max():%Y}" if len(df) else "none"
        print(f"[{i}/{len(todo)}] {t}: {len(df):,} 8-Ks, {span}")
        time.sleep(0.12)
        if i % 25 == 0:
            nf_file.write_text("\n".join(sorted(not_found)))
    nf_file.write_text("\n".join(sorted(not_found)))


def combine() -> pd.DataFrame:
    files = sorted((path("raw") / "sec_events").glob("*.parquet"))
    if not files:
        raise SystemExit("Nothing downloaded yet.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    bad = reused_sec_tickers(df.groupby("ticker")["filed"].max())
    if bad:
        print(f"Symbol now used by another company at the SEC ({len(bad)}), left out: {' '.join(bad)}")
        df = df[~df["ticker"].isin(bad)]
    out = path("clean") / "sec_events.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} 8-K filings for {df['ticker'].nunique()} companies, "
          f"{df['filed'].min():%Y-%m-%d} to {df['filed'].max():%Y-%m-%d} -> {out}")
    by = {k: int(df[f'ev_{k}'].sum()) for k in KINDS}
    print("Events: " + ", ".join(f"{k} {v:,}" for k, v in sorted(by.items(), key=lambda x: -x[1])))
    return df


def latest(tickers: list, days: int = 7) -> pd.DataFrame:
    """LIVE feed: re-read each company's most recent filings and list the 8-Ks of the last `days` days."""
    ciks = cik_map()
    since = pd.Timestamp.now(tz="UTC").normalize() - pd.to_timedelta(days, unit="D")
    parts = []
    for t in tickers:
        if t in ciks:
            df = fetch(t, ciks[t], full=False)
            if len(df):
                parts.append(df[df["accepted"] >= since])
            time.sleep(0.12)
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if ev.empty:
        print(f"No 8-K filings in the last {days} days.")
        return ev
    ev["what"] = ev["items"].map(describe)
    ev = ev.sort_values(["severity", "accepted"], ascending=[False, False])
    out = path("clean") / "sec_events_latest.parquet"
    ev.to_parquet(out, index=False)
    label = {0: "routine", 1: "notable", 2: "IMPORTANT", 3: "SEVERE"}
    print(f"{len(ev)} filings in the last {days} days ({(ev['severity'] >= 2).sum()} important or severe) -> {out}\n")
    for _, r in ev.head(40).iterrows():
        print(f"  {r['accepted']:%Y-%m-%d %H:%M} UTC  {r['ticker']:<6} [{label[int(r['severity'])]:>9}]  {r['what']}")
    return ev


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--latest", type=int, metavar="DAYS", help="live feed: 8-Ks filed in the last DAYS days")
    a = p.parse_args()
    u = pd.read_csv(ROOT / "config" / "universe.csv")
    tickers = a.tickers or sorted(u.loc[u.get("type", "stock") == "stock", "ticker"].dropna().unique())
    if a.latest:
        if not a.tickers:                              # live feed: current index members only
            tickers = sorted(u.loc[(u.get("type", "stock") == "stock") & u["end"].isna(), "ticker"].unique())
        latest(tickers, a.latest)
    else:
        if not a.combine:
            try:
                download(tickers, force=bool(a.tickers))   # tickers named on the command line are re-downloaded
            except KeyboardInterrupt:
                print("\nStopped; finished companies are saved. Run again to continue.")
        combine()
