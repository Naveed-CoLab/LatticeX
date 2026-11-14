"""Download company financials from SEC EDGAR (free, official), keeping the date each number was FILED.

    python -m src.ingest.edgar              all stocks in config/universe.csv (resumable)
    python -m src.ingest.edgar AAPL MSFT    just these
    python -m src.ingest.edgar --combine    rebuild data/clean/fundamentals.parquet

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (needs SEC_USER_AGENT in .env;
SEC allows max 10 requests/second). XBRL data starts around 2009-2011, so fundamentals are empty before.

Why the filing date matters: Apple's Q3 numbers describe June but are only public in late July.
We store `filed` with every value; features may only use a value from the day after it was filed.

Ticker -> company (CIK) uses SEC's current ticker list, so most companies that were delisted long ago
can't be matched; their fundamentals stay empty (the model handles missing values). Listed in
data/raw/edgar/_not_found.txt.

Output: data/raw/edgar/<TICKER>.parquet and data/clean/fundamentals.parquet with columns
  ticker, cik, item, concept, start, end, val, filed, form
"""
import argparse
import time

import pandas as pd
import requests

from src.config import ROOT, path, secret

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# item -> concepts in priority order (companies switch tags over the years)
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "shares": ["dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "20-F", "40-F"}

# Same business, earlier SEC registrant. After a re-incorporation or holding-company reorganisation the SEC
# gives the company a NEW number (CIK), and the old filings stay under the old one. Without this, Exxon's
# history would start in 2026 and Disney's in 2019. ticker -> earlier CIKs (company names are printed at
# download time so each match can be checked).
PREDECESSOR_CIKS = {
    "XOM": [34088], "DIS": [1001039], "GOOGL": [1288776], "GOOG": [1288776], "CI": [701221], "MDT": [64670],
    "BLK": [1364742], "XRX": [108772], "APA": [6769], "ETN": [31277], "AVGO": [1649338, 1441634],
    "ICE": [1174746], "MRVL": [1058057], "NAVI": [1593538], "FTI": [1135152],
}
# A word the earlier registrant's SEC name must contain. A predecessor whose name does not match is skipped
# (and reported), so a wrong number here can never mix another company's data in.
PREDECESSOR_WORDS = {
    34088: ("EXXON",), 1001039: ("DISNEY", "TWDC"), 1288776: ("GOOGLE",), 701221: ("CIGNA",),
    64670: ("MEDTRONIC",), 1364742: ("BLACKROCK",), 108772: ("XEROX",), 6769: ("APACHE",), 31277: ("EATON",),
    1649338: ("BROADCOM",), 1441634: ("AVAGO",), 1174746: ("INTERCONTINENTAL",),
    1058057: ("MARVELL",), 1593538: ("NAVIENT",), 1135152: ("FMC TECHNOLOGIES",),
}
REUSED_GRACE_DAYS = 400
# SEC matches known to be another company, which the date rule cannot see because the price file itself
# continues under the reused symbol: CPWR = Compuware until Dec 2014, a penny stock's filings since 2011.
SEC_WRONG_COMPANY = {"CPWR"}


def predecessor_ok(cik: int, name: str) -> bool:
    words = PREDECESSOR_WORDS.get(cik)
    ok = words is None or any(w in str(name).upper().replace(" ", "") or w in str(name).upper() for w in words)
    if not ok:
        print(f"      ! CIK {cik} is '{name}', expected {'/'.join(words)} -> skipped (fix PREDECESSOR_CIKS)")
    return ok


def reused_sec_tickers(last_filed: pd.Series, last_price: pd.Series | None = None) -> list:
    """Tickers whose SEC match is a DIFFERENT, later company: the registrant kept filing long after the stock's
    prices ended (e.g. old member delisted in 2014, another company uses the symbol today). SEC's ticker list
    only knows today's owner of a symbol, so these matches would attach the wrong company's numbers to the old
    stock. last_filed / last_price: latest date per ticker. Tickers without prices are left alone (unused)."""
    if last_price is None:
        p = path("clean") / "prices.parquet"
        if not p.exists():
            return []
        px = pd.read_parquet(p, columns=["ticker", "date"])
        last_price = px.groupby("ticker")["date"].max()
    lf = pd.to_datetime(last_filed, utc=True).dt.tz_localize(None)
    lp = pd.to_datetime(last_price, utc=True).dt.tz_localize(None).reindex(lf.index)
    late = lf > lp + pd.to_timedelta(REUSED_GRACE_DAYS, unit="D")
    return sorted(set(late[late].index) | (SEC_WRONG_COMPANY & set(lf.index)))


def _get(url: str) -> requests.Response:
    for attempt in range(6):
        try:
            r = requests.get(url, headers={"User-Agent": secret("SEC_USER_AGENT"), "Accept-Encoding": "gzip"},
                             timeout=60)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(10 * (attempt + 1))
            continue
        return r
    raise RuntimeError(f"SEC: too many failures for {url}")


def cik_map() -> dict:
    r = _get(TICKERS_URL)
    r.raise_for_status()
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}


def extract(facts_json: dict, ticker: str) -> pd.DataFrame:
    rows = []
    facts = facts_json.get("facts", {})
    for item, concepts in CONCEPTS.items():
        for concept in concepts:
            ns, name = concept.split(":") if ":" in concept else ("us-gaap", concept)
            units = facts.get(ns, {}).get(name, {}).get("units", {})
            for unit, vals in units.items():
                if unit not in ("USD", "shares"):
                    continue
                for v in vals:
                    if v.get("form") not in FORMS or "filed" not in v:
                        continue
                    rows.append((ticker, int(facts_json.get("cik", 0)), item, concept, v.get("start"), v["end"],
                                 float(v["val"]), v["filed"], v["form"]))
    df = pd.DataFrame(rows, columns=["ticker", "cik", "item", "concept", "start", "end", "val", "filed", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def download(tickers: list, force: bool = False) -> None:
    out = path("raw") / "edgar"
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
        r = _get(FACTS_URL.format(cik=cik))
        if r.status_code == 404:
            not_found.add(t)
            print(f"[{i}/{len(todo)}] {t}: no XBRL facts")
            continue
        r.raise_for_status()
        df = extract(r.json(), t)
        for old in PREDECESSOR_CIKS.get(t, []):          # earlier registrant of the same business
            time.sleep(0.12)
            ro = _get(FACTS_URL.format(cik=old))
            if ro.status_code == 200:
                jo = ro.json()
                if not predecessor_ok(old, jo.get("entityName", "?")):
                    continue
                print(f"      + earlier registrant CIK {old}: {jo.get('entityName', '?')}")
                df = pd.concat([extract(jo, t), df], ignore_index=True)
        df.to_parquet(out / f"{t}.parquet", index=False)
        span = f"{df['end'].min():%Y} to {df['end'].max():%Y}" if len(df) else "empty"
        print(f"[{i}/{len(todo)}] {t} (CIK {cik}): {len(df):,} values, {span}")
        time.sleep(0.12)                       # SEC limit: 10 requests/second
        if i % 25 == 0:
            nf_file.write_text("\n".join(sorted(not_found)))
    nf_file.write_text("\n".join(sorted(not_found)))
    print(f"Not matched ({len(not_found)}): see {nf_file}")


def combine() -> pd.DataFrame:
    files = [f for f in sorted((path("raw") / "edgar").glob("*.parquet"))]
    if not files:
        raise SystemExit("Nothing downloaded yet.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    bad = reused_sec_tickers(df.groupby("ticker")["filed"].max())
    if bad:
        print(f"Symbol now used by another company at the SEC ({len(bad)}), left out: {' '.join(bad)}")
        df = df[~df["ticker"].isin(bad)]
    out = path("clean") / "fundamentals.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} values for {df['ticker'].nunique()} companies -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*")
    p.add_argument("--combine", action="store_true")
    a = p.parse_args()
    if not a.combine:
        if a.tickers:
            tickers = a.tickers
        else:
            u = pd.read_csv(ROOT / "config" / "universe.csv")
            tickers = sorted(u.loc[u.get("type", "stock") == "stock", "ticker"].dropna().unique())
        try:
            download(tickers, force=bool(a.tickers))     # tickers named on the command line are re-downloaded
        except KeyboardInterrupt:
            print("\nStopped; finished companies are saved. Run again to continue.")
    combine()
