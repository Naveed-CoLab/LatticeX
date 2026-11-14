"""Build the point-in-time universe: every S&P 500 member since 2000 (with the dates it was in
the index) plus our ETFs, and check which of them Tiingo actually has prices for.

Run:  python -m scripts.build_universe

Sources (downloaded on your PC):
  - fja05680/sp500 on GitHub: daily S&P 500 member lists since 1996
  - Tiingo supported_tickers.zip: every symbol Tiingo has, with its first/last price date

Outputs:
  config/universe.csv          ticker,type,start,end,source_ticker   (one row per membership spell)
  config/universe_report.csv   every member with its Tiingo match status
  data/raw/sp500_history.csv   the raw downloaded file

How matching works: an index member is "covered" if Tiingo has the same symbol AND Tiingo's
price history overlaps the dates it was in the index. If Tiingo's history for that symbol starts
after the member left the index, the symbol was reused by another company -> not covered.

Fixed 12 Nov 2026 (the old version silently used a membership file that stopped in January 2019):
  1. Uses the repo's "(Updated)" file, which runs to the present, and refuses a file older than ~1 year.
  2. Ticker renames (FB -> META, ANTM -> ELV, ...) are joined into ONE continuous membership under the
     newest ticker, because price vendors keep the whole history under the newest ticker.
  3. A finished membership spell is rejected when the symbol's prices start more than 400 days after the
     stock joined the index and keep going for 3+ years after it left: that is another company using the
     old symbol (EP = El Paso in the index, Empire Petroleum in the price data).
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from src.config import ROOT, load_settings, path

GITHUB_API = "https://api.github.com/repos/fja05680/sp500/contents"
TIINGO_TICKERS = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
        "XLU", "XLV", "XLY", "TLT", "GLD", "VNQ"]


# Same company, new ticker (old -> new), from the repo's change log + company announcements. Chains are followed.
RENAMES = {
    "HRS": "LHX", "DWDP": "DD", "TMK": "GL", "BHGE": "BKR", "HCP": "PEAK", "PEAK": "DOC", "SYMC": "NLOK",
    "NLOK": "GEN", "CBS": "VIAC", "VIAC": "PARA", "BBT": "TFC", "JEC": "J", "UTX": "RTX", "ARNC": "HWM",
    "CTL": "LUMN", "MYL": "VTRS", "LB": "BBWI", "COG": "CTRA", "WLTW": "WTW", "DISCA": "WBD", "BLL": "BALL",
    "FB": "META", "ANTM": "ELV", "PKI": "RVTY", "FI": "FISV", "RE": "EG", "ABC": "COR", "CDAY": "DAY",
    "FLT": "CPAY", "MMC": "MRSH", "BK": "BNY",
}
# Renames that only apply before a date: until March 2020 "IR" was Ingersoll-Rand plc, which became Trane
# Technologies (TT); after that "IR" is a different company (the former Gardner Denver).
RENAMES_BEFORE = [("IR", "TT", "2020-03-03")]
# Spells to drop by hand: (ticker, any spell starting before this date). The price history under the symbol
# is ANOTHER company's (data vendors show a reused symbol's history as the latest owner's), and no rule on
# dates can tell. Found by scripts/audit_universe.py (trading volume far below other members) and by checking
# price levels against the member's known prices:
#   TT   before 2010 was Trane Inc., a different company from today's TT.
#   WEN  2000-2008 was Wendy's International; the history under WEN is Triarc's (the buyer, not a member).
#   CSR  in 2000 was Central & South West; the history is Centerspace's (a small REIT).
#   KMG  2000-2006 was Kerr-McGee; the history is KMG Chemicals' (a $4 stock).
#   MIL  2000-2010 was Millipore; the history is a small fund's.
#   LSI  2000-2014 was LSI Logic (about $11 when Avago bought it); the history is Life Storage's ($76).
#   GAS  2000-2011 was Nicor; the history is AGL Resources', which took the symbol and the index seat in
#        Dec 2011 (its later spell is correct and kept).
DROP_SPELLS = [("TT", "2010-01-01"), ("WEN", "2009-01-01"), ("CSR", "2001-01-01"), ("KMG", "2007-01-01"),
               ("MIL", "2011-01-01"), ("LSI", "2015-01-01"), ("GAS", "2011-06-01")]


def final_ticker(t: str) -> str:
    seen = set()
    while t in RENAMES and t not in seen:
        seen.add(t)
        t = RENAMES[t]
    return t


def apply_renames(hist: pd.DataFrame) -> pd.DataFrame:
    """Rewrite every daily member list with the newest ticker of each company."""
    out = []
    for d, s in zip(hist["date"], hist["tickers"]):
        m = set()
        for t in (x.strip() for x in str(s).split(",") if x.strip()):
            for old, new, before in RENAMES_BEFORE:
                if t == old and d < pd.Timestamp(before):
                    t = new
            m.add(final_ticker(t))
        out.append(",".join(sorted(m)))
    return hist.assign(tickers=out)


def download_sp500_history() -> pd.DataFrame:
    files = requests.get(GITHUB_API, timeout=60).json()
    if not isinstance(files, list):
        raise RuntimeError(f"GitHub API error: {files}")
    candidates = sorted(f["name"] for f in files
                        if f["name"].startswith("S&P 500 Historical Components") and f["name"].endswith(".csv"))
    if not candidates:
        raise RuntimeError("Could not find the history CSV in fja05680/sp500. Check the repo and update this script.")
    updated = [c for c in candidates if "updated" in c.lower()]
    name = updated[-1] if updated else candidates[-1]     # plain sorting used to pick the old 2019 file
    url = next(f["download_url"] for f in files if f["name"] == name)
    print(f"Downloading {name}")
    raw = requests.get(url, timeout=120).text
    (path("raw") / "sp500_history.csv").write_text(raw, encoding="utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    age = (pd.Timestamp.today() - df["date"].max()).days
    print(f"Membership history: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} ({len(df):,} dated lists)")
    if age > 400:
        raise RuntimeError(f"The membership file ends {df['date'].max():%Y-%m-%d} ({age} days ago): too old. "
                           "Check github.com/fja05680/sp500 for the current file name.")
    return df


def to_intervals(hist: pd.DataFrame, start: str) -> pd.DataFrame:
    """Daily member lists -> one row per continuous membership spell."""
    hist = hist.sort_values("date")
    dates = list(hist["date"])
    members = [set(t.strip() for t in str(s).split(",") if t.strip()) for s in hist["tickers"]]
    rows, open_since = [], {}
    for d, m in zip(dates, members):
        for t in m - open_since.keys():
            open_since[t] = d
        for t in list(open_since):
            if t not in m:
                rows.append((t, open_since.pop(t), d))   # left the index on this list date
    for t, s in open_since.items():
        rows.append((t, s, pd.NaT))                     # still a member
    iv = pd.DataFrame(rows, columns=["source_ticker", "start", "end"])
    iv = iv[(iv["end"].isna()) | (iv["end"] >= pd.Timestamp(start))]
    iv["start"] = iv["start"].clip(lower=pd.Timestamp(start))
    return iv.sort_values(["source_ticker", "start"]).reset_index(drop=True)


def tiingo_symbol(t: str) -> str:
    """fja05680 style -> Tiingo style. BRK.B -> BRK-B; drop '-YYYYMM' suffixes used for old tickers."""
    t = re.sub(r"-\d{6}$", "", t.strip().upper())
    return t.replace(".", "-").replace("/", "-")


def download_tiingo_tickers() -> pd.DataFrame:
    print("Downloading Tiingo supported tickers")
    z = zipfile.ZipFile(io.BytesIO(requests.get(TIINGO_TICKERS, timeout=120).content))
    df = pd.read_csv(z.open(z.namelist()[0]))
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("startDate", "endDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if "priceCurrency" in df:
        df = df[df["priceCurrency"].fillna("USD") == "USD"]
    return df


def match(iv: pd.DataFrame, tiingo: pd.DataFrame) -> pd.DataFrame:
    iv = iv.copy()
    iv["ticker"] = iv["source_ticker"].map(tiingo_symbol)
    t = tiingo.groupby("ticker").agg(t_start=("startDate", "min"), t_end=("endDate", "max")).reset_index()
    iv = iv.merge(t, on="ticker", how="left")
    end = iv["end"].fillna(pd.Timestamp.today())
    iv["status"] = "missing"
    has = iv["t_start"].notna()
    overlap = has & (iv["t_start"] <= end) & (iv["t_end"].fillna(pd.Timestamp.today()) >= iv["start"])
    iv.loc[has & ~overlap, "status"] = "symbol reused / no overlap"
    iv.loc[overlap, "status"] = "covered"
    full = overlap & (iv["t_start"] <= iv["start"] + np.timedelta64(10, "D"))
    iv.loc[overlap & ~full, "status"] = "covered (partial history)"
    # another company using an old symbol: the stock LEFT the index, its symbol's prices start long after it
    # joined AND keep going for years after it left. (A current member with a late price start is kept:
    # today's symbol is today's member, e.g. the new Fox Corp from 2019.)
    late = overlap & (iv["t_start"] > iv["start"] + np.timedelta64(400, "D"))
    goes_on = iv["end"].notna() & (iv["t_end"].fillna(pd.Timestamp.today()) > iv["end"] + np.timedelta64(1095, "D"))
    iv.loc[late & goes_on, "status"] = "symbol reused (prices start late)"
    for t, before in DROP_SPELLS:
        iv.loc[(iv["ticker"] == t) & (iv["start"] < pd.Timestamp(before)), "status"] = "dropped by hand (different company)"
    return iv


def main() -> None:
    s = load_settings()
    hist = apply_renames(download_sp500_history())
    iv = to_intervals(hist, s["start_date"])
    tiingo = download_tiingo_tickers()
    rep = match(iv, tiingo)
    rep.to_csv(ROOT / "config" / "universe_report.csv", index=False)

    ok = rep[rep["status"].str.startswith("covered")]
    stocks = ok[["ticker", "start", "end", "source_ticker"]].assign(type="stock")
    etfs = pd.DataFrame({"ticker": ETFS, "start": pd.Timestamp(s["start_date"]), "end": pd.NaT,
                         "source_ticker": ETFS, "type": "etf"})
    uni = pd.concat([etfs, stocks], ignore_index=True)[["ticker", "type", "start", "end", "source_ticker"]]
    uni["start"] = uni["start"].dt.date
    uni["end"] = uni["end"].dt.date
    uni.to_csv(ROOT / "config" / "universe.csv", index=False)

    n = rep["source_ticker"].nunique()
    gone = rep["end"].notna()
    print(f"\nS&P 500 members since {s['start_date']}: {n} tickers, {len(rep)} membership spells")
    print(rep["status"].value_counts().to_string())
    print(f"\nStill in the index today:   {(~gone).sum()} spells, "
          f"{(rep.loc[~gone, 'status'].str.startswith('covered')).mean():.0%} covered")
    print(f"Left the index since 2000: {gone.sum()} spells, "
          f"{(rep.loc[gone, 'status'].str.startswith('covered')).mean():.0%} covered   <- survivorship-bias check")
    print(f"\nSaved config/universe.csv ({uni['ticker'].nunique()} tickers) and config/universe_report.csv")
    have = {f.stem for f in (path("raw") / "prices").glob("*.parquet")} if (path("raw") / "prices").exists() else set()
    need = sorted(set(stocks["ticker"]) - have)
    print(f"Price files already on disk: {len(set(stocks['ticker']) & have)}; still to download: {len(need)}")
    if need:
        print("  " + " ".join(need))
    bad = rep[rep["status"].str.contains("reused|by hand")]
    print(f"Rejected as another company's symbol: {bad['ticker'].nunique()} tickers "
          f"({', '.join(sorted(bad['ticker'].unique())[:60])})")
    print("Next: python -m src.ingest.yahoo   (free; fills every ticker that has no price file yet)")


if __name__ == "__main__":
    main()
