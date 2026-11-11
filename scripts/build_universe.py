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


def download_sp500_history() -> pd.DataFrame:
    files = requests.get(GITHUB_API, timeout=60).json()
    if not isinstance(files, list):
        raise RuntimeError(f"GitHub API error: {files}")
    candidates = sorted(f["name"] for f in files
                        if f["name"].startswith("S&P 500 Historical Components") and f["name"].endswith(".csv"))
    if not candidates:
        raise RuntimeError("Could not find the history CSV in fja05680/sp500. Check the repo and update this script.")
    name = candidates[-1]
    url = next(f["download_url"] for f in files if f["name"] == name)
    print(f"Downloading {name}")
    raw = requests.get(url, timeout=120).text
    (path("raw") / "sp500_history.csv").write_text(raw, encoding="utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
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
    return iv


def main() -> None:
    s = load_settings()
    hist = download_sp500_history()
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
    print("Next: python -m src.ingest.tiingo")


if __name__ == "__main__":
    main()
