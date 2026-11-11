"""Download macro series from FRED, point-in-time.

Run: python -m src.ingest.fred
Output: data/clean/macro.parquet with columns: series, date, value, available_from

`available_from` is the first day the value could have been known:
  - revised series (CPI, unemployment): first-release value and its real publication date (ALFRED)
  - market series (VIX, yields, spreads): the next calendar day after the observation
Features must only use rows where available_from <= the feature date.
"""
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import path, secret

URL = "https://api.stlouisfed.org/fred/series/observations"

MARKET_SERIES = {
    "VIXCLS": "VIX",
    "DGS10": "10-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "T10Y2Y": "10y minus 2y yield-curve spread",
    "BAA10Y": "Baa corporate minus 10y Treasury (credit spread)",
}
REVISED_SERIES = {
    "CPIAUCSL": "CPI, all items",
    "UNRATE": "Unemployment rate",
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def _get(params: dict) -> list:
    params = {**params, "api_key": secret("FRED_API_KEY"), "file_type": "json"}
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["observations"]


def market_series(series_id: str) -> pd.DataFrame:
    df = pd.DataFrame(_get({"series_id": series_id, "observation_start": "1999-01-01"}))
    df["date"] = pd.to_datetime(df["date"])
    df["available_from"] = df["date"] + pd.Timedelta(days=1)
    return df[["date", "value", "available_from"]]


def revised_series(series_id: str) -> pd.DataFrame:
    # output_type=4 = initial release only; realtime_start = the day it was published
    obs = _get({"series_id": series_id, "observation_start": "1999-01-01",
                "realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "output_type": 4})
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["available_from"] = pd.to_datetime(df["realtime_start"])
    return df[["date", "value", "available_from"]]


def main() -> None:
    frames = []
    for sid in MARKET_SERIES:
        d = market_series(sid); d["series"] = sid; frames.append(d)
        print(f"{sid}: {len(d)} rows")
    for sid in REVISED_SERIES:
        d = revised_series(sid); d["series"] = sid; frames.append(d)
        print(f"{sid} (first release): {len(d)} rows")
    df = pd.concat(frames, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.dropna(subset=["value"])
    out = path("clean") / "macro.parquet"
    df[["series", "date", "value", "available_from"]].to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
