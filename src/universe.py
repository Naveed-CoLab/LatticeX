"""Universe helpers: which tickers are stocks vs ETFs, and when each stock was in the S&P 500.

config/universe.csv columns: ticker, type (stock/etf), and optionally start, end (membership
spell; a ticker can have several rows if it left and re-joined). No start/end = always a member.
"""
import pandas as pd

from src.config import ROOT


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(ROOT / "config" / "universe.csv")
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    u["type"] = u.get("type", pd.Series("stock", index=u.index)).fillna("stock").str.lower()
    for c in ("start", "end"):
        u[c] = pd.to_datetime(u[c]) if c in u else pd.NaT
    return u


def asset_types(universe: pd.DataFrame) -> dict:
    return universe.drop_duplicates("ticker").set_index("ticker")["type"].to_dict()


def filter_members(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Keep a (date, ticker) row only if the ticker was in the index on that date.
    Prevents survivorship bias: the model only sees stocks an investor could have picked then."""
    spells = universe[["ticker", "start", "end"]].copy()
    spells["start"] = spells["start"].fillna(pd.Timestamp("1900-01-01"))
    spells["end"] = spells["end"].fillna(pd.Timestamp("2100-01-01"))
    m = panel[["date", "ticker"]].reset_index().merge(spells, on="ticker", how="inner")
    m = m[(m["date"] >= m["start"]) & (m["date"] < m["end"])]
    return panel.loc[sorted(set(m["index"]))].reset_index(drop=True)
