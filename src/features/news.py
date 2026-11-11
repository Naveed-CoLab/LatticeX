"""Daily news features per stock, from FinBERT-scored Benzinga articles.

Timing rule (no look-ahead): an article counts for the trading day whose close comes AFTER it.
  - published before 4:00 pm New York time on a trading day -> that day
  - published at/after 4:00 pm, or on a weekend/holiday     -> the next trading day
So a feature on date t only uses news a trader could have read before the close of t.

Features (all from the past only):
  news_sent_1d / 5d / 20d   average sentiment of the stock's articles over the window
  news_cnt_5d / 20d         how much the stock is in the news (log of article count)
  news_abn_volume           this week's article count vs its normal rate over 60 days
  news_sent_surprise        last 5 days' tone vs the stock's usual tone over 120 days
  news_shock_3d             1 if a strongly positive/negative news day happened in the last 3 days
Before news coverage starts, features are left empty (NaN), not zero: "no data" is not "no news".
"""
import numpy as np
import pandas as pd

NEWS_FEATURES = [
    "news_sent_1d", "news_sent_5d", "news_sent_20d", "news_cnt_5d", "news_cnt_20d",
    "news_abn_volume", "news_sent_surprise", "news_shock_3d",
]
SHOCK = 0.6


def assign_trading_date(created_utc: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    et = pd.to_datetime(created_utc, utc=True).dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    day = day + pd.to_timedelta((et.dt.hour >= 16).astype(int), unit="D")
    td = pd.DatetimeIndex(sorted(trading_dates))
    idx = np.searchsorted(td.values, day.values.astype("datetime64[ns]"), side="left")
    out = pd.Series(pd.NaT, index=created_utc.index, dtype="datetime64[ns]")
    ok = idx < len(td)
    out[ok] = td.values[idx[ok]]
    return out


def news_features(articles: pd.DataFrame, scores: pd.DataFrame, trading_dates, tickers,
                  cross_sectional_rank: bool = True) -> pd.DataFrame:
    td = pd.DatetimeIndex(sorted(pd.unique(pd.Series(trading_dates))))
    a = articles[["id", "created_utc", "tickers"]].merge(scores[["id", "score"]], on="id", how="inner")
    a["date"] = assign_trading_date(a["created_utc"], td)
    a = a.dropna(subset=["date"])
    a = a.assign(ticker=a["tickers"].str.split(",")).explode("ticker")
    a = a[a["ticker"].isin(set(tickers))]

    daily = a.groupby(["ticker", "date"]).agg(n=("score", "size"), s=("score", "sum")).reset_index()
    grid = pd.MultiIndex.from_product([sorted(set(tickers)), td], names=["ticker", "date"])
    d = daily.set_index(["ticker", "date"]).reindex(grid, fill_value=0).reset_index()
    d["mean1"] = (d["s"] / d["n"]).where(d["n"] > 0)
    d["shock"] = (d["mean1"].abs() > SHOCK).astype(float)

    g = d.groupby("ticker", sort=False)
    roll = lambda col, w: g[col].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
    n5, s5, n20, s20 = roll("n", 5), roll("s", 5), roll("n", 20), roll("s", 20)
    n60, n120, s120 = roll("n", 60), roll("n", 120), roll("s", 120)
    d["news_sent_1d"] = d["mean1"]
    d["news_sent_5d"] = (s5 / n5).where(n5 > 0)
    d["news_sent_20d"] = (s20 / n20).where(n20 > 0)
    d["news_cnt_5d"] = np.log1p(n5)
    d["news_cnt_20d"] = np.log1p(n20)
    d["news_abn_volume"] = np.log((n5 + 1) / (n60 * 5 / 60 + 1))
    d["news_sent_surprise"] = d["news_sent_5d"] - (s120 / n120).where(n120 > 0)
    d["news_shock_3d"] = (roll("shock", 3) > 0).astype(float)

    # no coverage yet -> unknown, not zero (20-day warm-up after the first article)
    if len(daily):
        start = td[min(np.searchsorted(td.values, daily["date"].min().to_datetime64()) + 20, len(td) - 1)]
        d.loc[d["date"] < start, NEWS_FEATURES] = np.nan

    out = d[["date", "ticker"] + NEWS_FEATURES]
    if cross_sectional_rank:
        # ranks per date remove the steady growth in Benzinga coverage over the years
        out = out.copy()
        out[NEWS_FEATURES] = out.groupby("date")[NEWS_FEATURES].rank(pct=True) - 0.5
    return out
