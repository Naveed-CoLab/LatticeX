"""ARIMA baseline (statsforecast AutoARIMA) on each stock's daily EXCESS returns.

To stay fast across ~520 tickers, the model order is chosen and fitted once per month
per stock; on other evaluation dates in that month the same fitted model is applied to
the updated history (`forward`). The h-day forecast is the sum of the h daily forecasts.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsforecast.models import AutoARIMA


def _one_ticker(ticker: str, series: pd.Series, dates: list, h: int, window: int,
                max_p: int, max_q: int) -> list:
    out, model, month = [], None, None
    s = series.dropna()
    for d in dates:
        hist = s.loc[:d].values[-window:]
        if len(hist) < 250:
            continue
        try:
            if model is None or (d.year, d.month) != month:
                model = AutoARIMA(max_p=max_p, max_q=max_q, seasonal=False, d=0,
                                  stationary=True).fit(hist)
                month = (d.year, d.month)
                fc = model.predict(h)["mean"]
            else:
                fc = model.forward(y=hist, h=h)["mean"]
            out.append((d, ticker, float(np.sum(fc))))
        except Exception:
            out.append((d, ticker, 0.0))  # failed fit -> fall back to Random Walk
    return out


def arima_predictions(prices: pd.DataFrame, rows: pd.DataFrame, h: int, benchmark: str = "SPY",
                      window: int = 500, max_p: int = 3, max_q: int = 3, n_jobs: int = -1) -> pd.Series:
    """Return ARIMA forecasts aligned to `rows` (columns date, ticker)."""
    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    r = np.log(wide).diff()
    excess = r.sub(r[benchmark], axis=0)
    jobs = []
    for t, g in rows.groupby("ticker"):
        if t not in excess:
            continue
        jobs.append(delayed(_one_ticker)(t, excess[t], sorted(g["date"].unique()), h, window, max_p, max_q))
    results = Parallel(n_jobs=n_jobs)(jobs)
    preds = pd.DataFrame([x for res in results for x in res], columns=["date", "ticker", "y_pred"])
    return rows[["date", "ticker"]].merge(preds, on=["date", "ticker"], how="left")["y_pred"].fillna(0.0).values
