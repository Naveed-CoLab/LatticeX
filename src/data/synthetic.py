"""Fake price data for testing the pipeline before real data arrives.

signal="none":     pure random walks. A correct, leak-free pipeline must show ~0 skill here.
                   If a model "beats" this, something is leaking future data.
signal="momentum": each stock has a slowly drifting hidden trend, so past returns DO predict
                   future returns. A working pipeline must find this. Together the two
                   checks prove the pipeline is both honest and capable.
signal="volcluster": no return signal, but volatility moves through calm and stormy periods
                   (market-wide and per stock), like real markets. Used to test price brackets.
"""
import numpy as np
import pandas as pd


def make_prices(n_stocks: int = 60, start: str = "2004-01-01", end: str = "2026-10-30",
                signal: str = "none", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if signal == "volcluster":
        mvol = np.exp(_ar1(rng, n, 0.995, 0.06))                # market calm/stormy periods
        mkt = rng.normal(0.0003, 0.011, n) * mvol
    else:
        mvol = np.ones(n)
        mkt = rng.normal(0.0003, 0.011, n)  # SPY daily log returns

    frames = [_frame("SPY", dates, mkt, rng)]
    for i in range(n_stocks):
        beta = rng.uniform(0.6, 1.5)
        eps = rng.normal(0, 0.018, n)
        if signal == "momentum":
            drift = np.zeros(n)
            shocks = rng.normal(0, 0.00012, n)
            for t in range(1, n):
                drift[t] = 0.995 * drift[t - 1] + shocks[t]
            r = beta * mkt + drift + eps
        elif signal == "none":
            r = beta * mkt + eps
        elif signal == "volcluster":
            svol = np.exp(_ar1(rng, n, 0.99, 0.08))              # stock's own calm/stormy periods
            r = beta * mkt + eps * svol * np.sqrt(mvol)
        else:
            raise ValueError(signal)
        frames.append(_frame(f"STK{i:03d}", dates, r, rng))
    return pd.concat(frames, ignore_index=True)


def _ar1(rng, n, phi, sd):
    x = np.zeros(n)
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _frame(ticker, dates, log_ret, rng):
    close = 50 * np.exp(np.cumsum(log_ret))
    vol = rng.lognormal(14, 0.4, len(dates)).round()
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "adj_close": close, "volume": vol})
