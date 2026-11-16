"""What Wall Street analysts expect for one stock: 12-month price targets, buy / hold / sell counts and
whether recent earnings beat the estimates. Shown on the dashboard next to LatticeIQ's own ranges.

    python -m src.ingest.analysts NVDA        print what Yahoo returns for one ticker (use this to check it)

Source: Yahoo Finance through yfinance. One stock is fetched when it is opened in the dashboard and kept for a
day in data/live/analysts/<TICKER>.json (not committed). Each day's numbers are also added to a small history
in that file: Yahoo only gives today's consensus, so this is the only way to later test whether targets and
rating changes say anything about future returns.

These are analysts' opinions, not LatticeIQ forecasts, and nothing here has been backtested.
Yahoo data is for research and personal use; a commercial launch needs a licensed source.
"""
import json
import math
import sys
import time

import pandas as pd

from src.config import ROOT

DIR = ROOT / "data" / "live" / "analysts"
KEEP_HOURS = 24
RATINGS = ("strongBuy", "buy", "hold", "sell", "strongSell")


def _num(x):
    """A finite float, or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _try(get):
    try:
        return get()
    except Exception:
        return None


def parse(tk) -> dict:
    """Pull targets, ratings and earnings surprises out of a yfinance Ticker (or anything shaped like one).
    Every part is optional: yfinance versions differ and Yahoo leaves fields out for some stocks."""
    out = {"target": None, "ratings": None, "eps": []}

    # 1. price targets: the dedicated field in newer yfinance, else the company summary
    t = _try(lambda: tk.analyst_price_targets)
    target = {k: _num(t.get(k)) for k in ("low", "mean", "median", "high")} if isinstance(t, dict) else {}
    info = None
    if not target.get("mean"):
        info = _try(lambda: tk.info) or {}
        target = {"low": _num(info.get("targetLowPrice")), "mean": _num(info.get("targetMeanPrice")),
                  "median": _num(info.get("targetMedianPrice")), "high": _num(info.get("targetHighPrice"))}
    if target.get("mean") and target.get("low") and target.get("high"):
        n = _num((info or {}).get("numberOfAnalystOpinions")) if info is not None else None
        out["target"] = {**{k: round(v, 2) if v else None for k, v in target.items()}, "analysts": int(n) if n else None}

    # 2. buy / hold / sell counts for the current month
    r = _try(lambda: tk.recommendations_summary)
    if r is None or not hasattr(r, "columns") or "strongBuy" not in getattr(r, "columns", []):
        r = _try(lambda: tk.recommendations)
    if r is not None and hasattr(r, "columns") and all(c in r.columns for c in RATINGS) and len(r):
        row = r[r["period"] == "0m"].iloc[0] if "period" in r.columns and (r["period"] == "0m").any() else r.iloc[0]
        counts = {c: int(_num(row[c]) or 0) for c in RATINGS}
        if sum(counts.values()):
            out["ratings"] = counts

    # 3. the last four earnings reports: what was expected and what was reported (per share)
    h = _try(lambda: tk.earnings_history)
    if h is not None and hasattr(h, "columns") and {"epsActual", "epsEstimate"} <= set(h.columns):
        for when, row in h.tail(4).iterrows():
            a, e = _num(row["epsActual"]), _num(row["epsEstimate"])
            if a is not None and e is not None:
                quarter = _try(lambda: str(pd.Timestamp(when).date())) or str(when)
                out["eps"].append({"quarter": quarter, "actual": round(a, 2), "estimate": round(e, 2)})
    return out


def fetch(ticker: str) -> dict:
    import yfinance as yf
    return parse(yf.Ticker(ticker.replace(".", "-")))


def get(ticker: str, fetcher=None, now=None) -> dict:
    """Today's analyst view for one ticker, from the day's saved copy if there is one. Never raises:
    on failure it returns the last saved copy, or {'error': ...}."""
    fetcher = fetcher or fetch
    now = time.time() if now is None else now
    file = DIR / f"{ticker}.json"
    saved = _try(lambda: json.loads(file.read_text(encoding="utf-8"))) if file.exists() else None
    if saved and now - saved.get("fetched_at", 0) < KEEP_HOURS * 3600:
        return saved
    try:
        fresh = fetcher(ticker)
    except Exception as e:
        return saved or {"t": ticker, "error": f"{e!r:.160}"}
    if not fresh.get("target") and not fresh.get("ratings"):
        return saved or {"t": ticker, "error": "Yahoo returned no analyst data for this stock"}
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    history = [h for h in (saved or {}).get("history", []) if h.get("date") != day]
    history.append({"date": day, "target": fresh.get("target"), "ratings": fresh.get("ratings")})
    out = {"t": ticker, "source": "Yahoo Finance", "date": day, "fetched_at": now, **fresh, "history": history[-400:]}
    DIR.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    for sym in sys.argv[1:] or ["NVDA"]:
        got = fetch(sym)
        print(sym, json.dumps(got, indent=1))
        if not got["target"] and not got["ratings"]:
            import yfinance
            print(f"Nothing came back. yfinance version here: {yfinance.__version__}. Try: python -m pip install -U yfinance")
