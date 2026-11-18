"""The one-time final test of the price ranges on 2025 onward (docs/holdout-lock.md).

    python -m scripts.holdout_ranges

Exactly as in the product: range model v2 fitted on windows that ended by 31 Dec 2024, then every weekly forecast
made from 1 Jan 2025 whose outcome is known by the end of the frozen data is scored. v1 is scored alongside as the
secondary. Runs on the PC in a few minutes (about 4.5 GB of memory). Refuses to run a second time.
Output: results/<time>_ranges_HOLDOUT/summary.json and by_quarter.csv.
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ROOT, load_settings, path
from src.models import vol_ranges as vr

CALIB_END = pd.Timestamp("2024-12-31")


def v1_bounds(train: pd.DataFrame, test: pd.DataFrame, h: int) -> dict:
    """Range model v1 for comparison: blended volatility x one symmetric multiplier per level, earnings or not."""
    z = train["fwd"].abs() / (train["vol"] * np.sqrt(h))
    out, base = {}, test["vol"].to_numpy() * np.sqrt(h)
    for lv in vr.LEVELS:
        q = {flag: float(z[train["earn"] == flag].quantile(lv)) for flag in (False, True)}
        half = np.where(test["earn"], q[True], q[False]) * base
        out[str(int(lv * 100))] = (-half, half)
    return out


def score(test: pd.DataFrame, bounds: dict) -> dict:
    a, earn, res = test["fwd"].to_numpy(), test["earn"].to_numpy().astype(bool), {}
    for k, (lo, hi) in bounds.items():
        hit = (a >= lo) & (a <= hi)
        res[k] = {"held": float(hit.mean()), "avg_width": float(np.mean(np.exp(hi) - np.exp(lo))),
                  "held_earnings": float(hit[earn].mean()) if earn.any() else None,
                  "held_other": float(hit[~earn].mean()) if (~earn).any() else None}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--again", action="store_true", help="only after writing the reason into docs/holdout-lock.md")
    a = ap.parse_args()
    if not (ROOT / "docs" / "holdout-lock.md").exists():
        raise SystemExit("docs/holdout-lock.md is missing: the final test runs only after its rules are written down.")
    done = sorted(p.name for p in path("results").glob("*_ranges_HOLDOUT"))
    if done and not a.again:
        raise SystemExit(f"The range holdout has already been run ({', '.join(done)}). It is used once.")
    s = load_settings()
    hold = pd.Timestamp(s["holdout_start"])
    u = pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["start", "end"])
    u = u[u["type"] == "stock"]
    cols = ["ticker", "date", "open", "high", "low", "close", "adj_close"]
    allpx = pd.read_parquet(path("clean") / "prices.parquet", columns=cols)
    allpx = allpx[(allpx[cols[2:]] > 0).all(axis=1)]
    px = allpx[allpx["ticker"].isin(u["ticker"])].sort_values(["ticker", "date"]).reset_index(drop=True)
    spy = allpx[allpx["ticker"] == "SPY"].sort_values("date")
    ev = pd.read_parquet(path("clean") / "sec_events.parquet")
    est, et = vr.expected_earnings(ev), vr.earnings_typical(px, ev)
    print("Measuring volatility on the frozen research prices", flush=True)
    f = vr.features(px, spy)
    out_dir = path("results") / f"{datetime.now():%Y%m%d_%H%M}_ranges_HOLDOUT"
    summary, quarters = {"calibrated_to": f"{CALIB_END:%Y-%m-%d}", "holdout_start": f"{hold:%Y-%m-%d}",
                         "data_end": f"{px['date'].max():%Y-%m-%d}", "horizons": {}}, []
    for h in vr.HORIZONS:
        m = vr.panel(f, u, est, h, et)
        train, test = m[m["d_end"] <= CALIB_END], m[m["date"] >= hold]
        assert train["d_end"].max() <= CALIB_END < test["date"].min(), "training windows overlap the holdout"
        fitted = vr.fit(train, h)
        fc = vr.forecast(test, h, fitted)
        v2 = score(test, {k: fc[k] for k in ("50", "80", "90")})
        v1 = score(test, v1_bounds(train, test, h))
        a = test["fwd"].to_numpy()
        dh, dl = a + test["hi_end"].to_numpy(), a + test["lo_end"].to_numpy()
        ok = np.isfinite(dh) & np.isfinite(dl)
        day = {"high_held": float((dh <= fc["day_hi"])[ok].mean()), "low_held": float((dl >= fc["day_lo"])[ok].mean()),
               "close_above_middle": float(np.mean(a > fc["mid"]))}
        lo80, hi80 = fc["80"]
        q = pd.DataFrame({"quarter": test["date"].dt.to_period("Q").astype(str).to_numpy(), "hit": (a >= lo80) & (a <= hi80)})
        for qq, g in q.groupby("quarter"):
            quarters.append({"horizon": h, "quarter": qq, "held80": float(g["hit"].mean()), "n": int(len(g))})
        summary["horizons"][str(h)] = {"forecasts": int(len(test)), "first": f"{test['date'].min():%Y-%m-%d}",
                                       "last": f"{test['date'].max():%Y-%m-%d}", "share_with_earnings": float(test["earn"].mean()),
                                       "v2": v2, "v1": v1, "day": day}
        print(f"  {h:>2} days, {len(test):,} forecasts {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}: "
              f"80% range held {v2['80']['held']:.1%} (v1 {v1['80']['held']:.1%}), width {v2['80']['avg_width']:.1%} "
              f"(v1 {v1['80']['avg_width']:.1%}); 50% {v2['50']['held']:.1%}, 90% {v2['90']['held']:.1%}; "
              f"day high {day['high_held']:.1%}, low {day['low_held']:.1%}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(quarters).to_csv(out_dir / "by_quarter.csv", index=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
