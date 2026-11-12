"""TimesFM 2.5 zero-shot forecasts (Google's pretrained time-series model; we do NOT train it).

Runs on Kaggle's GPU via:  python -m scripts.kaggle_cloud timesfm --timing
                           python -m scripts.kaggle_cloud timesfm --years 2010 2024
For each stock in the S&P 500 on a date, every `step` trading days (default 5 = the weekly evaluation grid),
TimesFM reads the last `lookback` adjusted closes (only data up to and including that day) and forecasts
the next 20 trading days, with quantiles. We keep:
  timesfm_ret_5, timesfm_ret_20       forecast log return over 5 / 20 trading days (median path)
  timesfm_q10_20, timesfm_q90_20      10th / 90th percentile of the 20-day log return (uncertainty)
SPY is forecast too, so excess-vs-SPY forecasts can be formed later.

Unlike Kronos, TimesFM was NOT pretrained on stock prices (GIFT-Eval, Wikipedia, Google Trends, synthetic),
so it is a cleaner zero-shot test with less risk of having seen our test years.

Output: data/clean/timesfm_forecasts.parquet (date, ticker, timesfm_ret_5, timesfm_ret_20, timesfm_q10_20,
timesfm_q90_20), merged across runs.
--timing: a few dates only, prints speed and an estimate for the full run.
--mock:   no model (extends the recent trend) to test the plumbing without a GPU.
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.config import load_settings, path
from src.models.kronos_runner import adjusted_ohlcv
from src.universe import load_universe

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PIP = "timesfm==3.0.2"              # torch is already installed on Kaggle, so no [torch] extra
COLS = ["timesfm_ret_5", "timesfm_ret_20", "timesfm_q10_20", "timesfm_q90_20"]


def load_model(lookback: int, horizon: int, batch: int):
    try:
        import timesfm
        assert hasattr(timesfm, "TimesFM_2p5_200M_torch")
    except (ImportError, AssertionError):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIP], check=True)
        import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    m.compile(timesfm.ForecastConfig(
        max_context=lookback, max_horizon=max(32, horizon), normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True, per_core_batch_size=batch))
    return m


class MockModel:
    """Stand-in for TimesFM: extends the last 60-day trend, quantiles = +-1.28 x recent volatility."""
    def forecast(self, horizon, inputs):
        pts, qs = [], []
        for x in inputs:
            lx = np.log(np.asarray(x, dtype=float))
            slope = (lx[-1] - lx[-61]) / 60 if len(lx) > 61 else 0.0
            vol = np.std(np.diff(lx[-60:])) if len(lx) > 61 else 0.02
            k = np.arange(1, horizon + 1)
            mid = lx[-1] + slope * k
            q = np.stack([mid + z * vol * np.sqrt(k) for z in np.linspace(-1.28, 1.28, 9)], axis=1)
            pts.append(np.exp(mid))
            qs.append(np.exp(np.concatenate([mid[:, None], q], axis=1)))
        return np.array(pts), np.array(qs)


def to_returns(point: np.ndarray, quant: np.ndarray, last: np.ndarray, horizon: int = 20) -> np.ndarray:
    """point (N, H), quant (N, H, 10: mean then 10th..90th), last (N,) -> (N, 4) log returns in COLS order."""
    h = horizon - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.column_stack([np.log(point[:, 4] / last), np.log(point[:, h] / last),
                               np.log(quant[:, h, 1] / last), np.log(quant[:, h, 9] / last)])
    out[~np.isfinite(out)] = np.nan
    return out


def run(years=None, timing=False, mock=False, lookback=512, horizon=20, batch=256, benchmark="SPY", step=5) -> None:
    s = load_settings()
    t0 = time.time()
    px = adjusted_ohlcv(pd.read_parquet(path("clean") / "prices.parquet"))
    cal = pd.DatetimeIndex(sorted(px.loc[px["ticker"] == benchmark, "date"].unique()))
    grid = cal[::step]
    if years:
        grid = grid[(grid.year >= years[0]) & (grid.year <= years[-1])]
    grid = grid[grid < pd.Timestamp(s["holdout_start"])]          # holdout stays untouched
    if timing:
        grid = grid[[len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4]]
    uni = load_universe()
    members = uni[uni["type"] == "stock"].copy()
    members["start"] = members["start"].fillna(pd.Timestamp("1900-01-01"))
    members["end"] = members["end"].fillna(pd.Timestamp("2100-01-01"))
    by_t = {t: g["close"].to_numpy(float) for t, g in px.groupby("ticker")}
    dates_t = {t: g["date"].to_numpy() for t, g in px.groupby("ticker")}
    model = MockModel() if mock else load_model(lookback, horizon, batch)
    print(f"TimesFM: {len(grid)} dates {grid.min():%Y-%m-%d}..{grid.max():%Y-%m-%d}, lookback {lookback}, "
          f"horizon {horizon}, every {step} days, batch {batch}, {'MOCK' if mock else MODEL_ID}  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    min_hist = 128                                                 # need some history; shorter inputs are padded
    rows, n_done, t_model = [], 0, 0.0
    chunk_dates = 20                                               # forecast many dates in one call (faster on GPU)
    for c0 in range(0, len(grid), chunk_dates):
        keys, inputs = [], []
        for d in grid[c0:c0 + chunk_dates]:
            d64 = np.datetime64(d)
            act = members[(members["start"] <= d) & (members["end"] > d)]["ticker"].unique().tolist() + [benchmark]
            for t in act:
                if t not in by_t:
                    continue
                j = np.searchsorted(dates_t[t], d64, side="right")  # rows up to and including d
                if j < min_hist or dates_t[t][j - 1] < d64 - np.timedelta64(7, "D"):
                    continue
                keys.append((d, t))
                inputs.append(by_t[t][max(0, j - lookback):j])
        if not inputs:
            continue
        t1 = time.time()
        point, quant = model.forecast(horizon=horizon, inputs=list(inputs))   # list() : timesfm pads the list in place
        t_model += time.time() - t1
        last = np.array([x[-1] for x in inputs])
        r = to_returns(np.asarray(point), np.asarray(quant), last, horizon)
        rows.append(pd.DataFrame(r, columns=COLS).assign(date=[k[0] for k in keys], ticker=[k[1] for k in keys]))
        n_done += len(inputs)
        done = min(c0 + chunk_dates, len(grid))
        print(f"  {done}/{len(grid)} dates, {n_done:,} forecasts, {n_done / max(t_model, 1e-9):.0f}/s model speed  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    new = pd.concat(rows, ignore_index=True)[["date", "ticker"] + COLS].dropna(subset=["timesfm_ret_20"])
    out_f = path("clean") / "timesfm_forecasts.parquet"
    if timing:
        rate = n_done / max(t_model, 1e-9)
        g_all = cal[::step]
        full = len(g_all[(g_all.year >= 2010) & (g_all < pd.Timestamp(s["holdout_start"]))])
        est = full * (n_done / max(len(grid), 1)) / max(rate, 1e-9)
        print(f"\nTIMING: {rate:.0f} forecasts/s -> full 2010-2024 run (~{full} dates) ≈ {est / 3600:.2f} h")
        spy = new[new["ticker"] == benchmark][["date", "timesfm_ret_20"]].rename(columns={"timesfm_ret_20": "spy"})
        x = new.merge(spy, on="date")
        print(f"Forecast sanity: mean 20d {new['timesfm_ret_20'].mean():+.4f}, excess vs SPY mean "
              f"{(x['timesfm_ret_20'] - x['spy']).mean():+.4f}, spread (std) {(x['timesfm_ret_20'] - x['spy']).std():.4f}, "
              f"80% band width {(new['timesfm_q90_20'] - new['timesfm_q10_20']).mean():.3f}")
        print(new.describe().round(4).to_string())
        out_f = path("clean") / "timesfm_timing.parquet"
    elif out_f.exists():
        new = pd.concat([pd.read_parquet(out_f), new]).drop_duplicates(["date", "ticker"], keep="last")
    new.to_parquet(out_f, index=False)
    print(f"Saved {len(new):,} rows -> {out_f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, metavar=("FROM", "TO"))
    p.add_argument("--timing", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--lookback", type=int, default=512, help="days of history TimesFM reads (512 ≈ 2 trading years)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--step", type=int, default=5, help="forecast every N trading days (5 = weekly, the evaluation grid)")
    a = p.parse_args()
    run(a.years, a.timing, a.mock, a.lookback, batch=a.batch, step=a.step)
