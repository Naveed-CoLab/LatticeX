"""Kronos zero-shot forecasts (pretrained on billions of price candles; we do NOT train it).

Runs on Kaggle's GPU via:  python -m scripts.kaggle_cloud kronos --timing
                           python -m scripts.kaggle_cloud kronos --years 2008 2012
For each stock that was in the S&P 500 on a date, every `step` trading days (default 20 = monthly), Kronos reads the last
`lookback` days of split/dividend-adjusted open/high/low/close/volume (only data up to and including
that day) and forecasts the next 20 trading days of candles. We keep:
  kronos_ret_5, kronos_ret_20   forecast log return over 5 / 20 trading days
SPY is forecast too, so excess-vs-SPY forecasts can be formed later.

Output: data/clean/kronos_forecasts.parquet (date, ticker, kronos_ret_5, kronos_ret_20), merged across runs.
--timing: a few dates only, prints speed and an estimate for the full run.
--mock:   no model (extends the recent trend) to test the plumbing without a GPU.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.config import load_settings, path
from src.universe import load_universe

REPO = "https://github.com/shiyu-coder/Kronos"
MODEL_ID, TOKENIZER_ID = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"


def adjusted_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    """Raw OHLC scaled by adj_close/close so splits and dividends don't create fake jumps."""
    p = prices[["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]].dropna(
        subset=["close", "adj_close"]).copy()
    f = (p["adj_close"] / p["close"]).replace([np.inf, -np.inf], np.nan)
    for c in ("open", "high", "low"):
        p[c] = p[c] * f
    p["close"] = p["adj_close"]
    p["volume"] = p["volume"] / f.replace(0, np.nan)
    p = p.dropna(subset=["open", "high", "low", "close"])
    return p.drop(columns="adj_close").sort_values(["ticker", "date"])


def ensure_repo() -> str:
    """Download the Kronos code once (outside the job output folder)."""
    root = "/tmp/Kronos" if os.path.isdir("/kaggle") else str(path("raw") / "Kronos")
    if not os.path.isdir(os.path.join(root, "model")):
        subprocess.run(["git", "clone", "--depth", "1", REPO, root], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "safetensors", "huggingface_hub"],
                       check=False)
    return root


def load_predictor(device: str, max_context: int):
    root = ensure_repo()
    sys.path.insert(0, root)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402  (from the cloned repo)
    tok = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    mdl = Kronos.from_pretrained(MODEL_ID)
    return KronosPredictor(mdl, tok, device=device, max_context=max_context)


class MockPredictor:
    """Stand-in for Kronos when testing the plumbing: extends the last 60-day price trend.
    On fake data with a momentum signal this must show skill; on pure noise it must not."""
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        c = df["close"].values
        slope = np.log(c[-1] / c[-61]) / 60 if len(c) > 61 else 0.0
        return pd.DataFrame({"close": c[-1] * np.exp(slope * np.arange(1, pred_len + 1))})


def forecast_batch(predictor, windows: list, horizon: int, cal: pd.DatetimeIndex, T: float, top_p: float,
                   samples: int) -> list:
    """windows: list of (ticker, date, df_window). Returns list of predicted close arrays."""
    cols = ["open", "high", "low", "close", "volume"]
    dfs, xs, ys = [], [], []
    for _, d, w in windows:
        i = cal.searchsorted(d)
        fut = cal[i + 1:i + 1 + horizon]
        if len(fut) < horizon:                       # near the end of the data: business days
            fut = fut.append(pd.bdate_range(cal[-1] + pd.offsets.BDay(1), periods=horizon - len(fut)))
        win = w[cols].reset_index(drop=True).ffill()
        win["volume"] = win["volume"].fillna(0.0)
        dfs.append(win)
        xs.append(w["date"].reset_index(drop=True))
        ys.append(pd.Series(fut))
    kw = dict(pred_len=horizon, T=T, top_p=top_p, sample_count=samples)
    if hasattr(predictor, "predict_batch"):
        try:
            outs = predictor.predict_batch(df_list=dfs, x_timestamp_list=xs, y_timestamp_list=ys, verbose=False, **kw)
            return [np.asarray(o["close"], dtype=float) for o in outs]
        except TypeError:
            pass
    return [np.asarray(predictor.predict(df=a, x_timestamp=b, y_timestamp=c, **kw)["close"], dtype=float)
            for a, b, c in zip(dfs, xs, ys)]


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _free_gpu() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def run_sharded(argv: list, n: int) -> None:
    """Kaggle gives 2 GPUs: run half of the dates on each in parallel, then merge."""
    ensure_repo()                                              # once, before the workers start
    procs = [subprocess.Popen([sys.executable, "-u", "-m", "src.models.kronos_runner", *argv,
                               "--shard", str(i), "--nshards", str(n)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(n)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a GPU shard failed: exit codes {codes}")
    timing = "--timing" in argv
    parts = [path("clean") / f"kronos_{'timing' if timing else 'forecasts'}_part{i}.parquet" for i in range(n)]
    new = pd.concat([pd.read_parquet(f) for f in parts if f.exists()], ignore_index=True)
    out_f = path("clean") / ("kronos_timing.parquet" if timing else "kronos_forecasts.parquet")
    if not timing and out_f.exists():
        new = pd.concat([pd.read_parquet(out_f), new]).drop_duplicates(["date", "ticker"], keep="last")
    new.to_parquet(out_f, index=False)
    for f in parts:
        f.unlink(missing_ok=True)
    print(f"Merged {n} GPU shards: {len(new):,} rows -> {out_f}")


def run(years=None, timing=False, mock=False, lookback=256, horizon=20, batch=128, T=1.0, top_p=0.9,
        samples=2, benchmark="SPY", step=20, shard=0, nshards=1) -> None:
    s = load_settings()
    t0 = time.time()
    raw = pd.read_parquet(path("clean") / "prices.parquet")
    px = adjusted_ohlcv(raw)
    cal = pd.DatetimeIndex(sorted(px.loc[px["ticker"] == benchmark, "date"].unique()))
    grid = cal[::step]
    uni = load_universe()
    members = uni[uni["type"] == "stock"].copy()
    members["start"] = members["start"].fillna(pd.Timestamp("1900-01-01"))
    members["end"] = members["end"].fillna(pd.Timestamp("2100-01-01"))
    if years:
        grid = grid[(grid.year >= years[0]) & (grid.year <= years[-1])]
    grid = grid[grid < pd.Timestamp(s["holdout_start"])]      # holdout stays untouched
    if timing:
        grid = grid[[len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4, len(grid) // 4 + 1]]
    grid = grid[shard::nshards]                                # this GPU's share of the dates
    by_t = {t: g.reset_index(drop=True) for t, g in px.groupby("ticker")}
    predictor = MockPredictor() if mock else load_predictor(
        "cuda:0" if _cuda() else "cpu", max_context=512)       # each shard sees its own GPU as cuda:0
    print(f"Kronos: {len(grid)} dates {grid.min():%Y-%m-%d}..{grid.max():%Y-%m-%d}, lookback {lookback}, "
          f"horizon {horizon}, every {step} days, {samples} sample paths averaged, batch {batch}, "
          f"{'MOCK' if mock else MODEL_ID}  [{time.time() - t0:.0f}s]", flush=True)

    part = f"_part{shard}" if nshards > 1 else ""
    out_f = path("clean") / f"kronos_forecasts{part}.parquet"
    rows, n_done, t_model = [], 0, 0.0
    for gi, d in enumerate(grid, 1):
        act = members[(members["start"] <= d) & (members["end"] > d)]["ticker"].unique().tolist() + [benchmark]
        windows = []
        for t in act:
            g = by_t.get(t)
            if g is None:
                continue
            j = g["date"].searchsorted(d, side="right")       # rows up to and including d
            if j < lookback or g["date"].iloc[j - 1] < d - pd.Timedelta(days=7):
                continue
            windows.append((t, d, g.iloc[j - lookback:j]))
        b = 0
        while b < len(windows):
            chunk = windows[b:b + batch]
            t1 = time.time()
            try:
                preds = forecast_batch(predictor, chunk, horizon, cal, T, top_p, samples)
            except Exception as e:
                if not _is_oom(e) or batch <= 4:
                    raise
                _free_gpu()
                batch = max(batch // 2, 4)                  # GPU memory full: halve the batch and retry
                print(f"  GPU memory full -> batch size now {batch}", flush=True)
                continue
            t_model += time.time() - t1
            b += len(chunk)
            for (t, _, w), pc in zip(chunk, preds):
                last = float(w["close"].iloc[-1])
                if last > 0 and len(pc) >= horizon and np.all(pc[[4, horizon - 1]] > 0):
                    rows.append((d, t, np.log(pc[4] / last), np.log(pc[horizon - 1] / last)))
        n_done += len(windows)
        if gi % 10 == 0 or gi == len(grid):
            rate = n_done / max(t_model, 1e-9)
            print(f"  {gi}/{len(grid)} dates, {n_done:,} forecasts, {rate:.1f}/s model speed  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    new = pd.DataFrame(rows, columns=["date", "ticker", "kronos_ret_5", "kronos_ret_20"])
    if timing:
        rate = n_done / max(t_model, 1e-9)
        g_all = cal[::step]
        full = len(g_all[(g_all.year >= 2010) & (g_all < pd.Timestamp(s["holdout_start"]))])
        est = full * (n_done / max(len(grid), 1)) / max(rate * nshards, 1e-9)
        print(f"\nTIMING [GPU {shard}]: {rate:.1f} forecasts/s per GPU x {nshards} GPU(s) -> full 2010-2024 run "
              f"(~{full} dates) ≈ {est / 3600:.1f} h")
        x = new.merge(new[new["ticker"] == benchmark][["date", "kronos_ret_20"]].rename(columns={"kronos_ret_20": "spy"}), on="date")
        print(f"Forecast sanity: mean 20d {new['kronos_ret_20'].mean():+.3f}, excess vs SPY mean "
              f"{(x['kronos_ret_20'] - x['spy']).mean():+.3f}, spread (std) {(x['kronos_ret_20'] - x['spy']).std():.3f}")
        print(new.describe().round(4).to_string())
        out_f = path("clean") / f"kronos_timing{part}.parquet"
    elif out_f.exists() and nshards == 1:
        new = pd.concat([pd.read_parquet(out_f), new]).drop_duplicates(["date", "ticker"], keep="last")
    new.to_parquet(out_f, index=False)
    print(f"Saved {len(new):,} rows -> {out_f}  ({time.time() - t0:.0f}s)")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, metavar=("FROM", "TO"))
    p.add_argument("--timing", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--lookback", type=int, default=256, help="days of history Kronos reads (256 ≈ 1 trading year)")
    p.add_argument("--batch", type=int, default=128, help="starting batch; halves automatically if GPU memory runs out")
    p.add_argument("--samples", type=int, default=2, help="sample paths averaged per forecast (less noise)")
    p.add_argument("--step", type=int, default=20, help="forecast every N trading days (20 = monthly)")
    p.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--nshards", type=int, default=0, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.nshards == 0 and not a.mock and _gpu_count() >= 2:
        run_sharded([x for x in sys.argv[1:]], 2)                # both Kaggle GPUs in parallel
    else:
        run(a.years, a.timing, a.mock, a.lookback, batch=a.batch, samples=a.samples, step=a.step,
            shard=a.shard, nshards=max(a.nshards, 1))
