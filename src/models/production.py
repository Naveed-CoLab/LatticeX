"""The model in daily use: train once on everything, save it, score every stock after each close.

    python -m scripts.train_final        train LatticeIQ v1.0 for 5, 20 and 60 days and save it (needs PyTorch;
                                         on Kaggle: python -m scripts.kaggle_cloud final-model)
    python -m scripts.score_today        rank today's stocks with the saved model (the dashboard does this itself)

Training is the slow part and is done once (repeat it every month or quarter). It uses exactly the recipe that
was tested walk-forward: the 44 price, fundamental and sector features and the three-seed neural net, with the
same trust factor k. The only difference is that it learns from every date whose outcome is known, not from
one fold.

Scoring is fast and needs no PyTorch: the saved weights are plain arrays and the forward pass is a few lines of
numpy. It builds today's features with the SAME functions the research code uses, on a recent slice of prices
that is extended with the daily closing prices (src/ingest/live.py).

What is saved: models/latticeiq_v1.npz (not committed). What scoring writes: data/live/rank.json for the
dashboard and one file per day in data/live/forecasts/, so every forecast is on record before its outcome
is known.
"""
import json
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_settings, path

MODEL = ROOT / "models" / "latticeiq_v1.npz"
RANK = ROOT / "data" / "live" / "rank.json"
LOG = ROOT / "data" / "live" / "forecasts"
VERSION = "1.0"
RECENT_DAYS = 900            # calendar days of prices used for scoring (the longest feature looks back one year)


# ---------- the saved model ----------
def export_mlp(model) -> list:
    """MLPSeeds -> [{W: [...], b: [...], mu, sd}, ...] as numpy arrays (one entry per random seed)."""
    out = []
    for m in model.members:
        sd = {k: v.detach().cpu().numpy() for k, v in m.net.state_dict().items()}
        keys = sorted({k.split(".")[0] for k in sd}, key=int)
        out.append({"W": [sd[f"{k}.weight"] for k in keys], "b": [sd[f"{k}.bias"] for k in keys],
                    "mu": float(m.mu), "sd": float(m.sd)})
    return out


def forward(members: list, X: np.ndarray) -> np.ndarray:
    """The neural net's forecast in plain numpy: missing inputs count as 0 (an average stock), two hidden
    layers with ReLU, scaled back to return units, averaged over the seeds."""
    x = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    outs = []
    for m in members:
        a = x
        for j, (W, b) in enumerate(zip(m["W"], m["b"])):
            a = a @ W.T + b
            if j < len(m["W"]) - 1:
                a = np.maximum(a, 0.0)
        outs.append(a[:, 0] * m["sd"] + m["mu"])
    return np.mean(outs, axis=0)


def save(nets: dict, meta: dict, file=None) -> None:
    file = file or MODEL
    arrays = {}
    for h, members in nets.items():
        for i, m in enumerate(members):
            for j, (W, b) in enumerate(zip(m["W"], m["b"])):
                arrays[f"h{h}_m{i}_W{j}"], arrays[f"h{h}_m{i}_b{j}"] = W, b
    meta = {**meta, "scale": {str(h): [[m["mu"], m["sd"]] for m in members] for h, members in nets.items()}}
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, meta=np.array(json.dumps(meta)), **arrays)


def load(file=None) -> tuple:
    """-> (meta, {horizon: members}) or (None, None) if no model has been trained yet."""
    file = file or MODEL
    if not file.exists():
        return None, None
    z = np.load(file, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    nets = {}
    for h, scales in meta["scale"].items():
        nets[int(h)] = []
        for i, (mu, sd) in enumerate(scales):
            n = len([k for k in z.files if k.startswith(f"h{h}_m{i}_W")])
            nets[int(h)].append({"W": [z[f"h{h}_m{i}_W{j}"] for j in range(n)], "b": [z[f"h{h}_m{i}_b{j}"] for j in range(n)],
                                 "mu": mu, "sd": sd})
    return meta, nets


# ---------- training (once) ----------
def train(horizons=None, train_step: int = 1, verbose: bool = True, since: str | None = None) -> dict:
    """since: use prices from this date only (a quick trial on a small machine; the real model uses everything.
    The full run needs about 8 GB of memory, which is why it runs on Kaggle)."""
    from scripts.run_experiment import build_panel, fit_shrunk, load_prices
    from src.folds import eval_dates
    s = load_settings()
    horizons = list(horizons or s["horizons"])
    t0 = time.time()
    prices = load_prices("real", "none", 0, s["benchmark"], verbose)
    if since:
        prices = prices[prices["date"] >= pd.Timestamp(since)].reset_index(drop=True)
    panel, feats = build_panel(prices, s, "real", False, use_fund=True, use_sectors=True)
    panel = panel[panel["asset"] == "stock"].reset_index(drop=True)
    grid = set(eval_dates(panel["date"], train_step))
    if verbose:
        print(f"Panel: {len(panel):,} rows, {panel['ticker'].nunique()} stocks, {len(feats)} features, "
              f"{panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}, built in {time.time() - t0:.0f}s", flush=True)
    nets, per_h = {}, {}
    for h in horizons:
        y = f"y_{h}"
        tr = panel[panel[y].notna() & panel["date"].isin(grid)]
        model, k = fit_shrunk("mlp", tr, y, feats, s)
        nets[h] = export_mlp(model)
        check = float(np.max(np.abs(forward(nets[h], tr[feats].head(2000).values) - model.predict(tr[feats].head(2000)))))
        per_h[str(h)] = {"k": round(float(k), 4), "val_ic": round(float(model.val_ic), 4), "rows": int(len(tr)),
                         "last_target_date": f"{tr['date'].max():%Y-%m-%d}"}
        if verbose:
            print(f"  {h:>2}-day model: {len(tr):,} rows to {tr['date'].max():%Y-%m-%d}, trust k {k:.2f}, "
                  f"rank skill on its latest years {model.val_ic:+.3f}, saved-copy check {check:.1e}  [{time.time() - t0:.0f}s]", flush=True)
        assert check < 1e-4, "the saved copy of the model does not reproduce the trained model"
    meta = {"name": "LatticeIQ", "version": VERSION, "features": feats, "horizons": per_h, "train_step": train_step, "since": since,
            "trained_through": f"{panel['date'].max():%Y-%m-%d}", "stocks": int(panel["ticker"].nunique()),
            "created": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recipe": "3-seed MLP (64, 32), price + fundamentals + sector features, target = excess return vs SPY, winsorised"}
    save(nets, meta)
    if verbose:
        print(f"Saved {MODEL}  ({time.time() - t0:.0f}s total)")
    return meta


# ---------- today's prices for scoring ----------
def splice(frozen: pd.DataFrame, live: pd.DataFrame | None) -> pd.DataFrame:
    """Continue each ticker's research price history with the daily closing prices.

    frozen: date, ticker, close (as traded), adj_close, volume, split_factor.   live: date, ticker, close, adj_close, volume.
    The two sources adjust prices on different scales, so they are joined at the last day both have:
      adj_close  continues at the research scale (live value x research / live on the joining day);
      close      continues with the live price. If the two closes differ by more than 5% on the joining day,
                 a split happened since the research data was frozen; it is recorded as a split factor on the
                 first new day, so that market value (price x shares) stays right.
    A ticker with no research history is taken from the live file as it is.
    """
    cols = ["date", "ticker", "close", "adj_close", "volume", "split_factor"]
    if live is None or live.empty:
        return frozen[cols]
    parts, fz = [], {t: g for t, g in frozen.groupby("ticker", sort=False)}
    for t, g in live.groupby("ticker", sort=False):
        g = g.sort_values("date")
        f = fz.get(t)
        if f is None:
            parts.append(g.assign(split_factor=1.0)[cols])
            continue
        common = np.intersect1d(f["date"].values, g["date"].values)
        if len(common) == 0:
            continue                                          # no day in common: cannot be joined safely, keep research only
        c = common.max()
        fr, lr = f[f["date"] == c].iloc[0], g[g["date"] == c].iloc[0]
        new = g[g["date"] > c]
        if new.empty or not (fr["adj_close"] > 0 and lr["adj_close"] > 0 and fr["close"] > 0 and lr["close"] > 0):
            continue
        fz[t] = f[f["date"] <= c]
        ratio = float(fr["close"] / lr["close"])
        ext = pd.DataFrame({"date": new["date"].values, "ticker": t, "close": new["close"].values,
                            "adj_close": new["adj_close"].values * float(fr["adj_close"] / lr["adj_close"]),
                            "volume": new["volume"].values, "split_factor": 1.0})
        if abs(ratio - 1) > 0.05:
            ext.iloc[0, ext.columns.get_loc("split_factor")] = ratio
        parts.append(ext)
    return pd.concat([f[cols] for f in fz.values()] + parts, ignore_index=True)


def extended_prices() -> pd.DataFrame:
    from src.ingest import live
    from src.universe import load_universe
    s = load_settings()
    keep = set(load_universe()["ticker"]) | {s["benchmark"]}
    fz = pd.read_parquet(path("clean") / "prices.parquet", columns=["date", "ticker", "close", "adj_close", "volume", "split_factor"])
    fz = fz[fz["ticker"].isin(keep)]
    lv = live.load()
    end = max(fz["date"].max(), lv["date"].max()) if lv is not None and len(lv) else fz["date"].max()
    start = end - pd.to_timedelta(RECENT_DAYS, unit="D")
    fz = fz[fz["date"] >= start]
    if lv is not None:
        lv = lv[lv["ticker"].isin(keep) & (lv["date"] >= start)]
    return splice(fz, lv)


# ---------- scoring (every day) ----------
def score(verbose: bool = True) -> dict | None:
    """Rank the current index members for each horizon. Returns the summary written to rank.json, or None
    if there is no trained model yet."""
    from scripts.run_experiment import build_panel
    from src.data.clean import align_to_calendar
    meta, nets = load()
    if meta is None:
        if verbose:
            print("No trained model yet (models/latticeiq_v1.npz). Train it once: python -m scripts.kaggle_cloud final-model")
        return None
    t0 = time.time()
    s = load_settings()
    ext = extended_prices()
    prices = align_to_calendar(ext[["date", "ticker", "adj_close", "volume"]], s["benchmark"])
    # every filing, not only recent ones: for some companies the features depend on old filings, and scoring
    # must see exactly what training saw (checked: with all filings the features match to the last digit)
    facts = pd.read_parquet(path("clean") / "fundamentals.parquet")
    panel, feats = build_panel(prices, s, "real", False, use_fund=True, use_sectors=True,
                               raw_prices=ext[["date", "ticker", "close", "split_factor"]], fundamentals=facts)
    if feats != meta["features"]:
        raise SystemExit("The saved model was trained on a different feature list than the code builds now. Train it again.")
    panel = panel[panel["asset"] == "stock"]
    per_day = panel.groupby("date").size()
    asof = per_day[per_day >= 0.8 * per_day.tail(20).max()].index.max()      # the latest day most members have a price
    today = panel[panel["date"] == asof].reset_index(drop=True)
    X = today[feats]
    thin = int((X.isna().mean(axis=1) > 0.5).sum())
    px = ext[ext["date"] == asof].set_index("ticker")["close"]
    out = {"asof": f"{asof:%Y-%m-%d}", "n": int(len(today)), "scored": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "model": {k: meta[k] for k in ("name", "version", "trained_through", "created")}, "horizons": {}, "stocks": {}}
    rows = []
    for h, members in nets.items():
        raw = forward(members, X.values)
        k = meta["horizons"][str(h)]["k"]
        pct = pd.Series(raw).rank(pct=True).values                             # 1.0 = ranked highest
        pos = pd.Series(-raw).rank(method="first").astype(int).values          # 1 = ranked highest
        out["horizons"][str(h)] = {"k": k}
        for t, r, p, q in zip(today["ticker"], raw, pct, pos):
            out["stocks"].setdefault(t, {})[str(h)] = [int(q), round(float(p), 4), round(float(k * r), 5)]
            rows.append((f"{asof:%Y-%m-%d}", t, h, float(r), float(p), int(q), len(today), float(k * r), float(px.get(t, np.nan))))
    RANK.parent.mkdir(parents=True, exist_ok=True)
    RANK.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    LOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["asof", "ticker", "horizon", "score", "pct", "pos", "n", "expected_excess", "close"]).to_csv(
        LOG / f"{asof:%Y-%m-%d}.csv", index=False)
    if verbose:
        print(f"Model scores for the close of {asof:%a %d %b %Y}: {len(today)} stocks ranked for {', '.join(str(h) for h in nets)} days "
              f"(model trained on data to {meta['trained_through']})"
              + (f"; {thin} stocks have more than half their inputs missing" if thin else "") + f"  [{time.time() - t0:.0f}s]")
    return out
