"""Walk-forward experiment: build features + targets, train on each fold, score every model.

Examples
  python -m scripts.run_experiment --data synthetic                   # no signal: XGBoost must NOT beat Random Walk
  python -m scripts.run_experiment --data synthetic --signal momentum  # planted signal: XGBoost MUST beat Random Walk
  python -m scripts.run_experiment --data real                        # stocks, price features
  python -m scripts.run_experiment --data real --macro                # + macro features (ablation)
  python -m scripts.run_experiment --data real --asset etf            # score the ETFs on their own
  python -m scripts.run_experiment --data real --news                 # + news sentiment features
  python -m scripts.run_experiment --data real --fundamentals         # + SEC fundamentals
  python -m scripts.run_experiment --data real --models random_walk xgboost arima
  python -m scripts.run_experiment --data real --horizon 5
  python -m scripts.run_experiment --data real --fundamentals --sectors --models random_walk xgboost lightgbm mlp ridge ensemble

Output: results/<run_name>/metrics.csv, rank_ic_by_year.csv, dm_tests.json, importance.csv,
        shrink.csv, predictions.parquet, config.json
The 2025-26 holdout is never touched here.
"""
import argparse
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import load_settings, path
from src.data.clean import align_to_calendar
from src.data.synthetic import make_prices
from src.eval.metrics import diebold_mariano, summarize
from src.features.fundamentals import FUND_FEATURES, fundamental_features
from src.features.macro import MACRO_FEATURES, macro_features
from src.features.pretrained import FEATURES as PRETRAINED_FEATURES, pretrained_features
from src.features.sector import SECTOR_FEATURES, sector_features
from src.features.news import NEWS_FEATURES, news_features
from src.features.price import FEATURES as PRICE_FEATURES, price_features
from src.folds import eval_dates, make_folds
from src.models.ensemble import LGBMForecaster, MLPSeeds, RidgeForecaster, ensemble_rank
from src.models.random_walk import RandomWalk
from src.models.xgb_model import XGBForecaster
from src.targets import make_targets, transform_target
from src.universe import asset_types, filter_members, load_universe


def load_prices(source: str, signal: str, n_stocks: int, benchmark: str = "SPY",
                verbose: bool = False) -> pd.DataFrame:
    if source == "synthetic":
        return make_prices(n_stocks=n_stocks, signal=signal)
    raw = pd.read_parquet(path("clean") / "prices.parquet",
                          columns=["date", "ticker", "adj_close", "volume"])
    keep = set(load_universe()["ticker"]) | {benchmark}     # price files of symbols no longer in the universe
    dropped = raw["ticker"].nunique() - len(set(raw["ticker"]) & keep)
    raw = raw[raw["ticker"].isin(keep)]                     # (old names, other companies' symbols) are ignored
    if verbose and dropped:
        print(f"Ignoring {dropped} price files that are not in config/universe.csv")
    return align_to_calendar(raw, benchmark, verbose=verbose)


def build_panel(prices: pd.DataFrame, s: dict, source: str, use_macro: bool,
                use_news: bool = False, use_fund: bool = False, use_sectors: bool = False,
                use_kronos: bool = False, kronos_model: bool = False,
                use_timesfm: bool = False, timesfm_model: bool = False) -> tuple[pd.DataFrame, list]:
    universe = load_universe() if source == "real" else None
    groups = asset_types(universe) if universe is not None else None
    feats = price_features(prices, s["benchmark"], s["features"]["cross_sectional_rank"], groups)
    targ = make_targets(prices, s["horizons"], s["benchmark"])
    panel = feats.merge(targ, on=["date", "ticker"], how="inner")
    panel = panel[panel["ticker"] != s["benchmark"]]  # benchmark's excess return is always 0
    panel["asset"] = panel["ticker"].map(groups).fillna("stock") if groups else "stock"

    feature_cols = list(PRICE_FEATURES)
    if universe is not None:
        panel = filter_members(panel, universe)        # only dates each stock was really in the index
    if use_macro and source == "real" and not (path("clean") / "macro.parquet").exists():
        print("note: data/clean/macro.parquet not found -> continuing without macro features")
        use_macro = False
    if use_macro and source == "real":
        macro = macro_features(pd.read_parquet(path("clean") / "macro.parquet"), panel["date"])
        panel = panel.merge(macro, on="date", how="left")
        feature_cols += MACRO_FEATURES
    if use_fund and source == "real":
        ff = path("clean") / "fundamentals.parquet"
        if not ff.exists():
            raise SystemExit("--fundamentals needs data/clean/fundamentals.parquet. Run: python -m scripts.kaggle_cloud "
                             "fundamentals-download  (or locally: python -m src.ingest.edgar)")
        raw = pd.read_parquet(path("clean") / "prices.parquet", columns=["date", "ticker", "close", "split_factor"])
        stocks = panel.loc[panel["asset"] == "stock", ["date", "ticker"]]
        fund = fundamental_features(pd.read_parquet(ff), raw, stocks, s["features"]["cross_sectional_rank"],
                                    prices_adj=prices)
        panel = panel.merge(fund, on=["date", "ticker"], how="left")
        feature_cols += FUND_FEATURES
    for name, as_feat, as_model in (("kronos", use_kronos, kronos_model), ("timesfm", use_timesfm, timesfm_model)):
        if (as_feat or as_model) and source == "real":
            pf = pretrained_features(name, panel.loc[panel["asset"] == "stock", ["date", "ticker"]], s["benchmark"],
                                     s["features"]["cross_sectional_rank"])
            panel = panel.merge(pf, on=["date", "ticker"], how="left")
            if as_feat:
                feature_cols += PRETRAINED_FEATURES[name]
    if use_sectors:
        sf = sector_features(panel, prices, s["benchmark"])
        if sf["date"].nunique() == 0:
            print("note: no sector ETFs in the data -> sector features skipped")
        else:
            panel = panel.merge(sf.rename(columns={"sector": "sector_etf"}), on=["date", "ticker"], how="left")
            feature_cols += SECTOR_FEATURES
            # sector-neutral targets: stock excess return minus its (point-in-time) sector ETF's excess return
            etf_y = targ[targ["ticker"].isin(panel["sector_etf"].dropna().unique())]
            for hh in s["horizons"]:
                e = etf_y[["date", "ticker", f"y_{hh}"]].rename(columns={"ticker": "sector_etf", f"y_{hh}": "_ey"})
                panel = panel.merge(e, on=["date", "sector_etf"], how="left")
                panel[f"ys_{hh}"] = panel[f"y_{hh}"] - panel["_ey"]
                panel = panel.drop(columns="_ey")
    if use_news and source == "real":
        arts_f, scored_f = path("clean") / "news_articles.parquet", path("clean") / "news_scored.parquet"
        if not (arts_f.exists() and scored_f.exists()):
            raise SystemExit("--news needs data/clean/news_articles.parquet and news_scored.parquet. "
                             "See README: Benzinga download + Kaggle FinBERT.")
        stocks = panel.loc[panel["asset"] == "stock", "ticker"].unique()
        nf = news_features(pd.read_parquet(arts_f), pd.read_parquet(scored_f), panel["date"], stocks,
                           s["features"]["cross_sectional_rank"])
        panel = panel.merge(nf, on=["date", "ticker"], how="left")
        feature_cols += NEWS_FEATURES
    return panel.reset_index(drop=True), feature_cols


ENSEMBLE_MEMBERS = ("xgboost", "mlp")   # one tree model + one neural net: least correlated pair (see docs/decisions.md)
TRAINED = ("xgboost", "lightgbm", "mlp", "ridge")


def make_model(name: str, s: dict, objective: str = "regression"):
    if name == "xgboost":
        return XGBForecaster(s["xgb"], objective=objective)
    if name == "lightgbm":
        return LGBMForecaster()
    if name == "mlp":
        return MLPSeeds()
    if name == "ridge":
        return RidgeForecaster()
    raise ValueError(name)


def fit_xgb_shrunk(train: pd.DataFrame, y_col: str, feats: list, s: dict,
                   objective: str = "regression") -> tuple[XGBForecaster, float]:
    return fit_shrunk("xgboost", train, y_col, feats, s, objective)


def fit_shrunk(name: str, train: pd.DataFrame, y_col: str, feats: list, s: dict,
               objective: str = "regression"):
    """Train a model, then learn how much to trust it.

    An inner model is trained on the older part of the training years and checked on the most
    recent 20% (with an embargo gap). k = least-squares slope of real outcome on prediction, kept
    in [0, 1] for return-scale targets. Final forecast = k x prediction of a model trained on ALL
    training years. If the model has no skill, k -> 0 and the forecast falls back to Random Walk.
    """
    how = s["target_transform"]
    dates = np.sort(train["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * 0.8)])
    inner_end = cut - pd.offsets.BDay(s["folds"]["embargo_days"])
    inner, val = train[train["date"] <= inner_end], train[train["date"] >= cut]
    k, val_ic = 1.0, float("nan")
    if len(inner) > 1000 and len(val) > 200:
        m = make_model(name, s, objective).fit(inner[feats], transform_target(inner, y_col, how),
                                               dates=inner["date"].values)
        p, y = m.predict(val[feats]), val[y_col].values
        if objective == "rank":                       # ranker scores have no units: centre them per date first
            p = p - pd.Series(p).groupby(val["date"].values).transform("mean").values
        k = float(np.dot(p, y) / max(np.dot(p, p), 1e-12))
        k = max(k, 0.0) if (how == "rank" or objective == "rank") else min(max(k, 0.0), 1.0)
        # ranking skill on the held-out latest training years (mean per-date Spearman); used as ensemble weight
        v = pd.DataFrame({"d": val["date"].values, "p": p, "y": y})
        val_ic = float(v.groupby("d")[["p", "y"]].apply(lambda g: g["p"].rank().corr(g["y"].rank())).mean())
    model = make_model(name, s, objective).fit(train[feats], transform_target(train, y_col, how),
                                               dates=train["date"].values)
    model.val_ic = val_ic
    return model, k


def run(source="synthetic", signal="none", models=("random_walk", "xgboost"), horizon=None,
        n_stocks=60, train_step=1, name=None, verbose=True, asset="stock", use_macro=False,
        shrink=True, use_news=False, start_year=None, use_fund=False, use_sectors=False,
        objective="regression", use_kronos=False, use_timesfm=False, target="spy") -> dict:
    s = load_settings()
    h = horizon or s["main_horizon"]
    y, end_col = f"y_{h}", f"target_end_{h}"
    if target == "sector" and not use_sectors:
        raise SystemExit("--target sector needs --sectors (the sector of each stock comes from the sector step)")
    yt = f"ys_{h}" if target == "sector" else y    # what the models LEARN; scoring always uses y (excess vs SPY)
    t0 = time.time()

    prices = load_prices(source, signal, n_stocks, s["benchmark"], verbose)
    panel, feats = build_panel(prices, s, source, use_macro, use_news, use_fund, use_sectors,
                               use_kronos, kronos_model="kronos" in models,
                               use_timesfm=use_timesfm, timesfm_model="timesfm" in models)
    panel = panel[panel["asset"] == asset].reset_index(drop=True)
    if panel.empty:
        raise SystemExit(f"No rows for asset type '{asset}'. Check the 'type' column in config/universe.csv.")
    holdout = pd.Timestamp(s["holdout_start"])
    grid = set(eval_dates(panel["date"], s["eval_step"]))
    train_grid = set(eval_dates(panel["date"], train_step))
    folds = make_folds(panel["date"], start_year or s["folds"]["first_test_year"], s["folds"]["last_test_year"],
                       s["folds"]["embargo_days"], s["holdout_start"])
    lags = max(math.ceil(h / s["eval_step"]) - 1, 0)
    if verbose:
        print(f"Panel: {len(panel):,} rows, {panel['ticker'].nunique()} {asset}s, {len(feats)} features, "
              f"{len(folds)} folds, horizon {h}d, built in {time.time() - t0:.0f}s")

    all_preds, importances, shrinks = [], [], []
    for f in folds:
        train = panel[(panel["date"] <= f.train_end) & panel[y].notna() & panel[yt].notna()
                      & panel["date"].isin(train_grid)]
        test = panel[(panel["date"] >= f.test_start) & (panel["date"] <= f.test_end)
                     & panel["date"].isin(grid) & panel[y].notna() & (panel[end_col] < holdout)]
        if train.empty or test.empty:
            continue
        assert train[end_col].max() < f.test_start, "target leakage: a training target ends inside the test year"
        base = test[["date", "ticker", y]].rename(columns={y: "y_true"}).assign(fold=f.year)

        trained = {}                                   # name -> (shrunk prediction, raw prediction)
        need = [m for m in TRAINED if m in models or ("ensemble" in models and m in ENSEMBLE_MEMBERS)]
        for m in need:
            if shrink:
                model, k = fit_shrunk(m, train, yt, feats, s, objective if m == "xgboost" else "regression")
            else:
                model, k = make_model(m, s, objective if m == "xgboost" else "regression").fit(
                    train[feats], transform_target(train, yt, s["target_transform"]), dates=train["date"].values), 1.0
            raw = model.predict(test[feats])
            if m == "xgboost" and objective == "rank":
                raw = raw - pd.Series(raw).groupby(test["date"].values).transform("mean").values
            trained[m] = (k * raw, raw, k, getattr(model, "val_ic", float("nan")))
            if m == "xgboost":
                importances.append(model.importance().rename(f.year))
            shrinks.append({"fold": f.year, "k": k, "model": m, "val_ic": trained[m][3]})
        for m in models:
            raw = None
            if m == "random_walk":
                pred = RandomWalk().predict(test)
            elif m in TRAINED:
                pred, raw = trained[m][0], trained[m][1]
            elif m == "ensemble":
                # each member weighted by its RANKING skill on the latest training years (val_ic > 0 only);
                # no test-year data is used. Both <= 0 -> equal weights.
                ics = np.array([trained[x][3] for x in ENSEMBLE_MEMBERS], dtype=float)
                ics = np.where(np.isfinite(ics), np.clip(ics, 0, None), 0.0)
                wts = ics / ics.sum() if ics.sum() > 0 else np.full(len(ics), 1 / len(ics))
                pred = np.sum([wt * trained[x][0] for wt, x in zip(wts, ENSEMBLE_MEMBERS)], axis=0)
                raw = ensemble_rank([trained[x][1] for x in ENSEMBLE_MEMBERS], test["date"].values, weights=wts)
            elif m in ("kronos", "timesfm"):
                # zero-shot pretrained model used directly: its excess-return forecast IS the prediction (no training)
                pred = test[f"{m}_raw_5" if h == 5 else f"{m}_raw_20"].fillna(0.0).values
            elif m == "arima":
                from src.models.arima import arima_predictions
                a = s["arima"]
                pred = arima_predictions(prices, test, h, s["benchmark"], a["window"], a["max_p"], a["max_q"])
            else:
                raise ValueError(f"unknown model {m}")
            all_preds.append(base.assign(model=m, y_pred=pred, y_raw=raw if raw is not None else pred))
        if verbose:
            ks = [x for x in shrinks if x["fold"] == f.year]
            k_txt = (", k / val IC " + " ".join(f"{x['model']}={x['k']:.2f}/{x['val_ic']:+.3f}" for x in ks)) if ks else ""
            print(f"  fold {f.year}: train to {f.train_end.date()} ({len(train):,} rows), "
                  f"test {len(test):,} rows{k_txt}  [{time.time() - t0:.0f}s]")

    preds = pd.concat(all_preds, ignore_index=True)
    table = pd.DataFrame([summarize(preds[preds["model"] == m], lags, m) for m in models])
    dm = {}
    if "random_walk" in models:
        rw = preds[preds["model"] == "random_walk"]
        for m in models:
            if m != "random_walk":
                dm[m] = diebold_mariano(preds[preds["model"] == m], rw, lags)
    by_year = (preds.groupby(["model", "fold"])[["date", "ticker", "y_true", "y_pred", "y_raw"]]
               .apply(lambda g: summarize(g, lags)["rank_ic"])
               .unstack("model"))

    tag = (f"{source}{'_' + signal if source == 'synthetic' else ''}_{asset}_h{h}"
           f"{'_fund' if use_fund else ''}{'_sect' if use_sectors else ''}{'_kron' if use_kronos else ''}{'_tfm' if use_timesfm else ''}{'_sectgt' if target == 'sector' else ''}{'_rank' if objective == 'rank' else ''}{'_macro' if use_macro else ''}{'_news' if use_news else ''}{f'_from{start_year}' if start_year else ''}")
    out = path("results") / (name or f"{datetime.now():%Y%m%d_%H%M}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    by_year.to_csv(out / "rank_ic_by_year.csv")
    preds.to_parquet(out / "predictions.parquet", index=False)
    if importances:
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).to_csv(out / "importance.csv")
    pd.DataFrame(shrinks).to_csv(out / "shrink.csv", index=False)
    (out / "dm_tests.json").write_text(json.dumps(dm, indent=2))
    (out / "config.json").write_text(json.dumps({
        "source": source, "signal": signal, "asset": asset, "horizon": h, "models": list(models),
        "features": feats, "macro": use_macro, "news": use_news, "fundamentals": use_fund, "sectors": use_sectors, "kronos": use_kronos, "timesfm": use_timesfm, "target": target, "objective": objective, "start_year": start_year, "shrink": shrink, "train_step": train_step,
        "n_tickers": int(panel["ticker"].nunique()), "settings": s}, indent=2, default=str))

    if verbose:
        pd.set_option("display.width", 160)
        print("\n" + table.round(4).to_string(index=False))
        for m, r in dm.items():
            verdict = "better than" if r["dm_stat"] < 0 and r["p_value"] < 0.05 else "NOT significantly better than"
            print(f"Diebold-Mariano {m} vs random_walk: stat {r['dm_stat']:.2f}, p={r['p_value']:.3f} -> {verdict} Random Walk")
        if shrinks:
            sk = pd.DataFrame(shrinks)
            print("Average shrink k: " + ", ".join(f"{m} {v:.2f}" for m, v in sk.groupby("model")["k"].mean().items())
                  + "  (0 = no trust in the model, 1 = full trust)")
        if importances:
            top = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).head(8)
            print("Top features: " + ", ".join(top.index))
        print(f"\nRank IC by test year:\n{by_year.round(3).to_string()}")
        print(f"\nSaved to {out}  ({time.time() - t0:.0f}s total)")
    return {"metrics": table, "dm": dm, "by_year": by_year, "out": out}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--signal", choices=["none", "momentum", "volcluster"], default="none")
    p.add_argument("--models", nargs="+", default=["random_walk", "xgboost"])
    p.add_argument("--horizon", type=int)
    p.add_argument("--asset", choices=["stock", "etf"], default="stock")
    p.add_argument("--macro", action="store_true", help="add macro features (off by default: 8 Nov ablation showed no gain)")
    p.add_argument("--no-macro", action="store_true", help="(default now; kept so old commands still work)")
    p.add_argument("--no-shrink", action="store_true", help="use raw XGBoost predictions")
    p.add_argument("--news", action="store_true", help="add news sentiment features")
    p.add_argument("--fundamentals", action="store_true", help="add SEC EDGAR fundamental features")
    p.add_argument("--sectors", action="store_true", help="add sector momentum + within-sector ranks")
    p.add_argument("--rank", action="store_true", help="train XGBoost to rank stocks per date (rank:pairwise)")
    p.add_argument("--kronos", action="store_true", help="add Kronos forecasts as features (needs kronos_forecasts.parquet)")
    p.add_argument("--target", choices=["spy", "sector"], default="spy",
                   help="what models learn: excess vs SPY (default) or vs the stock's own sector ETF (scored vs SPY either way)")
    p.add_argument("--timesfm", action="store_true", help="add TimesFM forecasts as features (needs timesfm_forecasts.parquet)")
    p.add_argument("--start-year", type=int, help="first test year (e.g. the first year with news coverage)")
    p.add_argument("--n-stocks", type=int, default=60)
    p.add_argument("--train-step", type=int, default=1, help="use every Nth day for training (5 = 5x faster)")
    p.add_argument("--name")
    a = p.parse_args()
    run(a.data, a.signal, tuple(a.models), a.horizon, a.n_stocks, a.train_step, a.name,
        asset=a.asset, use_macro=a.macro, shrink=not a.no_shrink, use_news=a.news,
        start_year=a.start_year, use_fund=a.fundamentals, use_sectors=a.sectors,
        objective="rank" if a.rank else "regression", use_kronos=a.kronos, use_timesfm=a.timesfm, target=a.target)
