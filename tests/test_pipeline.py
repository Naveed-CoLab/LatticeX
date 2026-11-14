"""Leakage and sanity tests. Run: pytest -q
These are the 'leakage gate' from the design doc: nothing in Stage C starts until they pass.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.run_experiment import run
from src.data.synthetic import make_prices
from src.eval.metrics import daily_rank_ic, diebold_mariano, hit_rate
from src.features.price import FEATURES, price_features
from src.folds import make_folds
from src.targets import make_targets


@pytest.fixture(scope="module")
def prices():
    return make_prices(n_stocks=20, start="2010-01-01", end="2014-12-31", signal="momentum")


def test_features_ignore_the_future(prices):
    """Scramble every price after a cut-off date: no feature on or before the cut-off may change."""
    cut = pd.Timestamp("2013-06-28")
    rng = np.random.default_rng(1)
    changed = prices.copy()
    late = changed["date"] > cut
    changed.loc[late, "adj_close"] *= rng.uniform(0.5, 1.5, late.sum())
    changed.loc[late, "volume"] *= rng.uniform(0.5, 1.5, late.sum())
    a = price_features(prices).set_index(["date", "ticker"]).sort_index()
    b = price_features(changed).set_index(["date", "ticker"]).sort_index()
    early = a.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(a.loc[early, FEATURES], b.loc[early, FEATURES])


def test_target_is_future_excess_return(prices):
    t = make_targets(prices, horizons=(20,))
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    d0, d20 = wide.index[100], wide.index[120]
    expected = (np.log(wide.loc[d20, "STK003"] / wide.loc[d0, "STK003"])
                - np.log(wide.loc[d20, "SPY"] / wide.loc[d0, "SPY"]))
    got = t.set_index(["date", "ticker"]).loc[(d0, "STK003"), "y_20"]
    assert got == pytest.approx(expected)
    assert t.set_index(["date", "ticker"]).loc[(d0, "STK003"), "target_end_20"] == d20


def test_folds_have_embargo_and_skip_holdout(prices):
    dates = pd.Series(pd.bdate_range("2004-01-01", "2026-10-30"))
    folds = make_folds(dates, 2012, 2026, embargo_days=60, holdout_start="2025-01-01")
    idx = pd.DatetimeIndex(dates)
    for f in folds:
        gap = idx.get_loc(f.test_start) - idx.get_loc(f.train_end)
        assert gap > 60
        assert f.test_end < pd.Timestamp("2025-01-01")
    assert folds[-1].year == 2024


def test_metrics_basics():
    df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-01", periods=30), 10),
                       "y_true": np.random.default_rng(0).normal(size=300)})
    perfect = df.assign(y_pred=df["y_true"])
    zero = df.assign(y_pred=0.0)
    assert hit_rate(perfect["y_true"], perfect["y_pred"]) == 1.0
    assert hit_rate(zero["y_true"], zero["y_pred"]) == 0.5
    assert daily_rank_ic(perfect).mean() == pytest.approx(1.0)
    assert daily_rank_ic(zero).mean() == 0.0
    assert diebold_mariano(perfect, zero, lags=0)["dm_stat"] < 0


@pytest.mark.slow
def test_no_skill_on_pure_noise():
    """On random-walk prices a leak-free pipeline cannot predict anything."""
    r = run("synthetic", "none", n_stocks=30, train_step=5, name="pytest_none", verbose=False)
    xgb = r["metrics"].set_index("model").loc["xgboost"]
    assert abs(xgb["ic_tstat_nw"]) < 3, "XGBoost 'predicts' random noise: something is leaking"


@pytest.mark.slow
def test_finds_planted_signal():
    """When a real signal exists, the pipeline must find it."""
    r = run("synthetic", "momentum", n_stocks=30, train_step=5, name="pytest_momentum", verbose=False)
    xgb = r["metrics"].set_index("model").loc["xgboost"]
    assert xgb["rank_ic"] > 0.03 and xgb["ic_tstat_nw"] > 3


def test_membership_filter_respects_dates():
    from src.universe import filter_members
    dates = pd.bdate_range("2010-01-01", "2012-12-31")
    panel = pd.DataFrame([(d, t) for d in dates for t in ["AAA", "BBB", "CCC"]], columns=["date", "ticker"])
    uni = pd.DataFrame({"ticker": ["AAA", "BBB", "BBB"],
                        "start": pd.to_datetime([None, "2010-01-01", "2012-01-01"]),
                        "end": pd.to_datetime(["2011-06-30", "2011-01-01", None])})
    out = filter_members(panel, uni)
    assert out[out.ticker == "AAA"]["date"].max() < pd.Timestamp("2011-06-30")
    bbb = out[out.ticker == "BBB"]["date"]
    assert not ((bbb >= "2011-01-01") & (bbb < "2012-01-01")).any()   # gap between spells
    assert "CCC" not in set(out["ticker"])                              # never a member


def test_macro_features_use_only_published_values():
    from src.features.macro import macro_features
    d = pd.bdate_range("2019-01-01", "2021-12-31")
    months = pd.date_range("2018-01-01", "2021-12-01", freq="MS")
    base = [pd.DataFrame({"series": s, "date": d, "value": 1.0, "available_from": d + pd.Timedelta(days=1)})
            for s in ["VIXCLS", "T10Y2Y", "BAA10Y", "DGS10"]]
    cpi = pd.DataFrame({"series": "CPIAUCSL", "date": months, "value": np.arange(len(months)) + 100.0,
                        "available_from": months + pd.Timedelta(days=45)})
    un = cpi.assign(series="UNRATE")
    f = macro_features(pd.concat(base + [cpi, un]), d).set_index("date")
    # CPI for Jan 2020 is published ~15 Feb 2020: on 10 Feb the latest known month must be Dec 2019
    known = f.loc["2020-02-10", "cpi_yoy"]
    dec_2019 = (100 + 23) / (100 + 11) - 1
    assert known == pytest.approx(dec_2019)


def test_news_timing_4pm_rule():
    from src.features.news import assign_trading_date
    td = pd.bdate_range("2024-03-01", "2024-03-15")       # Fri 1 Mar ... (weekdays only)
    t = pd.Series(pd.to_datetime([
        "2024-03-01 15:59", "2024-03-01 16:00", "2024-03-02 11:00", "2024-03-04 08:00",
    ]).tz_localize("America/New_York").tz_convert("UTC"))
    got = assign_trading_date(t, td).dt.strftime("%Y-%m-%d").tolist()
    # before close Fri -> Fri; at 4pm Fri -> Mon; Saturday -> Mon; Mon pre-market -> Mon
    assert got == ["2024-03-01", "2024-03-04", "2024-03-04", "2024-03-04"]


def test_news_features_ignore_future_articles():
    from src.features.news import NEWS_FEATURES, news_features
    td = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    n = 3000
    arts = pd.DataFrame({"id": range(n), "tickers": rng.choice(["AAA", "BBB", "CCC"], n),
                         "created_utc": pd.to_datetime(rng.choice(td, n)) + pd.Timedelta(hours=15)})
    arts["created_utc"] = arts["created_utc"].dt.tz_localize("UTC")
    sc = pd.DataFrame({"id": range(n), "score": rng.uniform(-1, 1, n)})
    cut = pd.Timestamp("2020-08-31")
    a = news_features(arts, sc, td, ["AAA", "BBB", "CCC"], cross_sectional_rank=False).set_index(["date", "ticker"])
    sc2 = sc.copy()
    late = arts["created_utc"].dt.tz_convert(None) > cut + pd.Timedelta(days=1)
    sc2.loc[late.values, "score"] = 0.99                  # rewrite every future article
    b = news_features(arts, sc2, td, ["AAA", "BBB", "CCC"], cross_sectional_rank=False).set_index(["date", "ticker"])
    early = a.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(a.loc[early, NEWS_FEATURES], b.loc[early, NEWS_FEATURES])


def test_gaps_never_stretch_targets():
    """A ticker with a 120-day hole must not get targets that jump across it."""
    from src.data.clean import align_to_calendar
    p = make_prices(n_stocks=3, start="2010-01-01", end="2012-12-31", signal="none")
    dates = sorted(p["date"].unique())
    hole = (p["ticker"] == "STK001") & p["date"].between(dates[200], dates[320])
    p = p[~hole]
    aligned = align_to_calendar(p)
    t = make_targets(aligned, horizons=(20,)).dropna(subset=["y_20"])
    cal = pd.DatetimeIndex(dates)
    pos = cal.get_indexer(t["date"]), cal.get_indexer(t["target_end_20"])
    assert (pos[1] - pos[0] == 20).all()                      # always exactly 20 trading days
    s = t[t["ticker"] == "STK001"]
    # last real price before the hole is dates[199]; up to 5 days are carried forward (to dates[204]),
    # so no target may start between dates[185] and the end of the hole
    assert not s["date"].between(dates[185], dates[320]).any()


def test_prob_up_and_conformal_basics():
    from scripts.run_brackets import QS, conformal_shift, prob_up
    row = np.array([[-0.10, -0.06, -0.02, 0.01, 0.04, 0.08, 0.12]])
    p = prob_up(row)[0]
    assert 0.5 < p < 0.75                                   # median above 0 -> more likely up than down
    assert prob_up(np.array([[0.01] * 7]))[0] > 0.95        # whole distribution above 0
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    too_narrow = np.tile(np.array([-2, -0.5, -0.3, 0, 0.3, 0.5, 2.0]), (len(z), 1))
    assert conformal_shift(too_narrow, z, 0.10, 0.90, 0.80) > 0.5   # widens a too-narrow 80% range


@pytest.mark.slow
def test_brackets_are_calibrated():
    from scripts.run_brackets import run as run_brackets
    r = run_brackets("synthetic", "volcluster", n_stocks=25, name="pytest_brackets", verbose=False)
    m = r["metrics"].set_index("method").loc["model"]
    assert abs(m["coverage_80"] - 0.80) < 0.04 and abs(m["coverage_50"] - 0.50) < 0.04


def test_fundamentals_quarters_ttm_and_timing():
    from tests.fake_fund import make
    from src.features.fundamentals import first_filed, fundamental_features, quarters, ttm
    f = make()
    ff = first_filed(f)
    fl = ff[(ff["item"] == "revenue") & ff["start"].notna()]
    q = quarters(fl).set_index("end")["q"]
    assert q[pd.Timestamp("2014-12-31")] == 130            # Q4 = full year - 9 months YTD
    t = ttm(fl, quarters(fl)).set_index("end")
    assert t.loc[pd.Timestamp("2016-03-31"), "ttm"] == 570
    dates = pd.bdate_range("2014-01-01", "2017-06-30")
    panel = pd.DataFrame({"date": dates, "ticker": "FAKE"})
    px = pd.DataFrame({"date": dates, "ticker": "FAKE", "close": 50.0, "split_factor": 1.0})
    r = fundamental_features(f, px, panel, cross_sectional_rank=False).set_index("date")
    assert np.isnan(r.loc["2015-02-20", "f_sales_yield"])   # filing day: not usable yet
    assert r.loc["2015-02-23", "f_sales_yield"] == pytest.approx(460 / 500)


def test_fundamentals_ignore_later_filings():
    from tests.fake_fund import make
    from src.features.fundamentals import FUND_FEATURES, fundamental_features
    f = make()
    cut = pd.Timestamp("2015-12-31")
    changed = f.copy()
    changed.loc[changed["filed"] > cut, "val"] *= 3          # rewrite everything filed later
    dates = pd.bdate_range("2014-01-01", "2017-06-30")
    panel = pd.DataFrame({"date": dates, "ticker": "FAKE"})
    px = pd.DataFrame({"date": dates, "ticker": "FAKE", "close": 50.0, "split_factor": 1.0})
    a = fundamental_features(f, px, panel, cross_sectional_rank=False).set_index("date")
    b = fundamental_features(changed, px, panel, cross_sectional_rank=False).set_index("date")
    pd.testing.assert_frame_equal(a.loc[:cut, FUND_FEATURES], b.loc[:cut, FUND_FEATURES])


def _sector_prices(n_per=6, seed=3):
    """Fake market: SPY + 4 sector ETFs; each stock follows one sector factor."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2012-01-01", "2016-12-31")
    mkt = rng.normal(0.0003, 0.01, len(d))
    fac = {e: rng.normal(0, 0.008, len(d)) for e in ["XLK", "XLE", "XLF", "XLV"]}
    rows = [("SPY", mkt)] + [(e, mkt + f) for e, f in fac.items()]
    truth = {}
    for e, f in fac.items():
        for k in range(n_per):
            t = f"{e[-1]}S{k}"
            truth[t] = e
            rows.append((t, mkt + 1.2 * f + rng.normal(0, 0.012, len(d))))
    px = pd.concat([pd.DataFrame({"date": d, "ticker": t, "adj_close": 50 * np.exp(np.cumsum(r))}) for t, r in rows])
    return px, truth


def test_sectors_recovered_from_prices():
    from src.features.sector import assign_sectors
    px, truth = _sector_prices()
    sec = assign_sectors(px, list(truth))
    last = sec[sec["date"] == sec["date"].max()].set_index("ticker")["sector"]
    assert (last.reindex(list(truth)) == pd.Series(truth)).mean() > 0.9
    first_use = sec["date"].min()
    assert first_use > px["date"].min() + pd.Timedelta(days=200)   # needs a year of history first


def test_filing_reaction_timing():
    from tests.fake_fund import make
    from src.features.fundamentals import filing_reaction
    f = make()
    dates = pd.bdate_range("2014-01-01", "2017-06-30")
    px = pd.concat([pd.DataFrame({"date": dates, "ticker": t, "adj_close": 100.0}) for t in ("SPY", "FAKE")])
    jump = (px["ticker"] == "FAKE") & (px["date"] >= "2015-04-27")      # +10% on the first day after filing
    px.loc[jump, "adj_close"] = 110.0
    r = filing_reaction(f, px).set_index("end")
    assert r.loc[pd.Timestamp("2015-04-27"), "react"] == pytest.approx(np.log(1.1))
    assert r.loc[pd.Timestamp("2015-04-27"), "avail"] >= pd.Timestamp("2015-04-29")


@pytest.mark.slow
def test_rank_objective_finds_signal():
    r = run("synthetic", "momentum", n_stocks=30, train_step=5, name="pytest_rank", verbose=False, objective="rank")
    xgb_ = r["metrics"].set_index("model").loc["xgboost"]
    assert xgb_["rank_ic"] > 0.03 and xgb_["ic_tstat_nw"] > 3


def test_kronos_features_use_only_past_forecasts(tmp_path, monkeypatch):
    import src.features.pretrained as P
    d = pd.bdate_range("2020-01-01", "2020-03-31")
    grid = d[::5]
    k = pd.DataFrame([(g, t, 0.0, v) for g in grid for t, v in (("SPY", 0.01), ("AAA", 0.03), ("BBB", -0.02))],
                     columns=["date", "ticker", "kronos_ret_5", "kronos_ret_20"])
    monkeypatch.setattr(P, "path", lambda key: tmp_path)
    k.to_parquet(tmp_path / "kronos_forecasts.parquet")
    panel = pd.DataFrame([(x, t) for x in d for t in ("AAA", "BBB")], columns=["date", "ticker"])
    m = P.kronos_features(panel, cross_sectional_rank=False).set_index(["date", "ticker"])
    assert m.loc[(grid[1], "AAA"), "kronos_xret_20"] == pytest.approx(0.02)      # 0.03 - SPY 0.01
    before_first = d[d < grid[0]]
    assert len(before_first) == 0 or m.loc[before_first, "kronos_xret_20"].isna().all()
    one_after = d[d.get_loc(grid[1]) + 1]
    assert m.loc[(one_after, "BBB"), "kronos_xret_20"] == pytest.approx(-0.03)   # carried from grid[1], not grid[2]
    assert (m.index.get_level_values("date") >= grid[0]).all() or m.loc[d[d < grid[0]], "kronos_xret_20"].isna().all()


def test_timesfm_features_use_only_past_forecasts(tmp_path, monkeypatch):
    """TimesFM forecasts: excess vs SPY, uncertainty = q90 - q10, and never used before the day they were made."""
    import src.features.pretrained as P
    d = pd.bdate_range("2020-01-01", "2020-03-31")
    grid = d[::5]
    rows = []
    for i, g in enumerate(grid):
        for t, v in (("SPY", 0.01), ("AAA", 0.03 + i), ("BBB", -0.02)):
            rows.append((g, t, 0.0, v, v - 0.05, v + 0.05))
    k = pd.DataFrame(rows, columns=["date", "ticker", "timesfm_ret_5", "timesfm_ret_20", "timesfm_q10_20", "timesfm_q90_20"])
    monkeypatch.setattr(P, "path", lambda key: tmp_path)
    k.to_parquet(tmp_path / "timesfm_forecasts.parquet")
    panel = pd.DataFrame([(x, t) for x in d for t in ("AAA", "BBB")], columns=["date", "ticker"])
    m = P.timesfm_features(panel, cross_sectional_rank=False).set_index(["date", "ticker"])
    assert m.loc[(grid[1], "AAA"), "timesfm_xret_20"] == pytest.approx(0.03 + 1 - 0.01)
    assert m.loc[(grid[1], "AAA"), "timesfm_unc_20"] == pytest.approx(0.10)
    day_before = d[d.get_loc(grid[2]) - 1]          # must still use grid[1]'s forecast, not grid[2]'s
    assert m.loc[(day_before, "AAA"), "timesfm_raw_20"] == pytest.approx(0.03 + 1 - 0.01)


def test_timesfm_runner_ignores_future_prices(tmp_path, monkeypatch):
    """Scramble prices after a date: forecasts made on or before that date must not change (mock model, no GPU)."""
    import src.models.timesfm_runner as R
    px = make_prices(n_stocks=5, start="2012-01-01", end="2014-12-31", signal="none")
    uni = pd.DataFrame({"ticker": [t for t in px["ticker"].unique() if t != "SPY"], "type": "stock",
                        "start": pd.NaT, "end": pd.NaT})
    monkeypatch.setattr(R, "load_universe", lambda: uni)
    monkeypatch.setattr(R, "path", lambda key: tmp_path)

    def forecasts(prices):
        prices.to_parquet(tmp_path / "prices.parquet")
        (tmp_path / "timesfm_forecasts.parquet").unlink(missing_ok=True)
        R.run(years=(2013, 2014), mock=True, step=20)
        return pd.read_parquet(tmp_path / "timesfm_forecasts.parquet").set_index(["date", "ticker"]).sort_index()

    a = forecasts(px.copy())
    cut = pd.Timestamp("2013-12-31")
    b_px = px.copy()
    late = b_px["date"] > cut
    for c in ("open", "high", "low", "close", "adj_close"):
        b_px.loc[late, c] *= np.random.default_rng(3).uniform(0.7, 1.3, late.sum())
    b = forecasts(b_px)
    early = a.index.get_level_values("date") <= cut
    assert early.sum() > 50
    pd.testing.assert_frame_equal(a[early], b.loc[a.index[early]])
    assert not np.allclose(a[~early].values, b.loc[a.index[~early]].values)   # later ones did change
    assert set(R.COLS) <= set(a.columns) and (a["timesfm_q90_20"] > a["timesfm_q10_20"]).all()


def test_ensemble_rank_is_scale_free():
    from src.models.ensemble import ensemble_rank
    d = np.array(["a"] * 4 + ["b"] * 4)
    p1 = np.array([1, 2, 3, 4, 10, 20, 30, 40.0])
    p2 = p1 * 1000 - 7                                   # same order, different scale -> same ranks
    e = ensemble_rank([p1, p2], d)
    assert np.allclose(e, ensemble_rank([p1], d))
    assert np.allclose(pd.Series(e).groupby(d).mean().values, 0.0)
    assert np.allclose(ensemble_rank([p1, -p1], d, weights=[1, 0]), ensemble_rank([p1], d))   # weight 0 = ignored
    assert np.allclose(ensemble_rank([p1, -p1], d, weights=[0, 0]), 0.0)                      # all 0 -> equal


@pytest.mark.slow
def test_ensemble_finds_signal_and_not_noise():
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    models = ("random_walk", "xgboost", "lightgbm", "mlp", "ridge", "ensemble")
    good = run("synthetic", "momentum", models=models, n_stocks=30, train_step=5, name="pytest_ens_m", verbose=False)
    t = good["metrics"].set_index("model")
    for m in models[1:]:
        assert t.loc[m, "rank_ic"] > 0.03 and t.loc[m, "ic_tstat_nw"] > 3, m
    noise = run("synthetic", "none", models=models, n_stocks=30, train_step=5, name="pytest_ens_n", verbose=False)
    t = noise["metrics"].set_index("model")
    for m in models[1:]:
        assert t.loc[m, "ic_tstat_nw"] < 2.5, m          # no 'skill' on pure noise


def test_yahoo_rebuilds_raw_prices_around_splits():
    """Yahoo 'Close' is split-adjusted; we must rebuild the raw traded price like Tiingo's 'close'."""
    from src.ingest.yahoo import to_tiingo_format
    idx = pd.bdate_range("2020-01-01", periods=6)
    raw_close = np.array([100, 102, 51, 52, 26, 27.0])            # 2:1 split on day 3, 2:1 again on day 5
    splits = np.array([0, 0, 2, 0, 2, 0.0])
    later = np.array([4, 4, 2, 2, 1, 1.0])                         # splits after each day
    y = pd.DataFrame({"Open": raw_close / later, "High": raw_close / later, "Low": raw_close / later,
                      "Close": raw_close / later, "Adj Close": raw_close / later * 0.99,
                      "Volume": 1000 * later, "Dividends": 0.0, "Stock Splits": splits}, index=idx)
    out = to_tiingo_format(y, "TEST")
    assert np.allclose(out["close"], raw_close)
    assert np.allclose(out["volume"], 1000)                        # raw shares traded
    assert list(out["splitFactor"]) == [1, 1, 2, 1, 2, 1]
    assert set(out.columns) >= {"date", "open", "high", "low", "close", "volume", "adjClose", "adjVolume",
                                "divCash", "splitFactor", "ticker", "source"}


def test_yahoo_prune_sets_aside_recycled_symbols(tmp_path):
    """'S' = Sprint in the index from 2000, but Yahoo's 'S' history starts 2021 (SentinelOne) -> rejected."""
    from src.ingest.yahoo import prune
    def f(t, start, src, n=50):
        pd.DataFrame({"date": pd.bdate_range(start, periods=n).strftime("%Y-%m-%dT00:00:00.000Z"),
                      "adjClose": 1.0, "source": src}).to_parquet(tmp_path / f"{t}.parquet")
    f("S", "2021-06-30", "yahoo")         # recycled
    f("PEP", "2000-01-03", "yahoo")       # fine
    f("V", "2008-03-19", "yahoo")         # IPO 2008, joined index 2009 -> fine
    f("OLD", "2015-01-02", "tiingo")      # Tiingo files are never touched
    f("SPGI", "2001-01-02", "yahoo")      # history starts 1 year after our data window -> fine
    f("FOXA", "2019-03-12", "yahoo")      # current member, new company from 2019 -> kept (today's symbol)
    f("SNDK", "2025-02-13", "yahoo")      # old spell 2006-16 can't match, new spell from 2025 does -> kept
    f("KVUE", "2026-10-30", "yahoo", n=1)  # delisted: Yahoo keeps one stub row -> set aside even though "current"
    u = pd.DataFrame({"ticker": ["S", "PEP", "V", "OLD", "SPGI", "FOXA", "SNDK", "SNDK", "KVUE"],
                      "start": pd.to_datetime(["2000-01-01", "2000-01-01", "2009-12-21", "2000-01-01", "2000-01-01",
                                               "2004-12-20", "2006-04-20", "2025-11-28", "2023-08-25"]),
                      "end": pd.to_datetime(["2013-07-09", None, None, None, None, None, "2016-05-12", None, None])})
    moved = prune(tmp_path, u)
    assert [m.split()[0] for m in moved] == ["KVUE", "S"]
    assert (tmp_path / "_rejected" / "KVUE.parquet").exists()
    assert (tmp_path / "_rejected" / "S.parquet").exists() and (tmp_path / "PEP.parquet").exists()
    assert (tmp_path / "OLD.parquet").exists() and (tmp_path / "SPGI.parquet").exists()
    assert (tmp_path / "FOXA.parquet").exists() and (tmp_path / "SNDK.parquet").exists()
    assert [m.split()[0] for m in prune(tmp_path, u)] == ["KVUE", "S"]   # re-running is stable (restores, re-checks)


@pytest.mark.slow
def test_crash_risk_is_calibrated_and_beats_baserate():
    from scripts.run_crash_risk import run as crash_run
    r = crash_run("synthetic", "volcluster", n_stocks=30, name="pytest_crash", verbose=False)
    t = r["metrics"].set_index("method")
    assert t.loc["model", "auc"] > 0.55 and t.loc["model", "brier_skill"] > 0     # finds the volatility signal
    c = r["calibration"]
    assert (c["actual"] - c["predicted"]).abs().max() < 0.08                        # "20%" means about 20%


def test_sector_neutral_target_is_stock_minus_its_sector_etf():
    from scripts.run_experiment import build_panel
    from src.config import load_settings
    px = make_prices(n_stocks=12, start="2010-01-01", end="2013-12-31", signal="none")
    w = px.pivot(index="date", columns="ticker", values="adj_close")
    stocks = [c for c in w if c != "SPY"]
    etfs = []
    for name, members in (("XLK", stocks[:4]), ("XLE", stocks[4:8]), ("XLF", stocks[8:])):
        e = w[members].pct_change().mean(axis=1).fillna(0).add(1).cumprod().mul(100).rename(name)
        etfs.append(pd.DataFrame({"date": e.index, "ticker": name, "adj_close": e.values}))
    e = pd.concat(etfs)
    for c in ("open", "high", "low", "close"):
        e[c] = e["adj_close"]
    e["volume"] = 1e6
    s = load_settings()
    panel, _ = build_panel(pd.concat([px, e[px.columns]], ignore_index=True), s, "synthetic", False, use_sectors=True)
    p = panel[panel["sector_etf"].notna() & panel["ys_20"].notna()].sample(50, random_state=0)
    ey = panel.set_index(["date", "ticker"])["y_20"]
    for _, r in p.iterrows():
        assert r["ys_20"] == pytest.approx(r["y_20"] - ey[(r["date"], r["sector_etf"])])
    assert panel.loc[panel["ticker"].isin(stocks[:4]), "sector_etf"].dropna().eq("XLK").mean() > 0.8


def test_recovery_events_measure_days_back_to_pre_drop_price():
    """Flat at 100, falls to 85 in 5 days, climbs back to 100 thirty days after the low."""
    from scripts.run_recovery import find_events, recovered_within
    d = pd.bdate_range("2015-01-01", periods=400)
    p = np.full(400, 100.0)
    p[200:205] = [97, 94, 91, 88, 85]
    p[205:235] = np.linspace(85.5, 100.0, 30)             # back at 100 on index 234
    px = pd.concat([pd.DataFrame({"date": d, "ticker": "AAA", "adj_close": p}),
                    pd.DataFrame({"date": d, "ticker": "SPY", "adj_close": 100.0})])
    ev = find_events(px, "SPY", drop=0.10, window=5)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["date"] == d[203] and e["ev_drop"] == pytest.approx(0.88 - 1) and e["ev_market"] == 0
    assert e["days"] == 234 - 203                          # first close back at the pre-drop price (100)
    assert recovered_within(ev, 20, 10 ** 6).iloc[0] == 0 and recovered_within(ev, 60, 10 ** 6).iloc[0] == 1
    assert np.isnan(recovered_within(ev, 60, 203 + 30).iloc[0])     # window reaches the holdout -> unknown


@pytest.mark.slow
def test_recovery_model_has_no_skill_on_random_prices():
    from scripts.run_recovery import run as rec_run
    r = rec_run("synthetic", "volcluster", n_stocks=60, name="pytest_recovery", verbose=False)
    c = r["curve"].set_index("segment").loc["all crashes"]
    shares = [c[f"recovered_{h}d"] for h in (5, 10, 20, 60, 120, 250)]
    assert all(a <= b for a, b in zip(shares, shares[1:]))            # more time -> more recoveries
    assert 0.42 < r["metrics"].set_index("method").loc["model", "auc"] < 0.58   # random walk: nothing to learn


def test_sec_events_parse_8k_items():
    from src.ingest.sec_events import describe, parse_block
    block = {"form": ["8-K", "10-Q", "8-K", "4", "8-K/A"],
             "filingDate": ["2024-01-25", "2024-02-01", "2024-03-05", "2024-03-06", "2024-03-07"],
             "acceptanceDateTime": ["2024-01-25T21:05:11.000Z", "2024-02-01T12:00:00.000Z",
                                    "2024-03-05T13:30:00.000Z", "2024-03-06T10:00:00.000Z", "2024-03-07T22:00:00.000Z"],
             "items": ["2.02,9.01", "", "4.02,5.02", "", "8.01"],
             "accessionNumber": ["a1", "a2", "a3", "a4", "a5"]}
    df = parse_block(block, "TEST", 123)
    assert list(df["accession"]) == ["a1", "a3", "a5"]                     # only 8-K / 8-K/A kept
    assert df.loc[0, "ev_earnings"] and not df.loc[0, "ev_restatement"] and df.loc[0, "severity"] == 1
    assert df.loc[1, "ev_restatement"] and df.loc[1, "ev_management_change"] and df.loc[1, "severity"] == 3
    assert df.loc[2, "severity"] == 0 and str(df["accepted"].dt.tz) == "UTC"
    assert describe("4.02,5.02").startswith("earlier financial reports")   # most serious first
    assert describe("9.01") == "routine filing"


def test_universe_renames_and_recycled_symbols():
    """FB -> META becomes one continuous spell; a symbol whose prices start years late and run on after the
    stock left the index is rejected; a current member with a late price start is kept."""
    import scripts.build_universe as B
    hist = pd.DataFrame({"date": pd.to_datetime(["2014-01-02", "2022-06-08", "2022-06-09", "2024-01-02"]),
                         "tickers": ["FB,EP,IR,AAA", "FB,IR,AAA", "META,IR,AAA,TT", "META,IR,AAA,TT,FOXA"]})
    iv = B.to_intervals(B.apply_renames(hist), "2000-01-01")
    meta = iv[iv["source_ticker"] == "META"]
    assert len(meta) == 1 and pd.isna(meta["end"].iloc[0]) and meta["start"].iloc[0] == pd.Timestamp("2014-01-02")
    assert "FB" not in set(iv["source_ticker"])
    assert set(iv["source_ticker"]) >= {"TT", "IR"}                 # IR before Mar 2020 counted as TT
    assert iv[iv["source_ticker"] == "TT"]["start"].min() == pd.Timestamp("2014-01-02")
    tiingo = pd.DataFrame({"ticker": ["META", "EP", "AAA", "FOXA"],
                           "startDate": pd.to_datetime(["2012-05-18", "2018-02-01", "2000-01-03", "2023-06-01"]),
                           "endDate": pd.to_datetime(["2026-11-05"] * 4)})
    hist2 = pd.DataFrame({"date": pd.to_datetime(["2014-01-02", "2019-06-03", "2024-01-02"]),
                          "tickers": ["FB,EP,AAA,FOXA", "FB,AAA,FOXA", "META,AAA,FOXA"]})
    st = B.match(B.to_intervals(B.apply_renames(hist2), "2000-01-01"), tiingo).set_index("ticker")["status"]
    assert st["META"] == "covered" and st["AAA"] == "covered"
    assert st["EP"].startswith("symbol reused")        # left in 2019; its symbol's prices start 2018 and run to 2026
    assert st["FOXA"] == "covered (partial history)"   # current member whose prices start late: kept


def test_sec_matches_guard_against_reused_symbols_and_wrong_predecessors():
    """A symbol whose SEC registrant keeps filing years after the stock's prices ended belongs to another
    company today; and a predecessor number is only used if the SEC name matches."""
    from src.ingest.edgar import predecessor_ok, reused_sec_tickers
    filed = pd.Series(pd.to_datetime(["2026-10-30", "2026-11-01", "2014-11-05", "2026-08-01"]),
                      index=["CPWR", "AAPL", "OLDCO", "NOPX"])
    price = pd.Series(pd.to_datetime(["2014-12-15", "2026-11-11", "2014-12-01"]), index=["CPWR", "AAPL", "OLDCO"])
    assert reused_sec_tickers(filed, price) == ["CPWR"]          # no prices (NOPX) -> left alone
    price["CPWR"] = pd.Timestamp("2026-11-05")                    # price file continues under the reused symbol
    assert reused_sec_tickers(filed, price) == ["CPWR"]          # still left out: listed in SEC_WRONG_COMPANY
    assert predecessor_ok(1001039, "TWDC Enterprises 18 Corp.") and predecessor_ok(1001039, "WALT DISNEY CO/")
    assert predecessor_ok(1135152, "FMC TECHNOLOGIES INC") and not predecessor_ok(1135152, "Some Other Corp")
    assert predecessor_ok(999, "anything")                        # no expectation recorded -> allowed


def test_universe_audit_counts_members_with_prices():
    """Survivorship check: a member with no prices lowers the yearly coverage; a spell whose price file starts
    after the spell ended (later company under the same symbol) has zero coverage."""
    from scripts.audit_universe import coverage_by_year, relative_volume, spell_coverage
    days = pd.bdate_range("2001-01-02", "2003-12-31")
    px = pd.concat([pd.DataFrame({"ticker": "AAA", "date": days}),
                    pd.DataFrame({"ticker": "OLD", "date": pd.bdate_range("2003-07-01", "2003-12-31")})])
    u = pd.DataFrame({"ticker": ["AAA", "OLD"], "start": pd.to_datetime(["2000-01-01", "2000-01-01"]),
                      "end": pd.to_datetime([None, "2002-12-31"])})
    rep = pd.concat([u, pd.DataFrame({"ticker": ["GONE"], "start": pd.to_datetime(["2000-01-01"]),
                                      "end": pd.to_datetime([None])})])
    sc = spell_coverage(u, px).set_index("ticker")["coverage"]
    assert sc["AAA"] > 0.7 and sc["OLD"] == 0
    cy = coverage_by_year(rep, u, px).set_index("year")
    assert cy.loc[2002, "index_members"] == 3 and cy.loc[2002, "with_prices"] == 1    # OLD has no prices yet, GONE never
    assert cy.loc[2003, "index_members"] == 2 and cy.loc[2003, "with_prices"] == 1
    # an impostor history trades a tiny fraction of what real members trade
    pv = pd.concat([pd.DataFrame({"ticker": t, "date": days, "close": 50.0, "volume": v})
                    for t, v in [("AAA", 2e6), ("BBB", 3e6), ("CCC", 1e6), ("TINY", 1e3)]])
    uv = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "TINY"], "start": pd.to_datetime(["2000-01-01"] * 4),
                       "end": pd.to_datetime([None] * 4)})
    rv = relative_volume(uv, pv).set_index("ticker")["rel_volume"]
    assert rv["TINY"] < 0.05 and rv[["AAA", "BBB", "CCC"]].min() > 0.3
