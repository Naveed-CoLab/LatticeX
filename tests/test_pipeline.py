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
