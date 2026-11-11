"""Market-environment features from FRED, one row per date, point-in-time.

Each value is taken as of the last observation whose `available_from` <= the feature date,
so CPI and unemployment only appear after their real publication day.

These are the same for every stock on a date. They cannot rank stocks on their own; they help
XGBoost learn WHEN other signals work (e.g. momentum fails after VIX spikes, low-beta stocks
win when credit spreads widen). The ablation study measures whether that helps.
"""
import pandas as pd

MACRO_FEATURES = [
    "vix", "vix_chg_20", "term_spread", "credit_spread", "credit_chg_60",
    "y10_chg_60", "cpi_yoy", "unrate_chg_3m",
]


def _asof(dates: pd.DatetimeIndex, obs: pd.DataFrame, name: str) -> pd.Series:
    o = obs.sort_values("available_from")[["available_from", "value"]].rename(columns={"value": name})
    left = pd.DataFrame({"date": dates}).sort_values("date")
    out = pd.merge_asof(left, o, left_on="date", right_on="available_from", direction="backward")
    return out.set_index("date")[name]


def macro_features(macro: pd.DataFrame, dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    get = lambda sid: macro[macro["series"] == sid]
    f = pd.DataFrame(index=dates)
    f["vix"] = _asof(dates, get("VIXCLS"), "v")
    f["term_spread"] = _asof(dates, get("T10Y2Y"), "v")
    f["credit_spread"] = _asof(dates, get("BAA10Y"), "v")
    y10 = _asof(dates, get("DGS10"), "v")
    # changes are computed on the already point-in-time daily series, so they stay leak-free
    f["vix_chg_20"] = f["vix"] - f["vix"].shift(20)
    f["credit_chg_60"] = f["credit_spread"] - f["credit_spread"].shift(60)
    f["y10_chg_60"] = y10 - y10.shift(60)

    # monthly first-release series: change vs 12 / 3 months earlier, by observation month
    cpi = get("CPIAUCSL").sort_values("date").copy()
    cpi["value"] = cpi["value"] / cpi["value"].shift(12) - 1
    f["cpi_yoy"] = _asof(dates, cpi.dropna(subset=["value"]), "v")
    un = get("UNRATE").sort_values("date").copy()
    un["value"] = un["value"] - un["value"].shift(3)
    f["unrate_chg_3m"] = _asof(dates, un.dropna(subset=["value"]), "v")
    return f[MACRO_FEATURES].rename_axis("date").reset_index()
