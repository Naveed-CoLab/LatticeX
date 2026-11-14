"""Check the stock list against the price files: is every index member's price history really there?

    python -m scripts.audit_universe

Prints and saves
  results/universe_coverage_by_year.csv   each year: how many true S&P 500 members have prices (survivorship check)
  results/universe_spell_coverage.csv     each membership spell: share of its trading days that have a price
and lists the spells to look at by hand:
  - price file exists but (almost) no prices inside the membership dates (the symbol belongs to a later company),
  - trading volume in dollars far below the other members of the same years: an S&P 500 stock is heavily
    traded, so a tiny volume means the history under the symbol is some small company's (this found KMG, MIL,
    WEN, CSR and LSI on 13 Nov 2026; they are now in DROP_SPELLS in scripts/build_universe.py),
  - stocks that left the index while the price file runs on for years (check that the history is the member's).
"""
import numpy as np
import pandas as pd

from src.config import ROOT, path

FIRST = pd.Timestamp("2000-01-03")


def spell_coverage(u: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """u: ticker,start,end (stocks). px: ticker,date. One row per spell with the share of business days priced."""
    last = px["date"].max()
    by = {t: d.to_numpy() for t, d in px.groupby("ticker")["date"]}
    rows = []
    for t, a, b in zip(u["ticker"], u["start"], u["end"]):
        a0 = max(a, FIRST) if pd.notna(a) else FIRST
        b0 = b if pd.notna(b) else last
        d = by.get(t)
        if d is None:
            rows.append((t, a0, b0, pd.NaT, pd.NaT, 0.0))
            continue
        n = int(((d >= np.datetime64(a0)) & (d <= np.datetime64(b0))).sum())
        rows.append((t, a0, b0, pd.Timestamp(d.min()), pd.Timestamp(d.max()),
                     n / max(1, np.busday_count(a0.date(), b0.date()))))
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "prices_from", "prices_to", "coverage"])


def relative_volume(u: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Per spell: its median daily dollar volume as a share of the typical member's in the same years.
    px needs ticker, date, close, volume. Real members sit around 0.1 to 10; impostors far below 0.05."""
    p = px.assign(dv=px["close"] * px["volume"], year=px["date"].dt.year)
    m = p.merge(u[["ticker", "start", "end"]], on="ticker")
    m = m[(m["date"] >= m["start"].fillna(FIRST)) & (m["end"].isna() | (m["date"] <= m["end"]))]
    ty = m.groupby(["ticker", "start", "year"], dropna=False)["dv"].median().reset_index()
    ty["rel"] = ty["dv"] / ty.groupby("year")["dv"].transform("median")
    return (ty.groupby(["ticker", "start"], dropna=False).agg(rel_volume=("rel", "median"), first_year=("year", "min"),
            last_year=("year", "max")).reset_index().sort_values("rel_volume"))


def coverage_by_year(report: pd.DataFrame, u: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """report: every member spell (also the ones we could not match). Counted on the last trading day of June."""
    have = px.groupby("date")["ticker"].agg(set)
    rows = []
    for y in range(2001, int(have.index.max().year) + 1):
        i = have.index.searchsorted(pd.Timestamp(f"{y}-06-28"))
        if i >= len(have):
            break
        d = have.index[i]
        on = lambda f: f[(f["start"].fillna(FIRST) <= d) & (f["end"].isna() | (f["end"] >= d))]
        members, usable = len(on(report)), int(on(u)["ticker"].isin(have[d]).sum())
        rows.append({"year": y, "index_members": members, "with_prices": usable, "pct": round(100 * usable / members, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    u = pd.read_csv(ROOT / "config" / "universe.csv", parse_dates=["start", "end"])
    u = u[u["type"] == "stock"]
    rep = pd.read_csv(ROOT / "config" / "universe_report.csv", parse_dates=["start", "end"])
    px = pd.read_parquet(path("clean") / "prices.parquet", columns=["ticker", "date", "close", "volume"])
    out = ROOT / "results"
    sc = spell_coverage(u, px)
    sc.to_csv(out / "universe_spell_coverage.csv", index=False)
    cy = coverage_by_year(rep, u, px)
    cy.to_csv(out / "universe_coverage_by_year.csv", index=False)
    nofile = sc["prices_from"].isna()
    empty = ~nofile & (sc["coverage"] < 0.05)
    print(f"Stock spells in the list: {len(sc)} ({sc['ticker'].nunique()} tickers)")
    print(f"  no price file: {int(nofile.sum())}   file but (almost) no prices in the membership dates: {int(empty.sum())}   "
          f"partly covered (5-90%): {int(((sc['coverage'] >= 0.05) & (sc['coverage'] < 0.9)).sum())}   "
          f"covered (90%+): {int((sc['coverage'] >= 0.9).sum())}")
    print("\nTrue index members with prices, by year:")
    print(cy.to_string(index=False))
    test = cy[(cy["year"] >= 2012) & (cy["year"] <= 2024)]["pct"]
    print(f"\nTest years 2012-2024: {test.min():.0f}% to {test.max():.0f}% (average {test.mean():.0f}%); "
          f"training years before 2012: average {cy[cy['year'] < 2012]['pct'].mean():.0f}%")
    if empty.any():
        print("\nPrice file belongs to a later company (nothing inside the membership dates; harmless, unused):\n  "
              + " ".join(sorted(sc.loc[empty, "ticker"])))
    rv = relative_volume(u, px)
    low = rv[rv["rel_volume"] < 0.05]
    print("\nTrading volume far below other members (history is probably another company's -> add to DROP_SPELLS):"
          + ("\n  " + " ".join(f"{t}({a}-{b}: {r:.3f})" for t, a, b, r in
                               zip(low.ticker, low.first_year, low.last_year, low.rel_volume)) if len(low) else " none"))
    print("  lowest five kept: " + " ".join(f"{t}({r:.2f})" for t, r in
                                              zip(rv[rv["rel_volume"] >= 0.05].ticker[:5], rv[rv["rel_volume"] >= 0.05].rel_volume[:5])))
    cur = set(sc.loc[sc["end"] >= px["date"].max(), "ticker"])
    left = sc[(sc["end"] < px["date"].max() - pd.to_timedelta(150, unit="D")) & (sc["coverage"] >= 0.05)
              & (sc["prices_to"] > sc["end"] + pd.to_timedelta(1095, unit="D")) & ~sc["ticker"].isin(cur)]
    print(f"\nLeft the index but the price file runs on 3+ years ({len(left)}): check by hand that the history under "
          f"the symbol is the member's own\n  " + " ".join(f"{t}({a:%y}-{b:%y})" for t, a, b in zip(left.ticker, left.start, left.end)))
    print(f"\nSaved {out / 'universe_coverage_by_year.csv'} and universe_spell_coverage.csv")


if __name__ == "__main__":
    main()
