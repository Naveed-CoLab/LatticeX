"""How each 1-10 score did in the walk-forward test: the record shown next to a stock's score on the dashboard.

    python -m scripts.rank_scores

Score = the stock's rank among all members on a date, in tenths (10 = ranked highest by v1.0). For each score and
horizon, from the v1.0 walk-forward runs on 2012-2024 (results/*_real_stock_h<h>_fund_sect/predictions.parquet):
share of forecasts that beat SPY over the window, and the average excess return. Averages are taken per date first,
so every week counts once. Writes rank_scores_h<h>.csv next to each run (small, committed).
"""
import numpy as np
import pandas as pd

from src.config import path


def score_of(pct: pd.Series) -> pd.Series:
    """Rank as a fraction (1.0 = highest) -> score 1..10. Shared with the dashboard."""
    return np.ceil(pct * 10).clip(1, 10).astype(int)


def table(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred[pred["model"] == "mlp"].copy()
    p["score"] = score_of(p.groupby("date")["y_raw"].rank(pct=True))
    d = (p.groupby(["date", "score"])["y_true"]
         .agg(beat=lambda s: float((s > 0).mean()), excess="mean").reset_index())
    t = d.groupby("score").agg(beat_spy=("beat", "mean"), mean_excess=("excess", "mean"), dates=("date", "nunique"))
    t["forecasts"] = p.groupby("score").size()
    t["first"], t["last"] = f"{p['date'].min():%Y-%m-%d}", f"{p['date'].max():%Y-%m-%d}"
    return t.reset_index()


def main() -> None:
    for h in (5, 20, 60):
        runs = [r for r in sorted(path("results").glob(f"*_real_stock_h{h}_fund_sect")) if (r / "predictions.parquet").exists()]
        if not runs:
            print(f"{h} days: no v1.0 walk-forward run with predictions in results/")
            continue
        t = table(pd.read_parquet(runs[-1] / "predictions.parquet"))
        t.to_csv(runs[-1] / f"rank_scores_h{h}.csv", index=False)
        lo, hi = t.iloc[0], t.iloc[-1]
        print(f"{h:>2} days ({runs[-1].name}): score 1 beat SPY {lo['beat_spy']:.1%}, score 10 {hi['beat_spy']:.1%}")


if __name__ == "__main__":
    main()
