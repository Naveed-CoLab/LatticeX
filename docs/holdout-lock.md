# Holdout lock: the one-time final test on 2025–26

Written on 17 Nov 2026, **before any 2025–26 result was seen**. The Git commit that adds this file is the proof of when the rules were set. It must be pushed to GitHub before the test is run.

Since 6 Nov 2026, everything from 1 January 2025 has been kept out of every test and decision (`holdout_start` in `config/settings.yaml`, enforced in `src/folds.py`). This file fixes what is tested on that period, how, and what counts as passing. The results are published whatever they are.

## 1. What is final

| Part | Final version | Defined by |
|---|---|---|
| Ranking (primary) | **v1.0**: neural net 44 → 64 → 32 → 1, average of 3 seeds, trust factor *k*, 44 features (17 price and volume, 15 fundamentals, 12 sector), target = return minus SPY's with the extreme 1% clipped per day | `scripts/run_experiment.py`, `src/models/ensemble.py`, `config/settings.yaml` at this commit |
| Ranking (secondary) | Ensemble v2: XGBoost + neural net weighted by recent ranking skill. It was declared before v1.0 was chosen, so it is scored too | Same |
| Price ranges (primary) | **Range model v2**: volatility forecast by regression, separate up and down multipliers, earnings-aware; also the target day's high and low | `src/models/vol_ranges.py` at this commit |
| Price ranges (secondary) | Range model v1: blended volatility × one multiplier per side, earnings-aware | `scripts/holdout_ranges.py` |
| Not included | News sentiment, Kronos, TimesFM, macro inputs, ranking objective, quantile-XGBoost ranges | Tested and dropped on 2012–2024 (`decisions.md`) |

## 2. Data

The frozen research files, unchanged since the v1.0 runs:

| File | Size (bytes) | Content |
|---|---|---|
| `data/clean/prices.parquet` | 132,349,487 | Daily prices to 11 Nov 2026 (most stocks to 5 Nov) |
| `data/clean/fundamentals.parquet` | 7,726,023 | SEC financial statements, used from the day after each filing |
| `data/clean/sec_events.parquet` | 4,324,615 | 8-K filings (earnings dates for the ranges) |
| `config/universe.csv` | 26,297 | Audited S&P 500 membership ("universe v2"); the membership source runs to 18 Aug 2026 |

**Known limit:** stocks that joined the index after 18 Aug 2026 are missing. This affects the last weeks only.

## 3. Procedure

**Ranking.** For each horizon (5, 20, 60 trading days), one model is trained on every date whose target ended at least 60 trading days before 1 Jan 2025. Each weekly date from 2 Jan 2025 to the last date with a known outcome is then scored: about 93 dates at 5 days, 90 at 20 days and 82 at 60 days. There is no retraining inside the holdout. The commands are fixed in `scripts/kaggle_cloud.py` (`HOLDOUT_STEPS`):

```
python -m scripts.kaggle_cloud holdout --cpu
```

which runs, for h = 5, 20, 60:

```
python -m scripts.run_experiment --data real --fundamentals --sectors --models random_walk mlp ensemble --horizon h --holdout
```

**Ranges.** As in the product: weights and multipliers are fitted on windows that ended by 31 Dec 2024. Every weekly forecast from 1 Jan 2025 whose window has ended by the end of the data is scored.

```
python -m scripts.holdout_ranges
```

Both commands refuse to run a second time. A rerun is allowed only if a run failed for a technical reason, with that reason added to section 6 first.

## 4. How much this test can show

The holdout is short: about 90 weekly dates per horizon, against 650 in the walk-forward test. With v1.0's walk-forward skill (rank IC 0.015 at 5 days, 0.020 at 20 days) and its week-to-week spread (standard deviation of weekly rank IC 0.12 and 0.10), the **expected t-statistic on the holdout is only about 1.2, even if the edge is fully real**. The chance of a positive rank IC is about 88% in that case. A t of 2 or more is therefore not expected and is not the pass mark. The main question is the sign, and whether the size is in line with 2012–2024.

## 5. What counts as passing (decided now)

**Ranking (v1.0):**

| Result at 20 days (the main horizon) | Reading | What changes |
|---|---|---|
| Rank IC > 0 and t ≥ 2 | Edge confirmed out of sample | Nothing |
| Rank IC > 0, t < 2 | Consistent with the backtest, not proven on its own | Nothing; the report says both |
| Rank IC ≤ 0, t > −2 | Edge not seen in 2025–26 | The page and the report say so plainly next to the backtest |
| Rank IC < 0 and t ≤ −2 at any horizon | Evidence the edge is gone | Ranking taken off the page until a new version passes a new test |

The same table applies to 5 and 60 days as supporting evidence. Also reported: hit rate, Diebold-Mariano test against Random Walk, results by year (2025 and 2026), and the share of top-fifth and bottom-fifth stocks that beat SPY. The page's current statement is that the model is better at spotting laggards than winners. It holds if the bottom fifth beats SPY less often than the top fifth.

**Price ranges (v2):**

| Check | Pass band |
|---|---|
| "4 in 5" (80%) range held | 75–85% at each horizon |
| 50% and 90% ranges held | 45–55% and 85–95% |
| Day's high stayed below X, day's low stayed above Y | 85–95% each |

If a figure falls outside its band, the page's wording ("right 4 times in 5") will show the holdout figure next to it.

**Not allowed after seeing the results:** retuning anything on 2025–26, rerunning for a better number, swapping primary and secondary, or changing these bands.

## 6. Run log

| Date | What | Result folder | Notes |
|---|---|---|---|
| 17 Nov 2026 | Lock committed and pushed | Git commit `bc60f0a` | Before any 2025–26 number was seen |
| 18 Nov 2026 | Ranking holdout started on Kaggle (CPU) | `latticeiq-holdout-cpu` | |
| 18 Nov 2026 | Price ranges scored, first and only run | `results/20261118_2033_ranges_HOLDOUT` | **Pass** on every band (`results.md` section 8) |
