# Results

All results are walk-forward backtests on S&P 500 stocks, test years 2012–2024, scored weekly (~650 dates). The 2025–26 holdout has not been touched. Raw files: [`../results/`](../results/).

> **Which stock list each table uses.** Section 0 is measured on the **audited S&P 500 list, "universe v2"** (722 stocks, 14 Nov 2026). Every later section was measured earlier, on a list whose membership was frozen at January 2019 (bug in `build_universe.py`, see [`decisions.md`](decisions.md)); those tables are kept as the record of how the model was chosen. Price ranges are being re-run on universe v2.

## 0. v1.0 on universe v2 (14 Nov 2026) — final

722 stocks, 2.71 million stock-days, 44 features (price and volume 17, fundamentals 15, sector 12), about 298,000 weekly test forecasts. Runs `20261114_0752_…h5`, `20261114_0809_…h20`, `20261114_0827_…h60`.

| Model | 5 days | 20 days | 60 days |
|---|---|---|---|
| **Neural net, 3 seeds (v1.0)**: rank IC (t) | **0.0149 (3.07)** | **0.0197 (3.18)** | **0.0199 (2.23)** |
| v1.0 hit rate | 50.5% | 50.7% | 51.0% |
| v1.0 positive years | 12 of 13 | 10 of 13 | 10 of 13 |
| v1.0 vs Random Walk on forecast error (Diebold-Mariano p) | 0.47 | **0.02** | 0.41 |
| Ensemble v2 (XGBoost + neural net): rank IC (t) | 0.0141 (2.60) | 0.0173 (2.34) | 0.0123 (1.26) |
| XGBoost alone: rank IC (t) | 0.0094 (1.67) | 0.0067 (0.86) | 0.0016 (0.15) |

The same model on three versions of the stock list (v1.0 rank IC, t in brackets):

| Stock list | 5 days | 20 days | 60 days |
|---|---|---|---|
| Frozen at Jan 2019 (634 stocks, the bug) | 0.0136 (2.63) | 0.0180 (2.66) | 0.0229 (2.39) |
| Corrected, before the audit drops (727 stocks) | 0.0141 (2.84) | 0.0192 (2.95) | 0.0202 (2.23) |
| **Universe v2, audited (722 stocks)** | **0.0149 (3.07)** | **0.0197 (3.18)** | **0.0199 (2.23)** |

v1.0 rank IC by test year (universe v2):

| Year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 days | 0.012 | 0.019 | 0.042 | 0.005 | 0.008 | 0.019 | 0.011 | 0.013 | 0.032 | −0.001 | 0.029 | 0.001 | 0.003 |
| 20 days | 0.037 | 0.005 | 0.055 | 0.010 | 0.047 | 0.034 | 0.019 | −0.003 | 0.017 | −0.001 | 0.021 | 0.026 | −0.014 |
| 60 days | 0.006 | −0.030 | 0.066 | 0.003 | 0.083 | −0.005 | 0.030 | 0.008 | 0.023 | 0.024 | 0.037 | 0.030 | −0.029 |

Validation score inside training vs test score (v1.0, average over the 13 folds): 0.016 vs 0.015 (5 days), 0.020 vs 0.020 (20 days), 0.023 vs 0.020 (60 days). The two are close, so the model is not simply memorising its training data.

- The skill did not come from the wrong stock list: 5 and 20 days are slightly stronger on the true index, 60 days slightly weaker.
- 2024, the most recent test year, is negative at 20 and 60 days. 2019 and 2021 are about zero at 20 days.
- **How much a result moves by chance.** The last two rows of the stock-list table differ by five stocks out of 727, yet single years moved by up to 0.015 (20 days, 2016: 0.032 → 0.047) and the 13-year score by 0.0005–0.0008. Any add-on has to beat v1.0 by clearly more than that, in most years, to count.

### 0b. What the ranking is worth in plain numbers (universe v2)

Stocks sorted each week by the v1.0 forecast, test years 2012–2024. Excess = log return minus SPY's. Files `rank_groups_h*.csv` in each run folder.

| Rank group | 5d: beat SPY · mean excess | 20d: beat SPY · mean excess | 60d: beat SPY · mean excess |
|---|---|---|---|
| Bottom 20% | 48.4% · −0.15% | 47.7% · −0.60% | 47.4% · −1.37% |
| 20–40% | 49.1% · −0.10% | 48.9% · −0.36% | 48.1% · −1.02% |
| 40–60% | 49.6% · −0.08% | 49.5% · −0.31% | 48.9% · −0.89% |
| 60–80% | 49.8% · −0.04% | 49.8% · −0.17% | 49.5% · −0.71% |
| Top 20% | 50.2% · −0.03% | 49.9% · −0.14% | 50.1% · −0.62% |
| **Top minus bottom** | **+0.12%**, 11/13 years | **+0.46%**, 11/13 years | **+0.76%**, 10/13 years |
| All stocks (for reference) | 49.4% | 49.2% | 48.8% |

- The ordering is right at every step: each group does better than the one below it.
- The typical stock trails SPY slightly (an index is carried by a few big winners), so the top group only **matches** the market. The measurable value is **avoiding the bottom group**. Making the top group beat the market is the main accuracy goal.
- Before trading costs. Comparison with what other tools advertise: [`market-comparison.md`](market-comparison.md).

**How to read the metrics**

- **Hit rate:** share of stock-dates where the model got the direction vs SPY right (50% = coin flip).
- **Rank IC:** correlation between predicted and actual ranking of stocks on each date, averaged. 0.02–0.05 is considered useful in practice.
- **IC t-stat:** Newey-West t-statistic of the rank IC; about 2 means statistically significant.

### 0c. The model in daily use (trained 16 Nov 2026 on all data)

One training run, `python -m scripts.kaggle_cloud final-model`: 722 stocks, 2.7 million rows, prices 3 Jan 2000 to 5 Nov 2026, the v1.0 recipe unchanged. Saved as `models/latticeiq_v1.npz`.

| Horizon | Training rows | Trust factor k | Rank IC, mid-2021 to late 2026, model trained to mid-2021 |
|---|---|---|---|
| 5 days | 2,707,124 | 0.37 | +0.009 |
| 20 days | 2,698,770 | 0.20 | +0.001 |
| 60 days | 2,676,420 | 0.17 | +0.003 |

The last column is the check the recipe makes to set k: a copy trained on the first 80% of dates, never retrained, scored on the rest. It is harder than the walk-forward test above (retrained every year; rank IC 0.015 to 0.020 over 2012–2024) and it is not the holdout test, but it points the same way as the negative 2024 in the walk-forward runs: the edge has been thin in recent years and clearly positive only at 5 days. This is not a number to advertise and nothing is tuned from it.

## 1. Return ranking (20-day horizon): feature ablation

| Setup | Run folder | Hit rate | Rank IC | IC t-stat | Positive years |
|---|---|---|---|---|---|
| Random Walk | — | 50.0% | 0 | — | — |
| XGBoost, price only | `20261109_0940_real_stock_h20` | 50.4% | 0.0128 | 1.59 | — |
| + fundamentals | `20261109_0948_real_stock_h20_fund` | 50.7% | 0.0115 | 1.66 | — |
| **+ fundamentals + sectors** | `20261109_0956_real_stock_h20_fund_sect` | **50.7%** | **0.0140** | **1.80** | **11 / 13** |
| + ranking objective (dropped) | `20261109_1011_real_stock_h20_fund_sect_rank` | 50.3% | 0.0080 | 1.09 | — |
| + macro (earlier run, dropped) | — | 50.0% | 0.0118 | 0.99 | — |

Rank IC by year for the best setup: 2012 0.014 · 2013 0.006 · 2014 0.006 · 2015 0.026 · 2016 −0.020 · 2017 0.002 · 2018 0.035 · 2019 0.020 · 2020 0.040 · 2021 0.016 · 2022 0.030 · 2023 0.021 · 2024 −0.015.

## 2. Pretrained model: Kronos (zero-shot)

| Model (20-day) | Run folder | Hit rate | Rank IC | IC t-stat | Positive years |
|---|---|---|---|---|---|
| XGBoost, fundamentals + sectors | `20261112_1017_real_stock_h20_fund_sect` | 50.7% | 0.0140 | 1.80 | 11 / 13 |
| Kronos-small alone (no training) | same run | 50.1% | 0.0087 | 0.81 | 7 / 13 |
| XGBoost + Kronos forecasts as features | `20261112_1025_real_stock_h20_fund_sect_kron` | 50.8% | 0.0147 | 1.83 | 11 / 13 |

- Kronos on its own ranks stocks weakly and inconsistently (negative in 6 of 13 years) and its raw forecasts are far too large (RMSE 0.140 vs 0.074).
- Adding Kronos to XGBoost changes rank IC by +0.0007: better in 6 years, worse in 5. That is within noise, and Kronos features are not among the top features.
- **Decision:** Kronos is not in the model (no reliable gain, and it would need a daily GPU run). Interesting detail for later: Kronos did well in 2016 when XGBoost did badly, so it may help in an ensemble or after fine-tuning on US stocks.

### TimesFM 2.5 (zero-shot, weekly forecasts)

| Model (20-day) | Run folder | Hit rate | Rank IC | IC t-stat | Positive years |
|---|---|---|---|---|---|
| XGBoost, fundamentals + sectors | `20261112_1138_real_stock_h20_fund_sect` | 50.7% | 0.0140 | 1.80 | 11 / 13 |
| TimesFM 2.5 alone (no training) | same run | 49.95% | −0.0017 | −0.27 | 7 / 13 |
| XGBoost + TimesFM forecasts as features | `20261112_1147_real_stock_h20_fund_sect_tfm` | 50.8% | 0.0128 | 1.55 | 9 / 13 |

- TimesFM has no ranking skill: its forecasts barely differ between stocks (spread 1.6%), close to a random walk.
- Adding it to XGBoost makes the model slightly worse. **Decision: OUT.** See [`decisions.md`](decisions.md#key-finding-pretrained-models-vs-our-model).

## 3. Ensemble (20-day, fundamentals + sectors)

Run `20261112_1227_real_stock_h20_fund_sect`; combinations in `ensemble_combinations.csv` (same predictions, re-combined).

| Model / combination | Rank IC | IC t-stat | Hit rate | Positive years | IC 2012–17 | IC 2018–24 |
|---|---|---|---|---|---|---|
| XGBoost | 0.0140 | 1.80 | 50.7% | 11 | 0.0057 | 0.0209 |
| LightGBM | 0.0080 | 1.00 | 50.5% | 9 | 0.0001 | 0.0147 |
| MLP (neural net) | 0.0149 | 1.96 | 50.4% | 10 | 0.0240 | 0.0070 |
| Ridge | 0.0042 | 0.41 | 50.1% | 7 | 0.0017 | 0.0065 |
| **XGBoost + MLP** | **0.0168** | **2.04** | 50.5% | **11** | **0.0171** | **0.0163** |
| XGBoost + LightGBM + MLP | 0.0143 | 1.71 | 50.6% | 10 | 0.0116 | 0.0165 |
| All four | 0.0130 | 1.40 | 50.5% | 10 | 0.0103 | 0.0153 |

Correlation of the models' stock rankings: XGBoost–LightGBM 0.86, XGBoost–MLP 0.43, MLP–Ridge 0.55. Combining models that disagree (tree + neural net) is what helps. **Final model: XGBoost + MLP.**

## 3b. Full universe (634 stocks) and ensemble weighting

Run `20261112_1342_real_stock_h20_fund_sect` (634 stocks, 279,757 stock-dates). Comparison in `ensemble_weighting_comparison.csv`.

| Universe | Model | Rank IC | t-stat | Positive years | 2012–17 | 2018–24 |
|---|---|---|---|---|---|---|
| 466 stocks | XGBoost | 0.0140 | 1.80 | 11 | 0.0057 | 0.0209 |
| 466 stocks | MLP | 0.0149 | 1.96 | 10 | 0.0240 | 0.0070 |
| 466 stocks | Equal-weight XGBoost + MLP | 0.0168 | 2.04 | 11 | 0.0171 | 0.0163 |
| 466 stocks | **Trust-weighted XGBoost + MLP** | **0.0199** | **2.36** | **12** | 0.0226 | 0.0173 |
| 634 stocks | XGBoost | 0.0064 | 0.79 | 10 | 0.0003 | 0.0115 |
| 634 stocks | MLP | 0.0192 | 2.97 | 10 | 0.0262 | 0.0128 |
| 634 stocks | Equal-weight XGBoost + MLP | 0.0147 | 1.93 | 10 | 0.0147 | 0.0143 |
| 634 stocks | **Trust-weighted XGBoost + MLP (final)** | **0.0170** | **2.25** | **11** | 0.0240 | 0.0106 |

- On the 125 newly added stocks alone the neural net scores 0.0168 (t 2.0): the skill carries over to stocks it was never tuned on.
- Note on honesty: several ways of combining the two models were compared on the same test years. The trust-weighted rule was picked because it uses only training-year information and is the most stable across both universes; the locked 2025–26 holdout is the final check.

## 3c. All horizons on the full universe (634 stocks), trust-weighted ensemble v1

| Horizon | Run | XGBoost | MLP | Ensemble (k-weighted) |
|---|---|---|---|---|
| 5 days | `20261112_1409_real_stock_h5_fund_sect` | 0.0099 (t 1.69) | 0.0112 (t 2.20) | **0.0120 (t 2.10)** |
| 20 days | `20261112_1342_real_stock_h20_fund_sect` | 0.0064 (t 0.79) | **0.0192 (t 2.97)** | 0.0170 (t 2.25) |
| 60 days | `20261112_1422_real_stock_h60_fund_sect` | 0.0087 (t 0.82) | **0.0254 (t 2.81)**, hit 51.5% | 0.0161 (t 1.72) |

On the full universe XGBoost is weak at every horizon, and weighting by k (a calibration slope) still gave it too much say at 20 and 60 days. → Ensemble v2 (below): weights from each model's *ranking* skill on the latest training years, and 3 MLP seeds averaged.

## 3d. Final run — v1.0 (634 stocks, all horizons)

| Horizon | XGBoost | **MLP, 3 seeds (v1.0)** | Ensemble v2 (rank-skill weights) |
|---|---|---|---|
| 5 days | 0.0099 (t 1.69) | **0.0136 (t 2.63)**, hit 50.5% | 0.0135 (t 2.37) |
| 20 days | 0.0064 (t 0.79) | **0.0180 (t 2.66)**, hit 50.6% | 0.0159 (t 1.99) |
| 60 days | 0.0087 (t 0.82) | **0.0229 (t 2.39)**, hit 51.3% | 0.0176 (t 1.69) |

Runs `20261112_1525_…h5`, `20261112_1541_…h20`, `20261112_1558_…h60`. Why the MLP is primary: [`decisions.md`](decisions.md#locked-model--latticeiq-v10-12-nov-2026).

## 4. Other horizons (earlier, 466 stocks, XGBoost only) (fundamentals + sectors)

| Horizon | Hit rate | Rank IC | IC t-stat |
|---|---|---|---|
| 5 days | 50.5% | 0.0112 | 1.86 |
| 20 days | 50.7% | 0.0140 | 1.80 |
| 60 days | 50.9% | 0.0156 | 1.31 |

## 5. Price brackets (20-day)

| Metric | Model | Naive normal |
|---|---|---|
| 80% range: coverage (target 0.80) | **0.801** | 0.773 |
| 80% range: average width | 19.8% | 19.6% |
| 90% range: coverage (target 0.90) | **0.902** | 0.864 |
| 50% range: coverage (target 0.50) | 0.494 | 0.496 |
| Worst year (2020, COVID) 80% coverage | 0.725 | 0.762 |

Raw P(up) hit 54.7% vs a 57.7% "always up" base rate, so isotonic calibration was added (see 5b: it did not fix it).

## 5b. Price ranges on the full universe (634 stocks, all horizons) — final

Runs `20261112_1625_…h5_fund_brackets`, `20261112_1621_…h20_fund_brackets`, `20261112_1630_…h60_fund_brackets`.

| Horizon | 80% coverage (model / naive) | 90% coverage (model / naive) | 50% coverage (model / naive) | Avg width of 80% range |
|---|---|---|---|---|
| 5 days | **0.798** / 0.793 | **0.900** / 0.875 | **0.497** / 0.526 | 9.4% |
| 20 days | **0.803** / 0.775 | **0.904** / 0.866 | **0.498** / 0.497 | 19.8% |
| 60 days | **0.802** / 0.765 | **0.908** / 0.860 | 0.489 / 0.484 | 35.1% |

Weak year: 2020 (20-day 80% coverage 0.715): ranges were too narrow going into the COVID crash.

**P(up) from the range model: rejected.** Brier score (lower is better) vs the plain historical up-rate:

| Horizon | Model P(up) | Historical rate ("base rate") |
|---|---|---|
| 5 days | 0.2513 | **0.2478** |
| 20 days | 0.2513 | **0.2446** |
| 60 days | 0.2396 | **0.2378** |

It is worse at every horizon, has no skill at telling stocks apart on a given day (rank IC −0.009), and its extreme values are wrong (20-day: predicted 10% chance up → 57% actually rose). Whether stocks rise over the next weeks is mostly the market's direction, which these features do not predict.

## 5c. What the v1.0 ranking is worth in plain numbers

Stocks sorted each week by the v1.0 forecast (3-seed MLP), test years 2012–2024. Excess = log return minus SPY's. Files `rank_groups_h*.csv` in each v1.0 run folder.

| Rank group | 5d: beat SPY · mean excess | 20d: beat SPY · mean excess | 60d: beat SPY · mean excess |
|---|---|---|---|
| Bottom 20% | 48.6% · −0.14% | 48.0% · −0.54% | 47.6% · −1.29% |
| 20–40% | 48.7% · −0.11% | 48.9% · −0.38% | 47.8% · −1.14% |
| 40–60% | 49.6% · −0.05% | 49.2% · −0.31% | 48.7% · −0.91% |
| 60–80% | 50.0% · −0.05% | 49.8% · −0.16% | 49.7% · −0.67% |
| Top 20% | 50.0% · −0.03% | 49.9% · −0.14% | **50.3%** · −0.49% |
| **Top minus bottom** | **+0.11%**, 12/13 years | **+0.40%**, 12/13 years | **+0.80%**, 11/13 years |

- The ordering is right and steady: each group does better than the one below it, in almost every year.
- The typical stock trails SPY slightly (an index is carried by a few big winners, and log returns of a volatile stock average lower), so even the top group only matches SPY. The measurable value today is **avoiding the bottom group**, not picking market-beaters.
- The gap is before trading costs.

## 5d. Can the price ranges be narrower at the same hit rate? (17 Nov 2026, universe v2)

Replayed walk-forward on 2012–2024, weekly, about 318,000 forecasts per horizon (the study run, which also included the 18 index and sector funds; the stock-only replay in the product code is below): for each test year every weight and multiplier comes only from windows that ended before that year. Interval score: width plus a penalty for each miss (lower is better). "Years better": test years in which the interval score beat the current method.

| 80% range | 5 days | 20 days | 60 days |
|---|---|---|---|
| **Current** (blended volatility, earnings-aware): held · average width | 79.3% · 9.1% | 78.8% · 18.8% | 79.1% · 33.1% |
| **Candidate** (volatility forecast by regression + uneven up/down): held · average width | 79.0% · 8.9% | 78.7% · 18.1% | 79.1% · 31.5% |
| Narrower by | 2.4% | 3.7% | 5.0% |
| Interval score better by · years better | 1.0% · 11 of 13 | 1.7% · 11 of 13 | 2.2% · 10 of 13 |
| Worst year held, current → candidate | 75.6% → 77.2% | 75.7% → 76.5% | 73.3% → 75.6% |
| Earnings windows held, current → candidate | 77.3% → 77.4% | 78.2% → 78.5% | 79.3% → 79.2% |

The candidate predicts each stock's volatility over the window with one regression, refitted each year on past windows only: the four current measures plus one-week volatility, the market's mood (SPY's recent volatility against its own year), whether earnings fall in the window, and the stock's own typical earnings move (median of its last 8). The ranges then use the multipliers that would have held the price, separately for the upside and the downside, because prices drift up: the middle of the range is about +0.3% / +1.2% / +3.4% above today's price, and the close ended above that middle 51% of the time.

What each part adds (20 days, width change against current): the regression 3.3% (market mood is most of it: without it 2.4%), the stock's own earnings moves about 0%, one-week volatility about 0.3%, uneven up/down a further 0.4%.

Tested and rejected:

| Idea | Result | Why rejected |
|---|---|---|
| Separate multipliers by volatility level and market mood (18 groups) | 2.8–5.9% narrower, but held 78.6% / 78.7% at 20 / 60 days and a worse interval score than the candidate | Narrower by missing more; too many groups for the data |
| Blend in the last year's multipliers (adaptive) | Interval score worse at 20 and 60 days | Reacts to last year's market, which is often over |
| Separate multipliers for stocks with under two years of history | Those stocks held 77.7% / 76.7% / 74.3% (worse than without) | Too few such stocks per year to fit reliably. Without it they hold 78.6% / 77.8% / 77.6%, about one point under the rest |

**The target day's high and low** (new output, same replay): "the day's high stays below X" and "the day's low stays above Y", each meant to hold 9 times in 10.

| | 5 days | 20 days | 60 days |
|---|---|---|---|
| High stayed below X | 89.4% | 88.9% | 88.7% |
| Low stayed above Y | 89.6% | 90.0% | 90.5% |
| Both held | 79.0% | 78.9% | 79.2% |
| Typical X, Y against today's price | +5.6%, −5.2% | +10.9%, −9.4% | +19.4%, −14.4% |

Both bounds are close to the 90% they aim for, so they can be shown. They are wider than the close range because a day's high and low reach beyond its close.

**Now in the product (17 Nov).** The candidate is `src/models/vol_ranges.py` v2. Replayed in the product code on stocks only (the same 306,000 forecasts per horizon as v1), with v1 scored on the same rows:

| 80% range, 2012–2024 | 5 days | 20 days | 60 days |
|---|---|---|---|
| v1: held · average width | 79.5% · 9.2% | 79.0% · 19.1% | 79.1% · 33.6% |
| **v2: held · average width** | **79.2% · 9.0%** | **78.8% · 18.4%** | **79.1% · 32.0%** |
| Narrower by · interval score better by | 2.2% · 1.0% | 3.4% · 1.7% | 4.9% · 2.2% |
| Years with a better interval score | 11 of 13 | 11 of 13 | 10 of 13 |
| Worst year held, v1 → v2 | 75.8% → 77.4% | 75.9% → 76.6% | 73.3% → 75.6% |
| Earnings windows held, v1 → v2 | 77.3% → 77.4% | 78.2% → 78.4% | 79.3% → 79.2% |
| 50% and 90% ranges held (v2) | 49.7%, 89.3% | 48.6%, 89.2% | 48.8%, 89.4% |
| Day's high below X · low above Y · both | 89.6% · 89.6% · 79.2% | 89.0% · 90.1% · 79.1% | 88.7% · 90.5% · 79.2% |

**Reading this.** The gain is real but small: one to two percentage points of width out of ten to thirty, steady across years, with the same hit rate. Most of a stock's swing cannot be forecast from its past prices; the biggest known improvement would be the options market's own volatility forecast (implied volatility), for which no free history was found.

## 5e. News sentiment vs v1.0 (17 Nov 2026, universe v2) — not adopted

v1.0 with and without the 8 news inputs (FinBERT tone of 767,435 Benzinga articles, 2015 on), same code, same seeds, run on Kaggle's CPU, test years 2018–2024 (news starts in 2015; earlier years are needed to learn from), weekly dates. Runs: `20261117_1501_…h5…from2018`, `1536_…h5…news`, `1551_…h20…`, `1605_…h20…news`.

| Rank IC by test year | 5 d: v1.0 | 5 d: + news | 20 d: v1.0 | 20 d: + news |
|---|---|---|---|---|
| 2018 | 0.012 | 0.012 | 0.020 | 0.024 |
| 2019 | 0.013 | 0.012 | −0.003 | −0.003 |
| 2020 | 0.028 | 0.037 | 0.015 | 0.017 |
| 2021 | −0.006 | −0.000 | 0.003 | 0.011 |
| 2022 | 0.025 | 0.025 | 0.025 | 0.020 |
| 2023 | 0.002 | 0.007 | 0.023 | 0.036 |
| 2024 | 0.005 | 0.007 | −0.015 | −0.019 |
| **2018–2024** | **0.0111** (t 1.51) | **0.0142** (t 2.02) | **0.0099** (t 1.10) | **0.0125** (t 1.40) |

| Is the gain real? | 5 days | 20 days |
|---|---|---|
| News minus v1.0, mean of weekly rank ICs | +0.0031 | +0.0027 |
| Paired t (Newey-West, same dates) | 1.29 | 0.83 |
| Weeks in which news ranked better | 51.0% | 54.6% |
| Years better · same · worse | 4 · 2 · 1 | 4 · 0 · 3 |
| Same model run twice (CPU here vs the GPU run of 14 Nov): mean difference | −0.0015 | +0.0003 |
| Same model run twice: largest difference in one year | 0.005 | 0.004 |
| Top fifth minus average stock, per window (v1.0 → news) | +0.044% → +0.039% | +0.099% → +0.097% |
| Bottom fifth minus average stock, per window (v1.0 → news) | −0.086% → −0.086% | −0.186% → −0.210% |
| Rank correlation of the two models' forecasts | 0.84 | 0.81 |

**Verdict.** News helps a little in the years with big news-driven moves (2020, 2021, 2023), but the gain is about twice the difference between two runs of the very same model and well short of the bar (paired t about 2, most years better at both horizons). At 20 days it is worse in 3 of 7 years. The top and bottom fifths do not improve. **v1.0 stays; news is not added.** By the rule set before the test, a stronger language model for sentiment is not tried either. The download, scoring and features stay in the code for a later retest with more years or a licensed feed.

## 8. Holdout 2025–26: the one-time final test (rules in `holdout-lock.md`, locked 17 Nov)

### 8a. Price ranges — pass (18 Nov 2026)

Range model v2 as in the product, fitted on windows that ended by 31 Dec 2024, scored on every weekly forecast from 8 Jan 2025 whose window ended by the end of the data. Run `20261118_2033_ranges_HOLDOUT`.

| v2 (primary) | 5 days | 20 days | 60 days | Pass band |
|---|---|---|---|---|
| Forecasts | 47,323 | 45,798 | 41,614 | |
| "4 in 5" (80%) range held | **78.9%** | **78.1%** | **77.8%** | 75–85% ✅ |
| 50% range held | 48.5% | 48.0% | 47.2% | 45–55% ✅ |
| 90% range held | 89.2% | 88.9% | 88.9% | 85–95% ✅ |
| Day's high stayed below X | 88.7% | 88.8% | 89.1% | 85–95% ✅ |
| Day's low stayed above Y | 89.6% | 88.8% | 88.5% | 85–95% ✅ |
| 80% range held in earnings windows · other windows | 76.5% · 79.3% | 76.8% · 79.0% | 77.7% · 82.7% | |
| Average width of the 80% range | 10.7% | 21.7% | 36.8% | |
| Close ended above the range's middle | 49.2% | 47.3% | 44.6% | (50% ideal) |

| v1 (secondary), same forecasts | 5 days | 20 days | 60 days |
|---|---|---|---|
| 80% range held · width | 79.9% · 11.1% | 79.9% · 22.9% | 80.4% · 39.9% |
| 50% · 90% held | 49.6% · 90.1% | 49.8% · 90.1% | 50.6% · 90.4% |

80% range held by quarter (v2): 5 days 70–84%, 20 days 68–89%, 60 days 70–86%. The weakest quarters were 2025 Q4 and 2026 Q1; the partial last quarter (2026 Q4, 619 forecasts at 20 days) held 50%.

**Reading.** v2 passes every pre-set band, and its 2025–26 hit rates are close to its 2012–2024 replay (79.2% / 78.8% / 79.1%). Two honest points. First, v1 was better calibrated on the holdout (about 80% at every horizon) while v2 was 1.0 / 1.8 / 2.6 points under target, in exchange for ranges 3.2% / 5.2% / 7.7% narrower. Second, the main source of v2's shortfall is its upward tilt: it centres the range on the drift of 2001–2024, and in 2025–26 the close ended above the middle only 44.6% of the time at 60 days. By the lock's rules nothing is changed on the basis of this; it is recorded as the first thing to look at in a later version.

### 8b. Ranking — running

## 6. Single-feature ICs (2010–2024)

All single features have |IC| ≤ 0.022. Strongest: book-to-market (−0.022), short-term reversal (`ret_5`, `ret_20` negative), free-cash-flow yield (+0.013), sector-relative momentum (+0.013), sector-relative earnings surprise (+0.011), accruals (−0.011). File: [`../results/factor_ic_2010_2024.csv`](../results/factor_ic_2010_2024.csv).

## 7. Pending

- ARIMA baseline
- Volatility range model vs quantile XGBoost on universe v2 (interval score)
- Crash risk, recovery, sector-neutral target on real data

## Limitations

- Skill is small: v1.0's rank IC is 0.015–0.020 (t 2.2–3.2). This is normal for honest stock forecasting; large claimed accuracies usually come from look-ahead bias.
- The edge has been weak recently: 2024 is negative at 20 and 60 days, and the daily model's own check on mid-2021 to late 2026 is close to zero except at 5 days (section 0c).
- Survivorship bias in the early training years: 67% of true members have prices before 2012 (91% on average in the test years). See [`data-sources.md`](data-sources.md).
- Transaction costs are not modelled yet.
