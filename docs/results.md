# Results

All results are walk-forward backtests on S&P 500 stocks, test years 2012–2024, scored weekly (~650 dates). The 2025–26 holdout has not been touched. Raw files: [`../results/`](../results/).

> **Which stock list each table uses.** Section 0 is measured on the **corrected S&P 500 list** (727 stocks, 13 Nov 2026). Every later section was measured earlier, on a list whose membership was frozen at January 2019 (bug in `build_universe.py`, see [`decisions.md`](decisions.md)); those tables are kept as the record of how the model was chosen. Price ranges and rank-group figures are being re-run on the corrected list.

## 0. v1.0 on the corrected stock list (13 Nov 2026)

727 stocks, 2.72 million stock-days, 44 features (price and volume 17, fundamentals 15, sector 12), about 298,000 weekly test forecasts. Runs `20261113_1855_…h5`, `20261113_1914_…h20`, `20261113_1932_…h60`. These runs were made before six wrong-company price histories were removed (one of them, LSI, inside test years 2012–14); the final "universe v2" run replaces them.

| Model | 5 days | 20 days | 60 days |
|---|---|---|---|
| **Neural net, 3 seeds (v1.0)**: rank IC (t) | **0.0141 (2.84)** | **0.0192 (2.95)** | **0.0202 (2.23)** |
| v1.0 hit rate | 50.4% | 50.6% | 51.0% |
| v1.0 positive years | 11 of 13 | 12 of 13 | 9 of 13 |
| Ensemble v2 (XGBoost + neural net): rank IC (t) | 0.0137 (2.48) | 0.0175 (2.30) | 0.0130 (1.26) |
| XGBoost alone: rank IC (t) | 0.0092 (1.61) | 0.0086 (1.11) | 0.0033 (0.31) |

Old list vs corrected list, v1.0 rank IC: 0.0136 → 0.0141 (5 days), 0.0180 → 0.0192 (20 days), 0.0229 → 0.0202 (60 days).

v1.0 rank IC by test year:

| Year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 days | 0.006 | 0.026 | 0.041 | 0.011 | 0.007 | 0.016 | 0.013 | 0.006 | 0.029 | 0.000 | 0.028 | −0.007 | 0.006 |
| 20 days | 0.034 | 0.010 | 0.058 | 0.005 | 0.032 | 0.026 | 0.014 | 0.003 | 0.025 | 0.005 | 0.031 | 0.014 | −0.009 |
| 60 days | 0.012 | −0.026 | 0.061 | −0.003 | 0.088 | −0.008 | 0.039 | 0.018 | 0.014 | 0.008 | 0.047 | 0.036 | −0.039 |

Validation score inside training vs test score (v1.0, average over the 13 folds): 0.018 vs 0.014 (5 days), 0.021 vs 0.019 (20 days), 0.023 vs 0.020 (60 days). The two are close, so the model is not simply memorising its training data.

- The skill did not come from the wrong stock list: 5 and 20 days are slightly stronger on the true index.
- 60 days is weaker and less steady (four negative years).
- 2024, the most recent test year, is negative at 20 and 60 days.
- Diebold-Mariano tests on forecast error do not beat the Random Walk at any horizon (p 0.10–0.46): the skill is in the ranking, not in the size of the forecast.

How these numbers compare with other tools and research: [`market-comparison.md`](market-comparison.md).

**How to read the metrics**

- **Hit rate:** share of stock-dates where the model got the direction vs SPY right (50% = coin flip).
- **Rank IC:** correlation between predicted and actual ranking of stocks on each date, averaged. 0.02–0.05 is considered useful in practice.
- **IC t-stat:** Newey-West t-statistic of the rank IC; about 2 means statistically significant.

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
- **Decision:** Kronos is not in the final FYP-1 model (no reliable gain, and it would need a daily GPU run). Interesting detail for FYP-2: Kronos did well in 2016 when XGBoost did badly, so it may help in an ensemble or after fine-tuning on US stocks.

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

## 6. Single-feature ICs (2010–2024)

All single features have |IC| ≤ 0.022. Strongest: book-to-market (−0.022), short-term reversal (`ret_5`, `ret_20` negative), free-cash-flow yield (+0.013), sector-relative momentum (+0.013), sector-relative earnings surprise (+0.011), accruals (−0.011). File: [`../results/factor_ic_2010_2024.csv`](../results/factor_ic_2010_2024.csv).

## 7. Pending

- ARIMA baseline
- Brackets with calibrated P(up) for 5 / 60 days

## Limitations

- Skill is small: the final ensemble just passes the 5% significance bar (t 2.04). This is normal for honest stock forecasting; large claimed accuracies usually come from look-ahead bias.
- Survivorship bias and the A–PCAR universe (see [`data-sources.md`](data-sources.md)).
- Transaction costs are not modelled yet.
