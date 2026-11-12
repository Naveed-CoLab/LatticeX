# Results

All results are walk-forward backtests: 466 S&P 500 stocks, test years 2012–2024, scored weekly (~650 dates, ~208,000 stock-dates). The 2025–26 holdout has not been touched. Raw files: [`../results/`](../results/).

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

## 3. Other horizons (fundamentals + sectors)

| Horizon | Hit rate | Rank IC | IC t-stat |
|---|---|---|---|
| 5 days | 50.5% | 0.0112 | 1.86 |
| 20 days | 50.7% | 0.0140 | 1.80 |
| 60 days | 50.9% | 0.0156 | 1.31 |

## 4. Price brackets (20-day)

| Metric | Model | Naive normal |
|---|---|---|
| 80% range: coverage (target 0.80) | **0.801** | 0.773 |
| 80% range: average width | 19.8% | 19.6% |
| 90% range: coverage (target 0.90) | **0.902** | 0.864 |
| 50% range: coverage (target 0.50) | 0.494 | 0.496 |
| Worst year (2020, COVID) 80% coverage | 0.725 | 0.762 |

Raw P(up) hit 54.7% vs a 57.7% "always up" base rate, so isotonic calibration was added (re-run pending).

## 5. Single-feature ICs (2010–2024)

All single features have |IC| ≤ 0.022. Strongest: book-to-market (−0.022), short-term reversal (`ret_5`, `ret_20` negative), free-cash-flow yield (+0.013), sector-relative momentum (+0.013), sector-relative earnings surprise (+0.011), accruals (−0.011). File: [`../results/factor_ic_2010_2024.csv`](../results/factor_ic_2010_2024.csv).

## 6. Pending

- TimesFM zero-shot comparison (code ready)
- ARIMA baseline
- Brackets with calibrated P(up) for 5 / 60 days

## Limitations

- Skill is small and not yet significant at the 5% level (t ≈ 1.8). This is normal for honest stock forecasting; large claimed accuracies usually come from look-ahead bias.
- Survivorship bias and the A–PCAR universe (see [`data-sources.md`](data-sources.md)).
- Transaction costs are not modelled yet.
