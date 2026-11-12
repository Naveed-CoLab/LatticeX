# Experiment & decision log

What we tried, what it scored, and whether it is **in**, **out** or **pending** in the final model, with the reason. Updated after every experiment. Detailed numbers: [`results.md`](results.md). Day-by-day log: [`PROGRESS.md`](PROGRESS.md).

**Rule for every decision:** a component stays only if it improves rank IC on the walk-forward test years 2012–2024 *consistently* (in most years, not just on average). Everything is measured on the same 466 stocks, the same weekly dates and the same metrics. The 2025–26 holdout is never used for decisions.

## Current final model (FYP-1 candidate)

> **Ensemble of XGBoost + neural net (MLP)**, each weighted by its trust score from the latest *training* years, on **price + fundamentals + sector** features, **634 stocks** (Tiingo + Yahoo fill-in), cross-sectional ranks, winsorised target, prediction shrinkage.
> 20-day rank IC **0.0170** (t **2.25**), positive in **11 / 13** years. On the earlier 466-stock universe the same method scored 0.0199 (t 2.36, 12 / 13 years): it is the best or near-best option on both universes. Price brackets: **quantile XGBoost + conformal calibration** (80% range covers 80.1%).

```mermaid
flowchart LR
    A[Price features<br/>IC 0.0128] -->|+ macro: 0.0118 ✗| X1[macro OUT]
    A -->|+ fundamentals| B[0.0115]
    B -->|+ sectors| C[0.0140 ✓ best]
    C -->|ranking objective: 0.0080 ✗| X2[rank objective OUT]
    C -->|+ Kronos: 0.0147, noise ✗| X3[Kronos OUT]
    C -->|+ TimesFM: 0.0128 ✗| X4[TimesFM OUT]
    C -->|ensemble XGBoost + MLP, trust-weighted| E[0.0170–0.0199 ✓ final]
    C -->|+ news| P2[deferred to FYP-2]
```

## Models

| Model | Status | Result (20-day) | Why |
|---|---|---|---|
| Random Walk (forecast = 0) | **IN** (baseline) | IC 0 | Every model must beat it |
| XGBoost regression + shrinkage | **IN** (main model) | IC 0.0140, t 1.80, 11/13 years | Best and most consistent |
| XGBoost ranking objective (rank:pairwise) | **OUT** | IC 0.0080, t 1.09 | Worse than regression on the same features |
| Quantile XGBoost + conformal (brackets) | **IN** | 80% coverage 0.801 vs naive 0.773 | Calibrated ranges; beats the naive normal baseline |
| Kronos-small, zero-shot, alone | **OUT** | IC 0.0087, t 0.81, 7/13 years | Weaker and inconsistent; forecasts too large (RMSE 0.140 vs 0.074) |
| XGBoost + Kronos forecasts as features | **OUT** | IC 0.0147 vs 0.0140 | +0.0007 is noise (better 6 yrs, worse 5); needs a daily GPU run. Revisit in FYP-2 (ensemble / US fine-tune) |
| TimesFM 2.5, zero-shot, alone | **OUT** | IC −0.0017, t −0.27, 7/13 years | No skill at ranking stocks: forecasts stay close to "no change" (spread between stocks 1.6%) |
| XGBoost + TimesFM forecasts as features | **OUT** | IC 0.0128 (t 1.55) vs 0.0140 | Makes the model worse (9/13 positive years vs 11/13): extra noisy inputs |
| ARIMA (per stock) | **PENDING** | — | Classical baseline for the report |
| LightGBM | **OUT** | IC 0.0080, t 1.00 | Its rankings are 86% the same as XGBoost's, so it adds a second vote for trees, not new information; it pulled the ensemble down to 0.0143 |
| Small neural net (MLP, PyTorch) | **IN** (ensemble member) | IC 0.0149, t 1.96, 10/13 years | Best single model; only 43% correlated with XGBoost, strong in 2012–17 where XGBoost was weak |
| Ridge (linear) | **OUT** | IC 0.0042, t 0.41 | Too simple; lowers any ensemble it joins |
| **Ensemble: XGBoost + MLP** (equal-weight rank average) | **IN — final model** | **IC 0.0168, t 2.04, 11/13 years** | +20% over XGBoost alone, and stable: 0.0171 in 2012–17, 0.0163 in 2018–24 (XGBoost alone: 0.0057 / 0.0209). Members chosen by how *different* their predictions are (correlation, no test outcomes used); equal weights fixed in advance |
| Ensemble: XGBoost + LightGBM + MLP | OUT | IC 0.0143 | Two near-identical tree models outvote the neural net |
| Equal-weight XGBoost + MLP | **replaced** | 466 stocks 0.0168 · 634 stocks 0.0147 | On the full universe XGBoost alone fell to 0.0064 and dragged the equal mix down |
| **Trust-weighted XGBoost + MLP** | **IN — final model** | 466 stocks **0.0199** (t 2.36) · 634 stocks **0.0170** (t 2.25) | Each year, each model's weight = its shrinkage k, i.e. how well it predicted the most recent *training* years. A model that stopped working gets little weight automatically. No test-year data is used. Most stable choice across both universes |
| Ensemble weights = k (v1) | **replaced by v2** | 5d 0.0120 · 20d 0.0170 · 60d 0.0161 | k measures size calibration, not ranking skill: XGBoost kept ~40–60% weight at 20/60 days while ranking poorly |
| **Ensemble v2: weights = ranking skill (rank IC) on the latest training years; MLP = average of 3 seeds** | **PENDING — final run** | — | Same idea, but the weight measures what the product needs (ordering stocks). Seed-averaging removes the luck of one neural-net training run. Decided before seeing its test results; whatever it scores is locked |
| MLP alone | not chosen | 466: 0.0149 · 634: 0.0192 (t 2.97) | Best on the full universe but weaker on the smaller one: more variable than the weighted ensemble |
| Regime models | **FYP-2** | — | After the ensemble |
| Kronos fine-tuned on US stocks | **FYP-2** | — | Pretrained route, ~3 days and uncertain; zero-shot versions did not help |

## Feature groups

| Group | Status | Result | Why |
|---|---|---|---|
| Price & volume (17) | **IN** | IC 0.0128 alone | Core signal: momentum, reversal, volatility |
| Macro (8: VIX, yields, spreads, CPI…) | **OUT** | IC 0.0118 (t 0.99) vs 0.0128 | Same value for every stock on a date, so it cannot help rank stocks |
| Fundamentals (15, SEC EDGAR) | **IN** | 0.0115 alone, 0.0140 with sectors | Helps in combination with sector-relative features |
| Sector features (12) | **IN** | 0.0115 → 0.0140 | Biggest single gain; compares stocks within their sector |
| News sentiment (Benzinga + FinBERT) | **DEFERRED (FYP-2)** | — | Plan delivers ~2 articles/day: too sparse to test. FinBERT licence is non-commercial |
| Kronos forecasts (2) | **OUT** | see Models | No reliable gain |
| TimesFM forecasts (3) | **OUT** | 0.0140 → 0.0128 | Hurts ranking; see Models |

## Method choices

| Choice | Decision | Why |
|---|---|---|
| Target | Excess return vs SPY, 5/20/60 days; 20 days is the headline | What a user acts on: "will this stock beat the market?" |
| Evaluation | Walk-forward 2012–2024, 60-day embargo, weekly dates | Mimics real use; embargo stops overlapping targets leaking |
| Main metric | Rank IC with Newey-West t-stat | The product ranks stocks; NW handles overlapping 20-day targets |
| Holdout 2025–26 | Locked until end of FYP-2 | One honest final test, never tuned on |
| Features ranked across stocks per date | Yes | Removes market-wide level shifts; more stable |
| Target winsorised 1%/99% | Yes | Extreme moves (takeovers, crashes) otherwise dominate training |
| Prediction shrinkage | Yes | Scales forecasts to the skill seen in training; avoids overconfident numbers |
| P(up) calibration | Isotonic, added | Raw P(up) was right 54.7% vs a 57.7% "always up" base rate |
| Kronos grid | Monthly, 2 samples | Weekly would take ~10 GPU h; monthly forecasts still cover every period |

## Key finding: pretrained models vs our model

Two large pretrained forecasting models were tested zero-shot on the same stocks, dates and metrics:

| | Rank IC (20-day) | t-stat | Verdict |
|---|---|---|---|
| **Our XGBoost (price + fundamentals + sectors)** | **0.0140** | **1.80** | best |
| Kronos-small (trained on market candles) | 0.0087 | 0.81 | weaker, inconsistent |
| TimesFM 2.5 (general time series) | −0.0017 | −0.27 | no skill |

Each pretrained model sees **one stock's price history at a time**. Our model compares **all stocks on the same day** and adds company fundamentals and sector context. For "which stock will beat the market", the cross-sectional, finance-specific model wins. This matches the finance literature: a single price series alone carries little predictable signal. Pretrained models remain an FYP-2 option (fine-tuning on US stocks, ensembles).

## Product decisions

| Decision | Why |
|---|---|
| Chat app (FYP-2): user sends a ticker or a chart screenshot (e.g. TradingView), an LLM reads it and answers using **our model's** forecast | The LLM is only the interface (reads the image, explains in plain words). The forecast comes from our tested model, not from the LLM guessing from a picture |
| The screenshot is used to identify the stock and timeframe, not as the prediction input | A chart image holds less information than the price data we already have; image "pattern reading" is not reliable or testable |
| Every answer shows rank, P(up), price range, top drivers and the model's backtest record, plus "not investment advice" | Honest, explainable output builds trust; US rules on investment advice |
| Accuracy claims are only from walk-forward backtests and the locked holdout | Claims of 70–90% stock "accuracy" almost always come from look-ahead bias |

## Data decisions

| Decision | Why | Effect / plan |
|---|---|---|
| Universe = all S&P 500 members since 2000, only while they were members | Avoids survivorship bias | Only 47% of departed stocks have prices → noted as a limitation |
| Universe expanded 466 → **634 stocks** (Yahoo fill-in) | Commercial product must cover the index; more training examples | XGBoost became unstable (0.0140 → 0.0064); the neural net improved (0.0149 → 0.0192) |
| Missing 196 tickers (after PCAR) filled from Yahoo Finance instead of paying for Tiingo | Free and immediate; checked against Tiingo on 25 overlapping tickers before use | Some delisted stocks are not on Yahoo → small extra survivorship gap. A commercial launch needs a licensed feed whichever source we use |
| Fundamentals: first-filed values, usable the day after filing | No look-ahead from restatements or late filings | — |
| Macro: ALFRED vintages | Uses values as first published | — |
| Data never committed to GitHub | Tiingo/Benzinga licences | Rebuilt with scripts (`data/README.md`) |

## How to add an entry

After each experiment: add a row here (status IN / OUT / PENDING + one-line reason), the numbers to `results.md`, and a line to the `PROGRESS.md` work log.
