# Methodology

How LatticeIQ forecasts, how it is tested, and the published research each choice rests on. Written so that a reader with no finance background can follow it; every method has a "why" and a source. Numbers are from our own test runs (files in [`../results/`](../results/)); references are listed at the end.

Contents: [The question](#1-the-question) · [Data](#2-data-without-hindsight) · [Inputs](#3-inputs-features) · [Model](#4-the-model) · [Testing](#5-how-we-test) · [Scores](#6-how-we-score) · [Price ranges](#7-price-ranges) · [Dropped methods](#8-methods-we-tested-and-dropped-and-why) · [News](#9-news-sentiment-in-testing) · [Limits](#10-limits) · [References](#references)

## 1. The question

**What we forecast.** For each S&P 500 stock and each day: will it do better or worse than the market (SPY, a fund that holds the whole S&P 500) over the next 5, 20 and 60 trading days? Formally, the target is the stock's log return minus SPY's log return over that window ("excess return").

**Why relative to the market, not the price itself.** Most of a stock's daily move is the whole market moving, which no stock-level model can call. Removing it leaves the part that differs between stocks, which is what research on stock returns studies (the "cross-section of returns", Fama & French 1992; Gu, Kelly & Xiu 2020). It is also what an investor choosing between stocks needs.

**What we output.** An order of stocks (a rank), an expected excess return, and price ranges. We do not output buy or sell calls.

## 2. Data without hindsight

The most common reason published stock forecasts fail in practice is that the test used information that was not available on the forecast date ("look-ahead") or only included companies that survived ("survivorship bias", Brown, Goetzmann, Ibbotson & Ross 1992). Our rules:

| Data | Rule | Why |
|---|---|---|
| Index membership | A stock is scored on a day only if it was in the S&P 500 that day, including companies that later failed or were taken over | Testing only today's members would favour survivors |
| Prices | A feature on day *t* uses closing prices up to *t*; the outcome starts at *t + 1* | No future prices in the inputs |
| Company accounts (SEC filings) | The **first** value a company filed, usable from the **day after** the filing date | Later corrections were not known at the time |
| Company events (8-K filings) | Time-stamped to the second the SEC published them | Same |
| News | Articles at or after 4 pm New York time count for the next trading day | The market had closed |

Each rule has an automated test that changes future data and checks that no past input moves. Two further tests guard the whole pipeline: on **random data the model must show no skill**, and on data with a **planted pattern it must find it**.

We also found and fixed a real data error during the project (the membership list had stopped in January 2019) and then audited every price history against trading volume; six histories that belonged to a different company under a reused ticker were removed. See [`decisions.md`](decisions.md).

## 3. Inputs (features)

44 inputs per stock per day. Each input is turned into its **rank among all stocks that day** (from −0.5 to +0.5). Ranking removes market-wide shifts and extreme values and is standard in machine-learning studies of stock returns (Gu, Kelly & Xiu 2020). Each group comes from effects documented in published research:

| Group | Inputs | Research behind it |
|---|---|---|
| Momentum | Returns over 3, 6 and 12 months (the 12-month one skips the latest month); returns minus SPY's over 1 and 3 months | Stocks that rose over the past 3–12 months tend to keep outperforming for a while (Jegadeesh & Titman 1993) |
| Short-term reversal | Returns over 1 week and 1 month | Very recent winners tend to give some of it back (Jegadeesh 1990) |
| Risk | Volatility over 20 and 60 days, beta, stock-specific volatility | Highly volatile stocks have tended to do worse than their risk suggests (Ang, Hodrick, Xing & Zhang 2006) |
| Trend and activity | Distance from 50- and 200-day averages, unusual volume, RSI | Common technical measures, kept only because they help in our tests |
| Value | Earnings, sales, cash-flow yield; book-to-market | Cheap stocks relative to fundamentals have earned more over time (Fama & French 1992) |
| Profitability and quality | Gross profit to assets, return on equity, margins, leverage | More profitable firms have earned more (Novy-Marx 2013) |
| Accruals | Profit not backed by cash | Earnings made of accruals rather than cash tend to disappoint (Sloan 1996) |
| Earnings surprise | Latest quarter vs the same quarter a year earlier, scaled by how noisy that usually is; price reaction to the filing | Prices keep drifting after earnings surprises (Bernard & Thomas 1989) |
| Growth and size | Revenue and asset growth; market value | Standard controls |
| Sector | The sector's momentum; each stock's rank within its sector on 10 inputs above | Part of momentum is industry-wide (Moskowitz & Grinblatt 1999); comparing a stock with its own sector removes industry effects |

Sectors are worked out from prices alone: each month a stock is assigned to the sector fund (technology, energy, …) its returns followed most closely over the past year. No outside classification is needed and nothing from the future is used.

**What each group added (20 days, early stock list):** price inputs alone 0.0128; adding fundamentals and sector inputs 0.0140. Sector inputs gave the biggest single gain. Details: [`results.md`](results.md) section 1.

## 4. The model

**Type.** A small feed-forward neural network: 44 inputs → 64 → 32 → 1 output. Gu, Kelly & Xiu (2020), the largest published comparison of machine-learning methods on US stock returns, found neural networks among the best methods and shallow networks better than deep ones; our own comparison agreed (section 8).

**Training.**

- **Target clipping:** the most extreme 1% of outcomes on each side are clipped per day before training, so takeovers and crashes do not dominate what the model learns. Scores are always computed on the real, unclipped outcomes.
- **Optimiser and regularisation:** Adam (Kingma & Ba 2015), dropout of 10% (Srivastava et al. 2014), and early stopping on the latest 10% of the training dates.
- **Three seeds:** a neural network's result depends partly on its random starting point. We train three networks that differ only in that and average them, the same idea as the "deep ensembles" of Lakshminarayanan, Pritzel & Blundell (2017). It is chosen to remove the luck of one training run, not for a higher score: on the 634-stock list three seeds gave 0.0136 / 0.0180 / 0.0229 against 0.0112 / 0.0192 / 0.0254 for one seed.
- **Trust factor *k*:** after training, a copy trained on the first 80% of the training dates is checked on the last 20%. *k* is the slope of real outcomes on forecasts there, the standard check of whether forecasts are the right size (Mincer & Zarnowitz 1969), kept between 0 and 1. Every forecast is multiplied by *k*. A model with no skill gets *k* near 0, so its forecasts fall back to "no excess return".

**One model per horizon** (5, 20, 60 days). In daily use the model is trained once on all data with a known outcome and then applied every day; see [`architecture.md`](architecture.md#daily-model-train-once-score-daily).

## 5. How we test

**Walk-forward testing.** For each year from 2012 to 2024: train on all earlier years, test on that year, move one year forward. This is how the model would have been used in real time, and every forecast is made by a model that had not seen its outcome.

**A gap between training and testing.** The last 60 trading days before each test year are left out of training. Our targets look up to 60 days ahead, so without the gap a training example could overlap the test period. This is the "embargo" recommended for financial machine learning (López de Prado 2018).

**Same dates for every model.** All models are scored on the same weekly dates (every fifth trading day), so differences come from the models, not the sample.

**A locked holdout.** Everything from 1 January 2025 is excluded from all tests and decisions. It will be used once, when the model is final. Trying many variants on the same data and keeping the best one inflates results (Bailey, Borwein, López de Prado & Zhu 2014); an untouched holdout is the guard against that. One choice was made after seeing test results (the neural net alone over a pre-declared ensemble); we say so openly and both will be scored on the holdout.

**A bar for improvements.** Two runs that differed by only five stocks moved the 13-year score by 0.0005–0.0008 and single years by up to 0.015. An addition counts only if it beats the current model by clearly more than that, in most years, in the same run.

## 6. How we score

| Score | What it measures | Why we use it | Source |
|---|---|---|---|
| **Rank IC** | On each date, the correlation between the predicted order of stocks and the real order (Spearman), averaged over dates. 0 = no skill | The product ranks stocks, so the order is what matters. Honest values for stocks are small, typically 0.01–0.05 | Grinold & Kahn (2000), for the information coefficient |
| **t-statistic** of rank IC, Newey-West | How sure we are the rank IC is not zero. Overlapping 20- and 60-day windows make ordinary t-statistics too optimistic; Newey-West corrects for that | About 2 is the usual bar; Harvey, Liu & Zhu (2016) argue for about 3 when many ideas have been tried | Newey & West (1987); Harvey, Liu & Zhu (2016) |
| **Hit rate** | Share of forecasts with the right direction against the market | Easy to understand; 50% is a coin flip | — |
| **Diebold-Mariano test** | Whether the forecast errors are smaller than those of "no excess return" (Random Walk), with the small-sample correction | Tests the size of forecasts, not only their order | Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997) |
| **Rank groups** | Stocks sorted into five groups by forecast each week; how often each group beat SPY and by how much | Turns the score into plain numbers | — |
| **Positive years** | In how many of the 13 test years the rank IC was above 0 | A result that holds year after year is more trustworthy than a good average | — |

**v1.0 results:** rank IC 0.0149 / 0.0197 / 0.0199 at 5 / 20 / 60 days with t 3.07 / 3.18 / 2.23, positive in 12 / 10 / 10 of 13 years. The 5- and 20-day results pass the stricter t ≈ 3 bar; the 60-day result passes only the usual bar of 2.

## 7. Price ranges

**Goal.** Ranges that hold the price as often as they say (1 time in 2, 4 in 5, 9 in 10), as narrow as that allows.

**The model in the product: volatility ranges.**

1. Daily volatility is a blend of four measures, one of them based on each day's high and low (Garman & Klass 1980), which uses more of each day's information than closing prices alone.
2. Range for *H* trading days: price × exp(± *q* × volatility × √*H*).
3. The multiplier *q* is **not** taken from a bell curve. It is the value that would have held the price the stated share of the time on earlier windows (an empirical quantile), so fat tails are built in.
4. A separate multiplier applies when an earnings report is expected inside the window. In our data stocks move about 1.7 times more in earnings weeks; a single range held the price only 57% of the time there.

**Result (2012–2024, about 306,000 forecasts per horizon):** the "4 in 5" range held 79.5% / 79.0% / 79.1% at 5 / 20 / 60 days; in earnings windows 77–79%.

**The first model, for comparison:** quantile regression with XGBoost (Chen & Guestrin 2016) predicts several percentiles of the outcome; conformalised quantile regression then widens or narrows them using the previous years so the coverage is right (Romano, Patterson & Candès 2019). It held 79.8% / 80.3% / 80.2% on the earlier stock list. A planned improvement is adaptive calibration that recovers faster after shocks such as March 2020 (Gibbs & Candès 2021).

**Scores for ranges:** coverage (how often the price ended inside), average width, the pinball loss for percentiles, and the interval score, which rewards narrow ranges and penalises misses (Winkler 1972; Gneiting & Raftery 2007).

**The probability of going up was dropped.** We tested a "chance this stock rises" from the range model with the Brier score (Brier 1950). It did worse than simply saying "stocks rose 57% of the time", so it is not shown.

## 8. Methods we tested and dropped, and why

| Method | Result (20-day rank IC) | Why it was dropped | Context from research |
|---|---|---|---|
| XGBoost (gradient-boosted trees) | 0.0140 on the early list, 0.0067 on the full list | Unstable as the stock list grew | Trees were the other strong family in Gu, Kelly & Xiu (2020); on our data the neural net was steadier |
| LightGBM | 0.0080 | Ranked stocks 86% the same as XGBoost: no new information | Ke et al. (2017) |
| Ridge regression | 0.0042 | Too simple | Linear models trailed non-linear ones in Gu, Kelly & Xiu (2020) |
| Ranking objective for XGBoost | 0.0080 | Worse than plain regression on the same inputs | — |
| Ensembles of several models | 0.0130–0.0199 | Good on the small list; beaten by the neural net alone on the full list in 6 of 6 comparisons | — |
| Macro inputs (VIX, yields, credit spread, inflation) | 0.0118 vs 0.0128 without | The same value for every stock on a day cannot help put stocks in order | Macro variables also predict the overall market poorly out of sample (Welch & Goyal 2008) |
| Kronos (pretrained on market price data) | 0.0087 alone; +0.0007 as an extra input (noise) | Weak and inconsistent | Shi et al. (2025) |
| TimesFM 2.5 (pretrained, general time series) | −0.0017 alone; worse as an input | No skill: forecasts barely differ between stocks | Das et al. (2024) |
| Fine-tuning a general pretrained model on stock returns | not run | Published evidence shows little gain; models pretrained on financial data are the better route and are next to test | Rahimikia, Ni & Wang (2025); Chronos architecture: Ansari et al. (2024) |

**Why the pretrained forecasters lost.** They read one stock's price history at a time. Deciding which stocks will beat the market needs a comparison across all stocks on the same day, plus company and sector information, which is what our model does. The 2025 study of these models on financial data reached the same conclusion.

## 9. News sentiment (in testing)

Media tone has been shown to predict short-term stock and market moves (Tetlock 2007). Our test:

- **Source:** Alpaca's news API, Benzinga articles from 2015 (435–1,144 a day on sampled days, naming 13–33% of index members). Only one publisher is kept so the article counts mean the same thing in every year.
- **Scoring:** FinBERT, a language model trained on financial text, gives each headline and summary a score from −1 (negative) to +1 (positive) (Araci 2019). Its licence is non-commercial, so it would be replaced before any commercial use.
- **Inputs (8):** average tone over 1, 5 and 20 days; how much the stock is in the news; unusual news volume; tone compared with the stock's usual tone; a recent strong-news flag. Before coverage starts, inputs are left empty, not zero ("no data" is not "no news").
- **Test:** v1.0 with and without these inputs, on test years 2018–2024 (news begins in 2015 and the model needs earlier years to learn from), at 5 and 20 days. Kept only if it passes the bar in section 5.

## 10. Limits

- The edge is small (rank IC about 0.02) and has been weak recently: 2024 is negative at 20 and 60 days.
- The top-ranked group only matches the market; the measurable value today is in flagging likely laggards.
- Results are before trading costs; no trading strategy has been tested.
- In training years before 2012, 67% of true index members have price data, so those years lean towards survivors (91% on average in the test years).
- Free data sources (Yahoo, Tiingo) are for research only; a commercial version needs licensed data keyed by a permanent security ID.

---

## References

Finance: what predicts stock returns

- Ang, A., Hodrick, R. J., Xing, Y. & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299.
- Bernard, V. L. & Thomas, J. K. (1989). Post-earnings-announcement drift: delayed price response or risk premium? *Journal of Accounting Research*, 27 (Supplement), 1–36.
- Fama, E. F. & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427–465.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273. Working paper: [NBER w25398](https://www.nber.org/papers/w25398).
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers: implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.
- Moskowitz, T. J. & Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance*, 54(4), 1249–1290.
- Novy-Marx, R. (2013). The other side of value: the gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review*, 71(3), 289–315.
- Tetlock, P. C. (2007). Giving content to investor sentiment: the role of media in the stock market. *Journal of Finance*, 62(3), 1139–1168.
- Welch, I. & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.

Testing and avoiding false discoveries

- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the AMS*, 61(5), 458–471.
- Brown, S. J., Goetzmann, W., Ibbotson, R. G. & Ross, S. A. (1992). Survivorship bias in performance studies. *Review of Financial Studies*, 5(4), 553–580.
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68. Working paper: [NBER w20592](https://www.nber.org/papers/w20592).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Scores and statistics

- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Mincer, J. & Zarnowitz, V. (1969). The evaluation of economic forecasts. In J. Mincer (ed.), *Economic Forecasts and Expectations*. NBER.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Winkler, R. L. (1972). A decision-theoretic approach to interval estimation. *Journal of the American Statistical Association*, 67(337), 187–191.

Price ranges

- Garman, M. B. & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.
- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170).
- Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile regression. *NeurIPS 2019*. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222).

Machine learning

- Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD 2016*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754).
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- Kingma, D. P. & Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958.

Pretrained forecasting and language models

- Ansari, A. F. et al. (2024). Chronos: learning the language of time series. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).
- Araci, D. (2019). FinBERT: financial sentiment analysis with pre-trained language models. [arXiv:1908.10063](https://arxiv.org/abs/1908.10063).
- Das, A., Kong, W., Sen, R. & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). *ICML 2024*. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688).
- Rahimikia, E., Ni, H. & Wang, W. (2025). Re(Visiting) time series foundation models in finance. [arXiv:2511.18578](https://arxiv.org/abs/2511.18578).
- Shi, Y. et al. (2025). Kronos: a foundation model for the language of financial markets. [arXiv:2508.02739](https://arxiv.org/abs/2508.02739).

Data sources and their licences are listed in [`data-sources.md`](data-sources.md).
