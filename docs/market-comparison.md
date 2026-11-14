# How LatticeIQ compares with the market

What other AI stock tools and open research models say about their accuracy, how those numbers are measured, and where LatticeIQ stands. Checked on 14 Nov 2026; every figure has its source at the bottom.

**Short version:** nobody credible claims to call a stock's direction right 90% of the time. Commercial tools advertise backtested returns of their top picks or self-reported win rates. Published research reports direction accuracy of 51–52%. LatticeIQ's ranking edge is small and of the same kind, and it is measured more strictly than most marketing numbers.

## 1. What commercial tools claim

| Product | Published claim | What is actually measured | Audited? |
|---|---|---|---|
| Danelfin (AI Score) | US stocks scored 10/10 beat the S&P 500 by **+14.69% a year**, measured 3 months after the score, over 7 years. Stocks scored 1/10 trailed by −37.38% | The best-scored and worst-scored groups only, annualised | Self-published |
| Danelfin "Best Stocks" strategy | **+376%** vs +166% for the S&P 500, 3 Jan 2017 to 9 Jun 2025, after 0.15% trading costs | A 25-stock portfolio rebuilt from scores. The page states "Performance is based on backtested results" | Backtest |
| Prospero.ai | **54% win rate** against the S&P 500 over about 5,000 picks | Share of picks that beat the index; period not stated | Self-reported |
| AltIndex | **70% win rate**, average 22% gain over six-month windows | Share of picks that gained; no benchmark stated, so a rising market lifts it | Self-reported |

One 2026 review of these tools concluded: "there is no peer-reviewed evidence we're aware of that any retail AI stock picker consistently beats the market net of costs across a full market cycle." (That review is written by a company selling a portfolio-analysis product, so it is not neutral either.) For scale, the same review cites S&P's SPIVA scorecard: 65% of professional US large-cap funds trailed the S&P 500 in 2024, and 94% of US stock funds trailed their benchmark over 20 years.

## 2. What published research reports

| Source | Result | Market and horizon |
|---|---|---|
| Rahimikia, Ni & Wang (2025), finance-pretrained time-series models | Best model called direction right **51.74%** of the time; CatBoost 51.16%. General pretrained models (Chronos, TimesFM off the shelf) did worse than gradient boosting | All US stocks, next day |
| Same paper, portfolios | Long-short Sharpe ratio 5–7 **before trading costs**, daily rebalancing | All US stocks including small ones, which the paper finds more predictable |
| Microsoft Qlib benchmark | Rank IC of about 0.045 for LightGBM (*figure quoted from memory; re-check before citing in the report*) | Chinese CSI 300 stocks, short horizon |
| Our own tests of pretrained models | Kronos-small rank IC 0.0087 (t 0.81); TimesFM 2.5 rank IC −0.0017 | 466 S&P 500 stocks, 20 days |

## 3. Where LatticeIQ stands

Measured walk-forward on the corrected S&P 500 list (727 stocks, test years 2012–2024, about 298,000 weekly forecasts, model retrained for every test year, 2025–26 never touched):

| Measure | 5 days | 20 days | 60 days |
|---|---|---|---|
| Direction vs the market right | 50.4% | 50.6% | 51.0% |
| Rank IC (t-statistic) | 0.0141 (2.84) | 0.0192 (2.95) | 0.0202 (2.23) |
| Years with positive rank IC | 11 of 13 | 12 of 13 | 9 of 13 |
| 80% price range contains the real price* | 79.8% | 80.3% | 80.2% |
| 90% price range contains the real price* | 90.0% | 90.4% | 90.8% |

\* Price ranges and the group figures below were measured on the earlier 634-stock list; the re-run on the corrected list is pending.

**Measured the way competitors advertise** (how the top-ranked group did against the index, 20 days*): our top 20% beat the S&P 500 49.9% of the time, the bottom 20% only 48.0%. The top group roughly matches the market; the bottom group lags it. So today LatticeIQ is better at flagging likely laggards than at picking winners. Closing that gap is the main accuracy goal, and this number is reported for every model change.

### Why the numbers are not directly comparable

| | Typical marketing number | LatticeIQ number |
|---|---|---|
| Which stocks are scored | Only the top-scored picks | Every stock in the index, every week |
| Test period | Often the same years the model was tuned on; mostly a rising market since 2017 | 13 separate years, each forecast by a model trained only on earlier data, with a 60-day gap |
| Benchmark | Sometimes none (a "win" = the stock went up) | Always against the S&P 500 (SPY) |
| Costs | Sometimes included | Not a trading strategy; no return claim is made |
| Untouched final test | Rarely | 2025–26 holdout, to be run once |
| Data problems disclosed | Rarely | Yes: coverage of the true index is 91% in the test years, 67% in early training years ([`data-sources.md`](data-sources.md)) |

## 4. How to read any accuracy claim

1. **Accuracy of what?** Predicting the price *level* is trivially "99% accurate" (tomorrow's price is close to today's). Only the *change* matters.
2. **Against what?** A win rate without a benchmark mostly measures whether the market went up.
3. **Which stocks?** Results for the top picks say nothing about the rest.
4. **Backtest or live?** A backtest can be re-run until it looks good. A live record cannot.
5. **Was the test period ever used for tuning?**
6. **Before or after trading costs?**
7. **Who checked it?**

## 5. What LatticeIQ can and cannot claim

| Claim | Status |
|---|---|
| "Our ranking has a small, statistically significant edge that held in most of 13 test years" | **Yes** (t-statistic above 2 at all three horizons) |
| "Our 90% price range contains the real price about 90% of the time" | **Yes** (earlier list; being re-measured) |
| "Low-ranked stocks tend to lag the market" | **Yes** (earlier list; being re-measured) |
| "Our top picks beat the market" | **No, not yet** |
| "We predict whether a stock will go up" | **No**: tested and rejected |
| "X% accurate predictions" | **No**: not a meaningful or provable claim |
| Any return or profit claim | **No**: no trading strategy has been tested after costs |

## 6. Where the product can stand out

Not with a bigger backtest number, because anyone can print one. With things users can check:

- **A public live track record**: every forecast stored before the outcome is known and scored afterwards, losses included.
- **Calibrated price ranges**: a range that is right as often as it says.
- **Risk tools**: crash-risk warning, recovery statistics after a drop, a "likely laggards" list (where the measured skill is).
- **Reasons in plain language**, linked to official SEC filings.
- **Open method**: this repository shows what was tried, what failed and why ([`decisions.md`](decisions.md)).

A US launch also needs a legal review of every performance statement before it is shown to users.

## Sources

- Danelfin, "Are AI Scores reliable?": https://support.danelfin.com/hc/en-us/articles/4404382344465-Are-AI-Scores-reliable-How-successful-were-AI-Scores-predictions-in-the-past
- Danelfin, "Best Stocks Portfolio: +376% Return" (Nov 2025): https://blog.danelfin.com/?p=5049
- Truthifi, "What Is an AI Stock Picker?" (updated Nov 2026; cites Prospero.ai, AltIndex and SPIVA figures): https://truthifi.com/education/ai-stock-picker
- Walnut, "The Most Accurate AI Stock Predictor?" (Nov 2026): https://walnutinvest.com/resources/most-accurate-ai-stock-predictor
- Rahimikia, Ni & Wang, "Re(Visiting) Time Series Foundation Models in Finance" (2025): https://arxiv.org/abs/2511.18578
- Microsoft Qlib benchmarks: https://github.com/microsoft/qlib/tree/main/examples/benchmarks
- LatticeIQ runs: `results/20261113_1855_…h5`, `20261113_1914_…h20`, `20261113_1932_…h60`; earlier-list runs in [`results.md`](results.md)
