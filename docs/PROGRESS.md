# LatticeIQ — Progress & TODO

_Last updated: 17 Nov 2026 · Owner: project lead (all roles)_
Rule: update this file every time something is finished, decided or found. Newest log entries at the top.
**New to the project?** Read [`project-flow.md`](project-flow.md) first: the whole project as one step-by-step story with the status of each step. This file is the working to-do list and the day-by-day log.

---

## Where we are

**Steps 1–10 of 15 done.** The ranking model v1.0 is chosen and measured (rank IC 0.015 / 0.020 / 0.020 at 5 / 20 / 60 days, 13 test years, audited list of 722 stocks), trained once on all data and run every day on the PC behind a three-page dashboard with live prices, price ranges, filings and analyst views. **Now (step 13): the model is final (v1.0 + range model v2) and the one-time 2025–26 holdout test is locked in [`holdout-lock.md`](holdout-lock.md); after it, the report and slides (by 28 Nov).** News sentiment was tested and not added; further improvement tests (insider purchases, SEC 8-K events) are future work.

| # | Step (details in [`project-flow.md`](project-flow.md)) | Status |
|---|---|---|
| 1 | Define the question and how to score it | ✅ |
| 2 | Build the test bench before any model (48 tests) | ✅ |
| 3 | Collect the data (prices, fundamentals, 8-K filings, membership) | ✅ |
| 4 | Turn data into features (44) | ✅ |
| 5 | Compare models | ✅ |
| 6 | Choose and lock v1.0 | ✅ |
| 7 | Find the data error, audit, freeze "universe v2", re-measure | ✅ |
| 8 | Price ranges (volatility range model v2 in the product) | ✅ |
| 9 | Dashboard | ✅ |
| 10 | Run it every day (live prices, train once / score daily, forecast log) | ✅ |
| 11 | Improve the model, one test at a time | ✅ stopped 17 Nov (news ❌); the rest is future work |
| 12 | Live track record from the forecast log | ⏳ |
| 13 | Holdout test 2025–26, once | 🔄 ranges ✅ passed 18 Nov; ranking running |
| 14 | Report and presentation | ⏳ by 28 Nov |
| 15 | Before a launch | ⏳ |

## To do, in order

1. [ ] **Rotate keys** pasted in chat (Benzinga, FRED, Hugging Face, Tiingo, Kaggle, Alpaca) and update `.env`. Then delete the `Claude outputs` folder on the PC (it holds old copies of `.env`; it is never committed).
2. [ ] **Publish** the work since 14 Nov to GitHub (safety check first, see the README's quick start and `CONTRIBUTING.md`), then **host it**: `deploy/aws/README.md`.
3. [x] **News:** tested 17 Nov, not added (+0.003, paired t 1.29 / 0.83; `results.md` 5e). Next in the same slot: **insider purchases** from SEC Form 4.
4. [ ] **Sector-neutral target** on Kaggle (`--target sector`) vs v1.0.
5. [ ] **Price ranges on universe v2:** 5 and 20 days ran on 15 Nov (results to write up); 60 days failed on a typo, rerun: `python -m scripts.kaggle_cloud brackets "--data real --fundamentals --horizon 60"`. Then compare with the volatility range model.
6. [ ] **Crash risk and recovery** on real data (`kaggle_cloud crash ...`, `kaggle_cloud recovery ...`).
7. [ ] **Live track record:** score `data/live/forecasts/` against outcomes once a week of logs exists; show on the Track record page.
8. [ ] **Weekly fundamentals refresh** (`python -m src.ingest.edgar`); scoring uses fundamentals as of the last download.
9. [ ] SEC 8-K events as model inputs (filings already downloaded).
10. [ ] FinText finance-pretrained Chronos: runner and comparison.
11. [ ] ARIMA baseline (for the report).
12. [ ] Retrain the daily model monthly or quarterly: `python -m scripts.kaggle_cloud final-model`.
13. [ ] **Holdout test**, once: push `docs/holdout-lock.md` to GitHub first, then `python -m scripts.kaggle_cloud holdout --cpu` and `python -m scripts.holdout_ranges`.
14. [ ] **Report and slides.**

### Later / before a launch
- [ ] Licensed price and news data keyed by a permanent security ID.
- [ ] Replace FinBERT (non-commercial licence).
- [ ] Hosting, accounts, API; chat assistant that explains the model's forecast.
- [ ] Long-short backtest after trading costs.
- [ ] Securities lawyer review of every performance statement; no buy/sell suggestions.
- [ ] Regime models; insider trading (Form 4), short interest, option-implied volatility as inputs.

---

## Early results log (6–9 Nov, earlier stock lists; current numbers in [`results.md`](results.md))

| Date | Run | Universe | Features | Model | Hit rate | Rank IC (t) | Notes |
|---|---|---|---|---|---|---|---|
| 6 Nov 2026 | `20261106_1641_real_h20` | 27 tickers (17 ETFs + 10 stocks) | price | Random Walk | 50.0% | 0 | baseline |
| 6 Nov 2026 | `20261106_1641_real_h20` | same | price | XGBoost | 49.4% | −0.002 (−0.17) | no skill; cross-section too small, ETFs mixed in. DM +4.5 (slightly worse MSE) |
| 9 Nov 2026 | `20261109_0940_real_stock_h20` | 466 stocks | price 17 | XGBoost (reg) | 50.4% | 0.0128 (1.59) | baseline rerun, identical ✅ |
| 9 Nov 2026 | `20261109_0948_real_stock_h20_fund` | 466 | + fundamentals 15 (32) | XGBoost | 50.7% | 0.0115 (1.66) | positive 11/13 yrs |
| 9 Nov 2026 | `20261109_0956_real_stock_h20_fund_sect` | 466 | + sectors 12 (44) | XGBoost | **50.8%** | **0.0140 (1.80)** | **best so far**; positive 11/13 yrs (2017–2023 all +); not yet t>2 |
| 9 Nov 2026 | `20261109_1011_real_stock_h20_fund_sect_rank` | 466 | 44 | XGBRanker | 50.3% | 0.0080 (1.09) | worse → **ranking objective dropped** |
| 9 Nov 2026 | `20261109_0923_real_stock_h20_brackets` | 466 stocks | price + macro + raw vol | **Brackets** (quantile XGB + conformal) | — | — | **80% range coverage 80.1%** (naive 77.3%), 90%: 90.2% (naive 86.4%), 50%: 49.4%; width ±~10%; 2020 = 72% (COVID). P(up) hit 54.7% < base rate 57.7% → uncalibrated (fixed next) |
| 9 Nov 2026 | `20261109_0927_real_stock_h20_fund_brackets` | 466 stocks | + fundamentals | Brackets | — | — | 80.4 / 90.4 / 49.4% coverage; fundamentals don't change brackets (volatility drives width) |
| 8 Nov 2026 | `20261108_1527_real_stock_h20_nomacro` (Kaggle GPU, 5 min) | 466 stocks | price 17 | XGBoost + shrink (new scoring) | 50.4% | **+0.013 (1.6)** | best so far; positive 8/13 yrs, 54% of weeks; DM −1.55, p 0.12 (not yet significant) |
| 8 Nov 2026 | `20261108_1533_real_stock_h20` (Kaggle GPU) | 466 stocks | price 17 + macro 8 | XGBoost + shrink | 50.0% | +0.012 (1.0) | **macro adds nothing** for ranking; model leaned on macro (unrate, CPI, yields) without gain; DM p 0.77 |
| 7 Nov 2026 | `20261107_1853_real_stock_h20_nomacro` | **466 stocks** (500 tickers A–PCAR) | price 17 | XGBoost + shrink | 50.3% | **+0.008 (1.1)** | first full-universe baseline; positive 7/13 yrs; not significant. 3 yrs k=0 zeroed the ranking (fixed after this run). 28 min |
| 6 Nov 2026 | `20261106_1703_real_stock_h20_nomacro` | **10 stocks only** (download incomplete) | price 17 | XGBoost + shrink | 48.6% | −0.019 (−1.3) | not meaningful: too few stocks; k avg 0.06 → mostly falls back to RW |
| 6 Nov 2026 | `20261106_1711_real_stock_h20` | **10 stocks only** | price 17 + macro 8 | XGBoost + shrink | 50.1% | −0.007 (−0.4) | not meaningful; k avg 0.10. Rerun after full download |
| 6 Nov 2026 | synthetic, no signal | 60 fake | price 17 | XGBoost + shrink | 49.8% | −0.005 (−1.5) | k = 0.00 → falls back to RW; DM no longer worse (p 0.16) ✅ |
| 6 Nov 2026 | synthetic, planted momentum | 60 fake | price 17 | XGBoost + shrink | 53.6% | 0.101 (13.2) | k = 0.67; DM −5.4 (was −4.0) ✅ |
| 6 Nov 2026 | synthetic, no signal | 60 fake | price | XGBoost | 49.7% | −0.006 (−0.8) | leak check ✅ |
| 6 Nov 2026 | synthetic, planted momentum | 60 fake | price | XGBoost | 53.9% | 0.108 (14.3) | can-find-signal check ✅ |

Targets set on 6 Nov: rank IC 0.03–0.08 with t > 2, hit rate 52–56%. Reached so far (v1.0): rank IC 0.015–0.020 with t 2.2–3.2, hit rate 50.5–51.0%.

---

## Work log (newest first)

### 18 Nov 2026 — Holdout: price ranges pass
- `python -m scripts.holdout_ranges`, run once: range model v2 fitted to 2024, scored on 2025–26 (47,323 / 45,798 / 41,614 forecasts). The 80% range held **78.9% / 78.1% / 77.8%** at 5 / 20 / 60 days, 50% range 47–49%, 90% range 89%, day's high and low 88.5–89.6%. **Every pre-set band passed.**
- Recorded openly: v1 (the secondary) held about 80% everywhere; v2 is 1.1–2.2 points under 80% (up to 2.6 below v1), with ranges 3–8% narrower. v2's upward tilt was too high for 2025–26 (close above the middle 44.6% at 60 days). Nothing changed, as the lock requires; noted for a later version. Details: `results.md` section 8a.
- Server: Lightsail `latticeiq-web` (Sydney, 1 GB, $7 a month) created and set up with `deploy/aws/setup.sh`; site address `https://3-27-118-9.sslip.io`. Data upload and first start next.
- Ranking holdout running on Kaggle (CPU).
- First start on the server: Yahoo prices and the daily ranking work from AWS (500 stocks ranked in 170 s with swap). The saved range fit was not reused, because Git gives `config/universe.csv` Windows line endings on the PC (26,297 bytes) and Linux ones on the server (25,454), and the fit was keyed by file size. The server tried to refit (about 4.5 GB) and was killed for lack of memory. Fixed: text files are now fingerprinted with line endings removed, and a machine with under 3.5 GB stops with a clear message instead of refitting. New test (48).

### 17 Nov 2026 — Model final; holdout test locked
- Decision: no more improvement tests before the report. The last three ideas (news, Kronos, TimesFM) each added less than run-to-run noise; the holdout is the stronger result and must come after the model is final.
- Final: ranking **v1.0** (secondary: ensemble v2, as declared on 12 Nov), price ranges **v2** (secondary: v1).
- `docs/holdout-lock.md` written before any 2025–26 number was seen: what is tested, the data files and their sizes, the exact commands, how much the short holdout can show (about 90 weekly dates; a real edge is expected to give t ≈ 1.2 and a positive rank IC 88% of the time), and the pass marks (main: 20-day rank IC > 0; ranges: 80% range held 75–85%). Results get published whatever they are.
- Code: `run_experiment --holdout` (one model trained to 60 trading days before 1 Jan 2025, scored on every week from 2 Jan 2025), `kaggle_cloud holdout` (fixed commands, refuses a second run), `scripts/holdout_ranges.py` (ranges fitted to 2024, scored on 2025–26, refuses a second run). 47 tests pass (1 new).

### 17 Nov 2026 — News sentiment tested: not added
- Four runs on Kaggle's CPU, test years 2018–2024, v1.0 with and without the 8 news inputs. Rank IC 5 days **0.0111 → 0.0142**, 20 days **0.0099 → 0.0125**.
- Tested date by date on the same weeks: paired t **1.29** (5 days) and **0.83** (20 days). Two runs of the identical model (today's CPU run vs the GPU run of 14 Nov) differ by 0.0015 on average and up to 0.005 in a single year, so a +0.003 gain is within reach of noise. At 20 days news was worse in 3 of 7 years; the top and bottom fifths did not improve.
- Decision: **v1.0 stays**; news is not added; the stronger-language-model test, which depended on FinBERT news helping, is not run. Code and data kept for a retest. Full tables: `results.md` 5e.
- Note for every later test: compare on the same machine type (CPU vs GPU runs of the same model differ by up to 0.005 a year).

### 17 Nov 2026 — Ready to host on AWS
- AWS Agent Toolkit set up on the PC (AWS CLI, sign-in profile `latticeiq`, Region ap-southeast-2; skills and the AWS MCP server for Claude Code and Gemini CLI).
- 46 tests pass. Dashboard server mode: `--host` and a login (`LATTICEIQ_PASSWORD`, HTTP Basic, any user name); it refuses to listen beyond this computer without a password. New test: requests without the password get 401.
- The daily ranking now runs in its own process (it needs 1.7–1.8 GB for about a minute; the memory is handed back afterwards). Page build measured at 0.7 GB peak.
- The saved range fit is now keyed by the data files' sizes, not their times, so the fit copied from the PC is reused on the server.
- `deploy/aws/`: `setup.sh` (swap, Python packages, Caddy HTTPS with a free certificate, service that starts on boot), `upload_data.ps1` (copies the data and model from the PC), `requirements-server.txt` (pandas kept below 3.0, as tested), step-by-step README. Plan: Lightsail 1 GB in Sydney, $7 a month.

### 17 Nov 2026 — News comparison, first attempt (Kaggle CPU)
- Ran on Kaggle's CPU (the GPU queue had waited 45+ minutes). Run 1 of 4, v1.0 without news, 5 days, test years 2018–2024: rank IC **0.0111** (t 1.51), positive in 6 of 7 years; 983 s. This is the baseline the news run must beat.
- Run 2 (with news) stopped: the job's copy of the Kaggle dataset had `news_articles.parquet` but not `news_scored.parquet`, so runs 2–4 did not start. Next: check the dataset lists both files, then run the three remaining steps.
- `run_experiment.py` now checks for the news files before building the panel, so a missing file fails in seconds instead of after 4–5 minutes.

### 17 Nov 2026 — Range model v2 and the day's high and low on the dashboard
- `src/models/vol_ranges.py` v2: a regression forecasts each stock's volatility over the window (five volatility measures, market mood from SPY, expected earnings, the stock's typical earnings move), and the lower and upper multipliers are fitted separately. Weights and multipliers fitted on windows ended by 31 Dec 2024; the 2012–2024 record refits them each year on earlier windows only.
- Replayed in the product code, v1 scored on the same 306,000 stock forecasts per horizon: 80% range held **79.2% / 78.8% / 79.1%** (v1 79.5% / 79.0% / 79.1%) and is **2.2% / 3.4% / 4.9% narrower**; interval score better in 11 / 11 / 10 of 13 years; worst year 77.4% / 76.6% / 75.6% (v1 75.8% / 75.9% / 73.3%).
- New on the stock page: "On <day> itself" in the side panel, the likely middle, "day's high stays below" and "day's low stays above". Held 89.6% / 89.0% / 88.7% (high) and 89.6% / 90.1% / 90.5% (low), aimed at 90%. The Track record page has the same table. The cone shows the likely middle as a dotted line; ranges show as down / up (e.g. −8.4% / +9.8%).
- The range follows the price during the day as before, each side scaled separately. 45 tests pass (range tests rewritten, one new for the uneven sides). `data/live/range_fit.json` is rebuilt on the first run (a minute or two).

### 17 Nov 2026 — Benchmarks checked
- `market-comparison.md`: the Qlib figure, flagged earlier as quoted from memory, is now checked: CSI 300 rank IC 0.0485 (LightGBM), 0.0523 (DoubleEnsemble), 0.0573 (TRA, best). Price-range rows updated to the audited list (80% range held 79.5% / 79.0% / 79.1%).

### 17 Nov 2026 — Searched for what else could improve the forecasts
- v1.0 stays locked; every idea is tested against it and dropped if it does not beat it. Full table with sources: `decisions.md`, "Improvement search".
- Ranking, in order: news sentiment (running) → insider purchases (free SEC Form 4 data; studies show +1.0% over 5 days, gone after a month) → SEC 8-K events → short interest (FINRA) → sector-neutral target and FinText (queued) → option-implied signals and the 200+ published predictors only with university WRDS access. A stronger language model for news only if FinBERT news helps.
- Ranges: news counts next (shown to improve short-horizon volatility forecasts); implied volatility needs options history (Alpaca's starts in Feb 2024 and is paid; OptionMetrics via WRDS).
- Open question for the team: does the university have WRDS?

### 17 Nov 2026 — Narrower ranges and a day high/low: tested
- Replayed every idea on 2012–2024 (about 318,000 weekly forecasts per horizon including the 18 funds, all fitting on earlier windows only; an earlier version of this note said 570,000, which was wrong). Full tables: `results.md` section 5d.
- **Works, small:** a volatility forecast by regression (the current measures + one-week volatility + market mood + earnings) with uneven up/down multipliers: **2.4% / 3.7% / 5.0% narrower** at 5 / 20 / 60 days, same hit rate (79.0% / 78.7% / 79.1%), better in 10–11 of 13 years, worst year better everywhere. Market mood gives most of it; each stock's own earnings move adds about nothing.
- **Rejected:** 18 multiplier groups (narrower only by missing more), adaptive last-year blend (worse), separate multipliers for new stocks (worse; new stocks hold about one point under the rest without it).
- **Day high and low:** "high below X" held 88.7–89.4%, "low above Y" 89.6–90.5% (aimed at 90%). The middle of the range sits a little above today's price (+0.3% / +1.2% / +3.4%) because stocks drift up; the close ended above it 51% of the time.
- Put on the page the same day (entry above).

### 17 Nov 2026 — One price range by default
- The stock page now shows one range, the 80% range ("right 4 times in 5"), in the chart cone (one band with a clear edge, labelled "4 in 5") and in the side panel. "Show 50% and 90% ranges" under the chart or in the side panel brings back all three; the choice is remembered. The forecast cards already showed only the 80% range. The API and the Track record page keep all three.
- Checked in the simulation in light and dark, both modes.

### 17 Nov 2026 — News sentiment scored on Kaggle
- Merge run again with the fix: 4 articles dated before Jan 2015 left out; 767,435 Benzinga articles kept.
- `news-score` on Kaggle's GPU: all 767,435 headlines and summaries scored by FinBERT in about 10 minutes (`data/clean/news_scored.parquet`, 18 MB). No missing scores, no duplicates. Tone: clearly negative 17.6%, slightly negative 9.0%, neutral 27.8%, slightly positive 18.7%, clearly positive 26.9%; average +0.09 (company news leans positive).
- Data re-uploaded to Kaggle with the scores. Next: the four comparison runs (5 and 20 days, with and without news, test years 2018–2024), then keep or drop news.

### 16 Nov 2026 — News history downloaded: 767,439 Benzinga articles, 2015 to today
- `alpaca_news --start 2015-01` finished in 190 minutes (137 months; the first 6 were fetched the slower way earlier).
- `--combine`: publishers benzinga 430,546, not recorded (pre-2022, Benzinga) 336,893, fiscal.ai 71,024 (left out). Kept 767,439 articles → `data/clean/news_articles.parquet`; the earlier thin file is kept as `news_articles_benzinga_plan.parquet`.

| Year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to 16 Nov) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Articles | 47,051 | 48,024 | 37,036 | 38,848 | 41,055 | 53,756 | 70,834 | 83,188 | 89,312 | 90,679 | 88,110 | 79,542 |
| Our stocks named | 705 | 701 | 694 | 676 | 673 | 675 | 673 | 669 | 660 | 667 | 662 | 653 |
| Articles per stock | 67 | 69 | 53 | 57 | 61 | 80 | 105 | 124 | 135 | 136 | 133 | 122 |

- Nearly every stock has news every year, about 1 to 2.6 articles a week each. Coverage doubles from 2015 to 2023; the news features are ranked across stocks on each day, which takes out that steady growth.
- Found: 4 articles dated 2012–2014 (re-published pieces). Left in, the news features would have treated 2012–2014 as "covered, no news" instead of unknown, in the training years of the 2018 test. `--combine` now leaves out anything dated before the first month downloaded; run it once more before uploading.
- Next: `upload-data` → `news-score` → `upload-data` → the four comparison runs (5 and 20 days, with and without news, 2018–2024).

### 16 Nov 2026 — Logo
- Mark: three candlesticks (hollow = up, filled = down) running into the forecast cone, which widens with the square root of time in the chart's three band strengths: the dashboard's own picture reduced to a few shapes. Tried first: a binomial price lattice (read like a "share" icon) and a cone of dots (abstract, no link to stocks); candles were asked for.
- Files: `app/favicon.svg` (two candles on a blue tile for 16 px), `docs/assets/logo.svg` / `logo_dark.svg` (mark and name, letters as outlines), `logo_mark.svg` / `logo_mark_dark.svg`. The header draws the mark inline in the theme's colours. Checked in light and dark and at 16, 32 and 128 px.

### 16 Nov 2026 — Dashboard: sidebar filters, compact list, rank moved up
- **Sidebar filters** on the Stocks page: sector, today's move (up/down), the model's rank (top or bottom 20%) and range (calm = narrowest third, wild = widest third), with a count of how many are on, "Clear filters", a count of stocks shown, and an empty state that offers to clear them. Rank and range follow the look-ahead; filters are remembered in the browser; the panel folds away (closed by default on phones).
- **Compact list rows** (about 50 px instead of 90): logo, ticker, small chart, price and change. The second line shows the company name, or the number the list is sorted or filtered by (rank, range, earnings date). The range bar per row is gone; range width is now a filter and a sort.
- **"Against the market"** (the model's rank for 5, 20, 60 days and what that rank did in testing) moved from the bottom of the side panel to right under the price.
- **Sector** comes from the industry code each company reports to the SEC, now kept by the daily filings check (`data/live/companies.json`) and mapped to 11 sectors (`SIC_SECTORS` in `scripts/export_dashboard.py`; a few known exceptions such as Visa and Mastercard → Financials). Tried first: sectors from price behaviour (the model's own sector features); rejected for display because it put Apple and Union Pacific in Consumer staples. The filter appears after the next filings check; `run_dashboard` runs that check once if the codes are missing.
- Checked in the simulation: light and dark, 1440, 1280 and phone widths, filters on and off, no sectors file, Overview and Track record pages. 44 tests (1 new: industry codes of well-known companies map to the expected sector).

### 16 Nov 2026 — Methodology page with sources; the README's illustration replaced by a real result
- Published to GitHub: commit `12cd197` (daily model, dashboard, faster news download, docs).
- New `docs/methodology.md`: every method in plain words with why it was chosen and the research behind it: the question, data without hindsight, the 44 inputs and the papers that documented each effect, the model and its training, walk-forward testing with a gap and a locked holdout, each score, price ranges, the methods tested and dropped, the news test, limits. 36 references (journal articles, books and arXiv papers; arXiv numbers checked).
- README: the "what one forecast shows" picture used made-up prices. It is replaced by a chart of real test results (how often each v1.0 rank group beat SPY over 20 days, 2012–2024, from `rank_groups_h20.csv`). The banner stays as a labelled illustration: the free price data may not be republished, so no real-price pictures or dashboard screenshots are in the repo.

### 16 Nov 2026 — News download made about 5 times faster; price-range run at 60 days found failed
- The first version fetched every article (about 12,000 a month) to keep the third that name our stocks, one request at a time: about 1 request a second, 10 hours in all. It now asks Alpaca only for articles naming our stocks (in groups of 200 tickers, Alpaca spelling `BRK.B`) and fetches 3 months at once, staying under the 200-a-minute limit: about 95 requests a month instead of 240, three in parallel. The six months already saved are kept (same content, fetched the old way).
- If Alpaca refuses the long request, the message says to add `--group 50`.
- The 15 Nov price-range runs on universe v2 finished at 5 and 20 days; the 60-day run never started (a typo in the command, `--horizon 60python`). To rerun, and the 5/20-day results still have to be written up.

### 16 Nov 2026 — Docs cleaned up before publishing; one step-by-step guide added
- New `docs/project-flow.md`: the whole project as 15 steps (question, test bench, data, features, model comparison, v1.0, data error and audit, price ranges, dashboard, daily running, improvement tests, live record, holdout, report, launch), each with why, what was done, the result, the evidence and its status. Linked from the README ("start here") and from this file.
- This file: the status and to-do section at the top rewritten (it still said "waiting for the training run" and listed tasks finished days ago). Early results log, decisions and known issues brought up to date. The day-by-day log below is unchanged.
- `decisions.md`: models that were once "final" are marked *replaced*, v1.0 and the volatility range model stated as current, news marked in testing, SEC 8-K features marked "data downloaded, features not written".
- `results.md` limits and pending list, `data-sources.md`, `data/README.md` (folder layout incl. `data/live`), `kaggle-workflow.md` (`final-model`, `news-score`), `kaggle/README.md`, `models/README.md`, `CONTRIBUTING.md`, README status rows corrected.
- Checked before publishing: GitHub has the work up to 14 Nov and nothing private in its history; every file on the PC matches the working copy except `config/universe*.csv` (the PC's are the current ones); `.gitignore` keeps out `.env`, `kaggle.json`, `Claude outputs/`, `data/`, `models/*`, `app/data/`, logs, the design PDF and `test_keys.py`.

### 16 Nov 2026 — README and architecture doc rewritten for the current system
- `README.md`: status table; what the product gives and does not give; **models picked and dropped with their numbers** in one place (in the product, tested and dropped, written but not measured); system architecture diagram (PC, Kaggle, browser), the 12 components with what each reads and writes, the daily cycle, the server's three endpoints, every data file with its columns, technology; **requirements** (users, 21 functional requirements marked built or planned, 10 quality requirements, limits and assumptions); glossary. Enough to write an SRS and an SDS from.
- `docs/architecture.md`: brought up to date and extended: research half and daily half, features by name, the v1.0 network and its training, evaluation and look-ahead rules, both range models, the daily model (training, joining prices, scoring), daily price rules, the local server, the dashboard, failure handling.
- Corrected in passing: the range model shown in the product is the volatility range model (the README still named quantile XGBoost); the daily rank is connected (the dashboard README said it was not); the full training took 6 minutes, not 15.
- All three diagrams were rendered to check they draw.

### 16 Nov 2026 — News: Alpaca has enough; download and merge built
- Probe result (run on the PC): 435–869 articles on the sampled days of 2015–2021, naming 65–142 of about 500 index members; 843–1,144 a day in 2022 to March 2026, naming 125–168. Thinnest in 2018–2019 (about 13% of members), 15% or more on 13 of 15 sampled days. Enough to test, unlike the direct Benzinga plan (about 2 a day).
- Found in the probe: the last week has 2,800–3,000 articles a day because a second publisher, fiscal.ai, appears (1,500–2,000 a day). Only Benzinga articles are kept so the news counts mean the same thing in every year. The publisher is blank before 2022 (all Benzinga then) and is kept.
- `python -m src.ingest.alpaca_news --start 2015-01`: downloads month by month (headline and summary, only articles naming a stock of our list), about 60,000 requests, several hours. Each finished month is saved; stopping and running the same command again continues. It slows itself down before the per-minute limit.
- `python -m src.ingest.alpaca_news --combine`: one file in the columns the news features already read, `data/clean/news_articles.parquet`, and a table of articles and stocks per year. The earlier thin file is kept once as `news_articles_benzinga_plan.parquet`.
- Checked with a stand-in for the service: two months download, a second run asks for nothing, the merged file feeds the existing news features. 43 tests pass (1 new). The real download has not been run yet.
- Next, in order: download → combine → `python -m scripts.kaggle_cloud upload-data` → `news-score` (sentiment on Kaggle's GPU) → compare with and without `--news` on test years 2018–2024 at 5 and 20 days. News goes into the model only if it beats v1.0 by clearly more than chance (see results section 0).
- Known limit: articles carry the ticker used at the time, so a company that changed ticker loses its earlier news.

### 16 Nov 2026 — Chart widget improved
- Above the chart: change over the chosen period with its highest and lowest close. On the chart: the period's high and low, an "E" on each past earnings day, a dotted line at the current price, price and date labels on the axes when pointing.
- Controls: 1W added to the periods; a volume on/off button (remembered); a button that opens the chart across the whole window (Esc closes); a key under the chart explaining the three bands and the earnings marks.
- Checked in the simulation in light and dark, line and candles, all periods, and the full-window view.

### 16 Nov 2026 — News sentiment: Alpaca probe written (waiting for keys)
- News is still not in the model: the Benzinga plan gave about 2 articles a day market-wide, too thin to compare stocks. Alpaca's news API (Benzinga and other publishers) is the candidate replacement.
- Alpaca has a historical endpoint and a live stream. Only the historical one is needed: it serves both the back history and "everything since yesterday". The live stream matters only for showing headlines the second they appear.
- `python -m src.ingest.alpaca_news --probe` counts articles on one trading day per year (2015 on) and the last three days, and how many index members they cover. Needs `ALPACA_API_KEY` and `ALPACA_API_SECRET` in `.env` (blank entries added to `.env.example`). Nothing is saved. Decision after the probe: download and test news features against v1.0, or drop it.
- Still to solve before any commercial use: the sentiment scorer (FinBERT) is non-commercial, and Alpaca's terms for redistributing news text must be read.

### 16 Nov 2026 — ✅ Daily model trained on Kaggle (`models/latticeiq_v1.npz`)
- `python -m scripts.kaggle_cloud final-model` finished in 6 minutes: **722 stocks, 2.7 million rows, 44 features, prices from 3 Jan 2000 to 5 Nov 2026**. The saved copy matches the trained network to 1e-8. File checked after download: three horizons, three networks each, 44 → 64 → 32 → 1.

| Horizon | Rows | Last date with a known outcome | Trust factor k | Rank skill since mid-2021 (model trained to mid-2021) |
|---|---|---|---|---|
| 5 days | 2,707,124 | 29 Oct 2026 | 0.37 | +0.009 |
| 20 days | 2,698,770 | 8 Oct 2026 | 0.20 | +0.001 |
| 60 days | 2,676,420 | 12 Aug 2026 | 0.17 | +0.003 |

- **How to read it.** k only scales the forecast size (the expected return shown is 17–37% of the raw output); it does not change the order of stocks. The last column is a check made during training: a copy trained to about mid-2021 and never retrained is scored on mid-2021 to late 2026. It is a harder test than the walk-forward runs (which retrain every year and averaged 0.015–0.020 over 2012–2024), but it says the same thing as the weak 2024 in those runs: **the edge has been thin in recent years, clearly positive only at 5 days.** The daily rank is a mild tilt, not a strong signal. This is the main argument for the improvement tests.
- **Holdout note.** That check window includes 2025–26, with a model that had not seen those years. It is one blended number over five years, not the holdout test, and nothing will be chosen or tuned from it. Recorded here so the holdout claim stays exact.
- **Fixes after the first real run:** the price update now notices that the saved prices lack the former index members the model ranks against and downloads them once (remembered in `data/live/tickers_asked.json`); the dashboard re-ranks after a retrain even if prices have not changed; failed downloads are retried once, not twice.
- **Next start of the dashboard** downloads about 650 tickers once, ranks the 500 members with the new model and shows the ranks.

### 16 Nov 2026 — ✅ "Train once, score daily" built (the ranking model can now run every day)
- **What it is:** `python -m scripts.kaggle_cloud final-model` trains v1.0 once on all data (5, 20, 60 days) and copies `models/latticeiq_v1.npz` to the PC. From then on `python -m scripts.run_dashboard` ranks every stock with it after each close (about a minute, no PyTorch needed) and the page shows it: a "Rank against the market" panel on each stock, a "model rank" sort in the list, and highest / lowest ranked on the Overview. Retrain monthly or quarterly with the same command.
- **Every forecast is logged** to `data/live/forecasts/<day>.csv` before its outcome is known. Scoring these against what happened (the live record) is the next step; it needs at least 5 trading days of logs.
- **Checks done on the staged data (cloud workspace):** the saved numpy copy of the net matches PyTorch to 1e-9; research prices cut three weeks early and continued with a differently scaled daily file reproduce the scores exactly (difference 0.0); more price history changes nothing. Found: limiting filings to recent years changes fundamental features for some stocks, so scoring uses every filing (v1.1 item, see `decisions.md`).
- **Not done here:** the real training run. The full panel needs about 8 GB of memory, more than the cloud workspace has, so it was tried only on a reduced set (2017 onward, every 5th day) to prove the chain end to end. The real model comes from the Kaggle command above.
- **Known limits:** fundamentals are as of the last `python -m src.ingest.edgar` run (not refreshed daily yet); the daily model learns from 2025–26, the one-time holdout test will train its own model to 2024.
- 42 tests pass (2 new: prices are joined without a jump and a split after the freeze is recorded; the saved model equals the trained one).

### 16 Nov 2026 — Modern chart, ranges that follow the price, faster background updates
- **Chart:** line with a soft fill by default (green or red by the period's direction), candles as an option (remembered); a floating label follows the pointer (price and day's change, today so far, or the likely range on a future day); dotted grid; the line's end pulses while the market is open. Bug fixed on the way: a new range arriving during the cone's opening animation could freeze the cone half-drawn and hide the earnings marker.
- **Ranges follow the price:** while the market is open, the three ranges, the cone and the ladder are re-centred on the price now and sized for the time left to the same end date; the line under them shows the range as made at the close. See the decision in `decisions.md` (this replaces "fixed until the close").
- **Background updates:** the stock on screen refreshes every 20 seconds (was 60), all stocks every 10 minutes (was 15). After the close the finished day is downloaded and every forecast rebuilt without anyone touching it. `start_dashboard.bat` starts it minimised with a double-click.
- **Limits, stated on the page and in the README:** prices are Yahoo's and can be 15 minutes late; to-the-second quotes need a broker or paid feed. The live range is the tested range re-centred, not a separately tested model.
- **Model: what "updates itself" means today and what is next.** The range model already rebuilds from each finished day. The ranking model (v1.0) is not yet scored daily. Plan, in order: (1) daily scoring job: train once on all data, save the weights, score every stock after each close and show the rank on the page; (2) keep a live record of every forecast against what happened; (3) test, one at a time against v1.0 on the same years: 8-K event features, sector-neutral target, finance-pretrained Chronos, and (once a few months are saved) analyst target and rating changes; (4) ranges: adaptive calibration so the hit rate self-corrects after turbulent weeks.

### 16 Nov 2026 — "What analysts expect in 12 months" card (TradingView-style price targets)
- Request: a forecast view like TradingView's price-target page (past year of prices, then a fan to the high / average / low 12-month target, plus analyst ratings).
- Added to every stock page: that picture, the average / highest / lowest / median target with the change from today's price, the buy / hold / sell counts, and how often the last four earnings beat the estimates. LatticeIQ's own 60-day range is drawn on the same picture as a thick blue bar, so the two can be compared by eye.
- Source: Yahoo Finance, one stock at a time when it is opened, kept for a day (`src/ingest/analysts.py`, `/api/analysts` in `scripts/run_dashboard.py`). Check from the terminal: `python -m src.ingest.analysts NVDA`.
- **Stated on the page:** these are analysts' opinions, not LatticeIQ forecasts, and they have not been tested here. Each day's numbers are saved to a history per stock, because Yahoo only gives today's consensus; after a few months this can be tested as a model input (target versus price, rating changes).
- Not tested against the real Yahoo service from the cloud workspace; tested with stand-in data. 40 tests pass (1 new: fields are read defensively, a failed fetch falls back to the saved copy).

### 16 Nov 2026 — Logos and look brought closer to the big finance sites
- Request: logos were small, low quality and inside a white box; make the page look like Coinbase, Kalshi and Crypto.com.
- Logos: the white box and padding were ours and are gone. Every logo is now one rounded square; solid-square logos fill it edge to edge, see-through ones sit on a light tile. Coverage went from 297 to 492 of 500 stocks: every company address in `app/logos.js` was checked against the logo service from the PC's browser (many big companies are filed under addresses such as `corporate.walmart.com`). No logo found for AZO, CF, COF, EVRG, OKE, TTWO, ZTS, VMRK: letter tiles.
- Limits checked: the free service serves 128-pixel squares only, and its terms forbid bulk download and editing, so logos are not copied into the project or recoloured. Logos with their own white background show as a white square in dark mode. A free logo.dev key in `app/config.js` gives double-resolution, see-through logos in light and dark versions for all stocks.
- Look: company name first with the ticker as a small tag and the price large underneath; pill-shaped navigation, search, look-ahead, chart-period and list-filter controls; rounder cards and softer lines; cooler, lighter background.
- Bug found while checking on the PC: after opening a stock from another page, the candles could be squeezed into a corner with the cone cut off. The chart library was told the date range before it had noticed the chart's new size. The page now sizes the chart itself before setting the range.
- Company names are tidied for display (`tidy_name` in `scripts/export_dashboard.py`): "Cdw Corp" → "CDW", "Mckesson Corp" → "McKesson", "Bank Of America Corp /De/" → "Bank of America".

### 16 Nov 2026 — Layout fixed for laptop screens
- Reported: the page was not showing correctly. Opened the running dashboard on the PC and checked it at laptop sizes (1280×600, 1536×730). Found: the price ranges in the three forecast cards spilled out of their cards, the footer wrapped to two lines and took screen height, and the page was split into three small panes that each scrolled on their own, so the chart was mostly hidden.
- Fix: the page now scrolls as one page (the top bar, the stock list and the right-hand panel stay in place); the forecast cards size their text to the space and wrap between the two prices; the chart height follows the window so the whole chart fits on screen; thinner scrollbars; narrower side columns below 1480 px. Also: a pasted or edited address such as `#NVDA/20` now opens that stock.
- Checked at 1280×600, 1536×730, 1440×900, 900 and 390 px wide, light and dark.

### 16 Nov 2026 — Today's prices shown next to the forecasts
- Request: the page should show today's prices (16 Nov), not only the last close, together with the forecasts.
- While the dashboard runs and the US market is open, it now fetches today's prices so far: all stocks, the market strip and the Overview every 15 minutes; the stock on screen every minute, with today's candle on the chart. The big price becomes today's price with the last close under it. Before the open, the page says when the market opens in local time (7:30 pm in Pakistan) and shows the last close, since nothing newer exists.
- **Decision:** forecasts stay anchored to the last finished close and do not move during the day. That is how they were tested (close to close); re-anchoring on a half-finished day would make them a little too narrow. The page states whether today's price is inside, above or below the range. About 20 minutes after the close the dashboard downloads the finished day, rebuilds the forecasts and the page reloads itself.
- Code: `live.quotes()` and `live.market_state()` in `src/ingest/live.py`; `/api/today`, `/api/quote` and a background thread in `scripts/run_dashboard.py`. 39 tests pass (2 new).
- Not tested against the real Yahoo service from the cloud workspace (no access there); tested with a stand-in price source. First real run is on the PC.

### 16 Nov 2026 — ⚠ Dashboard prices were out of date: daily update added
- **Problem (reported from using the page):** prices looked wrong and the page seemed to show past prices, not future ones. Checked the data on the PC: the page said "11 Nov" but NVDA's price was its 6 Nov close and AAPL's its 5 Nov close, while the real last close was Fri 13 Nov. Cause: the research downloaders (`tiingo.py`, `yahoo.py`) fetch each ticker once and skip it afterwards, so `data/clean/prices.parquet` stops on whatever day each ticker was downloaded. Nothing updated it. The ranges were computed from those old prices, so they were off too. The range method itself was not the cause.
- **Fix:** `src/ingest/live.py` downloads the latest closing prices (three years, current members + 15 market funds, Yahoo) into `data/live/prices.parquet`; an unfinished trading day is never included. `python -m scripts.run_dashboard` now runs this on every start when a newer closing day should exist, refreshes SEC filings once a day, and rebuilds the page data only when something changed. The research file stays frozen (data freeze "universe v2"); the range multipliers are still fitted on it and are saved, so a daily rebuild takes seconds.
- **Page:** each stock now opens with "Where the price is likely to be": the range and end date for 5, 20 and 60 trading days. The big price is labelled as the close of a named day. A bar at the top warns when the prices are old or are saved ones. Candles now match public charts (adjusted for splits only).
- Correction to the note below: `python -m src.ingest.yahoo` does not bring existing tickers up to date; it only downloads tickers with no file.
- 37 tests pass (2 new: the update waits for the market to close; unfinished days, holes and missing tickers are handled, and a failed download keeps the file on disk).
- **Still true:** these are end-of-day prices from Yahoo, for research use. Live quotes and a commercial launch need a licensed feed.

### 16 Nov 2026 — Dashboard redesigned (Overview, search, logos, market strip)
- Three pages now. **Overview** (opens first): how many stocks rose and fell, a two-week earnings calendar, biggest moves of the day, calmest and wildest stocks for the chosen look-ahead, and important SEC filings. **Stocks**: the list (now with company logo, 30-day sparkline, day change, range width), the chart with 1M / 3M / 6M / 1Y buttons and a readout that shows prices on candles and the range on any future day in the cone, key figures (day range, 12-month range, volume, distance from high), then the ranges, earnings and filings. **Track record**: unchanged content.
- New: search by ticker or company name (press `/`), starred stocks (kept in the browser), "Earnings soon" filter, market strip of index and sector funds, up/down arrow keys to step through the list.
- Logos: about 300 well-known companies load from `cdn.tickerlogos.com` with no key (attribution link in the footer); a free logo.dev key in `app/config.js` covers the rest; letter tiles otherwise. `useTickerLogos: false` turns them off.
- Export fixes found while testing: (1) if a price download stopped half-way, stocks had different last days and "today's moves" mixed days. The export now shows every stock as of the latest day that 9 in 10 stocks have reached and prints a note. (2) A company that has already reported is shown as "Reported earnings on …" with the following report as next, instead of "expected". The range keeps its earnings width for those few days, exactly as it was tested.
- The data format changed: rebuild once with `python -m scripts.run_dashboard --refresh`. 35 tests pass.

### 15 Nov 2026 — Dashboard built (real candles, price cone, scheduled earnings, track record)
- `python -m scripts.run_dashboard` opens a local page (`app/`): list of all ~500 current members with each stock's range drawn as a width on one shared scale; candlestick chart (TradingView Lightweight Charts, vendored) with the 50 / 80 / 90% cone for 5 / 20 / 60 trading days; amber marker and wider range when an earnings report is expected in the window; past earnings-day moves; recent SEC filings in plain words; light and dark themes; works on a phone.
- Track-record page: every range replayed walk-forward on 2012–2024, said-vs-done table, year-by-year plot against the 80% aim, the ranking model's results, and what the site can and cannot tell you.
- New range model behind it, `src/models/vol_ranges.py` (blended volatility + earnings-aware multipliers; multipliers fitted on windows ending by 31 Dec 2024, holdout not scored). Measured: 80% range held 79.5% / 79.0% / 79.1% at 5 / 20 / 60 days; width ±4.6% / ±9.5% / ±16.6%; earnings windows 77–79% (a one-size range: 57%).
- `scripts/export_dashboard.py` builds `app/data/` (git-ignored: licensed prices). Company names from the SEC ticker list; logos optional via a free logo.dev key in `app/config.js` (letter tiles otherwise).
- 35 tests pass (2 new: no look-ahead in the volatility, hit rates near target on synthetic data, earnings-window flag).
- **Not connected yet:** the daily rank from the v1.0 model (needs the forecast job). Next: forecast job on Kaggle → rank panel; price-range baseline run on universe v2.

### 15 Nov 2026 (price ranges: how narrow can they honestly be?)
- Question from the product side: ranges must be narrow enough to act on, and still right. Answer in numbers: range width is set by the stock's volatility. Example from our data: NVDA at $186.64, 5-day 80% range ≈ ±7.0% ($174–$200); KO ±3.3%; TSLA ±8.4%. Nothing honest can be much narrower at the same confidence; the lever is a better volatility forecast and knowing when a big move is scheduled.
- Quick checks on the staged price data (5 days, 80% level, walk-forward multipliers, 2012–2024):
  - volatility input: 20-day close-to-close 9.64% average width → Garman-Klass 9.28% (−3.7%) → blend of four measures 9.13% (**−5.3%**, interval score −2.6%), same coverage (79%).
  - earnings weeks (7% of stock-weeks): typical move is 1.7× larger; a one-size 80% range covers only **56.6%** there vs 80.6% elsewhere. Earnings-aware ranges: 76% coverage in earnings weeks with the true date (64% with a crude "same week as last year" estimate), other weeks 3–4% narrower.
- Plan (in `docs/decisions.md` candidates): blended volatility features + earnings-in-window feature in `run_brackets`, adaptive conformal calibration, interval score and earnings-week coverage reported. Baseline = current ranges on universe v2 (run pending).

### 15 Nov 2026 — ✅ FINAL v1.0 numbers on universe v2 (audited list)
- Kaggle run v16 finished: **722 stocks** (cleaned list confirmed in the log), 2.71 M rows. Runs `20261114_0752_…h5`, `20261114_0809_…h20`, `20261114_0827_…h60`; summaries saved in `results/`.

| 3-seed MLP (v1.0) | Rank IC (t) | Hit rate | Positive years | Old list (634) | Before audit drops (727) |
|---|---|---|---|---|---|
| 5 days | **0.0149 (3.07)** | 50.5% | 12 of 13 | 0.0136 (2.63) | 0.0141 (2.84) |
| 20 days | **0.0197 (3.18)** | 50.7% | 10 of 13 | 0.0180 (2.66) | 0.0192 (2.95) |
| 60 days | **0.0199 (2.23)** | 51.0% | 10 of 13 | 0.0229 (2.39) | 0.0202 (2.23) |

- Ensemble v2 0.0141 / 0.0173 / 0.0123; XGBoost 0.0094 / 0.0067 / 0.0016. MLP best at all horizons. At 20 days the MLP now also beats Random Walk on forecast error (DM p 0.02).
- **Top vs bottom groups (the number competitors advertise):** top 20% beat SPY 50.2% / 49.9% / 50.1%; bottom 20% 48.4% / 47.7% / 47.4%; top minus bottom +0.12% / +0.46% / +0.76% in 11 / 11 / 10 of 13 years. Same conclusion as before: the top group matches the market, the bottom group lags.
- Validation vs test: 0.016 vs 0.015, 0.020 vs 0.020, 0.023 vs 0.020 → no sign of memorising.
- Weak points: 2024 negative at 20 and 60 days; 2019 and 2021 about zero at 20 days.
- **Noise floor learned:** five stocks fewer moved single years by up to 0.015 and the overall score by 0.0005–0.0008. Add-ons must beat v1.0 by clearly more than that.
- README, results.md (section 0 + 0b), decisions.md, market-comparison.md and the by-year chart updated to these numbers.
- **Next:** price ranges on universe v2 (`kaggle_cloud brackets`, 3 horizons) → crash risk + recovery → SEC 8-K event features → FinText → sector-neutral target.

### 14 Nov 2026 (market comparison + README brought up to date)
- New `docs/market-comparison.md`: what competitors publish (Danelfin +14.69%/yr for top scores and +376% vs +166% backtest; Prospero.ai 54% win rate; AltIndex 70% win rate, all self-reported or backtested), what research reports (51.74% next-day direction for the best finance-pretrained model), where we stand, why the numbers are not comparable, a checklist for reading accuracy claims, and what LatticeIQ can and cannot claim. Sources listed.
- README rewritten to the current state: corrected-list results first, "what we claim / don't claim" table, market comparison, 44 features, data-honesty section (index coverage, wrong-company audit), 33 tests, roadmap without stale items. Charts regenerated (rank IC by year now shows v1.0 on the corrected list).
- `docs/results.md`: new section 0 with the corrected-list tables, by-year values and validation vs test. `docs/decisions.md`: v1.0 re-measured table added.
- Run summaries saved in `results/20261113_*` (copied from the Kaggle logs).
- README images: new hero banner (candles, volume, forecast cone), an example forecast card and a "how wide the ranges are" strip (light and dark). Candles are synthetic and labelled so; the range widths (±4.7% / ±9.9% / ±17.6%) are measured. Up/down candle colours checked for colour-blind separation.
- Universe v2 rebuilt on the PC: 819 tickers, 801 stocks, 737 with price files; audit clean ("trading volume far below other members: none").
- **Next:** `upload-data` → final v1.0 run on universe v2 → price ranges → add-on tests. Goal for every add-on: make the top-ranked group beat the market, not only the bottom group lag.

### 14 Nov 2026 (v1.0 re-measured on the corrected stock list, before the six audit drops)
- Kaggle run v15 finished (727 stocks, 2.72 M rows, 3 × ~18 min). Runs `20261113_1855_…h5`, `20261113_1914_…h20`, `20261113_1932_…h60`.

| 3-seed MLP (v1.0) | Old list (634 stocks, frozen at Jan 2019) | Corrected list (727 stocks) | Positive years |
|---|---|---|---|
| 5 days | IC 0.0136, t 2.63, hit 50.5% | **IC 0.0141, t 2.84, hit 50.4%** | 11 of 13 (2021 flat, 2023 negative) |
| 20 days | IC 0.0180, t 2.66, hit 50.6% | **IC 0.0192, t 2.95, hit 50.6%** | 12 of 13 (2024 negative) |
| 60 days | IC 0.0229, t 2.39, hit 51.3% | **IC 0.0202, t 2.23, hit 51.0%** | 9 of 13 (2013, 2015, 2017, 2024 negative) |

- Ensemble v2: 0.0137 (t 2.48) / 0.0175 (t 2.30) / 0.0130 (t 1.26). XGBoost alone: 0.0092 / 0.0086 / 0.0033. MLP stays the best single model at all three horizons.
- Reading: the skill did not come from the wrong stock list. 5 and 20 days are slightly stronger on the true index, 60 days slightly weaker and less steady. **2024, the most recent test year, is negative at 20 and 60 days**: to be watched on the final run and the holdout.
- This run still contains the six wrong-company spells (LSI in 2012–14). Final v1.0 numbers = the next run on "universe v2".

### 13 Nov 2026 (universe audit, part 2: impostor histories → data freeze "universe v2")
- Hand fixes applied on the PC (WEN, CSR dropped; CPWR left out of SEC data): 822 tickers, 740 stocks with prices; audit table unchanged (91% of true members in test years).
- Went through the audit's hand-check list and added a new automatic check: **dollar trading volume of each spell vs other members of the same years**. Real members sit at 0.06–10× the typical member; five spells sit below 0.05: KMG (Kerr-McGee's symbol, KMG Chemicals' prices), CSR, MIL (Millipore's symbol, a small fund's prices), WEN and **LSI (LSI Logic's symbol, Life Storage's prices, 2000–May 2014)**. Plus GAS 2000–11 (Nicor's symbol, AGL Resources' prices; found from the missing July-2002 crash).
- LSI is the only one inside test years (1 stock of ~405, 2012–May 2014). All six spells now dropped in `build_universe.py`; check added to `audit_universe.py` (33 tests pass).
- Decision: **freeze the data as "universe v2" after this**; re-run v1.0 once more on it so every headline number comes from one audited data set. The run started earlier today (Kaggle v15, before these drops) is kept as a before/after check.

### 13 Nov 2026 (universe audit)
- MRVL / NAVI / FTI earlier registrations confirmed by name (Marvell Technology Group, Navient, FMC Technologies) ✅. Fundamentals 1,246,413 values; 8-K 201,884 filings. Data uploaded to Kaggle.
- New `scripts/audit_universe.py` (33 tests pass): checks every membership spell against the price files and counts true index members with prices per year. **Coverage of the real index: test years 2012–2024 81% → 98% (average 91%); training years before 2012 average 67%.** Table in `docs/data-sources.md`.
- Audit findings: 18 spells have a price file that belongs to a later company (no prices inside the membership dates → harmless, simply absent). Three cases where the history under the symbol is another company's, fixed by hand: WEN 2000–08 (Triarc's prices), CSR 2000 (Centerspace) → spells dropped in `build_universe`; CPWR fundamentals (a penny stock's filings) → left out. All three only touch training years before 2012, never test years.
- Corrected an earlier note: JCI / CB / PLD before their mergers carry the merger partner's prices **and** fundamentals (the legal survivor), so the series is consistent; the absorbed company is what is missing.
- The v1.0 re-run started today was launched before the three hand fixes; they change < 0.3% of training rows. Later runs include them, and every add-on test runs its own v1.0 baseline in the same job.

### 13 Nov 2026 (universe rebuild: company data done)
- `yahoo --prune`: 13 files set aside (10 wrong-company + 3 one-row stubs KVUE, WBD, QRVO) → **774 tickers** in `prices.parquet`.
- `build_universe`: 1,066 members since 2000, 1,108 spells; `universe.csv` 824 tickers; **742 stocks with prices** (was 634), 64 without (53 not on Yahoo, 8 wrong-company, 3 stubs).
- Fundamentals re-downloaded for all 806 stocks: **634 companies, 1,242,870 values**; 172 not matched. 8-K events: **201,179 filings, 634 companies**.
- All 12 earlier SEC registrations checked by name: Exxon, Disney (TWDC), Google, Cigna, Medtronic, BlackRock, Xerox, Apache, Eaton, Broadcom/Avago, ICE ✅.
- Found while reading the output and fixed in code (32 tests pass): (1) SEC's ticker list maps an old symbol to today's owner of that symbol → a symbol whose registrant keeps filing 400+ days after the stock's prices ended is now left out; (2) 3 more companies changed SEC number (MRVL, NAVI, FTI) → earlier registration added, and every earlier registration is now accepted only if its SEC name matches.
- Known and written down, not fixed yet (`docs/data-sources.md`): JCI / CB / PLD carry the merger partner's fundamentals before their merger dates; Mylan history not joined to VTRS.
- **Next:** refresh MRVL NAVI FTI → `upload-data` → re-run v1.0 (5/20/60d) + price ranges.

### 13 Nov 2026 (universe rebuild: prices done)
- Yahoo fill-in finished for the corrected stock list: **776 tickers, 4,225,730 rows** in `prices.parquet` (was 634 stocks + ETFs). New: TSLA, META, MRNA, UBER, PLTR, NOW, PANW, CRWD and ~110 more.
- **53 not on Yahoo** (taken over / delisted, e.g. TWX, TWTR, WFM, RTN, XLNX, PXD, WBA, X, TIF, PX): still missing → survivorship gap stays, to be stated with the final coverage numbers.
- 11 wrong-company (recycled) files set aside as before (S, STI, SE, POM, SUN, SII, SPLS, UN, QRVO, PARA, SBNY).
- Found 2 **stub files** (KVUE, WBD: one row each, companies delisted in Oct 2026 and Yahoo removed their history). Fix: files with < 20 rows are now skipped at download and set aside by `--prune` (test added; 31 tests pass).
- PC problem solved: Windows **Smart App Control** blocked pyarrow 26.0.1 ("Application Control policy has blocked this file"). pyarrow 22.0.0 loads → pinned `pyarrow>=15,<23` in `requirements.txt`. Do not upgrade pyarrow on this PC.
- **Next:** `yahoo --prune` → `build_universe` (final coverage) → fundamentals + 8-K for new tickers → `upload-data` → re-run v1.0 (5/20/60d) + price ranges, old vs new side by side.

### 13 Nov 2026 (open-source model survey)
- Searched Hugging Face + recent papers for the best open-source model to fine-tune. Full table in `docs/decisions.md` → "Open-source model survey".
- **Pick: FinText finance-pretrained Chronos-Small** (Apache-2.0; trained from scratch on US daily excess returns; one version per year 2000–2023, so test year Y uses the version trained to Y−1, no look-ahead). Second: Chronos-2 (Apache-2.0). Optional: TabICL v2 (BSD-3) as a third ensemble member.
- Ruled out by licence: TabPFN-2.5, Moirai 2.0 (non-commercial), TiRex (custom licence). Ruled out by evidence: fine-tuning Kronos/TimesFM ourselves (paper 2511.18578: fine-tuning general models on stock returns barely helps; pre-training on finance data is what helps).
- **Next (after the universe rebuild):** `fintext` Kaggle job (same pattern as Kronos/TimesFM) → forecasts as features → compare with v1.0 on the same folds.

### 12 Nov 2026 — ⚠ DATA ERROR FOUND AND FIXED IN CODE: membership list stopped in Jan 2019
- SEC 8-K history downloaded: 169,811 filings, 494 companies (44,227 earnings). Live feed `--latest 7` works (145 filings, 2 important).
- Checking it exposed a bug in `build_universe.py`: `sorted(...)[-1]` picked "…Changes.csv" (ends 2019-01-11) over "…Changes (Updated).csv" (to 2026-08-18). Consequences: no post-2019 entrants (TSLA, MRNA, UBER, PLTR, NOW… ~150), Jan-2019 members treated as members to 2026, renamed tickers (FB→META, ANTM→ELV, HCP→DOC, ~30) lost from the rename date, and three confirmed wrong-company price series inside membership dates (EP 2011–12 = Empire Petroleum, LB 2024 = LandBridge, HCP 2021–25 = HashiCorp; APC 2026, FB 2025 in the holdout period).
- No look-ahead was involved; 2012–2018 test years used the right list; 2019–2024 used the frozen Jan-2019 list.
- **Fixes (31 tests pass):** build_universe uses the Updated file + refuses a file > 400 days old; `RENAMES` (31 renames + IR→TT before Mar 2020) joined into continuous spells; spells rejected when the symbol's prices start > 400 days late and run on 3+ years after the stock left; Yahoo recycled check is now per spell; `PREDECESSOR_CIKS` for 12 re-incorporated companies in both SEC downloaders; date-parsing warnings fixed.
- **Next:** rebuild universe → Yahoo fill-in (~170 tickers) → fundamentals + 8-K for new tickers → re-upload → re-run v1.0 (3 horizons) and price ranges on the corrected universe; report old vs new side by side. 8-K features after that.

### 12 Nov 2026 (price ranges final; P(up) rejected; plain-number check of v1.0)
- **Price ranges on 634 stocks ✅ final:** 80% coverage 0.798 / 0.803 / 0.802 and 90% coverage 0.900 / 0.904 / 0.908 at 5 / 20 / 60 days; beats naive at all horizons. 2020 weak (0.715).
- **P(up) OUT:** Brier worse than the base rate at all horizons (20d 0.2513 vs 0.2446), no cross-sectional skill (IC −0.009), extremes wrong (predicted 10% → 57% actual). Shrinking it toward the base rate gave no real gain either.
- **v1.0 in plain numbers** (rank groups): top-minus-bottom 20% = +0.11% / +0.40% / +0.80% (5 / 20 / 60d), right in 12 / 12 / 11 of 13 years; bottom group beats SPY 48%, top group ~50%. Skill = avoiding laggards; top group only matches SPY. Added "what v1.0 can honestly claim" table to `decisions.md`.
- README coverage chart regenerated on 634 stocks.

### 12 Nov 2026 (recovery model + SEC event feed)
- **Recovery after a crash** `scripts/run_recovery.py`: crash = −10% or worse in 5 days (one per stock per 20 days); recovered = close back at the pre-drop price. Output A: history table (share recovered within 5/10/20/60/120/250 days + median days, by market-wide vs stock-specific, drop size, one-day shock vs slide). Output B: calibrated P(recover within 60 or 20 days), walk-forward, vs base rate and drop-size-only baselines. On random synthetic prices the model shows no skill (AUC 0.51), as it should. Kaggle: `kaggle_cloud recovery "..."`.
- **Official news feed** `src/ingest/sec_events.py`: all 8-K filings per company from SEC submissions API with event kinds + severity 0–3 (restatement, bankruptcy, delisting notice = severe). `--latest N` = live feed of the last N days. History will be tested as features (earnings timing, negative events).
- Tests 27 → 30 (event timing of recovery, no-skill-on-noise, 8-K item parsing).
- Next: run crash + recovery + sector-target on Kaggle; download 8-K history on the PC; then 8-K features and insider data.

### 12 Nov 2026 (v1.1 work started)
- **Crash-risk model** `scripts/run_crash_risk.py`: P(drop > 10% in 20 days), XGBoost classifier + isotonic calibration; baselines = base rate and a volatility-only logistic model. Metrics: AUC, Brier skill, crash rate in the riskiest 10% (lift), calibration by decile, AUC by year. Kaggle: `kaggle_cloud crash "..."`. Synthetic check: calibrated within a few % (test added).
- **Sector-neutral target** `run_experiment --target sector`: models learn stock minus its point-in-time sector ETF; scored vs SPY like v1.0. Test added (target = stock − its ETF). 27 tests pass.
- Next: run both on Kaggle; then insider trading (Form 4).

### 12 Nov 2026 — 🔒 MODEL LOCKED: LatticeIQ v1.0
- Final run (634 stocks): MLP 3-seed **5d 0.0136 (t 2.63) · 20d 0.0180 (t 2.66) · 60d 0.0229 (t 2.39, hit 51.3%)**; ensemble v2 0.0135 / 0.0159 / 0.0176; XGBoost 0.0099 / 0.0064 / 0.0087.
- **v1.0 = 3-seed MLP** (primary) — significant at every horizon. Ensemble v2 kept as secondary. Deviation from the pre-declared ensemble written up openly in `decisions.md`; both get the one-time holdout test.
- Next: price brackets on 634 stocks (3 horizons), then `forecast_today` job + Streamlit app. Accuracy work continues as v1.1+ (sector-neutral target, insider trading, short interest).

### 12 Nov 2026 (horizons + ensemble v2)
- 5d: XGB 0.0099 · MLP 0.0112 · ensemble 0.0120 (t 2.10). 60d: XGB 0.0087 · **MLP 0.0254 (t 2.81, hit 51.5%)** · ensemble 0.0161 (t 1.72).
- Lesson: on 634 stocks XGBoost is weak at all horizons; k-weights (calibration slope) still gave it ~half the vote.
- **Ensemble v2 (final, pre-declared):** weights = each model's rank IC on the held-out latest training years (`val_ic`, clipped at 0); MLP = mean of 3 seeds (`MLPSeeds`). Synthetic check OK, 25 tests pass. Logged k and val_ic per fold in shrink.csv.
- Next: one final run for 5/20/60 days → **lock the model** with those numbers, whatever they are.

### 12 Nov 2026 (full universe results)
- Full universe run (634 stocks, 279,757 stock-dates): XGBoost 0.0064 (t 0.79, unstable) · **MLP 0.0192 (t 2.97)** · equal-weight ensemble 0.0147 (t 1.93).
- MLP holds up on the 125 new stocks alone (0.0168, t 2.0). XGBoost's trust score k is 0 in several early years → noisy.
- New ensemble rule: weight each model by its shrinkage k (skill on latest training years). Trust-weighted XGBoost + MLP: **0.0199 (t 2.36, 12/13) on 466 stocks, 0.0170 (t 2.25, 11/13) on 634 stocks** → most stable → **final model**. Code: `ensemble_rank(weights=...)`; test added (25 tests).
- Next: 5/60-day horizons + price brackets on the full universe, then the daily forecast job + dashboard.

### 12 Nov 2026 (Yahoo fill-in)
- 196 universe tickers were never downloaded (Tiingo free cap stopped at PCAR): 133 still members (PEP, PG, UNH, V, WMT, XOM, …), 63 left the index.
- Free alternative instead of Tiingo upgrade: `src/ingest/yahoo.py` (yfinance) writes the same raw format as Tiingo, rebuilds raw prices from Yahoo's split-adjusted ones, cuts to Tiingo's date range per ticker. `--check` compares Yahoo vs Tiingo on 25 tickers first. Test added (split handling).
- `--check` ✅: 18 overlapping tickers compared: median daily-return correlation **0.9999**, only 0.08% of days differ by >0.5%, raw close ratio 1.0000 (split rebuild correct). Two tickers lower (FITB 0.945, MDLZ 0.981) from a handful of outlier days, which the cleaning step (`align_to_calendar`: blank one-day moves > ×5) handles. 7 sampled tickers not on Yahoo (delisted / acquired). Result file: `results/data_check_yahoo_vs_tiingo.csv`.
- Download: 141 tickers from Yahoo, total **641 tickers** in prices.parquet. 54 not on Yahoo: mostly delisted/acquired (PXD, TWTR, XLNX, RTN, TIF…), plus some current names that timed out (STX, STZ, likely UPS, STT, SEE) → retry added (3 attempts).
- Found recycled symbols (Yahoo gives a newer company: S = SentinelOne not Sprint, STI, SE, POM, SUN, SII, SPLS, UN, SNDK, TT…). Added `prune()`: a Yahoo history that starts >30 days after the stock joined the index is set aside in `data/raw/prices/_rejected/` (+ known case TT). Test added.
- Retry run: STT, STX, STZ, UPS recovered. 50 not on Yahoo (delisted/acquired: PXD, TWTR, XLNX, RTN, TIF, WBA, X…; current-looking SEE, SNV, TGNA likely taken over in 2025–26).
- Recycled check set aside 12: POM, QRVO, S, SE, SII, SNDK, SPLS, STI, SUN, TT, UN, plus SPGI by mistake → grace period 30 → 400 days (all real cases are 3+ years late), `--prune` now re-checks earlier rejects. **633 tickers** (≈615 stocks + 18 ETFs) vs 484 before.
- Next: re-upload, then re-upload data and re-run the ensemble on the full universe.

### 12 Nov 2026 (ensemble results)
- Real-data run: XGBoost 0.0140 (t 1.80) · LightGBM 0.0080 · **MLP 0.0149 (t 1.96)** · Ridge 0.0042 · 3-model ensemble 0.0143.
- Re-combined the saved predictions: **XGBoost + MLP = IC 0.0168, t 2.04, 11/13 years**, stable across 2012–17 (0.0171) and 2018–24 (0.0163). LightGBM ranks 86% like XGBoost (redundant); MLP only 43% → real diversification.
- **Decision:** final model = XGBoost + MLP rank-average ensemble (`ENSEMBLE_MEMBERS` updated). LightGBM, Ridge OUT. First result past t = 2.
- Next: sector-neutral target, full universe (Tiingo upgrade, waiting on the project lead), SEC insider/8-K features.

### 12 Nov 2026 (ensemble)
- Decision (with Naveed): push accuracy with an **ensemble** now; Kronos fine-tuning + live news in FYP-2.
- New `src/models/ensemble.py`: LightGBM, Ridge, MLP (PyTorch, GPU) + `ensemble_rank` (equal-weight average of per-date ranks). Every member gets the same shrinkage step as XGBoost.
- `run_experiment --models ... lightgbm mlp ridge ensemble`. Synthetic check: all find the planted momentum signal (IC 0.08–0.12, t > 7) and none show skill on noise.
- Tests 21 → 23. `lightgbm` added to requirements; CI installs CPU torch.
- Next: run the ensemble on real data.

### 12 Nov 2026 (TimesFM results)
- **TimesFM alone:** hit 49.95%, rank IC −0.0017 (t −0.27): no skill. **XGBoost + TimesFM:** IC 0.0128 (t 1.55), 9/13 positive years vs 0.0140 / 11/13 without → worse.
- **Decision:** TimesFM OUT. Both pretrained models tested; neither beats our cross-sectional XGBoost. Final FYP-1 model stays XGBoost (price + fundamentals + sectors). Written up as a key finding in `decisions.md`.
- Next: brackets with calibrated P(up) for 5/20/60 days, ARIMA baseline, then lock the final model.

### 12 Nov 2026 (TimesFM full run)
- TimesFM 2.5 zero-shot 2010–2024 ✅: 755 weekly dates, **239,117 forecasts** at 180/s, 23 min on one T4. `timesfm_forecasts.parquet` (9 MB) on PC and Kaggle. Next: comparison run.

### 12 Nov 2026 (TimesFM timing)
- TimesFM 2.5 loads and runs on Kaggle T4: 136 forecasts/s → weekly 2010–2024 run ≈ 0.45 GPU h (Kronos monthly took 1.7 h).
- Sanity: excess vs SPY mean −0.05%, cross-stock spread 1.6% (much more cautious than Kronos's 12%), average 80% band 16.4% wide (our brackets: 19.8%).

### 12 Nov 2026 (decision log)
- New `docs/decisions.md`: every model, feature group, method and data choice with status IN / OUT / PENDING and the reason. Updated after each experiment from now on.

### 12 Nov 2026 (Kronos results)
- **Kronos alone:** hit 50.1%, rank IC 0.0087, t 0.81, positive 7/13 years; raw forecasts too large (RMSE 0.140 vs 0.074).
- **XGBoost + Kronos features:** IC 0.0147 (t 1.83) vs 0.0140 (t 1.80) without: +0.0007, better in 6 years / worse in 5 → noise. Kronos features not in top features.
- **Decision:** Kronos left out of the final FYP-1 model (no reliable gain + needs daily GPU). Kept for FYP-2 (ensemble / US fine-tune; it was strong in 2016 when XGBoost was weak).

### 12 Nov 2026 (TimesFM)
- Kronos comparison job was cancelled during the Kaggle outage → re-pushed (version 9), running.
- **TimesFM 2.5 zero-shot runner** (`src/models/timesfm_runner.py`): weekly forecasts (same grid as evaluation), 512-day context, 5/20-day returns + 10th/90th percentile of the 20-day return. Not pretrained on stock prices → cleaner zero-shot test than Kronos. API checked offline (shapes); `--mock` mode for tests.
- Features: `timesfm_xret_5`, `timesfm_xret_20` (excess vs SPY), `timesfm_unc_20` (its own 80% band width). `run_experiment --timesfm` (features) and model `timesfm` (standalone). `pretrained.py` now handles both models.
- Tests 19 → 21: TimesFM features only use past forecasts; runner output unchanged when future prices are scrambled.
- Next: `kaggle_cloud timesfm --timing`, then full run, then comparison.

### 11 Nov 2026 (README visuals)
- README redesigned: banner (candlesticks + forecast cone, synthetic), app-structure diagram (data → features → models → daily forecast → dashboard/API), example forecast output, result charts (ablation, rank IC by year, bracket coverage) in light + dark versions in `docs/assets/`.
- Removed real API keys from `.env.example` (GitHub push protection caught the HF token). Keys to be rotated.

### 11 Nov 2026 (GitHub repo)
- Organised the code into GitHub repo `Naveed-CoLab/LatticeX`: new README (overview, results, quick start, layout), `docs/` (architecture, data sources, results, Kaggle workflow, design PDF), `data/`, `models/`, `results/` READMEs, CONTRIBUTING, issue + PR templates, GitHub Actions running the 19 tests on every push.
- `.gitignore` now also blocks `kaggle.json`; data, parquet files, models and logs are never committed. `test_keys.py` → `scripts/check_keys.py`; old `scripts/kaggle_experiment.py` removed (replaced by `kaggle_cloud.py`).
- Kronos comparison job (`experiments`) stuck in Kaggle's queue during a Kaggle outage.
- h5 / h60 results logged: h5 IC 0.0112 (t 1.86), h60 IC 0.0156 (t 1.31).

### 9 Nov 2026 (Kronos full run)
- Kronos-small zero-shot 2010–2024 ✅: 189 monthly dates × ~316 members = **59,699 forecasts**, both GPUs at 4.8/s, 104 min. `kronos_forecasts.parquet` (1 MB) saved on PC and uploaded to Kaggle. Next: comparison run.

### 9 Nov 2026 (Kronos timing v4)
- Dual-GPU run ✅: 4.5 + 4.7 forecasts/s (3× faster per GPU than v3, 6× overall) → full 2010–2024 run ≈ **1.6 GPU h** (was 9.8). Sanity: excess vs SPY mean −0.9% / −1.3%, spread 11–12% → sane.
- Waited ~1 h 40 min in Kaggle's T4×2 queue (Kaggle busy, not our error). Plan: one job for all of 2010–2024.

### 9 Nov 2026 (Kronos timing v2/v3)
- v2: CUDA out of memory (256 × 4 samples = 1,024 sequences) → batch now starts at 128 and halves automatically on OOM (tested offline 128→64→32).
- v3 ✅: forecasts sane — excess vs SPY mean −0.3%, cross-stock spread 12% (raw mean −4.6% cancels against SPY). Speed 1.5 forecasts/s (4 samples) → ≈9.8 GPU h: too much.
- Fixes: run on **both Kaggle T4 GPUs** in parallel (dates split, merged after; tested identical to single run), 2 sample paths (was 4), lookback 256 days ≈ 1 trading year (was 400). Expected ≈6× faster.

### 9 Nov 2026 (Kronos timing)
- `kronos --timing` on Kaggle T4 ✅ (Kronos loads and runs): 6.5 forecasts/s with weekly dates → ≈10.5 GPU h (too much). Forecasts biased (mean −5% 5d, −7% 20d, some −60%): single random sample path + Kronos's pull toward the 400-day mean in its per-window normalisation.
- Read Kronos source (`auto_regressive_inference`, `predict_batch`): recomputes the full context every step (no cache) → slow by design; `predict_batch` signature matches ours.
- Fixes: monthly forecast dates (every 20 trading days; 20-day forecasts still cover every period; features carried ≤31 days, never backwards), 4 sample paths averaged, batch 256, Kronos repo cloned to /tmp (no longer pollutes the job output). Years 2010–2024. Re-run timing to confirm.

### 9 Nov 2026 (Kronos pipeline)
- Built Kronos zero-shot pipeline (design doc: "Kronos, used as-is" = required): `src/models/kronos_runner.py` (clones Kronos repo + NeoQuasar/Kronos-small on Kaggle GPU; split/dividend-adjusted OHLCV; 400-day lookback; members only; weekly grid 2008–2024, holdout excluded; 20-day forecast → kronos_ret_5/20; batches; chunked by years; `--timing`, `--mock`), `src/features/pretrained.py` (excess vs SPY's Kronos forecast; ranked), `run_experiment` model `kronos` (standalone) + `--kronos` (features), Kaggle job `kronos`. Tests 19/19. Mock end-to-end OK.
- Searched Hugging Face: **no public Kronos fine-tuned on US stocks** (only BTC, Egypt EGX, Vietnam VN30, Taiwan, MLX ports). Original Kronos is pretrained on 45 global exchanges incl. US → used as-is in FYP-1. **US-only fine-tune (2000–2017 train, later years test) = FYP-2** (design-doc optional extension).
- Note for report: Kronos pretraining data may overlap test years (possible contamination) → also report XGBoost without Kronos.

### 9 Nov 2026 (feature comparison + single-factor ICs)
- 4-way comparison (above): best = price + fundamentals + sectors, IC 0.014 (t 1.80). Rank objective worse → dropped.
- Single-feature IC table 2010–2024 (`results/factor_ic_2010_2024.csv`): fundamentals coverage 76–85% (gross profit 36%, leverage 52%). All factors weak (|IC| ≤ 0.022): value negative (book-to-market −0.022, the 2010s "value drought"), short-term reversal (ret_5/ret_20 ≈ −0.015), momentum 12-1 +0.012, FCF yield +0.013, sector-relative SUE +0.011, accruals −0.011 — consistent with academic findings. Explains why combined IC ≈ 0.014.
- Incident: `data\clean\fundamentals.parquet` on the PC became 0 bytes (cause unknown) after the Kaggle download; Kaggle runs used the good 6 MB copy. Restored from `data\kaggle_out\latticeiq-fundamentals-download\...`. `upload-data` now refuses to upload empty files.
- Next: horizons 5d / 60d with best setup; brackets rerun with calibrated P(up).

### 9 Nov 2026 (brackets on real data)
- Brackets on real data ✅: 80% ranges hold the real 20-day outcome **80.1%** of the time (13 unseen years, 208k forecasts); 90% → 90.2%; naive under-covers tails (86.4%). Headline FYP-1 accuracy result.
- Found: stocks rose over 20 days 57.7% of the time → "always up" beats raw P(up) (54.7%). Added isotonic calibration of P(up) on calibration years + a base-rate baseline row + P(up) calibration table (`p_up_calibration.csv`). Synthetic check: calibrated P(up) beats base rate (Brier 0.2489 vs 0.2495).

### 8 Nov 2026 (fundamentals downloaded + model upgrades)
- `fundamentals-download` on Kaggle ✅ (5 min): 1,007,535 values, **499/677 companies** (current members 422/478, departed 77/199); 400–490 companies/yr from 2010.
- Built upgrades (each a switch): `--rank` (XGBRanker rank:pairwise on per-date quintiles; scores centred per date), `--sectors` (`src/features/sector.py`: sector = sector ETF with highest market-neutral return correlation over past 252 d, set monthly; features sec_mom_20/60 + 10 within-sector ranks), `f_filing_reaction` added to fundamentals (3-day excess return around latest 10-Q/10-K, usable 3 trading days later).
- Tests 18/18 (+ sectors recovered >90%, reaction timing, ranker finds planted signal). Simulation with all switches runs (17 → 29 → 44 features).

### 8 Nov 2026 (news diagnosis)
- `--diagnose 2019-03` on Kaggle: batching loses nothing (10 tickers in one request = 55 vs 56 one-by-one). **Plan limit**: whole-market news for 2019-03-08 = **2 articles** (Benzinga publishes hundreds/day). Per big company 0–23 articles/month.
- **Decision: news sentiment deferred to FYP-2** (needs a fuller feed: higher Benzinga tier / Polygon / Tiingo news, or free SEC 8-K event filings). News code (download, FinBERT, features, tests) stays ready. Skip news-score/news experiments in FYP-1.

### 8 Nov 2026 (EDGAR fundamentals built)
- `src/ingest/edgar.py`: SEC companyfacts per company (current-ticker→CIK map; delisted companies mostly unmatched → NaN), keeps `filed` date. Kaggle job `fundamentals-download`.
- `src/features/fundamentals.py`: first-filed values only; quarters from YTD (Q4 = FY − 9M); TTM; usable the day AFTER filing; expire after 15 months; split-adjusted market value. 14 features: earnings/sales/book/FCF yield, ROE, gross profitability, op margin, accruals, leverage, revenue & asset growth, SUE, days since filing, size.
- Flags: `run_experiment --fundamentals`, `run_brackets --fundamentals`. Tests 15/15 (Q4/TTM/timing + later-filings-don't-leak). 60-company simulation runs end-to-end.

### 8 Nov 2026 (brackets saved)
- Saved price brackets: `scripts/run_brackets.py` (quantile XGBoost on volatility-scaled returns, normalized conformal calibration, P(up), $ ranges), raw volatility features, future-return target `fret_h`, synthetic `volcluster` data, Kaggle job `brackets`. Synthetic checks: 80/50/90% coverage 79.4/49.6/89.6% (volcluster) and 80.2/50.3/90.1% (momentum; P(up) hit 53.1%). Tests 13/13.
- Macro now opt-in for the ranking model (`--macro`); brackets always use macro.
- Benzinga chunk 2015–2019: 5,479 articles (total 12,729), thinning over time → running `--diagnose 2019-03` to test batching vs plan limits.

### 8 Nov 2026 (first GPU results)
- First Kaggle GPU experiments ✅: ~5 min per run (PC: ~28 min).
- Price-only: IC +0.013 (t 1.6), hit 50.4% → small consistent edge, not yet significant.
- +Macro: IC +0.012 (t 1.0) → no gain. **Decision: drop macro from the return-ranking model** (same value for every stock on a date, can't rank stocks); keep macro for price brackets (market-wide volatility matters there). Next accuracy lever: stock-specific fundamentals (EDGAR).
- Price brackets code written + tested in the sandbox (80% range coverage 79–80% on synthetic data); not yet saved to the PC, waiting for go-ahead.

### 8 Nov 2026 (scope)
- Split into FYP-1 (credible core, 20 days) and FYP-2 (scale + deployment). Holdout 2025–26 stays locked until end of FYP-2. Added price brackets + P(up) to FYP-1 as the core user-facing output.

### 8 Nov 2026 (news chunk 1)
- `news-download 2010-01..2014-12` on Kaggle ✅ (21 min): **7,250 articles**, sparse — 124–236 stocks/yr with any news, 5–12 articles per stock per year; AAPL = 1/3 of all; Jul–Oct 2011 nearly empty (archive gap). Pagination verified (AAPL 49 in one month > 25 page cap).
- Decision: test news features only on years where Benzinga coverage is broad (rule: 300+ stocks, 20+ articles/stock); decide after chunks 2–3 with `python -m src.ingest.benzinga --report`.
- Added `--report` (articles / stocks covered / articles per stock, per year).

### 8 Nov 2026 (first Kaggle jobs)
- `upload-secrets` ✅ and `upload-data` ✅ (prices 78 MB + macro) → private datasets on Kaggle.
- First Kaggle job `news-probe` ✅ (15 s): **Benzinga history starts 2010** (AAPL: 3 articles Jan 2010, 25+ from 2011).
- Found: plan returns max **25 articles per page** (asked 100) → downloader would have stopped after page 1. Fixed pagination (page until empty/short page, dedupe guard), batch 10 tickers, up to 400 pages. Tested offline.
- News download will be chunked into ≤12 h Kaggle runs: 2010–2014, 2015–2019, 2020–2026.

### 8 Nov 2026 (all on Kaggle)
- Built `scripts/kaggle_cloud.py`: upload-secrets (keys -> private dataset), upload-data, news-probe, news-download (CPU), news-score (GPU), experiments (GPU), status/download. Code is packed into each job; outputs come back to the PC; chunked news merges; data auto re-uploaded after news jobs. Simulated locally ✅ (secrets, experiments, merge).
- Claude's cloud session can't reach kaggle.com → the PC runs the one-line commands.
- FinBERT scorer: GPU batch 256, length-sorted batches.

### 8 Nov 2026
- Built Kaggle GPU experiment runner (`scripts/kaggle_experiment.py`): upload-data / run "jobs" / status / download / all. Code + job list embedded in one script; XGBoost uses the GPU on Kaggle (`LATTICEIQ_DEVICE=cuda`). Simulated locally end-to-end ✅. ARIMA import made lazy; dotenv optional.
- Agreed 20-day plan to demo (see top).

### 7 Nov 2026 (first full-universe result)
- Calendar alignment on real data: 6,752 trading days; 454 short gaps filled, 2,555 days left empty, 22 bad one-day jumps removed.
- Baseline (price only, 466 stocks): rank IC +0.008, t 1.1, hit 50.3%. Weak but plausible: price signals alone are weak in large caps. This is the row every feature group must beat.
- Fix: shrinkage k=0 had set all predictions to 0, erasing the ranking. Hit rate / rank IC / long-short now use the raw model output (`y_raw`); RMSE / Diebold–Mariano use the shrunk forecast. Runs before this fix aren't directly comparable on IC → rerun the pair with `--train-step 5`.

### 7 Nov 2026 (gap fix)
- Combined 500 tickers (2,748,234 rows); panel 466 stocks. First full-universe run stopped at fold 2013 by the leakage guard: tickers with holes in their history made "20 rows ahead" span months.
- Fix: `src/data/clean.py` puts every ticker on SPY's trading calendar (≤5-day gaps carried forward, longer gaps left empty, x5 one-day jumps blanked). Targets now always exactly 20 trading days. New test: gaps never stretch targets. 11/11 tests pass; simulated run with random 30–200-day holes completes all folds.

### 7 Nov 2026 (news pipeline)
- Built news pipeline: `src/ingest/benzinga.py` (--probe history depth, monthly × 20-ticker batches of members only, resumable, --combine), `scripts/kaggle_news.py` (upload / run on T4 / status / download / all), `kaggle/finbert/finbert_score.py`, `src/news/sentiment.py` (local CPU for daily), `src/features/news.py` (8 features, 4 pm New York rule, NaN before coverage, ranked per date).
- Runner: `--news`, `--start-year` (compare with/without news on the same years).
- Tests 10/10 (added 4 pm rule + future-article test). Fake-news end-to-end check: random prices + planted news → without news IC 0.01, with news IC 0.50, top features all news ✅ (fake signal deliberately strong; real gain will be small).
- Not yet run against the real Benzinga API (cloud session can't reach it).

### 7 Nov 2026
- Hit Tiingo FREE plan **monthly cap: 500 unique symbols/month**. 502 of 695 tickers downloaded (A → ~PCAR); 193 left (alphabetical tail P–Z). Downloader now stops cleanly on the monthly limit instead of waiting hourly.
- Consequence: free plan cannot support the daily update of ~500 tickers either (each counts toward 500/month) → paid plan needed for the product.
- Resumed Tiingo download (317 left). Crashed at PCAR+1: Tiingo returned an error message (HTTP 200) instead of prices. Fixed `tiingo.py`: reads such messages, waits if it's a limit, skips the ticker if not, never crashes mid-run (always saves). Tested offline.
- Found that some earlier file updates hadn't reached the PC; verified all code files now match.

### 6 Nov 2026 (late)
- Ran `build_universe`: 979 tickers / 998 membership spells since 2000. Tiingo status: 671 covered, 41 partial, 113 symbol reused, 173 missing. Current members 95% covered; departed members 47% covered → survivorship-bias risk.
- Tiingo download hit 429 rate limit after ~50 new tickers. Rewrote `src/ingest/tiingo.py`: waits out the limit (61 min) and resumes, `--no-wait`, `--combine-only`, Ctrl+C-safe, remembers not-found tickers. Tested offline.
- Two real runs happened on only 10 stocks (combine step never ran after the crash) → results not meaningful; logged anyway.
- Fixed a date warning in `build_universe.py`.

### 6 Nov 2026 (evening)
- Built `scripts/build_universe.py`: downloads fja05680/sp500 daily member lists, converts to membership spells since 2000, matches each to Tiingo's supported-ticker list (detects reused symbols), writes `config/universe.csv` + `universe_report.csv`.
- Added `src/universe.py` (membership filter, stock/ETF types), `src/features/macro.py` (VIX, VIX 20d change, 10y–2y spread, credit spread + 60d change, 10y yield 60d change, CPI YoY first-release, unemployment 3m change).
- Added price features `beta_60`, `idio_vol_60` (17 total). Ranks now computed within stocks and within ETFs separately.
- Added prediction shrinkage: inner train/validation split per fold → trust factor k in [0,1]. Fixes "XGBoost worse than Random Walk" on noise.
- New flags: `--asset stock|etf`, `--no-macro`, `--no-shrink`. Each run saves `config.json` and `shrink.csv`.
- Tests: 8/8 pass (added membership-date test and CPI publication-date test). Simulated real run with ETFs, delisted stocks and macro works end-to-end.
- Note: macro features can rank high in importance even when they add nothing; only the ablation (with vs `--no-macro`) shows whether they help.

### 6 Nov 2026
- Team change: One person is doing all roles (data, models, app). Timeline needs adjusting: see Decisions.
- First real experiment ran (275 s). Pipeline works end-to-end on real prices. No skill yet (see results log).
- `pytest -q`: 6/6 pass on the development PC.
- Fixed Windows issues: installed VC++ 2015–2022 x64 redistributable (XGBoost needs `vcomp140.dll`); added scikit-learn to `requirements.txt`.
- Kaggle CLI working (token in the user's `.kaggle` folder). `kaggle` added to requirements.
- Downloaded data: Tiingo prices for 28/29 starter tickers, 2000-01-03 → 2026-11-05 (LEH missing). FRED: VIX, 10y, 2y, 10y–2y, Baa spread (daily), CPI + unemployment first-release (ALFRED).
- All API keys verified with `test_keys.py`: FRED, Hugging Face, Benzinga, SEC EDGAR, Tiingo.
- Python 3.13 + `.venv` set up on the development PC (Windows).
- Built repo: config, ingest (Tiingo, FRED), price features, targets, walk-forward folds with 60-day embargo, holdout lock, shared metrics (hit rate, rank IC with Newey-West t, long-short, RMSE, Diebold–Mariano), Random Walk, ARIMA, XGBoost, experiment runner, leakage tests.
- Validated on synthetic data: no false skill on noise; finds planted signal.

### 5 Nov 2026
- Reviewed design doc. Key changes: weekly Kronos/TimesFM runs, report XGBoost without pretrained features, need historical S&P membership source, EDGAR first-filed values, Newey-West/HAC stats, regime-weight shrinkage, calibration, live prediction log.
- Checked model licences on Hugging Face: Kronos (MIT), TimesFM (Apache 2.0), Chronos-2 (Apache 2.0) OK to sell; **FinBERT non-commercial** → FYP only.
- Set up `.env.example`, `.gitignore`, `test_keys.py`. Got Benzinga, FRED, Hugging Face, Tiingo, Kaggle keys.
- Decided: Kaggle (free GPU) for Kronos/TimesFM/FinBERT; PC for everything else. PyTorch, not TensorFlow.

---

## Decisions

| Date | Decision | Why |
|---|---|---|
| 5 Nov | Forecast excess return vs SPY at 5/20/60 days (not raw price, not 5-year) | Defensible, matches design doc; long horizons not forecastable |
| 5 Nov | PyTorch, not TensorFlow | Kronos, TimesFM, FinBERT all use PyTorch |
| 5 Nov | Kaggle for GPU jobs, PC for the rest | Free ~30 GPU h/week, runs in background |
| 5 Nov | Portfolio engine shows risk only, no buy/sell calls | Avoids investment-adviser registration |
| 6 Nov | Weekly evaluation grid for all models | Same dates for every model; Kronos affordable |
| 6 Nov | Rank features across stocks per date | Removes market-wide shifts; standard in quant |
| 6 Nov | 2025-01-01 onward = locked holdout | Run once after model freeze |
| 8 Nov | Holdout stays locked until the model is final | Strongest final evidence for credibility |
| 8 Nov | News set aside | Benzinga plan ≈ 2 articles/day market-wide; too sparse to rank 466 stocks |
| 8 Nov | Macro features dropped from return ranking, kept for brackets | Ablation: +macro IC 0.012 vs 0.013 price-only, noisier |
| 6 Nov | One person does all roles; optional extras (Kronos fine-tuning, TimesFM LoRA, Chronos-2, FinBERT fine-tuning, AV sentiment check) dropped unless ahead of schedule | Design doc's 11–16 weeks assumed 3 people working in parallel |
| 12 Nov | v1.0 = neural net averaged over 3 seeds (ensemble v2 kept as the secondary for the holdout) | Best at all three horizons, 6 of 6 comparisons on the full list |
| 13 Nov | Research data frozen as "universe v2" after the audit | Every later result is measured on the same data |
| 15 Nov | Volatility range model in the product | As well calibrated, simpler, and right in earnings weeks |
| 16 Nov | Train once, score daily; daily prices kept apart from the research data | Fast daily runs without changing the data the results were measured on |
| 16 Nov | News retried with Alpaca (Benzinga articles only) | 435–1,144 articles a day from 2015; keeping one publisher keeps counts comparable across years |

## Known issues
- Tiingo FREE plan: 50 req/h, 1,000/day, 2 GB/month; possibly a monthly unique-ticker cap. Downloader resumes automatically. Free licence = personal use only.
- ~~Benzinga plan returns only ≈2 news articles a day~~ → news retried with Alpaca (16 Nov).
- ~~Only 47% of departed S&P members have prices~~ → after the Yahoo fill-in and rebuild, 98% of true members in the test years (91% on average); 67% before 2012, so early training years still lean towards survivors.
- The edge has been thin recently: 2024 negative at 20 and 60 days; the daily model's check on mid-2021 to late 2026 is near zero except at 5 days.
- Market holidays are not known to the daily update; on a holiday it finds no new prices and waits.
- Scoring uses all filings; limiting filings to 7 or 14 years changed some fundamental features (cause not traced).
- Delisted stocks whose Tiingo symbol differs (e.g. LEH → LEHMQ) are counted as missing; manual mapping possible later.
- ~~Price gaps stretched targets across holes~~ fixed 7 Nov (calendar alignment).
- ~~Universe mixes ETFs with stocks~~ fixed 6 Nov. ~~XGBoost MSE worse than RW~~ fixed with shrinkage 6 Nov.
- `.env` can't be written by Claude's file tools; edit it by hand.
