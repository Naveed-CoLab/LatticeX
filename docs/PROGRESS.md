# LatticeIQ — Progress & TODO

_Last updated: 11 Nov 2026 · Owner: project lead (all roles)_
Rule: update this file every time something is finished, decided or found. Newest log entries at the top.

---

## Where we are (one-line status)

**Stage A (collect data) ~85% · Stage B (dataset) ~95% · Stage C (models) ~60% · App 0%.**
Data, features, leakage tests, walk-forward evaluation, XGBoost, price brackets and Kronos forecasts are done. Now: Kronos comparison run (queued on Kaggle), TimesFM, then lock the final model and build the dashboard.

| Stage (design doc) | Status |
|---|---|
| A. Collect data | Prices 466 stocks + 18 ETFs (Tiingo, A–PCAR) ✅ · FRED macro ✅ · EDGAR fundamentals ✅ · Benzinga news ⏸ (too sparse, FYP-2) · Kronos forecasts 2010–2024 ✅ |
| B. Build dataset | Price 17 + macro 8 + fundamentals 15 + sector 12 + Kronos 2 features ✅ · Targets ✅ · Folds + holdout lock ✅ · 19 leakage tests ✅ |
| C. Models | Random Walk ✅ · XGBoost + shrinkage ✅ · Price brackets + P(up) ✅ · Kronos zero-shot ✅ (comparison queued) · ARIMA ⏳ · TimesFM ⏳ · ensemble ⏳ |
| D. Holdout test (2025–26) | Locked, untouched until end of FYP-2 ✅ |
| E. Daily run + app | Not started (dashboard planned 14–20 Nov) |
| Repo | GitHub `Naveed-CoLab/LatticeX` organised: README, docs, CI tests, issue/PR templates ✅ |

## FYP-1 vs FYP-2 scope (agreed 8 Nov)
**FYP-1 (now → ~28 Nov): built right, measured honestly.** Data pipeline + data-quality report · leakage tests, walk-forward, baselines (RW + ARIMA) · return-ranking model (price + macro + fundamentals) · **price brackets + P(up), calibrated** · ablation + statistics · simple dashboard demo · report with limitations + FYP-2 plan.
**FYP-2 (next semester): bigger, better, deployed.** Full ~680 stocks + commercial data · Kronos/TimesFM · news at scale + own sentiment model · ensemble/regimes · API with keys, daily automation, deployment · LLM explanations, full portfolio engine · **final 2025–26 holdout test (kept locked until then)**.

## 20-day plan (FYP-1, 8 Nov → ~28 Nov 2026)
| Days | Dates | Work | Done when |
|---|---|---|---|
| 1–3 | 8–11 Nov | Kaggle runs (price vs +macro) · news chunks downloading · **price brackets + volatility + P(up), conformal calibration** | bracket coverage/width table logged |
| 3–7 | 11–15 Nov | **EDGAR fundamentals** (code ✅ 8 Nov) → run download → ablation on returns AND brackets | +fundamentals rows logged |
| 7–9 | 15–17 Nov | ARIMA baseline once · light ensemble (XGB seeds + LightGBM) · feature importance/SHAP · data-quality report · (stretch) news ablation on well-covered years | FYP-1 config chosen |
| 9–12 | 17–20 Nov | **Streamlit demo**: stock page (forecast, price brackets, P(up), drivers), ranking table, simple portfolio risk | demo runs |
| 12–17 | 20–25 Nov | Report: data, method, results, limitations, FYP-2 plan | draft done |
| 18–20 | 26–28 Nov | Slides, rehearsal, buffer | — |
Actions this week: ask supervisor for the marking criteria + confirm scope; consider 1-month Tiingo upgrade to remove the A–PCAR gap.

## Accuracy roadmap (agreed 6 Nov) — add one at a time, measure each (ablation)
| # | Step | Expected gain |
|---|---|---|
| 1 | Full universe (~680 stocks) | makes results measurable |
| 2 | EDGAR fundamentals: earnings yield, ROE, gross margin, growth, debt, earnings surprise | large (20–60 d) |
| 3 | Benzinga news + FinBERT: sentiment 1/5/20 d, article count, sentiment surprise, news shock; after 4 pm ET → next day | medium (5–20 d) |
| 4 | Earnings-date features: days since/until earnings, last earnings reaction (post-earnings drift) | medium |
| 5 | Sector-relative ranks (sector from SEC SIC codes) | small–medium |
| 6 | Kronos + TimesFM forecasts as features | research question |
| 7 | Tuning: ranking objective, Optuna inside folds, seed averaging | small |
| 8 | Regime ensemble + isotonic calibration | fewer big errors |
Realistic final target: rank IC 0.03–0.06, hit rate 52–55%. Choose features on 2012–2024 folds only; never peek at 2025–26.

Daily automation (later): Windows Task Scheduler, weekdays ~2 pm PT → Tiingo + Benzinga update → FinBERT on new articles (CPU) → features → frozen model → forecasts.

---

## TODO

> **Team note (6 Nov):** One person is doing all roles (data, models, app). Tasks are in the order to do them.

### Now — in this order
1. [x] Code: build S&P 500 universe since 2000 with membership dates (`scripts/build_universe.py`)
2. [x] Code: Tiingo coverage check for delisted stocks (inside build_universe → `config/universe_report.csv`)
3. [x] **Run** `python -m scripts.build_universe` → 979 members since 2000; current members 95% covered, **departed members only 47% covered**
4. [x] **Check Tiingo plan** → FREE tier: 50 requests/hour, 1,000/day, 2 GB bandwidth/month. Full download ≈13 h (run overnight), ≈1 GB. Free = personal use only → paid plan needed before public demo/selling
4b. [~] **Run** `python -m src.ingest.tiingo` → 502/695 done; blocked by 500-symbol monthly cap
4c. [ ] **Decide**: upgrade Tiingo (finish today) OR wait until 1 Dec for the remaining 193 (P–Z). Meanwhile run experiments on the 502
5. [x] Membership filter (`src/universe.py`, used in run_experiment)
6. [x] Score stocks and ETFs separately (`--asset stock|etf`, ranks within each group)
7. [x] Macro features (`src/features/macro.py`, 8 features, point-in-time) + beta & idiosyncratic volatility
8. [x] Prediction shrinkage (k per fold; 0 = fall back to Random Walk)
9. [ ] **Run** the ablation on the full universe and record both rows below:
       `python -m scripts.run_experiment --data real --no-macro` then `python -m scripts.run_experiment --data real`
10. [ ] **Run** `python -m scripts.run_experiment --data real --asset etf`
11. [ ] Delisted coverage is 47% (< 70%) → decide: (a) accept + document survivorship bias and measure it, (b) map renamed symbols by hand for the biggest departed names, or (c) buy a delisted-inclusive source (Norgate / Sharadar). See universe_report.csv

### App track (after the model reruns, or when you need a break from models)
- [ ] Database tables (forecasts, user holdings) + FastAPI skeleton

### Next
- [x] Comparison: price / +fund / +sectors / +rank → best = fund + sectors (IC 0.014, t 1.80)
- [ ] **Run** horizons: `experiments "--data real --fundamentals --sectors --horizon 5" "--data real --fundamentals --sectors --horizon 60"`
- [ ] Run ARIMA baseline on real data (`--models random_walk xgboost arima`, ~1–2 h)
- [ ] Try `target_transform: rank` in `config/settings.yaml`, compare
- [ ] Kaggle scripts: upload data, run Kronos-small vs Kronos-mini timing, download results
- [x] Kronos pipeline code (zero-shot) ✅
- [x] `kronos --timing` (v1): works, too slow + noisy → fixed
- [x] `kronos --timing` v4 ✅: 4.6 forecasts/s per GPU × 2 → full 2010–2024 ≈ 1.6 GPU h
- [x] `kronos --years 2010 2024` ✅: 59,699 forecasts, 189 dates, 1.74 GPU h
- [ ] **Run** comparison: Random Walk vs XGBoost vs Kronos alone vs XGBoost + Kronos features
- [ ] TimesFM 2.5 zero-shot (same pattern) → comparison row
- [ ] FYP-2: fine-tune Kronos on US stocks 2000–2017 (our own US-only Kronos)
- [ ] EDGAR fundamentals ingest (first-filed values, TTM) → fundamental features
- [x] Code: Benzinga news ingest → FinBERT on Kaggle → news features (`--news`)
- [x] Price brackets code (run on real data: `python -m scripts.kaggle_cloud brackets "--data real"`)
- [x] Brackets on real data (20d): coverage 80.1 / 90.2 / 49.4 % ✅
- [ ] Rerun brackets with calibrated P(up), plus 5d and 60d horizons
- [x] **Run** news-probe on Kaggle → news from **2010** (plan: 25 articles/page)
- [~] **Run** news-download on Kaggle: 2010–2014 ✅ (7,250, sparse) · 2015–2019 ⏳ · 2020–2026 ⏳ → then `--report`
- [ ] **Run** `python -m scripts.kaggle_news all` (FinBERT on Kaggle)
- [ ] **Run** ablation: `run_experiment --data real --start-year <Y>` vs `--news --start-year <Y>`; record both
- [ ] Ablation table: price → +macro → +fundamentals → +news → +Kronos/TimesFM

### Later
- [ ] Regime rules (VIX high/low × SPY vs 200-day average)
- [ ] Regime-weighted ensemble (shrink weights toward equal)
- [ ] Probability calibration (isotonic) + calibration plot
- [ ] SHAP explanations → LLM explanation layer (needs LLM API key)
- [ ] Long-short backtest after trading costs
- [ ] Model freeze → run 2025–26 holdout ONCE
- [ ] Daily batch job, portfolio engine, dashboard, live prediction log

### Before launch / housekeeping
- [ ] Create private GitHub repo
- [ ] Get LLM API key (Anthropic or OpenAI)
- [ ] Ask Benzinga how many years of history the plan includes
- [ ] Rotate ALL keys (Benzinga, FRED, Hugging Face, Tiingo, Kaggle) — they were pasted in chat
- [ ] Delete `Claude outputs` folder (contains key copies)
- [ ] Replace FinBERT before selling (non-commercial licence)
- [ ] Securities lawyer review before showing buy/sell suggestions to users

---

## Results log

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

Targets to beat: rank IC 0.03–0.08 with t > 2, hit rate 52–56%, DM p < 0.05 vs Random Walk.

---

## Work log (newest first)

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
| 8 Nov | Holdout stays locked until end of FYP-2 | Strongest final evidence for credibility |
| 8 Nov | News deferred to FYP-2 | Benzinga plan ≈ 2 articles/day market-wide; too sparse to rank 466 stocks |
| 8 Nov | Macro features dropped from return ranking, kept for brackets | Ablation: +macro IC 0.012 vs 0.013 price-only, noisier |
| 6 Nov | One person does all roles; optional extras (Kronos fine-tuning, TimesFM LoRA, Chronos-2, FinBERT fine-tuning, AV sentiment check) dropped unless ahead of schedule | Design doc's 11–16 weeks assumed 3 people working in parallel |

## Known issues
- Tiingo FREE plan: 50 req/h, 1,000/day, 2 GB/month; possibly a monthly unique-ticker cap. Downloader resumes automatically. Free licence = personal use only.
- Benzinga plan returns only a small sample of news (≈2/day market-wide) → news not used in FYP-1.
- Only 47% of departed S&P members have Tiingo prices → survivorship bias in backtests until fixed or measured.
- Delisted stocks whose Tiingo symbol differs (e.g. LEH → LEHMQ) are counted as missing; manual mapping possible later.
- ~~Price gaps stretched targets across holes~~ fixed 7 Nov (calendar alignment).
- ~~Universe mixes ETFs with stocks~~ fixed 6 Nov. ~~XGBoost MSE worse than RW~~ fixed with shrinkage 6 Nov.
- `.env` can't be written by Claude's file tools; edit it by hand.
