# Data sources

No data is stored in this repository. Every dataset is rebuilt by a script (see [`../data/README.md`](../data/README.md)).

| Data | Source | Script | Output | Key needed | Licence / notes |
|---|---|---|---|---|---|
| Daily prices (OHLCV, adjusted) | [Tiingo](https://www.tiingo.com) | `src/ingest/tiingo.py` | `data/clean/prices.parquet` (~80 MB) | `TIINGO_API_KEY` | Not redistributable → never committed. Free plan: 500 symbols/month |
| Daily prices, fill-in (tickers after PCAR) | Yahoo Finance via `yfinance` | `src/ingest/yahoo.py` | same `data/raw/prices/` files, `source = yahoo` | — | Research use only. Checked against Tiingo first (`--check`); raw prices rebuilt from split-adjusted ones; cut to Tiingo's date range per ticker |
| S&P 500 membership history | [fja05680/sp500](https://github.com/fja05680/sp500) | `scripts/build_universe.py` | `config/universe.csv` | — | Public; membership start/end dates per ticker |
| Macro (VIX, yields, credit spread, CPI, unemployment) | [FRED / ALFRED](https://fred.stlouisfed.org) | `src/ingest/fred.py` | `data/clean/macro.parquet` | `FRED_API_KEY` | Public; vintages used so values are as first released |
| Company fundamentals | [SEC EDGAR XBRL](https://www.sec.gov/edgar) companyfacts | `src/ingest/edgar.py` | `data/clean/fundamentals.parquet` | `SEC_USER_AGENT` (name + email) | Public; first-filed values with filing dates |
| Official company events (8-K) | [SEC EDGAR submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | `src/ingest/sec_events.py` | `data/clean/sec_events.parquet`, live: `sec_events_latest.parquet` | `SEC_USER_AGENT` | Public, commercial use allowed; event type from 8-K item numbers; acceptance time to the second |
| News | [Benzinga](https://www.benzinga.com/apis) | `src/ingest/benzinga.py` | `data/clean/news_articles.parquet` | `BENZINGA_API_KEY` | Not redistributable; current plan too sparse (~2 articles/day) → deferred to FYP-2 |
| Kronos forecasts | [Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) run on Kaggle | `src/models/kronos_runner.py` | `data/clean/kronos_forecasts.parquet` | — | Model MIT; 59,699 forecasts, 2010–2024 |

## Universe

- `config/universe.csv`: 695 tickers (677 stocks that were ever in the S&P 500 since 2000, plus 18 ETFs: SPY and sector ETFs).
- Prices found for **466 stocks + 18 ETFs** from Tiingo (free monthly cap stopped at A–PCAR). The other **196** (133 still listed as members, 63 that left the index) are filled from Yahoo Finance where available.
- `config/universe_report.csv` lists every member and whether Tiingo had its prices.

## Known gaps (stated in the report)

| Gap | Effect | Plan |
|---|---|---|
| Only 47% of stocks that left the S&P 500 have prices (survivorship bias) | Backtests may look slightly better than reality | Commercial data in FYP-2 |
| Tickers after PCAR come from Yahoo, not Tiingo | Two sources; Yahoo lacks some delisted stocks | Checked against Tiingo (`yahoo --check`); licensed feed for launch |
| Fundamentals coverage 76–85% of stock-days | Missing values handled by XGBoost | — |
| News very sparse | News features not used in FYP-1 | Better source in FYP-2 |
| Kronos was pretrained on global market data that may include our test years | Possible optimistic Kronos results | Report XGBoost with and without Kronos; US fine-tune in FYP-2 |

## Secrets

Keys live in `.env` (copy `.env.example`). `.env` and `kaggle.json` are in `.gitignore` and must never be committed. On Kaggle, keys are read from a **private** dataset `latticeiq-secrets`.
