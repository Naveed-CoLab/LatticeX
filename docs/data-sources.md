# Data sources

No data is stored in this repository. Every dataset is rebuilt by a script (see [`../data/README.md`](../data/README.md)).

| Data | Source | Script | Output | Key needed | Licence / notes |
|---|---|---|---|---|---|
| Daily prices (OHLCV, adjusted) | [Tiingo](https://www.tiingo.com) | `src/ingest/tiingo.py` | `data/clean/prices.parquet` (~80 MB) | `TIINGO_API_KEY` | Not redistributable → never committed. Free plan: 500 symbols/month |
| Daily prices, fill-in (tickers after PCAR) | Yahoo Finance via `yfinance` | `src/ingest/yahoo.py` | same `data/raw/prices/` files, `source = yahoo` | — | Research use only. Checked against Tiingo first (`--check`); raw prices rebuilt from split-adjusted ones; cut to Tiingo's date range per ticker |
| S&P 500 membership history | [fja05680/sp500](https://github.com/fja05680/sp500), file `… (Updated).csv` (to Aug 2026) | `scripts/build_universe.py` | `config/universe.csv` | — | Public; membership start/end dates per ticker; ticker renames joined; recycled symbols rejected. **Until 12 Nov 2026 the script used the old file ending Jan 2019** |
| Macro (VIX, yields, credit spread, CPI, unemployment) | [FRED / ALFRED](https://fred.stlouisfed.org) | `src/ingest/fred.py` | `data/clean/macro.parquet` | `FRED_API_KEY` | Public; vintages used so values are as first released |
| Company fundamentals | [SEC EDGAR XBRL](https://www.sec.gov/edgar) companyfacts | `src/ingest/edgar.py` | `data/clean/fundamentals.parquet` | `SEC_USER_AGENT` (name + email) | Public; first-filed values with filing dates |
| Official company events (8-K) | [SEC EDGAR submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | `src/ingest/sec_events.py` | `data/clean/sec_events.parquet`, live: `sec_events_latest.parquet` | `SEC_USER_AGENT` | Public, commercial use allowed; event type from 8-K item numbers; acceptance time to the second |
| News | [Benzinga](https://www.benzinga.com/apis) | `src/ingest/benzinga.py` | `data/clean/news_articles.parquet` | `BENZINGA_API_KEY` | Not redistributable; current plan too sparse (~2 articles/day) → deferred to FYP-2 |
| Kronos forecasts | [Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) run on Kaggle | `src/models/kronos_runner.py` | `data/clean/kronos_forecasts.parquet` | — | Model MIT; 59,699 forecasts, 2010–2024 (tested, not used) |

## Universe (rebuilt 13 Nov 2026 on the corrected membership list)

- S&P 500 members since 2000: **1,066 tickers, 1,108 membership spells** (a stock that left and came back has two spells). Renamed tickers (FB→META, ANTM→ELV, …) are joined into one spell.
- `config/universe.csv`: about **820 tickers** = about 800 stocks whose symbol and dates match a price source + 18 ETFs (SPY and sector ETFs).
- Price files on disk: **738 stocks + 18 ETFs** after the audit drops below (742 before) (Tiingo for A–PCAR, Yahoo for the rest). `prices.parquet`: 4.2 million rows.
- **64 stocks in the list have no prices**: 53 are not on Yahoo any more (taken over or delisted: TWX, TWTR, WFM, RTN, XLNX, PXD, WBA, X, TIF, PX …), 8 are wrong-company symbols set aside (S, STI, SE, POM, SPLS, UN, PARA, SBNY) and 3 are one-row stubs of stocks delisted in autumn 2026 (KVUE, WBD, QRVO).
- **18 more have a price file that belongs to a later company** using the same symbol (ADT, ADCT, APC, ALTR, BEAM, CA, CSRA, DELL's first spell, DNB, DO, DOW's first spell, EMC, FTR, INFO, MON, NFX, PBY, SNDK's first spell): no prices inside the membership dates, so they are simply absent.
- The other 260 tickers were never usable: no symbol match, or the symbol belongs to another company now (`config/universe_report.csv` lists every member with its status).
- Fundamentals: **634 companies** matched at the SEC; 172 not matched (SEC's ticker list only knows today's symbols, so most long-delisted companies cannot be matched). 8-K events: 201,884 filings for 634 companies.

### How much of the real index we have (`python -m scripts.audit_universe`)

True S&P 500 members that have prices on the last trading day of June:

| Year | Members | With prices | | Year | Members | With prices | |
|---|---|---|---|---|---|---|---|
| 2001 | 495 | 282 | 57% | 2014 | 498 | 421 | 85% |
| 2003 | 493 | 299 | 61% | 2016 | 505 | 449 | 89% |
| 2005 | 495 | 308 | 62% | 2018 | 506 | 470 | 93% |
| 2007 | 496 | 344 | 69% | 2020 | 505 | 485 | 96% |
| 2009 | 498 | 374 | 75% | 2022 | 503 | 487 | 97% |
| 2011 | 497 | 392 | 79% | 2024 | 503 | 493 | 98% |
| 2012 | 497 | 402 | 81% | 2026 | 504 | 500 | 99% |

Test years 2012–2024: 81% → 98%, average **91%**. Training years before 2012: average 67%. The missing ones are mostly companies that were taken over, merged or failed, so the early training years lean towards survivors.

## Known gaps (stated in the report)

| Gap | Effect | Plan |
|---|---|---|
| Stocks that left the index are under-represented: we have 91% of true members in the test years but only 67% in the early training years (survivorship bias, table above) | Backtests may look somewhat better than reality: failed and acquired companies are missing more often | Licensed delisted-inclusive data before launch |
| Tickers after PCAR come from Yahoo, not Tiingo | Two sources; Yahoo drops delisted stocks | Checked against Tiingo (`yahoo --check`); licensed feed for launch |
| Fundamentals missing for 172 delisted companies and before ~2009 (XBRL start) | Missing values are handled by the models | Map old companies to SEC numbers by hand, later |
| Merger survivors: before the merger date, JCI carries Tyco's prices and fundamentals (to Sep 2016), CB carries ACE's (to Jan 2016), PLD carries AMB's (to Jun 2011) | The series is one real company throughout (the legal survivor), but the absorbed company (old Johnson Controls, old Chubb, old ProLogis) is missing before the merger | Accepted; same cause as the survivorship gap |
| Symbols whose history a data vendor filled with another company's (found by the audit, dropped by hand in `build_universe.py`): WEN 2000–08 (Triarc), CSR 2000 (Centerspace), KMG 2000–06 (KMG Chemicals), MIL 2000–10 (a small fund), GAS 2000–11 (AGL Resources), **LSI 2000–14 (Life Storage)**; CPWR fundamentals (a penny stock's filings) | Would have scored a non-member as a member. Five are before 2012 (training only); LSI reached into test years 2012–14 | Spells dropped / SEC match left out. Run `audit_universe` after every rebuild; a launch needs data keyed by a permanent security ID |
| Mylan's history (MYL, before Nov 2020) is not joined to VTRS | One stock missing for 2004–2020 | Stitch the two price files, later |
| News very sparse | News features not used | Better source later; SEC 8-K events used meanwhile |
| Kronos was pretrained on global market data that may include our test years | Possible optimistic Kronos results | Kronos is not in the model |

Safety checks built into the downloads: wrong-company price files are set aside (`yahoo --prune`), a symbol whose SEC registrant keeps filing 400+ days after the stock's prices ended is left out of fundamentals and events, and an earlier SEC registration is only merged in when its SEC name matches the expected company.

## Secrets

Keys live in `.env` (copy `.env.example`). `.env` and `kaggle.json` are in `.gitignore` and must never be committed. On Kaggle, keys are read from a **private** dataset `latticeiq-secrets`.
