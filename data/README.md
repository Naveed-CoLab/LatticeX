# data/

This folder is **not committed** (too large, and price and news data may not be redistributed). Rebuild it with the scripts below. Sources and licences: [`../docs/data-sources.md`](../docs/data-sources.md).

```
data/
├── raw/        per-ticker / per-month downloads (resumable)
├── clean/      research data, frozen as "universe v2" (the files every result was measured on)
│   ├── prices.parquet             Tiingo + Yahoo daily prices, 2000-2026        (4.2 million rows)
│   ├── fundamentals.parquet       SEC EDGAR first-filed values                  (1.2 million rows)
│   ├── sec_events.parquet         SEC 8-K filings by event type                 (170,000 rows)
│   ├── sec_events_latest.parquet  filings since then, refreshed daily by the dashboard
│   ├── macro.parquet              FRED/ALFRED with release dates (tested, not used)
│   ├── kronos_forecasts.parquet   Kronos forecasts (tested, not used)
│   └── news_articles.parquet      Alpaca news headlines, 2015 on (in testing)
└── live/       daily data for the dashboard and the daily model
    ├── prices.parquet             last 3 years of closing prices, replaced after every close
    ├── rank.json                  today's model ranking
    ├── forecasts/<day>.csv        every day's forecasts, kept as a record
    └── analysts/<TICKER>.json     analyst views, one stock at a time
```

## Rebuild (needs keys in `.env`)

```
python -m scripts.build_universe                   # config/universe.csv from S&P 500 history
python -m src.ingest.tiingo                        # prices      (resumable; free plan = 500 symbols/month)
python -m src.ingest.yahoo                         # prices Tiingo lacks (checked against Tiingo first)
python -m scripts.audit_universe                   # after any rebuild: drop histories that belong to another company
python -m src.ingest.fred                          # macro
python -m scripts.kaggle_cloud fundamentals-download   # fundamentals on Kaggle (or: python -m src.ingest.edgar)
python -m src.ingest.sec_events                    # 8-K filings
python -m src.ingest.alpaca_news --start 2015-01   # news (several hours), then --combine
python -m scripts.run_dashboard                    # fills data/live/ (needs models/latticeiq_v1.npz for the ranking)
```

No keys? Everything runs on synthetic data: `python -m scripts.run_experiment --data synthetic --signal momentum`.
