# data/

This folder is **not committed** (too large, and Tiingo/Benzinga data may not be redistributed). Rebuild it with the scripts below. Sources and licences: [`../docs/data-sources.md`](../docs/data-sources.md).

```
data/
├── raw/        per-ticker / per-month downloads (resumable)
└── clean/      combined files the pipeline reads
    ├── prices.parquet             Tiingo daily OHLCV, adjusted       (~80 MB)
    ├── macro.parquet              FRED/ALFRED with release dates
    ├── fundamentals.parquet       SEC EDGAR first-filed values       (~6 MB)
    ├── kronos_forecasts.parquet   Kronos 5/20-day forecasts          (~1 MB)
    └── news_articles.parquet      Benzinga (deferred)
```

## Rebuild (needs keys in `.env`)

```
python -m scripts.build_universe                   # config/universe.csv from S&P 500 history
python -m src.ingest.tiingo                        # prices      (resumable; free plan = 500 symbols/month)
python -m src.ingest.fred                          # macro
python -m scripts.kaggle_cloud fundamentals-download   # fundamentals on Kaggle (or: python -m src.ingest.edgar)
python -m scripts.kaggle_cloud kronos --years 2010 2024 # Kronos forecasts on Kaggle GPU
```

No keys? Everything runs on synthetic data: `python -m scripts.run_experiment --data synthetic --signal momentum`.
