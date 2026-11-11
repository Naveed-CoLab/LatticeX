# results/

One folder per experiment run, named `YYYYMMDD_HHMM_<data>_<asset>_h<horizon>_<options>`.
Only small summary files are committed; `predictions.parquet` and logs stay local. A readable summary of all runs is in [`../docs/results.md`](../docs/results.md).

| File | Contents |
|---|---|
| `config.json` | Exact flags and settings used (reproducibility) |
| `metrics.csv` | Hit rate, rank IC, Newey-West t, long-short, RMSE per model |
| `rank_ic_by_year.csv` | Rank IC for each test year |
| `dm_tests.json` | Diebold-Mariano test vs Random Walk |
| `importance.csv` | XGBoost feature importance |
| `shrink.csv` | Shrinkage factor chosen per fold |
| `bracket_metrics.csv` | Brackets: coverage, width, pinball, Brier (bracket runs) |
| `coverage_by_year.csv`, `conformal_shifts.csv` | Bracket calibration details |

Runs before 9 Nov 2026 used an earlier, smaller pipeline and are kept locally only.
