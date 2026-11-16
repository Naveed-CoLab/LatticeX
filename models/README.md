# models/

Trained model files are **not committed**.

## The model in daily use

`latticeiq_v1.npz` is LatticeIQ v1.0 trained once on all data (three horizons: 5, 20 and 60 trading days), written by `python -m scripts.train_final` or, on Kaggle, `python -m scripts.kaggle_cloud final-model`. It holds the neural-net weights as plain arrays plus a description (feature list, trust factor k per horizon, the last date it learned from). `src/models/production.py` loads it and ranks every stock after each close without PyTorch. Train it again every month or quarter.

## Research models

In the walk-forward tests every model is retrained inside each fold by `scripts/run_experiment.py` and `scripts/run_brackets.py`; none of those is saved.

v1.0 is the neural net (MLP) averaged over 3 seeds. The others were compared and not chosen; see [`../docs/decisions.md`](../docs/decisions.md).

| Model | Code | Trained here? |
|---|---|---|
| Random Walk (predicts 0 excess return) | `src/models/random_walk.py` | no training |
| ARIMA (per stock) | `src/models/arima.py` | per fold |
| XGBoost excess-return model | `src/models/xgb_model.py` | per fold |
| LightGBM, Ridge, MLP (ensemble members) | `src/models/ensemble.py` | per fold |
| Ensemble (rank average of XGBoost, LightGBM, MLP) | `src/models/ensemble.py` | combines the above |
| Quantile XGBoost (price brackets) | `scripts/run_brackets.py` | per fold |
| Kronos-small (pretrained, zero-shot) | `src/models/kronos_runner.py` | no, downloaded from Hugging Face `NeoQuasar/Kronos-small` |
| TimesFM 2.5 200M (pretrained, zero-shot) | `src/models/timesfm_runner.py` | no, downloaded from Hugging Face `google/timesfm-2.5-200m-pytorch` |
