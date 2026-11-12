# models/

Trained model files are **not committed**. In FYP-1 every model is retrained inside each walk-forward fold by `scripts/run_experiment.py` and `scripts/run_brackets.py`, so there is no single saved model to store.

| Model | Code | Trained here? |
|---|---|---|
| Random Walk (predicts 0 excess return) | `src/models/random_walk.py` | no training |
| ARIMA (per stock) | `src/models/arima.py` | per fold |
| XGBoost excess-return model | `src/models/xgb_model.py` | per fold |
| Quantile XGBoost (price brackets) | `scripts/run_brackets.py` | per fold |
| Kronos-small (pretrained, zero-shot) | `src/models/kronos_runner.py` | no, downloaded from Hugging Face `NeoQuasar/Kronos-small` |
| TimesFM 2.5 200M (pretrained, zero-shot) | `src/models/timesfm_runner.py` | no, downloaded from Hugging Face `google/timesfm-2.5-200m-pytorch` |

The final frozen model (FYP-1 end) will be saved here as `models/final_<date>/` and published as a GitHub release asset rather than committed.
