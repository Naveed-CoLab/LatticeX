"""XGBoost: the main model we train. Learns which feature combinations came before
a stock beating or trailing the benchmark."""
import os

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBForecaster:
    name = "xgboost"

    def __init__(self, params: dict, seed: int = 42, objective: str = "regression"):
        self.params = params
        self.seed = seed
        self.objective = objective          # "regression" (predict excess return) or "rank" (order stocks per date)
        self.model = None

    def fit(self, X: pd.DataFrame, y, dates=None) -> "XGBForecaster":
        # LATTICEIQ_DEVICE=cuda (set automatically on Kaggle GPU runs) trains on the GPU
        device = os.getenv("LATTICEIQ_DEVICE", "cpu")
        if self.objective == "rank":
            # learn to ORDER stocks within each date: labels = per-date quintile of the outcome (0..4)
            d = pd.Series(np.asarray(dates))
            order = np.argsort(d.values, kind="stable")
            qid = pd.factorize(d.values[order])[0]
            lab = pd.Series(np.asarray(y)[order]).groupby(qid).rank(pct=True)
            lab = np.minimum((lab * 5).astype(int), 4)
            self.model = xgb.XGBRanker(**self.params, objective="rank:pairwise", tree_method="hist", n_jobs=-1,
                                       random_state=self.seed, device=device)
            self.model.fit(X.iloc[order], lab.values, qid=qid)
        else:
            self.model = xgb.XGBRegressor(**self.params, tree_method="hist", n_jobs=-1,
                                          random_state=self.seed, device=device)
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def importance(self) -> pd.Series:
        """Gain-based importance; use SHAP later for per-forecast explanations."""
        imp = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False)
