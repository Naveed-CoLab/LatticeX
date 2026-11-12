"""Extra models for the ensemble. Each has fit(X, y, dates) / predict(X), like XGBForecaster.

  LGBMForecaster   LightGBM: gradient-boosted trees grown leaf-wise (different errors from XGBoost)
  RidgeForecaster  linear model on the ranked features: simple, stable baseline
  MLPForecaster    small neural net (PyTorch, GPU if available)

All features are already cross-sectional ranks in [-0.5, 0.5] (0 = median), so a missing value is
filled with 0 = "average stock" for the linear and neural models; tree models handle NaN natively.

ensemble_rank(): combines members by averaging each member's per-date rank of its prediction, weighted by
each member's trust score k from the shrinkage step (measured on the latest training years only).
"""
import os

import numpy as np
import pandas as pd

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=500,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


class LGBMForecaster:
    name = "lightgbm"

    def __init__(self, params: dict | None = None, seed: int = 42):
        self.params, self.seed, self.model = params or LGBM_PARAMS, seed, None

    def fit(self, X: pd.DataFrame, y, dates=None) -> "LGBMForecaster":
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(**self.params, random_state=self.seed, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class RidgeForecaster:
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha, self.model = alpha, None

    def fit(self, X: pd.DataFrame, y, dates=None) -> "RidgeForecaster":
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha).fit(X.fillna(0.0).values, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.fillna(0.0).values)


class MLPForecaster:
    """Two hidden layers, dropout, early stop on the last 10% of training dates (by time, never shuffled
    across the split). Targets are standardised for training and scaled back on predict."""
    name = "mlp"

    def __init__(self, hidden=(64, 32), dropout=0.1, lr=1e-3, epochs=8, batch=4096, seed=42):
        self.hidden, self.dropout, self.lr, self.epochs, self.batch, self.seed = hidden, dropout, lr, epochs, batch, seed
        self.net, self.mu, self.sd = None, 0.0, 1.0

    def _device(self):
        import torch
        want = os.getenv("LATTICEIQ_DEVICE", "cpu")
        return torch.device("cuda" if want.startswith("cuda") and torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y, dates=None) -> "MLPForecaster":
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self._device()
        x = X.fillna(0.0).values.astype(np.float32)
        y = np.asarray(y, dtype=np.float64)
        self.mu, self.sd = float(np.mean(y)), float(np.std(y) or 1.0)
        t = ((y - self.mu) / self.sd).astype(np.float32)
        d = np.asarray(dates) if dates is not None else np.arange(len(t))
        cut = np.sort(np.unique(d))[int(len(np.unique(d)) * 0.9)]
        tr, va = d < cut, d >= cut
        layers, n_in = [], x.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(self.dropout)]
            n_in = h
        self.net = nn.Sequential(*layers, nn.Linear(n_in, 1)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        xt, yt = torch.from_numpy(x[tr]).to(dev), torch.from_numpy(t[tr]).to(dev)
        xv, yv = torch.from_numpy(x[va]).to(dev), torch.from_numpy(t[va]).to(dev)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g).to(dev)
            for i in range(0, len(xt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = ((self.net(xt[idx]).squeeze(1) - yt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(((self.net(xv).squeeze(1) - yv) ** 2).mean()) if len(xv) else 0.0
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        x = X.fillna(0.0).values.astype(np.float32)
        with torch.no_grad():
            for i in range(0, len(x), 65536):
                out.append(self.net(torch.from_numpy(x[i:i + 65536]).to(dev)).squeeze(1).cpu().numpy())
        return np.concatenate(out) * self.sd + self.mu


def ensemble_rank(raw_preds: list, dates, weights=None) -> np.ndarray:
    """Weighted average of each member's per-date percentile rank (centred at 0). Scale-free.
    weights: one number per member, e.g. each member's shrinkage k (its skill on the most recent TRAINING
    years, so no test-year information is used). All zero / None -> equal weights."""
    d = np.asarray(dates)
    w = np.ones(len(raw_preds)) if weights is None else np.clip(np.asarray(weights, dtype=float), 0, None)
    w = np.ones(len(raw_preds)) if w.sum() <= 0 else w
    w = w / w.sum()
    ranks = []
    for p in raw_preds:
        r = pd.Series(p).groupby(d).rank(pct=True)
        ranks.append((r - r.groupby(d).transform("mean")).values)      # centred: average stock = 0 each date
    return np.average(ranks, axis=0, weights=w)
