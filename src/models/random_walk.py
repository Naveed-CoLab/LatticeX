"""Random Walk baseline: every stock matches the market, so the forecast is always 0.
This is the 'no skill' line every other model must beat."""
import numpy as np
import pandas as pd


class RandomWalk:
    name = "random_walk"

    def fit(self, X: pd.DataFrame, y) -> "RandomWalk":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))
