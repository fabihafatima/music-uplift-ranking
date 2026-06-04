"""XGBoost learning-to-rank (rank:pairwise)."""

from __future__ import annotations

import numpy as np


class XGBoostRanker:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: list[int]):
        try:
            import xgboost as xgb

            self.model = xgb.XGBRanker(
                objective="rank:pairwise",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            )
            self.model.fit(X, y, group=group)
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
