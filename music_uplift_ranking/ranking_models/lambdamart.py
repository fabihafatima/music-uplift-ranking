"""LambdaMART via LightGBM LambdaRank objective."""

from __future__ import annotations

import numpy as np


class LambdaMARTRanker:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray):
        import lightgbm as lgb

        # group: relevance grades per query; here single query = all tracks
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X, y, group=group)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
