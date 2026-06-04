from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseUpliftLearner(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> BaseUpliftLearner:
        ...

    @abstractmethod
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """Individual Treatment Effect (incremental lift)."""
        ...

    def predict_mu_0(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_mu_1(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_all(self, X: np.ndarray) -> dict[str, np.ndarray]:
        ite = self.predict_ite(X)
        try:
            mu_0 = self.predict_mu_0(X)
            mu_1 = self.predict_mu_1(X)
        except NotImplementedError:
            mu_0 = np.zeros_like(ite)
            mu_1 = ite
        return {"ite": ite, "predicted_lift_score": ite, "mu_0": mu_0, "mu_1": mu_1}
