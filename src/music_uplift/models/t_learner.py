"""
T-Learner: separate outcome models for treatment and control groups.

  mu_1(x) = E[Y | X, T=1]  fitted on promoted subset
  mu_0(x) = E[Y | X, T=0]  fitted on non-promoted subset
  tau(x)  = mu_1(x) - mu_0(x)
"""

from __future__ import annotations

import numpy as np

from music_uplift.config import ModelConfig
from music_uplift.models.base import UpliftModel
from music_uplift.models.estimators import make_regressor


class TLearner(UpliftModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_treated: object | None = None
        self.model_control: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> TLearner:
        mask_t = treatment.astype(bool)
        mask_c = ~mask_t

        if mask_t.sum() < 10 or mask_c.sum() < 10:
            raise ValueError("Insufficient samples in treatment or control arm")

        self.model_treated = make_regressor(self.config)
        self.model_control = make_regressor(self.config)
        self.model_treated.fit(X[mask_t], y[mask_t])
        self.model_control.fit(X[mask_c], y[mask_c])
        self.is_fitted = True
        return self

    def predict_mu_1(self, X: np.ndarray) -> np.ndarray:
        if self.model_treated is None:
            raise RuntimeError("Model not fitted")
        return np.maximum(self.model_treated.predict(X), 0.0)

    def predict_mu_0(self, X: np.ndarray) -> np.ndarray:
        if self.model_control is None:
            raise RuntimeError("Model not fitted")
        return np.maximum(self.model_control.predict(X), 0.0)
