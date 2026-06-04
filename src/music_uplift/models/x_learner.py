"""
X-Learner (Künzel et al.): robust meta-learner for observational data.

Stages:
  1. Fit mu_1, mu_0 via T-learner on respective arms
  2. Impute pseudo-effects: D1 = Y - mu_0(X) for treated, D0 = mu_1(X) - Y for control
  3. Fit tau_1 on treated, tau_0 on control
  4. Combine: tau(x) = g(x)*tau_1(x) + (1-g(x))*tau_0(x) where g = propensity
"""

from __future__ import annotations

import numpy as np

from music_uplift.config import ModelConfig
from music_uplift.models.base import UpliftModel
from music_uplift.models.estimators import make_propensity_model, make_regressor
from music_uplift.models.t_learner import TLearner


class XLearner(UpliftModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.t_learner = TLearner(config)
        self.tau_model_treated: object | None = None
        self.tau_model_control: object | None = None
        self.propensity_model: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> XLearner:
        mask_t = treatment.astype(bool)
        mask_c = ~mask_t

        self.t_learner.fit(X, y, treatment)
        mu_1 = self.t_learner.predict_mu_1(X)
        mu_0 = self.t_learner.predict_mu_0(X)

        # Pseudo treatment effects
        d1 = y[mask_t] - mu_0[mask_t]
        d0 = mu_1[mask_c] - y[mask_c]

        self.tau_model_treated = make_regressor(self.config)
        self.tau_model_control = make_regressor(self.config)
        self.tau_model_treated.fit(X[mask_t], d1)
        self.tau_model_control.fit(X[mask_c], d0)

        self.propensity_model = make_propensity_model(self.config)
        self.propensity_model.fit(X, treatment)
        self.is_fitted = True
        return self

    def predict_mu_1(self, X: np.ndarray) -> np.ndarray:
        mu_0 = self.predict_mu_0(X)
        tau = self.predict_uplift(X)
        return np.maximum(mu_0 + tau, 0.0)

    def predict_mu_0(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.t_learner.predict_mu_0(X), 0.0)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        g = self.predict_propensity(X)
        tau_1 = self.tau_model_treated.predict(X)
        tau_0 = self.tau_model_control.predict(X)
        return g * tau_1 + (1.0 - g) * tau_0

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        if self.propensity_model is None:
            raise RuntimeError("Model not fitted")
        return self.propensity_model.predict_proba(X)[:, 1]
