"""
R-Learner (Robinson decomposition): orthogonalized effect estimation.

  tau(x) from regressing (Y - mu(x)) / (T - e(x)) residuals,
  where mu = E[Y|X] and e = P(T|X).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from music_uplift.config import ModelConfig
from music_uplift.models.base import UpliftModel
from music_uplift.models.estimators import make_propensity_model, make_regressor


class RLearner(UpliftModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.outcome_model: object | None = None
        self.propensity_model: object | None = None
        self.effect_model: object | None = None
        self._mu_0_model: object | None = None
        self._mu_1_model: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> RLearner:
        t = treatment.astype(np.float64)
        e = np.clip(t.mean(), 0.05, 0.95)  # placeholder until propensity fit

        self.outcome_model = make_regressor(self.config)
        self.outcome_model.fit(X, y)
        mu = self.outcome_model.predict(X)

        self.propensity_model = make_propensity_model(self.config)
        self.propensity_model.fit(X, treatment)
        e_hat = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.05, 0.95)

        residual_y = y - mu
        residual_t = t - e_hat
        # Pseudo-outcome for effect: avoid division by near-zero
        denom = np.where(np.abs(residual_t) < 0.05, np.sign(residual_t) * 0.05, residual_t)
        pseudo_effect = residual_y / denom
        pseudo_effect = np.clip(pseudo_effect, -500, 500)

        weights = (residual_t**2)
        self.effect_model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators // 2,
            max_depth=min(self.config.max_depth, 6),
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_child_samples,
            random_state=self.config.random_state,
        )
        self.effect_model.fit(X, pseudo_effect, sample_weight=weights)

        # Auxiliary T-learner style models for interpretable mu_0, mu_1
        mask_t = treatment.astype(bool)
        self._mu_1_model = make_regressor(self.config)
        self._mu_0_model = make_regressor(self.config)
        self._mu_1_model.fit(X[mask_t], y[mask_t])
        self._mu_0_model.fit(X[~mask_t], y[~mask_t])

        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if self.effect_model is None:
            raise RuntimeError("Model not fitted")
        return self.effect_model.predict(X)

    def predict_mu_0(self, X: np.ndarray) -> np.ndarray:
        if self._mu_0_model is None:
            raise RuntimeError("Model not fitted")
        return np.maximum(self._mu_0_model.predict(X), 0.0)

    def predict_mu_1(self, X: np.ndarray) -> np.ndarray:
        mu_0 = self.predict_mu_0(X)
        return np.maximum(mu_0 + self.predict_uplift(X), 0.0)

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        if self.propensity_model is None:
            raise RuntimeError("Model not fitted")
        return self.propensity_model.predict_proba(X)[:, 1]
