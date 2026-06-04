import numpy as np

from music_uplift_ranking.uplift_models.base import BaseUpliftLearner
from music_uplift_ranking.uplift_models.config import UpliftConfig
from music_uplift_ranking.uplift_models.estimators import make_classifier, make_regressor
from music_uplift_ranking.uplift_models.t_learner import TLearner


class XLearner(BaseUpliftLearner):
    def __init__(self, cfg: UpliftConfig):
        self.cfg = cfg
        self.t_learner = TLearner(cfg)
        self.tau_1 = None
        self.tau_0 = None
        self.propensity = None

    def fit(self, X, y, treatment):
        t = treatment.astype(bool)
        self.t_learner.fit(X, y, treatment)
        mu_1 = self.t_learner.predict_mu_1(X)
        mu_0 = self.t_learner.predict_mu_0(X)
        d1 = y[t] - mu_0[t]
        d0 = mu_1[~t] - y[~t]
        self.tau_1 = make_regressor(self.cfg)
        self.tau_0 = make_regressor(self.cfg)
        self.tau_1.fit(X[t], d1)
        self.tau_0.fit(X[~t], d0)
        self.propensity = make_classifier(self.cfg)
        self.propensity.fit(X, treatment)
        return self

    def predict_mu_0(self, X):
        return self.t_learner.predict_mu_0(X)

    def predict_mu_1(self, X):
        return self.predict_mu_0(X) + self.predict_ite(X)

    def predict_ite(self, X):
        g = self.propensity.predict_proba(X)[:, 1]
        return g * self.tau_1.predict(X) + (1 - g) * self.tau_0.predict(X)
