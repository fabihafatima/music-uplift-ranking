import numpy as np

from music_uplift_ranking.uplift_models.base import BaseUpliftLearner
from music_uplift_ranking.uplift_models.config import UpliftConfig
from music_uplift_ranking.uplift_models.estimators import make_regressor


class TLearner(BaseUpliftLearner):
    def __init__(self, cfg: UpliftConfig):
        self.cfg = cfg
        self.model_1 = None
        self.model_0 = None

    def fit(self, X, y, treatment):
        t = treatment.astype(bool)
        self.model_1 = make_regressor(self.cfg)
        self.model_0 = make_regressor(self.cfg)
        self.model_1.fit(X[t], y[t])
        self.model_0.fit(X[~t], y[~t])
        return self

    def predict_mu_1(self, X):
        return np.maximum(self.model_1.predict(X), 0)

    def predict_mu_0(self, X):
        return np.maximum(self.model_0.predict(X), 0)

    def predict_ite(self, X):
        return self.predict_mu_1(X) - self.predict_mu_0(X)
