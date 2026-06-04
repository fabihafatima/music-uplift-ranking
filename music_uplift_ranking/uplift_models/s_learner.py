import numpy as np

from music_uplift_ranking.uplift_models.base import BaseUpliftLearner
from music_uplift_ranking.uplift_models.config import UpliftConfig
from music_uplift_ranking.uplift_models.estimators import make_regressor


class SLearner(BaseUpliftLearner):
    """Single model with treatment as an input feature."""

    def __init__(self, cfg: UpliftConfig):
        self.cfg = cfg
        self.model = None

    def fit(self, X, y, treatment):
        Xt = np.column_stack([X, treatment.astype(float)])
        self.model = make_regressor(self.cfg)
        self.model.fit(Xt, y)
        return self

    def predict_mu_1(self, X):
        Xt = np.column_stack([X, np.ones(len(X))])
        return np.maximum(self.model.predict(Xt), 0)

    def predict_mu_0(self, X):
        Xt = np.column_stack([X, np.zeros(len(X))])
        return np.maximum(self.model.predict(Xt), 0)

    def predict_ite(self, X):
        return self.predict_mu_1(X) - self.predict_mu_0(X)
