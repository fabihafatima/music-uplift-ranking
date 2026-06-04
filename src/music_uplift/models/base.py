"""Base interfaces for counterfactual outcome and uplift estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from music_uplift.config import ModelConfig
from music_uplift.data.features import FeaturePreprocessor


@dataclass
class UpliftModelArtifact:
    """Serialized production artifact."""

    learner_name: str
    model: Any
    preprocessor: FeaturePreprocessor
    config: ModelConfig
    feature_columns: list[str]
    version: str = "0.1.0"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> UpliftModelArtifact:
        return joblib.load(path)


class UpliftModel(ABC):
    """
    Estimates counterfactual engagement:
      mu_1(x) = E[Y | X=x, T=1]  (if promoted)
      mu_0(x) = E[Y | X=x, T=0]  (if not promoted)
      tau(x)  = mu_1(x) - mu_0(x)  (incremental lift)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> UpliftModel:
        ...

    @abstractmethod
    def predict_mu_1(self, X: np.ndarray) -> np.ndarray:
        """Expected engagement under promotion."""
        ...

    @abstractmethod
    def predict_mu_0(self, X: np.ndarray) -> np.ndarray:
        """Expected engagement without promotion."""
        ...

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.predict_mu_1(X) - self.predict_mu_0(X)

    def predict_propensity(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def predict_all(self, X: np.ndarray) -> dict[str, np.ndarray]:
        mu_1 = self.predict_mu_1(X)
        mu_0 = self.predict_mu_0(X)
        result = {
            "mu_1": mu_1,
            "mu_0": mu_0,
            "tau": mu_1 - mu_0,
        }
        prop = self.predict_propensity(X)
        if prop is not None:
            result["propensity"] = prop
        return result

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        preprocessor: FeaturePreprocessor,
        track_ids: pd.Series | None = None,
    ) -> pd.DataFrame:
        X = preprocessor.transform(df)
        preds = self.predict_all(X)
        out = pd.DataFrame(
            {
                "track_id": track_ids if track_ids is not None else df["track_id"],
                "engagement_if_promoted": preds["mu_1"],
                "engagement_if_not_promoted": preds["mu_0"],
                "incremental_lift": preds["tau"],
            }
        )
        if "propensity" in preds:
            out["promotion_propensity"] = preds["propensity"]
        return out
