"""Factory for base regression / classification estimators."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from music_uplift.config import ModelConfig


def make_regressor(config: ModelConfig) -> object:
    if config.base_estimator == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            verbose=-1,
            n_jobs=-1,
        )
    if config.base_estimator == "sklearn_gb":
        return GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_samples_leaf=config.min_child_samples,
            random_state=config.random_state,
        )
    raise ValueError(f"Unknown base_estimator: {config.base_estimator}")


def make_propensity_model(config: ModelConfig) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=config.random_state,
    )
