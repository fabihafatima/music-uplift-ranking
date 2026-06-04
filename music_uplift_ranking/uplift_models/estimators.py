from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from music_uplift_ranking.uplift_models.config import UpliftConfig


def make_regressor(cfg: UpliftConfig):
    if cfg.base_estimator == "xgboost":
        try:
            import xgboost as xgb

            return xgb.XGBRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                random_state=cfg.random_state,
                n_jobs=-1,
                verbosity=0,
            )
        except Exception:
            pass
    if cfg.base_estimator == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                random_state=cfg.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        except Exception:
            pass
    return GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        random_state=cfg.random_state,
    )


def make_classifier(cfg: UpliftConfig):
    return LogisticRegression(max_iter=1000, random_state=cfg.random_state)
