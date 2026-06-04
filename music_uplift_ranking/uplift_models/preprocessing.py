from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LATENT_COLS = {
    "true_ite_streams",
    "true_ite_engagement",
    "control_outcome_streams",
    "treatment_outcome_streams",
    "observed_outcome_streams",
    "control_outcome_engagement",
    "treatment_outcome_engagement",
    "observed_outcome_engagement",
    "propensity_score",
}

CATEGORICAL = ["genre", "artist_tier", "promotion_type"]
NUMERIC = [
    "artist_popularity_score",
    "release_age_days",
    "num_followers",
    "playlist_appearances",
    "listener_demo_gen_z_share",
    "listener_demo_millennial_share",
    "listener_demo_premium_share",
    "hist_streams_sum_6m",
    "hist_streams_mean_6m",
    "hist_streams_std_6m",
    "hist_saves_sum_6m",
    "stream_velocity_6m",
    "promotion_duration_days",
    "promotion_budget_usd",
    "promo_type_homepage",
    "promo_type_playlist",
    "promo_type_push",
    "promo_type_recommendation",
    "historical_streams",
    "save_rate",
    "share_rate",
    "skip_rate",
    "completion_rate",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def prepare_features(df: pd.DataFrame, fit: bool = False, pipeline: Pipeline | None = None):
    use_cols = [c for c in CATEGORICAL + NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in use_cols]
    num = [c for c in NUMERIC if c in use_cols]
    X = df[use_cols]
    if pipeline is not None:
        return pipeline.transform(X), pipeline
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
        ]
    )
    pipe = Pipeline([("prep", prep)])
    if fit:
        return pipe.fit_transform(X), pipe
    return pipe.transform(X), pipe
