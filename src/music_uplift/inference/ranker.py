"""Rank tracks by expected incremental promotional lift."""

from __future__ import annotations

import pandas as pd

from music_uplift.config import InferenceConfig
from music_uplift.inference.scorer import UpliftScorer


def rank_tracks_by_uplift(
    df: pd.DataFrame,
    scorer: UpliftScorer,
    top_k: int | None = None,
    min_uplift: float | None = None,
) -> pd.DataFrame:
    """
    Identify tracks that benefit most from promotion.

    Returns ranked DataFrame with:
      - engagement_if_promoted (mu_1)
      - engagement_if_not_promoted (mu_0)
      - incremental_lift (tau)
      - rank
    """
    config = scorer.inference_config
    k = top_k if top_k is not None else config.top_k
    threshold = min_uplift if min_uplift is not None else config.min_uplift

    scored = scorer.score(df)
    if threshold > 0:
        scored = scored[scored["incremental_lift"] >= threshold]
    scored = scored.head(k).copy()
    scored["rank"] = range(1, len(scored) + 1)
    return scored
