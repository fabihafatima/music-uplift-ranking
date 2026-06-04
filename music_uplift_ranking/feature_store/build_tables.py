from __future__ import annotations

from pathlib import Path

import pandas as pd

from music_uplift_ranking.feature_store.bigquery_local import LocalBigQueryStore


def load_feature_tables(feature_store_dir: str | Path) -> dict[str, pd.DataFrame]:
    store = LocalBigQueryStore(feature_store_dir)
    return {
        "track_features": store.load_table("track_features"),
        "promotion_features": store.load_table("promotion_features"),
        "engagement_features": store.load_table("engagement_features"),
        "master": store.master_training_view(),
    }
