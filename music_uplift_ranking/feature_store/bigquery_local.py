"""
Local BigQuery simulation using Parquet + DuckDB SQL interface.

Swap `use_duckdb=False` to use pure pandas joins when DuckDB unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class LocalBigQueryStore:
    """Simulates BigQuery dataset with materialized Parquet tables."""

    def __init__(self, feature_store_dir: str | Path, dataset: str = "music_uplift_local"):
        self.feature_store_dir = Path(feature_store_dir)
        self.dataset = dataset
        self._conn = None

    def _table_path(self, table: str) -> Path:
        return self.feature_store_dir / f"{table}.parquet"

    def load_table(self, table: str) -> pd.DataFrame:
        path = self._table_path(table)
        if not path.exists():
            raise FileNotFoundError(f"Table not found: {path}")
        return pd.read_parquet(path)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            import duckdb
        except ImportError:
            return self._pandas_query_fallback(sql)

        conn = duckdb.connect()
        for table in ["track_features", "promotion_features", "engagement_features"]:
            path = self._table_path(table)
            if path.exists():
                conn.register(table, pd.read_parquet(path))
        return conn.execute(sql).df()

    def _pandas_query_fallback(self, sql: str) -> pd.DataFrame:
        """Minimal fallback for the master training join."""
        track = self.load_table("track_features")
        promo = self.load_table("promotion_features")
        engage = self.load_table("engagement_features")
        campaigns = pd.read_parquet(self.feature_store_dir.parent / "raw" / "campaigns.parquet")
        latent_cols = [
            c
            for c in campaigns.columns
            if c.startswith("true_ite") or c.startswith("control_") or c.startswith("treatment_")
        ]
        df = track.merge(promo, on="track_id").merge(engage, on="track_id", suffixes=("", "_eng"))
        if latent_cols:
            df = df.merge(campaigns[["track_id"] + latent_cols], on="track_id", how="left")
        return df

    def register_tables(self) -> None:
        """Ensure tables exist (no-op if parquet present)."""
        for t in ["track_features", "promotion_features", "engagement_features"]:
            if not self._table_path(t).exists():
                raise FileNotFoundError(f"Run beam pipeline first. Missing {t}")

    def master_training_view(self) -> pd.DataFrame:
        sql = """
        SELECT
            t.track_id,
            t.genre,
            t.artist_tier,
            t.artist_popularity_score,
            t.release_age_days,
            t.num_followers,
            t.playlist_appearances,
            t.listener_demo_gen_z_share,
            t.listener_demo_millennial_share,
            t.listener_demo_premium_share,
            t.hist_streams_sum_6m,
            t.hist_streams_mean_6m,
            t.hist_streams_std_6m,
            t.hist_saves_sum_6m,
            t.stream_velocity_6m,
            p.promoted,
            p.T,
            p.promotion_type,
            p.promotion_duration_days,
            p.promotion_budget_usd,
            p.propensity_score,
            p.promo_type_homepage,
            p.promo_type_playlist,
            p.promo_type_push,
            p.promo_type_recommendation,
            e.streams,
            e.likes,
            e.saves,
            e.shares,
            e.completion_rate,
            e.engagement_index,
            e.historical_streams,
            e.save_rate,
            e.share_rate,
            e.skip_rate
        FROM track_features t
        JOIN promotion_features p ON t.track_id = p.track_id
        JOIN engagement_features e ON t.track_id = e.track_id
        """
        return self.query(sql)
