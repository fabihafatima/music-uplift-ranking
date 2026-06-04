"""BigQuery-style table schemas (documentation + DDL)."""

TRACK_FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS `{dataset}.track_features` (
  track_id STRING NOT NULL,
  genre STRING,
  artist_tier STRING,
  artist_popularity_score FLOAT64,
  release_age_days INT64,
  num_followers INT64,
  playlist_appearances INT64,
  listener_demo_gen_z_share FLOAT64,
  listener_demo_millennial_share FLOAT64,
  listener_demo_premium_share FLOAT64,
  hist_streams_sum_6m FLOAT64,
  hist_streams_mean_6m FLOAT64,
  hist_streams_std_6m FLOAT64,
  hist_saves_sum_6m FLOAT64,
  hist_shares_sum_6m FLOAT64,
  hist_skips_sum_6m FLOAT64,
  hist_completion_mean_6m FLOAT64,
  stream_velocity_6m FLOAT64
);
"""

PROMOTION_FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS `{dataset}.promotion_features` (
  track_id STRING NOT NULL,
  promoted INT64,
  T INT64,
  promotion_type STRING,
  promotion_duration_days INT64,
  promotion_budget_usd FLOAT64,
  propensity_score FLOAT64,
  promo_type_homepage INT64,
  promo_type_playlist INT64,
  promo_type_push INT64,
  promo_type_recommendation INT64
);
"""

ENGAGEMENT_FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS `{dataset}.engagement_features` (
  track_id STRING NOT NULL,
  streams INT64,
  likes INT64,
  saves INT64,
  shares INT64,
  completion_rate FLOAT64,
  engagement_index FLOAT64,
  historical_streams INT64,
  historical_saves INT64,
  historical_shares INT64,
  historical_skips INT64,
  save_rate FLOAT64,
  share_rate FLOAT64,
  skip_rate FLOAT64
);
"""
