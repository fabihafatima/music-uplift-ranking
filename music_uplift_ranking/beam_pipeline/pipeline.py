"""
Apache Beam ETL pipeline (DirectRunner = local Dataflow simulation).

Stages: ingest → aggregate time windows → impute → normalize → feature tables
"""

from __future__ import annotations

import json
from pathlib import Path

import apache_beam as beam
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions

from music_uplift_ranking.beam_pipeline.transforms import (
    ExtractEngagementFeatures,
    ExtractPromotionFeatures,
    ExtractTrackFeatures,
    ImputeMissing,
    NormalizeNumericFeatures,
    aggregate_history,
)


def _compute_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for col in columns:
        if col in df.columns:
            stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std() or 1.0)}
    return stats


def _compute_medians(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {col: float(df[col].median()) for col in columns if col in df.columns}


def run_beam_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    runner: str = "DirectRunner",
) -> dict[str, Path]:
    import sys

    # Prevent Beam from parsing parent script CLI args (e.g. --n-tracks)
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        return _run_beam_pipeline_impl(raw_dir, output_dir, runner)
    finally:
        sys.argv = saved_argv


def _run_beam_pipeline_impl(
    raw_dir: str | Path,
    output_dir: str | Path,
    runner: str = "DirectRunner",
) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_parquet(raw_dir / "tracks.parquet")
    history = pd.read_parquet(raw_dir / "engagement_history.parquet")
    campaigns = pd.read_parquet(raw_dir / "campaigns.parquet")

    # Pre-compute history aggregates for Beam join
    history_by_track: dict[str, list[dict]] = {}
    for row in history.to_dict("records"):
        history_by_track.setdefault(row["track_id"], []).append(row)

    bundles = []
    track_lookup = tracks.set_index("track_id").to_dict("index")
    for tid in tracks["track_id"]:
        bundles.append(
            (
                tid,
                {
                    "track": track_lookup[tid],
                    "history_agg": aggregate_history(history_by_track.get(tid, [])),
                },
            )
        )

    track_numeric = [
        "artist_popularity_score",
        "release_age_days",
        "num_followers",
        "playlist_appearances",
        "hist_streams_sum_6m",
        "hist_streams_mean_6m",
    ]
    track_df_preview = pd.DataFrame(
        [b[1]["track"] | b[1]["history_agg"] for b in bundles[:1000]]
    )
    # Build preview with proper keys
    preview_rows = []
    for tid, bundle in bundles[:5000]:
        row = {**bundle["track"], **bundle["history_agg"]}
        preview_rows.append(row)
    track_df_preview = pd.DataFrame(preview_rows)
    norm_stats = _compute_stats(track_df_preview, track_numeric)
    medians = _compute_medians(
        campaigns,
        ["promotion_budget_usd", "promotion_duration_days", "historical_streams"],
    )

    options = PipelineOptions(
        runner=runner,
        direct_running_mode="multi_processing",
    )

    track_features_path = output_dir / "track_features.parquet"
    promotion_features_path = output_dir / "promotion_features.parquet"
    engagement_features_path = output_dir / "engagement_features.parquet"

    with beam.Pipeline(options=options) as p:
        track_pcoll = (
            p
            | "TrackBundles" >> beam.Create(bundles)
            | "ExtractTrack" >> beam.ParDo(ExtractTrackFeatures())
            | "ImputeTrack" >> beam.ParDo(ImputeMissing(medians))
            | "NormalizeTrack" >> beam.ParDo(NormalizeNumericFeatures(norm_stats, track_numeric))
        )
        promo_pcoll = (
            p
            | "Campaigns" >> beam.Create(campaigns.to_dict("records"))
            | "ExtractPromo" >> beam.ParDo(ExtractPromotionFeatures())
        )
        engage_pcoll = (
            p
            | "CampaignsEng" >> beam.Create(campaigns.to_dict("records"))
            | "ExtractEngage" >> beam.ParDo(ExtractEngagementFeatures())
        )

        _ = (
            track_pcoll
            | "TrackToList" >> beam.combiners.ToList()
            | "WriteTracks" >> beam.Map(_write_parquet, str(track_features_path))
        )
        _ = (
            promo_pcoll
            | "PromoToList" >> beam.combiners.ToList()
            | "WritePromo" >> beam.Map(_write_parquet, str(promotion_features_path))
        )
        _ = (
            engage_pcoll
            | "EngToList" >> beam.combiners.ToList()
            | "WriteEng" >> beam.Map(_write_parquet, str(engagement_features_path))
        )

    stats_path = output_dir / "normalization_stats.json"
    with stats_path.open("w") as f:
        json.dump({"norm_stats": norm_stats, "medians": medians}, f, indent=2)

    return {
        "track_features": track_features_path,
        "promotion_features": promotion_features_path,
        "engagement_features": engagement_features_path,
        "normalization_stats": stats_path,
    }


def _write_parquet(rows: list[dict], path: str) -> str:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
