"""Apache Beam transforms for feature engineering."""

from __future__ import annotations

import math
from typing import Any, Iterable

import apache_beam as beam


class ExtractTrackFeatures(beam.DoFn):
    """Build track_features row from catalog + aggregated history."""

    def process(self, element: tuple[str, dict[str, Any]]) -> Iterable[dict]:
        track_id, bundle = element
        track = bundle["track"]
        hist = bundle.get("history_agg", {})
        yield {
            "track_id": track_id,
            "genre": track["genre"],
            "artist_tier": track["artist_tier"],
            "artist_popularity_score": track["artist_popularity_score"],
            "release_age_days": track["release_age_days"],
            "num_followers": track["num_followers"],
            "playlist_appearances": track["playlist_appearances"],
            "listener_demo_gen_z_share": track["listener_demo_gen_z_share"],
            "listener_demo_millennial_share": track["listener_demo_millennial_share"],
            "listener_demo_premium_share": track["listener_demo_premium_share"],
            "hist_streams_sum_6m": hist.get("streams_sum", 0),
            "hist_streams_mean_6m": hist.get("streams_mean", 0),
            "hist_streams_std_6m": hist.get("streams_std", 0),
            "hist_saves_sum_6m": hist.get("saves_sum", 0),
            "hist_shares_sum_6m": hist.get("shares_sum", 0),
            "hist_skips_sum_6m": hist.get("skips_sum", 0),
            "hist_completion_mean_6m": hist.get("completion_mean", 0),
            "stream_velocity_6m": hist.get("stream_velocity", 0),
        }


class ExtractPromotionFeatures(beam.DoFn):
    def process(self, campaign: dict) -> Iterable[dict]:
        yield {
            "track_id": campaign["track_id"],
            "promoted": campaign["promoted"],
            "T": campaign["T"],
            "promotion_type": campaign["promotion_type"],
            "promotion_duration_days": campaign["promotion_duration_days"],
            "promotion_budget_usd": campaign["promotion_budget_usd"],
            "propensity_score": campaign["propensity_score"],
            "promo_type_homepage": int(campaign["promotion_type"] == "homepage"),
            "promo_type_playlist": int(campaign["promotion_type"] == "playlist"),
            "promo_type_push": int(campaign["promotion_type"] == "push_notification"),
            "promo_type_recommendation": int(campaign["promotion_type"] == "recommendation_slot"),
        }


class ExtractEngagementFeatures(beam.DoFn):
    def process(self, campaign: dict) -> Iterable[dict]:
        yield {
            "track_id": campaign["track_id"],
            "streams": campaign["streams"],
            "likes": campaign["likes"],
            "saves": campaign["saves"],
            "shares": campaign["shares"],
            "completion_rate": campaign["completion_rate"],
            "engagement_index": campaign["engagement_index"],
            "historical_streams": campaign["historical_streams"],
            "historical_saves": campaign["historical_saves"],
            "historical_shares": campaign["historical_shares"],
            "historical_skips": campaign["historical_skips"],
            "save_rate": _safe_div(campaign["saves"], campaign["streams"]),
            "share_rate": _safe_div(campaign["shares"], campaign["streams"]),
            "skip_rate": _safe_div(campaign["historical_skips"], campaign["historical_streams"]),
        }


class NormalizeNumericFeatures(beam.DoFn):
    """Z-score normalize selected numeric columns using precomputed stats."""

    def __init__(self, stats: dict[str, dict[str, float]], columns: list[str]):
        self.stats = stats
        self.columns = columns

    def process(self, row: dict) -> Iterable[dict]:
        out = dict(row)
        for col in self.columns:
            if col in out and col in self.stats:
                mu = self.stats[col]["mean"]
                sigma = self.stats[col]["std"] or 1.0
                val = out[col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    out[f"{col}_normalized"] = (float(val) - mu) / sigma
                else:
                    out[f"{col}_normalized"] = 0.0
        yield out


class ImputeMissing(beam.DoFn):
    """Fill missing numeric values with column medians."""

    def __init__(self, medians: dict[str, float]):
        self.medians = medians

    def process(self, row: dict) -> Iterable[dict]:
        out = dict(row)
        for col, median in self.medians.items():
            if col in out and (out[col] is None or (isinstance(out[col], float) and math.isnan(out[col]))):
                out[col] = median
        yield out


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def aggregate_history(history_rows: list[dict]) -> dict[str, float]:
    if not history_rows:
        return {}
    streams = [r["streams"] for r in history_rows]
    saves = [r["saves"] for r in history_rows]
    shares = [r["shares"] for r in history_rows]
    skips = [r["skips"] for r in history_rows]
    completions = [r["completion_rate"] for r in history_rows]
    n = len(streams)
    mean_s = sum(streams) / n
    var_s = sum((x - mean_s) ** 2 for x in streams) / max(n - 1, 1)
    return {
        "streams_sum": sum(streams),
        "streams_mean": mean_s,
        "streams_std": var_s**0.5,
        "saves_sum": sum(saves),
        "shares_sum": sum(shares),
        "skips_sum": sum(skips),
        "completion_mean": sum(completions) / n,
        "stream_velocity": (streams[-1] - streams[0]) / max(n - 1, 1) if n > 1 else 0,
    }
