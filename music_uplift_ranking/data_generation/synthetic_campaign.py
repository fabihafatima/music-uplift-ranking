"""
Synthetic campaign dataset: 50,000 tracks × 6 months historical engagement.

Generates observational data with confounded treatment assignment and
heterogeneous treatment effects on multi-outcome engagement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["pop", "hip_hop", "rock", "electronic", "r_and_b", "country", "latin", "indie"]
ARTIST_TIERS = ["new", "emerging", "mid", "established", "star"]
PROMO_TYPES = ["none", "homepage", "playlist", "push_notification", "recommendation_slot"]
DEMO_SEGMENTS = ["gen_z", "millennial", "gen_x", "global_south", "premium_subscriber"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _generate_track_catalog(rng: np.random.Generator, n_tracks: int) -> pd.DataFrame:
    genre = rng.choice(GENRES, n_tracks)
    artist_tier = rng.choice(ARTIST_TIERS, n_tracks, p=[0.12, 0.28, 0.30, 0.22, 0.08])
    tier_mult = {"new": 0.15, "emerging": 0.4, "mid": 0.8, "established": 1.3, "star": 2.8}
    mult = np.array([tier_mult[t] for t in artist_tier])

    artist_popularity_score = np.clip(rng.beta(2, 4, n_tracks) * mult / 3 + 0.05, 0, 1)
    release_age_days = np.where(
        rng.random(n_tracks) < 0.15,
        rng.integers(1, 60, n_tracks),
        np.where(
            rng.random(n_tracks) < 0.5,
            rng.integers(60, 365, n_tracks),
            rng.integers(365, 2500, n_tracks),
        ),
    )
    num_followers = (rng.lognormal(8, 1.5, n_tracks) * mult * 100).astype(int)
    playlist_appearances = rng.poisson(np.where(artist_popularity_score > 0.5, 4, 1), n_tracks)

    demo_primary = rng.choice(DEMO_SEGMENTS, n_tracks)
    demo_gen_z = rng.uniform(0, 1, n_tracks) * (demo_primary == "gen_z") * 0.8 + rng.uniform(0, 0.3, n_tracks)
    demo_millennial = rng.uniform(0, 1, n_tracks) * (demo_primary == "millennial") * 0.8 + rng.uniform(0, 0.25, n_tracks)
    demo_premium = rng.uniform(0, 1, n_tracks) * (demo_primary == "premium_subscriber") * 0.7 + rng.uniform(0, 0.2, n_tracks)

    return pd.DataFrame(
        {
            "track_id": [f"TRK_{i:07d}" for i in range(n_tracks)],
            "genre": genre,
            "artist_tier": artist_tier,
            "artist_popularity_score": artist_popularity_score.round(4),
            "release_age_days": release_age_days,
            "num_followers": num_followers,
            "playlist_appearances": playlist_appearances,
            "listener_demo_gen_z_share": demo_gen_z.round(4),
            "listener_demo_millennial_share": demo_millennial.round(4),
            "listener_demo_premium_share": demo_premium.round(4),
            "listener_demo_primary": demo_primary,
        }
    )


def _generate_engagement_history(
    rng: np.random.Generator,
    tracks: pd.DataFrame,
    n_months: int = 6,
) -> pd.DataFrame:
    """Monthly historical engagement per track (6 months)."""
    rows = []
    base_streams = (
        200
        + tracks["artist_popularity_score"].values * 8000
        + tracks["playlist_appearances"].values * 120
        - tracks["release_age_days"].values * 0.05
    )
    base_streams = np.maximum(base_streams, 50)

    for month_idx in range(n_months):
        month = f"2025-{month_idx + 1:02d}"
        trend = 1 + 0.03 * month_idx
        noise_scale = 0.12
        streams = base_streams * trend * (1 + rng.normal(0, noise_scale, len(tracks)))
        streams = np.maximum(streams, 0).astype(int)
        saves = (streams * rng.uniform(0.02, 0.12, len(tracks))).astype(int)
        shares = (streams * rng.uniform(0.005, 0.04, len(tracks))).astype(int)
        skips = (streams * rng.uniform(0.1, 0.35, len(tracks))).astype(int)
        likes = (streams * rng.uniform(0.03, 0.15, len(tracks))).astype(int)
        completion_rate = np.clip(rng.beta(3, 2, len(tracks)) * 0.5 + 0.35, 0.1, 0.99)

        rows.append(
            pd.DataFrame(
                {
                    "track_id": tracks["track_id"].values,
                    "month": month,
                    "month_index": month_idx,
                    "streams": streams,
                    "likes": likes,
                    "saves": saves,
                    "shares": shares,
                    "skips": skips,
                    "completion_rate": completion_rate.round(4),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _genre_lift(genre: np.ndarray) -> np.ndarray:
    boost = {"pop": 30, "latin": 25, "hip_hop": 20, "electronic": 15, "indie": -20, "country": -5}
    return np.array([boost.get(g, 0) for g in genre])


def _generate_campaign_observations(
    rng: np.random.Generator,
    tracks: pd.DataFrame,
    history: pd.DataFrame,
    promotion_rate: float,
) -> pd.DataFrame:
    """Campaign-period outcomes with latent potential outcomes and biased treatment."""
    n = len(tracks)
    hist_agg = (
        history.groupby("track_id")
        .agg(
            historical_streams=("streams", "sum"),
            historical_saves=("saves", "sum"),
            historical_shares=("shares", "sum"),
            historical_skips=("skips", "sum"),
            historical_likes=("likes", "sum"),
            avg_completion_rate=("completion_rate", "mean"),
        )
        .reset_index()
    )
    df = tracks.merge(hist_agg, on="track_id")

    # Promotion features (assigned before treatment realization)
    promo_type_idx = rng.choice(len(PROMO_TYPES), n, p=[0.55, 0.12, 0.15, 0.10, 0.08])
    promo_type = np.array([PROMO_TYPES[i] for i in promo_type_idx])
    promotion_duration_days = np.where(
        promo_type == "none", 0, rng.integers(3, 30, n)
    )
    promotion_budget_usd = np.where(
        promo_type == "none",
        0.0,
        rng.uniform(500, 50000, n) * (1 + df["artist_popularity_score"].values),
    ).round(2)

    # Baseline engagement Y(0) — composite anchored on streams
    genre_mult = _genre_lift(df["genre"].values)
    mu_0_streams = (
        80
        + df["historical_streams"].values * 0.08
        + df["artist_popularity_score"].values * 1200
        + df["playlist_appearances"].values * 25
        - df["release_age_days"].values * 0.02
        + genre_mult
    )
    mu_0_streams = mu_0_streams * (1 + rng.normal(0, 0.1, n))
    mu_0_streams = np.maximum(mu_0_streams, 10)

    # Heterogeneous treatment effect on streams
    tau_streams = (
        40
        + df["artist_popularity_score"].values * 150
        + np.where(df["artist_tier"].values == "new", 120, 0)
        + np.where(df["artist_tier"].values == "emerging", 80, 0)
        + np.where(df["artist_tier"].values == "star", -60, 0)
        + np.where(df["release_age_days"].values < 60, 90, 0)
        + _genre_lift(df["genre"].values)
        + np.where(promo_type == "homepage", 50, 0)
        + np.where(promo_type == "playlist", 35, 0)
        + np.where(promo_type == "push_notification", 25, 0)
        + np.where(promo_type == "recommendation_slot", 40, 0)
        + promotion_budget_usd * 0.001
    )
    tau_streams = tau_streams * (1 + rng.normal(0, 0.08, n))

    mu_1_streams = mu_0_streams + tau_streams

    # Propensity P(T=1|X) — selection bias toward popular / new tracks
    logit_p = (
        -2.0
        + df["artist_popularity_score"].values * 3.5
        + np.where(df["artist_tier"].values.isin(["star", "established"]), 1.0, 0)
        + np.where(df["release_age_days"].values < 90, 0.9, 0)
        + df["playlist_appearances"].values * 0.08
        + np.where(promo_type != "none", 0.5, -0.8)
    )
    propensity = _sigmoid(logit_p)
    propensity = propensity * (promotion_rate / propensity.mean())
    propensity = np.clip(propensity, 0.02, 0.98)

    promoted = rng.binomial(1, propensity).astype(bool)
    # If not promoted, clear promotion features
    promo_type = np.where(promoted, promo_type, "none")
    promotion_duration_days = np.where(promoted, promotion_duration_days, 0)
    promotion_budget_usd = np.where(promoted, promotion_budget_usd, 0.0)

    # Multi-outcome generation
    def _outcomes(mu_s: np.ndarray, tau_s: np.ndarray, treated: np.ndarray) -> dict:
        base = np.where(treated, mu_s + tau_s, mu_s)
        noise = 1 + rng.normal(0, 0.07, n)
        streams = np.maximum(base * noise, 0)
        saves = streams * rng.uniform(0.04, 0.14, n)
        likes = streams * rng.uniform(0.05, 0.18, n)
        shares = streams * rng.uniform(0.01, 0.06, n)
        completion = np.clip(
            0.4 + df["avg_completion_rate"].values * 0.4 + np.where(treated, 0.05, 0) + rng.normal(0, 0.05, n),
            0.1,
            0.99,
        )
        return {"streams": streams, "saves": saves, "likes": likes, "shares": shares, "completion_rate": completion}

    y0 = _outcomes(mu_0_streams, np.zeros(n), np.zeros(n, dtype=bool))
    y1 = _outcomes(mu_1_streams, tau_streams, np.ones(n, dtype=bool))
    y_obs = _outcomes(mu_0_streams, tau_streams, promoted)

    engagement_index = (
        y_obs["streams"] * 1.0
        + y_obs["saves"] * 5
        + y_obs["likes"] * 2
        + y_obs["shares"] * 8
    )

    return pd.DataFrame(
        {
            "track_id": df["track_id"],
            "promoted": promoted.astype(int),
            "T": promoted.astype(int),
            "promotion_type": promo_type,
            "promotion_duration_days": promotion_duration_days,
            "promotion_budget_usd": promotion_budget_usd,
            "propensity_score": propensity.round(4),
            # Observed outcomes
            "streams": y_obs["streams"].round(0).astype(int),
            "likes": y_obs["likes"].round(0).astype(int),
            "saves": y_obs["saves"].round(0).astype(int),
            "shares": y_obs["shares"].round(0).astype(int),
            "completion_rate": y_obs["completion_rate"].round(4),
            "engagement_index": engagement_index.round(2),
            # Latent potential outcomes (streams primary for ITE)
            "control_outcome_streams": y0["streams"].round(0).astype(int),
            "treatment_outcome_streams": y1["streams"].round(0).astype(int),
            "observed_outcome_streams": y_obs["streams"].round(0).astype(int),
            "control_outcome_engagement": (
                y0["streams"] + y0["saves"] * 5 + y0["likes"] * 2
            ).round(2),
            "treatment_outcome_engagement": (
                y1["streams"] + y1["saves"] * 5 + y1["likes"] * 2
            ).round(2),
            "observed_outcome_engagement": engagement_index.round(2),
            "true_ite_streams": tau_streams.round(2),
            "true_ite_engagement": (tau_streams * 1.2 + tau_streams * 0.05 * 5).round(2),
            # Pass-through for feature joins
            "genre": df["genre"],
            "artist_tier": df["artist_tier"],
            "artist_popularity_score": df["artist_popularity_score"],
            "release_age_days": df["release_age_days"],
            "num_followers": df["num_followers"],
            "playlist_appearances": df["playlist_appearances"],
            "historical_streams": df["historical_streams"],
            "historical_saves": df["historical_saves"],
            "historical_shares": df["historical_shares"],
            "historical_skips": df["historical_skips"],
            "listener_demo_gen_z_share": df["listener_demo_gen_z_share"],
            "listener_demo_millennial_share": df["listener_demo_millennial_share"],
            "listener_demo_premium_share": df["listener_demo_premium_share"],
        }
    )


def generate_full_dataset(
    n_tracks: int = 50_000,
    n_months: int = 6,
    promotion_rate: float = 0.25,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tracks = _generate_track_catalog(rng, n_tracks)
    history = _generate_engagement_history(rng, tracks, n_months)
    campaigns = _generate_campaign_observations(rng, tracks, history, promotion_rate)
    return {
        "tracks": tracks,
        "engagement_history": history,
        "campaigns": campaigns,
    }


def save_raw_datasets(datasets: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in datasets.items():
        p = output_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths
