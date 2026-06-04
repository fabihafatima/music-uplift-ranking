import pandas as pd

from music_uplift.data.features import strip_latent_columns
from music_uplift.data.synthetic import generate_campaign_dataset


def test_generate_campaign_dataset_shape():
    df = generate_campaign_dataset(n_tracks=1000, seed=0)
    assert len(df) == 1000
    assert "promoted" in df.columns
    assert "engagement" in df.columns
    assert "_tau" in df.columns


def test_promotion_rate_approximate():
    df = generate_campaign_dataset(n_tracks=20_000, promotion_rate=0.25, seed=1)
    rate = df["promoted"].mean()
    assert 0.18 < rate < 0.32


def test_latent_lift_positive_on_average():
    df = generate_campaign_dataset(n_tracks=5000, seed=2)
    assert df["_tau"].mean() > 0


def test_strip_latent_columns():
    df = generate_campaign_dataset(n_tracks=10, seed=3)
    clean = strip_latent_columns(df)
    assert "_tau" not in clean.columns
    assert "track_id" in clean.columns
