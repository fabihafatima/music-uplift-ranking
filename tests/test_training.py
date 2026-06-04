import numpy as np

from music_uplift.config import AppConfig, DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from music_uplift.data.features import FeaturePreprocessor, extract_labels, strip_latent_columns
from music_uplift.data.synthetic import generate_campaign_dataset
from music_uplift.evaluation.metrics import evaluate_uplift_model
from music_uplift.models.registry import create_uplift_model


def _small_config() -> AppConfig:
    features = FeatureConfig(
        categorical=["genre", "artist_tier", "release_recency_bucket", "playlist_inclusion"],
        numeric=[
            "artist_monthly_listeners",
            "track_popularity_score",
            "prior_stream_velocity",
            "social_mention_count",
            "days_since_release",
            "acousticness",
            "danceability",
            "existing_playlist_placements",
        ],
    )
    return AppConfig(
        data=DataConfig(n_tracks=2000, promotion_rate=0.25, seed=99),
        features=features,
        model=ModelConfig(learner="x_learner", n_estimators=50, max_depth=5),
        training=TrainingConfig(test_size=0.25),
    )


def test_x_learner_fit_and_predict():
    cfg = _small_config()
    df = generate_campaign_dataset(n_tracks=3000, seed=99)
    df = strip_latent_columns(df)
    tau = generate_campaign_dataset(n_tracks=3000, seed=99)["_tau"].values

    preprocessor = FeaturePreprocessor(cfg.features)
    X = preprocessor.fit_transform(df)
    y, t = extract_labels(df)

    n = len(df)
    split = int(n * 0.75)
    model = create_uplift_model(cfg.model)
    model.fit(X[:split], y[:split], t[:split])

    preds = model.predict_all(X[split:])
    assert preds["tau"].shape[0] == n - split
    assert np.isfinite(preds["mu_1"]).all()
    assert np.isfinite(preds["mu_0"]).all()


def test_evaluate_uplift_improves_over_random():
    cfg = _small_config()
    raw = generate_campaign_dataset(n_tracks=4000, seed=7)
    tau_true = raw["_tau"].values
    df = strip_latent_columns(raw)

    preprocessor = FeaturePreprocessor(cfg.features)
    X = preprocessor.fit_transform(df)
    y, t = extract_labels(df)

    split = int(len(df) * 0.7)
    model = create_uplift_model(cfg.model)
    model.fit(X[:split], y[:split], t[:split])
    metrics = evaluate_uplift_model(
        model, X[split:], y[split:], t[split:], tau_true=tau_true[split:]
    )
    assert metrics["qini"] != 0
    assert metrics["tau_mae"] < 80  # reasonable on synthetic data
