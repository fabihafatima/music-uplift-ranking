from pathlib import Path

import pandas as pd

from music_uplift.config import load_config, project_root
from music_uplift.data.features import strip_latent_columns
from music_uplift.data.synthetic import generate_campaign_dataset
from music_uplift.inference.ranker import rank_tracks_by_uplift
from music_uplift.models.base import UpliftModelArtifact
from music_uplift.training.pipeline import TrainingPipeline


def test_rank_returns_ordered_lift(tmp_path, monkeypatch):
    root = project_root()
    cfg = load_config(root / "config" / "default.yaml")
    cfg.data.n_tracks = 3000
    cfg.model.n_estimators = 40
    cfg.training.model_dir = str(tmp_path / "models")
    cfg.training.metrics_dir = str(tmp_path / "metrics")
    cfg.data.output_dir = str(tmp_path / "data")

    pipeline = TrainingPipeline(config=cfg)
    pipeline.run(force_regenerate=True)

    artifact_path = tmp_path / "models" / "uplift_model.joblib"
    artifact = UpliftModelArtifact.load(artifact_path)

    from music_uplift.inference.scorer import UpliftScorer

    scorer = UpliftScorer(artifact, cfg.inference)
    df = strip_latent_columns(generate_campaign_dataset(n_tracks=500, seed=1))
    ranked = rank_tracks_by_uplift(df, scorer, top_k=50)
    assert len(ranked) <= 50
    lifts = ranked["incremental_lift"].values
    assert (lifts[:-1] >= lifts[1:]).all() or len(lifts) <= 1
