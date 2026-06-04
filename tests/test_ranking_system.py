import tempfile
from pathlib import Path


def test_synthetic_dataset_scale():
    from music_uplift_ranking.data_generation import generate_full_dataset

    ds = generate_full_dataset(n_tracks=500, n_months=6, seed=0)
    assert len(ds["tracks"]) == 500
    assert len(ds["engagement_history"]) == 500 * 6
    assert "true_ite_streams" in ds["campaigns"].columns
    assert "control_outcome_streams" in ds["campaigns"].columns


def test_uplift_learners_smoke():
    from music_uplift_ranking.data_generation import generate_full_dataset
    from music_uplift_ranking.uplift_models.config import UpliftConfig
    from music_uplift_ranking.uplift_models.train import train_uplift_models

    ds = generate_full_dataset(n_tracks=800, seed=1)
    cfg = UpliftConfig(base_estimator="sklearn_gb", n_estimators=30, max_depth=4)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        train_uplift_models(ds["campaigns"], cfg, Path(tmp), primary_learner="x_learner")
