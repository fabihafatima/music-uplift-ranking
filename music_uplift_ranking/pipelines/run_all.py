#!/usr/bin/env python3
"""End-to-end pipeline: data → Beam → features → uplift → ranking → evaluation → dashboards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve_estimator(name: str) -> str:
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except OSError:
            return "sklearn_gb"
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            return "sklearn_gb"
    return name


def main():
    parser = argparse.ArgumentParser(description="Music uplift ranking pipeline")
    parser.add_argument("--config", default=str(ROOT / "music_uplift_ranking" / "config.yaml"))
    parser.add_argument("--skip-beam", action="store_true")
    parser.add_argument("--n-tracks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    n_tracks = args.n_tracks or cfg["data"]["n_tracks"]
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    feature_dir = ROOT / cfg["data"]["feature_store_dir"]

    print("=" * 60)
    print("1. Dataset generation")
    from music_uplift_ranking.data_generation import generate_full_dataset, save_raw_datasets

    datasets = generate_full_dataset(
        n_tracks=n_tracks,
        n_months=cfg["data"]["n_months"],
        promotion_rate=cfg["data"]["promotion_rate"],
        seed=cfg["data"]["seed"],
    )
    paths = save_raw_datasets(datasets, raw_dir)
    print(f"   Tracks: {len(datasets['tracks']):,}")
    print(f"   History rows: {len(datasets['engagement_history']):,} (6 months)")
    print(f"   Campaigns: {len(datasets['campaigns']):,}")
    for k, p in paths.items():
        print(f"   -> {p}")

    if not args.skip_beam:
        print("=" * 60)
        print("2. Apache Beam feature pipeline (local DirectRunner)")
        from music_uplift_ranking.beam_pipeline import run_beam_pipeline

        feat_paths = run_beam_pipeline(raw_dir, feature_dir)
        for k, p in feat_paths.items():
            print(f"   {k}: {p}")
    else:
        print("   (skipped beam)")

    print("=" * 60)
    print("3. BigQuery feature store (local)")
    from music_uplift_ranking.feature_store import LocalBigQueryStore

    store = LocalBigQueryStore(feature_dir, cfg["data"]["bq_dataset"])
    master = store.master_training_view()
    campaigns = pd.read_parquet(raw_dir / "campaigns.parquet")
    train_df = master.merge(
        campaigns[
            [
                "track_id",
                "true_ite_streams",
                "true_ite_engagement",
                "control_outcome_streams",
                "treatment_outcome_streams",
                "observed_outcome_streams",
            ]
        ],
        on="track_id",
    )
    print(f"   Master view: {len(train_df):,} rows × {len(train_df.columns)} cols")

    print("=" * 60)
    print("4. Uplift modeling (T / S / X learners)")
    from music_uplift_ranking.uplift_models.config import UpliftConfig
    from music_uplift_ranking.uplift_models.train import train_uplift_models

    uplift_cfg = UpliftConfig(
        base_estimator=_resolve_estimator(cfg["uplift"]["base_estimator"]),
        n_estimators=cfg["uplift"]["n_estimators"],
        max_depth=cfg["uplift"]["max_depth"],
        learning_rate=cfg["uplift"]["learning_rate"],
    )
    uplift_dir = ROOT / "artifacts" / "uplift"
    train_uplift_models(
        train_df,
        uplift_cfg,
        uplift_dir,
        primary_learner=cfg["uplift"]["primary_learner"],
    )
    ite_preds = pd.read_parquet(uplift_dir / "ite_predictions.parquet")
    print(f"   ITE predictions: {len(ite_preds):,}")

    print("=" * 60)
    print("5. Second-stage ranking (LambdaMART + XGB Ranker)")
    from music_uplift_ranking.ranking_models import generate_top_n, train_ranking_models

    rank_df = train_df.merge(ite_preds, on="track_id")
    ranking_dir = ROOT / "artifacts" / "ranking"
    train_ranking_models(rank_df, output_dir=ranking_dir, primary=cfg["ranking"]["primary_ranker"])
    ranking_scores = pd.read_parquet(ranking_dir / "ranking_scores.parquet")
    top_n = generate_top_n(ranking_scores, n=cfg["ranking"]["top_n"])
    top_path = ROOT / "artifacts" / "top_promotion_candidates.parquet"
    top_n.to_parquet(top_path, index=False)
    print(f"   Top-{cfg['ranking']['top_n']} saved: {top_path}")

    print("=" * 60)
    print("6. Evaluation + plots + dashboards")
    from music_uplift_ranking.dashboards.generate_dashboard import generate_dashboard
    from music_uplift_ranking.evaluation import run_full_evaluation

    eval_dir = ROOT / cfg["evaluation"]["output_dir"]
    plots_dir = ROOT / cfg["evaluation"]["plots_dir"]
    metrics = run_full_evaluation(
        campaigns,
        ite_preds,
        ranking_scores,
        uplift_dir / f"{cfg['uplift']['primary_learner']}.joblib",
        eval_dir,
        plots_dir,
    )
    dash_path = generate_dashboard(metrics, ranking_scores, top_n, ROOT / cfg["evaluation"]["dashboards_dir"])
    print("   Copying plots to docs/images/ for README...")
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_readme_plots.py")], check=False)
    print(json.dumps(metrics, indent=2)[:2000])
    print(f"   Dashboard: {dash_path}")
    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
