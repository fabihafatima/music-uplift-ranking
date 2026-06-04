"""End-to-end training pipeline for promotional uplift models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from music_uplift.config import AppConfig, load_config, project_root
from music_uplift.data.features import (
    FeaturePreprocessor,
    extract_labels,
    strip_latent_columns,
)
from music_uplift.data.synthetic import generate_campaign_dataset, load_dataset, save_dataset
from music_uplift.evaluation.metrics import evaluate_uplift_model, save_metrics
from music_uplift.models.base import UpliftModelArtifact
from music_uplift.models.registry import create_uplift_model


@dataclass
class TrainingResult:
    artifact_path: Path
    metrics_path: Path
    metrics: dict
    train_size: int
    test_size: int


class TrainingPipeline:
    def __init__(self, config: AppConfig | None = None, config_path: str | Path | None = None):
        if config is None:
            root = project_root()
            default_path = root / "config" / "default.yaml"
            config = load_config(config_path or default_path)
        self.config = config

    def ensure_data(self, force_regenerate: bool = False) -> Path:
        root = project_root()
        data_dir = root / self.config.data.output_dir
        data_path = data_dir / "campaign_observations.parquet"
        if force_regenerate or not data_path.exists():
            df = generate_campaign_dataset(
                n_tracks=self.config.data.n_tracks,
                promotion_rate=self.config.data.promotion_rate,
                seed=self.config.data.seed,
            )
            save_dataset(df, data_path)
        return data_path

    def load_training_data(self) -> pd.DataFrame:
        data_path = self.ensure_data()
        return load_dataset(data_path)

    def run(self, force_regenerate: bool = False) -> TrainingResult:
        if force_regenerate:
            self.ensure_data(force_regenerate=True)
        df = self.load_training_data()

        has_latent = "_tau" in df.columns
        df_model = strip_latent_columns(df)
        tau_true = df["_tau"].values if has_latent else None

        stratify = (
            df_model["promoted"].astype(int)
            if self.config.training.stratify_treatment
            else None
        )
        train_df, test_df = train_test_split(
            df_model,
            test_size=self.config.training.test_size,
            random_state=self.config.model.random_state,
            stratify=stratify,
        )
        if tau_true is not None:
            train_idx = train_df.index
            test_idx = test_df.index
            tau_train = tau_true[train_idx]
            tau_test = tau_true[test_idx]
        else:
            tau_test = None

        preprocessor = FeaturePreprocessor(self.config.features)
        X_train = preprocessor.fit_transform(train_df)
        X_test = preprocessor.transform(test_df)
        y_train, t_train = extract_labels(train_df)
        y_test, t_test = extract_labels(test_df)

        model = create_uplift_model(self.config.model)
        model.fit(X_train, y_train, t_train)

        metrics = evaluate_uplift_model(model, X_test, y_test, t_test, tau_true=tau_test)
        metrics["learner"] = self.config.model.learner
        metrics["train_promotion_rate"] = float(t_train.mean())
        metrics["test_promotion_rate"] = float(t_test.mean())
        metrics["trained_at"] = datetime.now(timezone.utc).isoformat()

        root = project_root()
        model_dir = root / self.config.training.model_dir
        metrics_dir = root / self.config.training.metrics_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        artifact = UpliftModelArtifact(
            learner_name=self.config.model.learner,
            model=model,
            preprocessor=preprocessor,
            config=self.config.model,
            feature_columns=preprocessor.feature_columns,
        )
        artifact_path = artifact.save(model_dir / "uplift_model.joblib")
        metrics_path = save_metrics(metrics, metrics_dir / "evaluation.json")

        manifest = {
            "artifact": str(artifact_path),
            "metrics": str(metrics_path),
            "feature_columns": preprocessor.feature_columns,
            "learner": self.config.model.learner,
        }
        with (model_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        return TrainingResult(
            artifact_path=artifact_path,
            metrics_path=metrics_path,
            metrics=metrics,
            train_size=len(train_df),
            test_size=len(test_df),
        )
