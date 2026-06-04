"""Configuration loading for the uplift pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    n_tracks: int = 50_000
    promotion_rate: float = 0.25
    seed: int = 42
    output_dir: str = "data/raw"


@dataclass
class FeatureConfig:
    categorical: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    learner: str = "x_learner"
    base_estimator: str = "sklearn_gb"
    propensity_model: str = "logistic"
    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.05
    min_child_samples: int = 50
    random_state: int = 42


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    val_size: float = 0.15
    stratify_treatment: bool = True
    model_dir: str = "artifacts/models"
    metrics_dir: str = "artifacts/metrics"


@dataclass
class InferenceConfig:
    top_k: int = 100
    min_uplift: float = 0.0


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def _merge_dict(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config; defaults apply when file is missing."""
    defaults = {
        "data": {},
        "features": {},
        "model": {},
        "training": {},
        "inference": {},
        "serving": {},
    }
    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open() as f:
                loaded = yaml.safe_load(f) or {}
            defaults = _merge_dict(defaults, loaded)

    return AppConfig(
        data=DataConfig(**defaults.get("data", {})),
        features=FeatureConfig(**defaults.get("features", {})),
        model=ModelConfig(**defaults.get("model", {})),
        training=TrainingConfig(**defaults.get("training", {})),
        inference=InferenceConfig(**defaults.get("inference", {})),
        serving=ServingConfig(**defaults.get("serving", {})),
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
