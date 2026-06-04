"""Production inference: score tracks with counterfactual engagement estimates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from music_uplift.config import AppConfig, InferenceConfig, load_config, project_root
from music_uplift.data.schema import UpliftPrediction
from music_uplift.models.base import UpliftModelArtifact


class UpliftScorer:
    def __init__(
        self,
        artifact: UpliftModelArtifact,
        inference_config: InferenceConfig | None = None,
    ):
        self.artifact = artifact
        self.inference_config = inference_config or InferenceConfig()

    @classmethod
    def from_artifact_path(
        cls,
        path: str | Path,
        config: AppConfig | None = None,
    ) -> UpliftScorer:
        artifact = UpliftModelArtifact.load(path)
        inf_cfg = config.inference if config else InferenceConfig()
        return cls(artifact, inf_cfg)

    @classmethod
    def from_default(cls, config: AppConfig | None = None) -> UpliftScorer:
        root = project_root()
        if config is None:
            config = load_config(root / "config" / "default.yaml")
        path = root / config.training.model_dir / "uplift_model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: music-uplift train"
            )
        return cls.from_artifact_path(path, config)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions with engagement_if_promoted, engagement_if_not_promoted, incremental_lift."""
        required = set(self.artifact.feature_columns) | {"track_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        preds = self.artifact.model.predict_dataframe(
            df,
            self.artifact.preprocessor,
            track_ids=df["track_id"],
        )
        if self.inference_config.min_uplift > 0:
            preds = preds[preds["incremental_lift"] >= self.inference_config.min_uplift]
        return preds.sort_values("incremental_lift", ascending=False).reset_index(drop=True)

    def score_to_records(self, df: pd.DataFrame) -> list[UpliftPrediction]:
        scored = self.score(df)
        records = []
        for i, row in scored.iterrows():
            records.append(
                UpliftPrediction(
                    track_id=row["track_id"],
                    mu_1=row["engagement_if_promoted"],
                    mu_0=row["engagement_if_not_promoted"],
                    tau=row["incremental_lift"],
                    promotion_propensity=row.get("promotion_propensity"),
                    rank=int(i) + 1,
                )
            )
        return records
