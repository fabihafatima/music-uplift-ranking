"""FastAPI service for uplift scoring and promotion targeting."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from music_uplift.config import load_config, project_root
from music_uplift.data.schema import UpliftPrediction
from music_uplift.inference.ranker import rank_tracks_by_uplift
from music_uplift.inference.scorer import UpliftScorer


class TrackFeatures(BaseModel):
    track_id: str
    genre: str
    artist_tier: str
    release_recency_bucket: str
    playlist_inclusion: str
    artist_monthly_listeners: float = Field(ge=0)
    track_popularity_score: float = Field(ge=0, le=1)
    prior_stream_velocity: float = Field(ge=0)
    social_mention_count: float = Field(ge=0)
    days_since_release: int = Field(ge=0)
    acousticness: float = Field(ge=0, le=1)
    danceability: float = Field(ge=0, le=1)
    existing_playlist_placements: int = Field(ge=0)


class ScoreRequest(BaseModel):
    tracks: list[TrackFeatures]


class RankRequest(ScoreRequest):
    top_k: int = Field(default=100, ge=1, le=10_000)
    min_uplift: float = Field(default=0.0)


class HealthResponse(BaseModel):
    status: str
    learner: str | None = None
    version: str


_scorer: UpliftScorer | None = None


def get_scorer() -> UpliftScorer:
    global _scorer
    if _scorer is None:
        config = load_config(project_root() / "config" / "default.yaml")
        _scorer = UpliftScorer.from_default(config)
    return _scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_scorer()
    except FileNotFoundError:
        pass
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Music Promotional Uplift API",
        description=(
            "Estimates counterfactual engagement and incremental lift from "
            "promotional campaigns using causal uplift modeling."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            scorer = get_scorer()
            return HealthResponse(
                status="ok",
                learner=scorer.artifact.learner_name,
                version=scorer.artifact.version,
            )
        except FileNotFoundError:
            return HealthResponse(status="no_model", learner=None, version="0.1.0")

    @app.post("/score", response_model=list[UpliftPrediction])
    def score_tracks(request: ScoreRequest) -> list[UpliftPrediction]:
        try:
            scorer = get_scorer()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        df = pd.DataFrame([t.model_dump() for t in request.tracks])
        return scorer.score_to_records(df)

    @app.post("/rank", response_model=list[UpliftPrediction])
    def rank_tracks(request: RankRequest) -> list[UpliftPrediction]:
        try:
            scorer = get_scorer()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        df = pd.DataFrame([t.model_dump() for t in request.tracks])
        ranked = rank_tracks_by_uplift(
            df, scorer, top_k=request.top_k, min_uplift=request.min_uplift
        )
        return [
            UpliftPrediction(
                track_id=row["track_id"],
                mu_1=row["engagement_if_promoted"],
                mu_0=row["engagement_if_not_promoted"],
                tau=row["incremental_lift"],
                promotion_propensity=row.get("promotion_propensity"),
                rank=int(row["rank"]),
            )
            for _, row in ranked.iterrows()
        ]

    @app.get("/metrics")
    def get_training_metrics() -> dict[str, Any]:
        metrics_path = project_root() / "artifacts" / "metrics" / "evaluation.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="No evaluation metrics found")
        import json

        with metrics_path.open() as f:
            return json.load(f)

    return app


app = create_app()
