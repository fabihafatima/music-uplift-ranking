"""Command-line interface for the music uplift system."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import uvicorn

from music_uplift.config import load_config, project_root
from music_uplift.data.synthetic import generate_campaign_dataset, save_dataset
from music_uplift.inference.ranker import rank_tracks_by_uplift
from music_uplift.inference.scorer import UpliftScorer
from music_uplift.training.pipeline import TrainingPipeline

app = typer.Typer(
    name="music-uplift",
    help="Causal uplift modeling for music promotional campaign impact.",
)


def _config_path_option() -> Path:
    return project_root() / "config" / "default.yaml"


@app.command()
def generate(
    n_tracks: int = typer.Option(None, help="Number of tracks to simulate"),
    output: Path = typer.Option(None, help="Output parquet path"),
    config: Path = typer.Option(None, help="Config YAML path"),
) -> None:
    """Generate synthetic promotional campaign observational data."""
    cfg = load_config(config or _config_path_option())
    n = n_tracks or cfg.data.n_tracks
    df = generate_campaign_dataset(
        n_tracks=n,
        promotion_rate=cfg.data.promotion_rate,
        seed=cfg.data.seed,
    )
    out = output or (project_root() / cfg.data.output_dir / "campaign_observations.parquet")
    save_dataset(df, out)
    typer.echo(f"Generated {len(df):,} tracks -> {out}")
    typer.echo(f"  Promotion rate: {df['promoted'].mean():.1%}")
    typer.echo(f"  Mean engagement: {df['engagement'].mean():.1f}")


@app.command()
def train(
    regenerate: bool = typer.Option(False, "--regenerate", "-r", help="Regenerate data first"),
    config: Path = typer.Option(None, help="Config YAML path"),
) -> None:
    """Train uplift model and persist artifact + evaluation metrics."""
    pipeline = TrainingPipeline(config_path=config or _config_path_option())
    result = pipeline.run(force_regenerate=regenerate)
    typer.echo(f"Trained on {result.train_size:,} / tested on {result.test_size:,}")
    typer.echo(f"  Artifact: {result.artifact_path}")
    typer.echo(f"  Metrics:  {result.metrics_path}")
    typer.echo(json.dumps(result.metrics, indent=2))


@app.command()
def score(
    input_path: Path = typer.Argument(..., help="Parquet/CSV of tracks to score"),
    output: Path = typer.Option(None, help="Output path for scores"),
    config: Path = typer.Option(None, help="Config YAML path"),
) -> None:
    """Score tracks with counterfactual engagement and incremental lift."""
    cfg = load_config(config or _config_path_option())
    scorer = UpliftScorer.from_default(cfg)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    preds = scorer.score(df)
    out = output or input_path.with_suffix(".scored.parquet")
    preds.to_parquet(out, index=False)
    typer.echo(f"Scored {len(preds):,} tracks -> {out}")
    typer.echo(preds.head(10).to_string(index=False))


@app.command()
def rank(
    input_path: Path = typer.Argument(..., help="Parquet/CSV of candidate tracks"),
    top_k: int = typer.Option(100, help="Return top K by incremental lift"),
    min_uplift: float = typer.Option(0.0, help="Minimum predicted lift threshold"),
    output: Path = typer.Option(None, help="Output path"),
    config: Path = typer.Option(None, help="Config YAML path"),
) -> None:
    """Rank tracks by expected incremental promotional lift."""
    cfg = load_config(config or _config_path_option())
    scorer = UpliftScorer.from_default(cfg)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    ranked = rank_tracks_by_uplift(df, scorer, top_k=top_k, min_uplift=min_uplift)
    out = output or input_path.with_suffix(".ranked.parquet")
    ranked.to_parquet(out, index=False)
    typer.echo(f"Top {len(ranked)} promotion candidates -> {out}")
    typer.echo(ranked.head(min(15, len(ranked))).to_string(index=False))


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    config: Path = typer.Option(None),
) -> None:
    """Start FastAPI inference service."""
    cfg = load_config(config or _config_path_option())
    h = host or cfg.serving.host
    p = port or cfg.serving.port
    typer.echo(f"Serving on http://{h}:{p}")
    uvicorn.run(
        "music_uplift.serving.api:app",
        host=h,
        port=p,
        reload=False,
    )


if __name__ == "__main__":
    app()
