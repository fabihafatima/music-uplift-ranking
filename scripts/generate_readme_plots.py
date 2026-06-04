#!/usr/bin/env python3
"""Regenerate evaluation plots for README (docs/images/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music_uplift_ranking.evaluation.plots import (
    plot_baseline_qini_comparison,
    plot_business_metrics,
    plot_feature_importance,
    plot_ite_distribution,
    plot_lift_curve,
    plot_qini_curve,
    plot_ranking_comparison,
)


def main():
    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)

    report_path = ROOT / "artifacts" / "evaluation" / "evaluation_report.json"
    if not report_path.exists():
        print("Run pipeline first: python music_uplift_ranking/pipelines/run_all.py")
        sys.exit(1)

    with report_path.open() as f:
        metrics = json.load(f)

    campaigns = pd.read_parquet(ROOT / "data" / "raw" / "campaigns.parquet")
    ite = pd.read_parquet(ROOT / "artifacts" / "uplift" / "ite_predictions.parquet")
    df = campaigns.merge(ite, on="track_id")

    t = df["T"].values
    y = df["streams"].values
    tau_pred = df["predicted_lift_score"].values
    tau_true = df["true_ite_streams"].values if "true_ite_streams" in df.columns else None

    plot_qini_curve(tau_pred, t, y, out / "qini_curve.png")
    if tau_true is not None:
        plot_lift_curve(tau_pred, tau_true, out / "lift_curve.png")
        plot_ite_distribution(tau_pred, tau_true, out / "ite_distribution.png")
    plot_ranking_comparison(metrics, out / "ranking_comparison.png")
    plot_baseline_qini_comparison(metrics, out / "baseline_qini.png")
    plot_business_metrics(metrics, out / "business_metrics.png")

    if metrics.get("feature_importance_top5"):
        plot_feature_importance(metrics["feature_importance_top5"], out / "feature_importance.png", top_k=5)

    # Copy to artifacts/plots for dashboard paths
    artifacts_plots = ROOT / "artifacts" / "plots"
    artifacts_plots.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        (artifacts_plots / p.name).write_bytes(p.read_bytes())

    print(f"Plots written to {out}/")
    for p in sorted(out.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
