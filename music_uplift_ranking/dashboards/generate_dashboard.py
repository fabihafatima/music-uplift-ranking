"""Generate interactive HTML ranking dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_dashboard(
    metrics: dict,
    ranking_scores: pd.DataFrame,
    top_n: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_display = top_n.head(50)[
        [
            c
            for c in [
                "track_id",
                "promotion_rank",
                "ranking_score",
                "predicted_lift_score",
                "genre",
                "artist_tier",
                "streams",
                "historical_streams",
            ]
            if c in top_n.columns
        ]
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Music Promotion Uplift Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f0f12; color: #e8e8ec; }}
    h1, h2 {{ color: #a78bfa; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }}
    .card {{ background: #1a1a22; padding: 1rem; border-radius: 8px; border: 1px solid #333; }}
    .card .value {{ font-size: 1.5rem; font-weight: bold; color: #34d399; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #333; padding: 0.5rem; text-align: left; }}
    th {{ background: #252530; }}
    img {{ max-width: 100%; margin: 1rem 0; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Music Promotional Uplift — Ranking Dashboard</h1>
  <div class="metrics">
    <div class="card"><div>Qini</div><div class="value">{metrics.get("uplift", {}).get("qini_coefficient", 0):.0f}</div></div>
    <div class="card"><div>AUUC</div><div class="value">{metrics.get("uplift", {}).get("auuc", 0):.0f}</div></div>
    <div class="card"><div>ITE R²</div><div class="value">{metrics.get("uplift", {}).get("ite_r2", 0):.3f}</div></div>
    <div class="card"><div>nDCG@10 (ranking)</div><div class="value">{metrics.get("ranking", {}).get("ranking_model", {}).get("ndcg@10", 0):.3f}</div></div>
    <div class="card"><div>Incr. streams top-500</div><div class="value">{metrics.get("business", {}).get("incremental_streams_top500", 0):,.0f}</div></div>
  </div>
  <h2>Evaluation Plots</h2>
  <img src="../../docs/images/qini_curve.png" alt="Qini" width="800">
  <img src="../../docs/images/lift_curve.png" alt="Lift" width="800">
  <img src="../../docs/images/ranking_comparison.png" alt="Ranking" width="800">
  <img src="../../docs/images/business_metrics.png" alt="Business" width="800">
  <h2>Top Promotion Candidates</h2>
  {top_display.to_html(index=False)}
  <h2>Full Metrics</h2>
  <pre>{json.dumps(metrics, indent=2)}</pre>
</body>
</html>
"""
    out = output_dir / "ranking_dashboard.html"
    out.write_text(html)
    ranking_scores.head(1000).to_parquet(output_dir / "dashboard_sample.parquet", index=False)
    return out
