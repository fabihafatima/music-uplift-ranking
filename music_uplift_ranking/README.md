# music_uplift_ranking

Production-style **causal uplift + learning-to-rank** system for music promotional campaigns.

## System overview

```
Raw events (synthetic or warehouse)
        │
        ▼
┌───────────────────┐
│ Apache Beam ETL   │  aggregation, imputation, normalization
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Feature Store     │  track_features | promotion_features | engagement_features
│ (BigQuery local)  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Uplift models     │  T-Learner, S-Learner, X-Learner (primary)
│ XGBoost/LightGBM  │  → ITE / predicted_lift_score per track
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Ranking models    │  LambdaMART, XGBoost Ranker
└─────────┬─────────┘
          ▼
   Top-N promotion queue + evaluation + dashboards
```

## Dataset (50,000 tracks × 6 months)

| Layer | Content |
|-------|---------|
| **Track features** | Genre, artist popularity, release age, followers, playlist appearances, listener demographics |
| **6-month history** | Monthly streams, likes, saves, shares, skips, completion rate |
| **Promotion features** | Type (homepage / playlist / push / recommendation), duration, budget |
| **Treatment** | `T=1` promoted, `T=0` otherwise (biased propensity) |
| **Outcomes** | Streams, likes, saves, shares, completion rate |
| **Latent** | `control_outcome_*`, `treatment_outcome_*`, `true_ite_streams` |

## Quick start

```bash
cd music-promotion
pip install -e ".[ranking]"
python music_uplift_ranking/pipelines/run_all.py
```

Outputs:

- `data/raw/` — tracks, engagement_history, campaigns
- `data/feature_store/` — three feature tables
- `artifacts/uplift/` — T/S/X models + ITE predictions
- `artifacts/ranking/` — LambdaMART / XGB ranker scores
- `artifacts/top_promotion_candidates.parquet`
- `artifacts/plots/` — Qini, lift, SHAP, feature importance
- `artifacts/dashboards/ranking_dashboard.html`

## Module map

| Directory | Responsibility |
|-----------|----------------|
| `data_generation/` | Synthetic 50k-track dataset with confounding + heterogeneous effects |
| `beam_pipeline/` | Apache Beam feature extraction (local DirectRunner) |
| `feature_store/` | BigQuery DDL + local Parquet/DuckDB store |
| `uplift_models/` | T/S/X learners, ITE estimation |
| `ranking_models/` | LambdaMART, XGBoost Ranker second stage |
| `evaluation/` | Qini, AUUC, nDCG, MAP, P@10, business metrics |
| `notebooks/` | Analysis scripts |
| `dashboards/` | HTML ranking dashboard |

## Evaluation

**Uplift:** Qini curve, Qini coefficient, AUUC  
**Ranking:** nDCG@10, MAP, Precision@10  
**Business:** Incremental streams/saves, promotion efficiency  
**Baselines:** Historical streams ranking, engagement prediction ranking

### Latest results (50k tracks)

See the [project README](../README.md#results-50000-track-run) for full tables and charts.

| Plot | Description |
|------|-------------|
| ![Qini](../docs/images/qini_curve.png) | Model vs random targeting |
| ![Lift](../docs/images/lift_curve.png) | True ITE captured by targeting fraction |
| ![Ranking](../docs/images/ranking_comparison.png) | nDCG / MAP / P@10 by approach |
| ![Business](../docs/images/business_metrics.png) | Top-500 incremental impact |

Regenerate plots after a new run:

```bash
python scripts/generate_readme_plots.py
```

## GCP production

1. Run `beam_pipeline` on Dataflow (`--runner=DataflowRunner`)
2. Load feature tables to BigQuery (`feature_store/bigquery_schema.sql`)
3. Schedule uplift + ranking training from BQ master view
