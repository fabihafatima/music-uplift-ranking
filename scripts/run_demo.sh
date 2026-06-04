#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing package (editable)"
pip install -e ".[dev]" -q

echo "==> Generating synthetic campaign data"
music-uplift generate

echo "==> Training X-Learner uplift model"
music-uplift train

echo "==> Scoring and ranking promotion candidates"
ROOT=$(pwd)
DATA="${ROOT}/data/raw/campaign_observations.parquet"
music-uplift score "$DATA" --output "${ROOT}/artifacts/scored.parquet"
music-uplift rank "$DATA" --top-k 20 --output "${ROOT}/artifacts/top_promotion_candidates.parquet"

echo "==> Done. See artifacts/ for model, metrics, and rankings."
