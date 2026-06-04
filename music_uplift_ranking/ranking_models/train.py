from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from music_uplift_ranking.ranking_models.lambdamart import LambdaMARTRanker
from music_uplift_ranking.ranking_models.xgb_ranker import XGBoostRanker


RANKING_FEATURES = [
    "predicted_lift_score",
    "track_quality_score",
    "artist_popularity_score",
    "historical_streams",
    "hist_streams_mean_6m",
    "hist_saves_sum_6m",
    "engagement_index",
    "completion_rate",
    "save_rate",
]


def _build_ranking_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    if "track_quality_score" not in df.columns:
        df["track_quality_score"] = (
            df.get("completion_rate", 0.5) * 0.4
            + df.get("save_rate", 0.05) * 0.3
            + df.get("artist_popularity_score", 0.5) * 0.3
        )
    cols = [c for c in RANKING_FEATURES if c in df.columns]
    X = df[cols].fillna(0).values.astype(float)
    return X, cols


def train_ranking_models(
    df: pd.DataFrame,
    relevance_col: str = "true_ite_streams",
    output_dir: str | Path = "artifacts/ranking",
    primary: str = "lambdamart",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if relevance_col not in df.columns:
        relevance_col = "predicted_lift_score"

    X, feat_cols = _build_ranking_matrix(df)
    y = df[relevance_col].fillna(df["predicted_lift_score"]).values

    # Binarize relevance for ranking metrics
    y_rank = (y > np.percentile(y, 75)).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_rank[train_idx], y_rank[test_idx]

    # Single group = full catalog ranking
    group_train = [len(X_train)]
    group_test = [len(X_test)]

    models = {}

    def _sklearn_ranker_fallback():
        from sklearn.ensemble import GradientBoostingRegressor

        m = GradientBoostingRegressor(n_estimators=200, random_state=42)
        m.fit(X_train, y_train)
        return m

    try:
        lmart = LambdaMARTRanker()
        lmart.fit(X_train, y_train, group=np.array(group_train))
        models["lambdamart"] = lmart
        joblib.dump(lmart, output_dir / "lambdamart.joblib")
    except Exception as e:
        models["lambdamart_error"] = str(e)
        models["lambdamart"] = _sklearn_ranker_fallback()

    try:
        xgb_r = XGBoostRanker()
        xgb_r.fit(X_train, y_train, group=group_train)
        models["xgb_ranker"] = xgb_r
        joblib.dump(xgb_r, output_dir / "xgb_ranker.joblib")
    except Exception as e:
        models["xgb_ranker_error"] = str(e)
        if "lambdamart" not in models or isinstance(models.get("lambdamart"), str):
            models["xgb_ranker"] = _sklearn_ranker_fallback()

    joblib.dump({"scaler": scaler, "features": feat_cols}, output_dir / "ranking_preprocess.joblib")

    primary_model = models.get(primary) or models.get("xgb_ranker") or models.get("lambdamart")
    if primary_model is None:
        raise RuntimeError("No ranking model trained successfully")

    scores = primary_model.predict(X_scaled)
    df_out = df.copy()
    df_out["ranking_score"] = scores
    df_out = df_out.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    df_out.to_parquet(output_dir / "ranking_scores.parquet", index=False)

    manifest = {
        "primary": primary,
        "features": feat_cols,
        "output": str(output_dir / "ranking_scores.parquet"),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_top_n(
    df: pd.DataFrame,
    n: int = 500,
    score_col: str = "ranking_score",
) -> pd.DataFrame:
    if score_col not in df.columns:
        df = df.sort_values("predicted_lift_score", ascending=False)
    else:
        df = df.sort_values(score_col, ascending=False)
    top = df.head(n).copy()
    top["promotion_rank"] = range(1, len(top) + 1)
    return top
