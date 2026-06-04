from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from music_uplift_ranking.evaluation.business_metrics import (
    incremental_saves,
    incremental_streams,
    promotion_efficiency,
)
from music_uplift_ranking.evaluation.plots import (
    plot_baseline_qini_comparison,
    plot_business_metrics,
    plot_feature_importance,
    plot_ite_distribution,
    plot_lift_curve,
    plot_qini_curve,
    plot_ranking_comparison,
    plot_shap_summary,
)
from music_uplift_ranking.evaluation.ranking_metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
)
from music_uplift_ranking.evaluation.uplift_metrics import auuc, qini_coefficient


def run_full_evaluation(
    campaigns: pd.DataFrame,
    ite_preds: pd.DataFrame,
    ranking_scores: pd.DataFrame,
    uplift_model_path: Path,
    output_dir: Path,
    plots_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = campaigns.merge(ite_preds, on="track_id", how="inner")
    df = df.merge(
        ranking_scores[["track_id", "ranking_score"]],
        on="track_id",
        how="left",
    )
    df["ranking_score"] = df["ranking_score"].fillna(df["predicted_lift_score"])

    t = df["T"].values
    y = df["streams"].values
    tau_pred = df["predicted_lift_score"].values
    tau_true = df["true_ite_streams"].values if "true_ite_streams" in df.columns else tau_pred

    metrics: dict = {
        "uplift": {
            "qini_coefficient": qini_coefficient(tau_pred, t, y),
            "auuc": auuc(tau_pred, t, y),
            "ite_mae": float(mean_absolute_error(tau_true, tau_pred)),
            "ite_r2": float(r2_score(tau_true, tau_pred)),
        },
        "baselines": {},
        "ranking": {},
        "business": {},
    }

    # Baselines
    for name, score in [
        ("historical_streams", df["historical_streams"].values),
        ("engagement_index", df["engagement_index"].values),
        ("engagement_prediction", df["engagement_index"].values * 0.9 + df["streams"].values * 0.1),
    ]:
        metrics["baselines"][name] = {
            "qini": qini_coefficient(score, t, y),
            "ndcg@10": ndcg_at_k((tau_true > np.percentile(tau_true, 75)).astype(int), score, 10),
        }

    relevance = (tau_true > np.percentile(tau_true, 75)).astype(int)
    for name, score in [
        ("uplift_model", tau_pred),
        ("ranking_model", df["ranking_score"].values),
        ("historical_streams", df["historical_streams"].values),
        ("engagement_prediction", df["engagement_index"].values),
    ]:
        metrics["ranking"][name] = {
            "ndcg@10": ndcg_at_k(relevance, score, 10),
            "map": average_precision(relevance, score),
            "precision@10": precision_at_k(relevance, score, 10),
        }

    top_k = 500
    order_uplift = np.argsort(-tau_pred)[:top_k]
    top_mask = np.zeros(len(df), dtype=bool)
    top_mask[order_uplift] = True
    budget = df.get("promotion_budget_usd", pd.Series(0, index=df.index)).fillna(0).values

    metrics["business"] = {
        "incremental_streams_top500": incremental_streams(top_mask, tau_true),
        "incremental_saves_top500": incremental_saves(top_mask, tau_true),
        "promotion_efficiency": promotion_efficiency(
            incremental_streams(top_mask, tau_true), budget, top_mask
        ),
    }

    plot_qini_curve(tau_pred, t, y, plots_dir / "qini_curve.png")
    plot_lift_curve(tau_pred, tau_true, plots_dir / "lift_curve.png")
    plot_ranking_comparison(metrics, plots_dir / "ranking_comparison.png")
    plot_baseline_qini_comparison(metrics, plots_dir / "baseline_qini.png")
    plot_business_metrics(metrics, plots_dir / "business_metrics.png")
    plot_ite_distribution(tau_pred, tau_true, plots_dir / "ite_distribution.png")

    # Feature importance from primary model
    artifact = joblib.load(uplift_model_path)
    model = artifact["model"]
    imp_path = plots_dir / "feature_importance.png"
    try:
        tlearner = model.t_learner
        imp = tlearner.model_1.feature_importances_
        pipe = artifact["pipeline"]
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        imp_dict = dict(zip(names[: len(imp)], imp))
        plot_feature_importance(imp_dict, imp_path)
        metrics["feature_importance_top5"] = dict(
            sorted(imp_dict.items(), key=lambda x: -x[1])[:5]
        )
    except Exception:
        pass

    # SHAP on sample
    try:
        import shap

        from music_uplift_ranking.uplift_models.preprocessing import CATEGORICAL, NUMERIC, prepare_features

        eval_df = df[[c for c in CATEGORICAL + NUMERIC if c in df.columns]].head(2000)
        X, _ = prepare_features(eval_df, fit=False, pipeline=artifact["pipeline"])
        explainer = shap.TreeExplainer(tlearner.model_1)
        sv = explainer.shap_values(X[:500])
        names = list(artifact["pipeline"].named_steps["prep"].get_feature_names_out())
        plot_shap_summary(sv, names, plots_dir / "shap_summary.png")
        metrics["shap"] = "generated"
    except Exception as e:
        metrics["shap"] = f"skipped: {e}"

    with (output_dir / "evaluation_report.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
