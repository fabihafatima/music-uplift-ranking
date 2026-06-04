from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from music_uplift_ranking.evaluation.uplift_metrics import qini_curve_points


def plot_qini_curve(tau_pred, treatment, outcome, path: Path, title: str = "Qini Curve"):
    frac, curve = qini_curve_points(tau_pred, treatment, outcome)
    random_curve = curve[-1] * frac if len(curve) else np.zeros_like(frac)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frac, curve, label="Model", linewidth=2)
    ax.plot(frac, random_curve, "--", label="Random", linewidth=1.5)
    ax.set_xlabel("Fraction targeted")
    ax.set_ylabel("Cumulative uplift")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lift_curve(tau_pred, tau_true, path: Path):
    order = np.argsort(-tau_pred)
    cum_true = np.cumsum(tau_true[order]) / max(len(tau_true), 1)
    fracs = np.arange(1, len(cum_true) + 1) / len(cum_true)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, cum_true, linewidth=2)
    ax.set_xlabel("Fraction targeted (by predicted ITE)")
    ax.set_ylabel("Cumulative true incremental streams")
    ax.set_title("Lift Curve — True ITE Captured")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: dict[str, float], path: Path, top_k: int = 20):
    items = sorted(importances.items(), key=lambda x: -abs(x[1]))[:top_k]
    names = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, vals)
    ax.set_title("Feature Importance (X-Learner)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ranking_comparison(metrics: dict, path: Path, k: int = 10):
    """Bar chart: nDCG@k, MAP, Precision@k across ranking approaches."""
    ranking = metrics.get("ranking", {})
    labels = []
    ndcg, map_scores, prec = [], [], []
    display_names = {
        "uplift_model": "Uplift (ITE)",
        "ranking_model": "LambdaMART / Ranker",
        "historical_streams": "Historical streams",
        "engagement_prediction": "Engagement prediction",
    }
    for key, label in display_names.items():
        if key not in ranking:
            continue
        labels.append(label)
        ndcg.append(ranking[key].get(f"ndcg@{k}", ranking[key].get("ndcg@10", 0)))
        map_scores.append(ranking[key].get("map", 0))
        prec.append(ranking[key].get(f"precision@{k}", ranking[key].get("precision@10", 0)))

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, ndcg, width, label=f"nDCG@{k}")
    ax.bar(x, map_scores, width, label="MAP")
    ax.bar(x + width, prec, width, label=f"Precision@{k}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ranking quality by approach")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_baseline_qini_comparison(metrics: dict, path: Path):
    """Compare Qini coefficient: uplift vs baselines."""
    uplift_qini = metrics.get("uplift", {}).get("qini_coefficient", 0)
    baselines = metrics.get("baselines", {})
    names = ["X-Learner (ITE)"] + [
        {"historical_streams": "Historical streams", "engagement_index": "Engagement index",
         "engagement_prediction": "Engagement pred."}.get(k, k)
        for k in baselines
    ]
    values = [uplift_qini] + [baselines[k].get("qini", 0) for k in baselines]
    colors = ["#34d399"] + ["#94a3b8"] * len(baselines)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Qini coefficient")
    ax.set_title("Uplift targeting vs baselines")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_business_metrics(metrics: dict, path: Path):
    biz = metrics.get("business", {})
    labels = ["Incremental\nstreams (top 500)", "Incremental\nsaves (top 500)"]
    values = [
        biz.get("incremental_streams_top500", 0) / 1000,
        biz.get("incremental_saves_top500", 0) / 1000,
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#6366f1", "#a78bfa"])
    ax.set_ylabel("Thousands")
    ax.set_title("Business impact — top 500 promoted tracks")
    eff = biz.get("promotion_efficiency", 0)
    ax.text(0.5, 0.95, f"Promotion efficiency: {eff:.4f} streams/$", transform=ax.transAxes, ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ite_distribution(tau_pred: np.ndarray, tau_true: np.ndarray | None, path: Path):
    fig, axes = plt.subplots(1, 2 if tau_true is not None else 1, figsize=(12 if tau_true is not None else 6, 5))
    if tau_true is None:
        axes = [axes]
    axes[0].hist(tau_pred, bins=50, color="#6366f1", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Predicted ITE (streams)")
    axes[0].set_title("Predicted lift distribution")
    if tau_true is not None:
        axes[1].scatter(tau_true, tau_pred, alpha=0.15, s=8, c="#34d399")
        lim = max(tau_true.max(), tau_pred.max()) * 1.05
        axes[1].plot([0, lim], [0, lim], "--", color="#94a3b8", label="Perfect")
        axes[1].set_xlabel("True ITE")
        axes[1].set_ylabel("Predicted ITE")
        axes[1].set_title("ITE calibration")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary(shap_values, feature_names, path: Path):
    try:
        import shap
    except ImportError:
        return
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
