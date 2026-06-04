"""
Uplift model evaluation metrics.

Uses true tau when available (simulation); otherwise reports ranking metrics
on held-out data with observed treatment randomization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from music_uplift.models.base import UpliftModel


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _cumulative_uplift_curve(
    tau_pred: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by predicted uplift descending; compute cumulative incremental outcome."""
    order = np.argsort(-tau_pred)
    t_sorted = treatment[order]
    y_sorted = outcome[order]
    n = len(tau_pred)
    fractions = np.arange(1, n + 1) / n
    cum_uplift = []
    for k in range(1, n + 1):
        sub_t = t_sorted[:k]
        sub_y = y_sorted[:k]
        treated = sub_t == 1
        control = sub_t == 0
        if treated.sum() == 0 or control.sum() == 0:
            cum_uplift.append(0.0)
            continue
        uplift = sub_y[treated].mean() - sub_y[control].mean()
        cum_uplift.append(uplift * k)
    return fractions, np.array(cum_uplift)


def qini_coefficient(
    tau_pred: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """
    Qini coefficient: area between model curve and random targeting curve.
    Higher is better for ranking high-uplift units.
    """
    frac, curve = _cumulative_uplift_curve(tau_pred, treatment, outcome)
    random_curve = curve[-1] * frac if len(curve) else np.zeros_like(frac)
    return _trapz(curve - random_curve, frac)


def auuc_score(
    tau_pred: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """Area Under the Uplift Curve — normalized Qini-style metric."""
    frac, curve = _cumulative_uplift_curve(tau_pred, treatment, outcome)
    if len(curve) == 0:
        return 0.0
    max_area = _trapz(curve, frac)
    random_baseline = curve[-1] * 0.5 if len(curve) else 0.0
    denom = max_area - random_baseline * len(frac) / len(frac) if max_area != 0 else 1.0
    qini = qini_coefficient(tau_pred, treatment, outcome)
    return float(qini / (abs(denom) + 1e-8))


def uplift_at_k(
    tau_pred: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    k: float = 0.1,
) -> float:
    """Observed uplift in top k fraction when targeted by predicted tau."""
    n = len(tau_pred)
    top_n = max(1, int(n * k))
    order = np.argsort(-tau_pred)[:top_n]
    t = treatment[order]
    y = outcome[order]
    treated = t == 1
    control = t == 0
    if treated.sum() == 0 or control.sum() == 0:
        return 0.0
    return float(y[treated].mean() - y[control].mean())


def evaluate_uplift_model(
    model: UpliftModel,
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    tau_true: np.ndarray | None = None,
) -> dict[str, Any]:
    preds = model.predict_all(X)
    tau_pred = preds["tau"]
    metrics: dict[str, Any] = {
        "qini": qini_coefficient(tau_pred, treatment, y),
        "auuc": auuc_score(tau_pred, treatment, y),
        "uplift_at_10pct": uplift_at_k(tau_pred, treatment, y, k=0.1),
        "uplift_at_20pct": uplift_at_k(tau_pred, treatment, y, k=0.2),
        "mean_predicted_lift": float(tau_pred.mean()),
    }
    if tau_true is not None:
        metrics["tau_mae"] = float(mean_absolute_error(tau_true, tau_pred))
        metrics["tau_rmse"] = float(np.sqrt(mean_squared_error(tau_true, tau_pred)))
        metrics["tau_r2"] = float(r2_score(tau_true, tau_pred))
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return path
