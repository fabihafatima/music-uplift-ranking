from __future__ import annotations

import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def cumulative_uplift_curve(tau_pred, treatment, outcome):
    order = np.argsort(-tau_pred)
    t_s = treatment[order]
    y_s = outcome[order]
    n = len(tau_pred)
    fracs = np.arange(1, n + 1) / n
    curve = []
    for k in range(1, n + 1):
        sub_t, sub_y = t_s[:k], y_s[:k]
        tr, ct = sub_t == 1, sub_t == 0
        if tr.sum() == 0 or ct.sum() == 0:
            curve.append(0.0)
        else:
            curve.append((sub_y[tr].mean() - sub_y[ct].mean()) * k)
    return fracs, np.array(curve)


def qini_coefficient(tau_pred, treatment, outcome) -> float:
    frac, curve = cumulative_uplift_curve(tau_pred, treatment, outcome)
    random_curve = curve[-1] * frac if len(curve) else np.zeros_like(frac)
    return _trapz(curve - random_curve, frac)


def auuc(tau_pred, treatment, outcome) -> float:
    return qini_coefficient(tau_pred, treatment, outcome)


def qini_curve_points(tau_pred, treatment, outcome):
    return cumulative_uplift_curve(tau_pred, treatment, outcome)
