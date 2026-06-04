from __future__ import annotations

import numpy as np


def incremental_streams(top_mask: np.ndarray, tau_true: np.ndarray) -> float:
    return float(tau_true[top_mask].sum())


def incremental_saves(top_mask: np.ndarray, tau_true: np.ndarray, save_ratio: float = 0.08) -> float:
    return float(tau_true[top_mask].sum() * save_ratio)


def promotion_efficiency(
    incremental_value: float,
    budget: np.ndarray,
    top_mask: np.ndarray,
) -> float:
    total_budget = budget[top_mask].sum()
    return incremental_value / total_budget if total_budget > 0 else 0.0
