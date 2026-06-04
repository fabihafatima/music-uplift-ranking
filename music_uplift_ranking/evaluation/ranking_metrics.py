from __future__ import annotations

import numpy as np


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k]
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2**rel - 1) / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    rel_sorted = y_true[order]
    dcg = dcg_at_k(rel_sorted, k)
    ideal_order = np.argsort(-y_true)
    idcg = dcg_at_k(y_true[ideal_order], k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    hits = y_sorted == 1
    if hits.sum() == 0:
        return 0.0
    precisions = []
    hit_count = 0
    for i, rel in enumerate(y_sorted, start=1):
        if rel == 1:
            hit_count += 1
            precisions.append(hit_count / i)
    return float(np.mean(precisions))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    top = y_true[order[:k]]
    return float(top.sum() / k)
