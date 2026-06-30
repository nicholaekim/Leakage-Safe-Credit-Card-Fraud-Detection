"""Imbalance-appropriate metrics, cost-sensitive thresholding, aggregation.

Why not accuracy? At a 0.17% fraud rate, predicting "never fraud" is 99.83%
accurate and useless. The primary metric here is PR-AUC (average precision);
ROC-AUC is reported but is optimistic under extreme imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from . import config


def precision_recall_at_k(y_true, scores, k):
    """Operational view: if analysts can review the top-k flagged
    transactions, how many frauds do we catch? Returns (precision@k,
    recall@k)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(y_true[idx].sum())
    return tp / max(k, 1), tp / max(int(y_true.sum()), 1)


def expected_cost(y_true, scores, threshold, amounts=None, c_fp=None, c_fn=None):
    """Total cost at a threshold. With config.AMOUNT_AWARE, a missed fraud
    costs that transaction's Amount; otherwise a flat c_fn."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    c_fp = config.COST_FP if c_fp is None else c_fp
    c_fn = config.COST_FN if c_fn is None else c_fn
    yhat = (scores >= threshold).astype(int)
    fp = (yhat == 1) & (y_true == 0)
    fn = (yhat == 0) & (y_true == 1)
    if amounts is not None and config.AMOUNT_AWARE:
        return float(c_fp * fp.sum() + np.asarray(amounts)[fn].sum())
    return float(c_fp * fp.sum() + c_fn * fn.sum())


def pick_threshold(y_val, scores_val, objective="cost", amounts=None, grid=None):
    """Choose the decision threshold on VALIDATION data — never on test.
    objective: 'cost' (minimise expected_cost) or 'f1' (maximise F1)."""
    scores_val = np.asarray(scores_val)
    grid = np.unique(scores_val) if grid is None else np.asarray(grid)
    if len(grid) > 2000:                       # cap for speed
        grid = np.quantile(scores_val, np.linspace(0, 1, 2000))
    best_t, best_obj = 0.5, None
    for t in grid:
        if objective == "cost":
            val = expected_cost(y_val, scores_val, t, amounts)
            better = best_obj is None or val < best_obj
        elif objective == "f1":
            val = f1_score(y_val, (scores_val >= t).astype(int), zero_division=0)
            better = best_obj is None or val > best_obj
        else:
            raise ValueError(f"Unknown objective: {objective!r}")
        if better:
            best_obj, best_t = val, float(t)
    return best_t


def evaluate_predictions(y_true, scores, threshold, amounts=None, k=100) -> dict:
    """Full metric row for one fitted model on one split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    yhat = (scores >= threshold).astype(int)
    p_at_k, r_at_k = precision_recall_at_k(y_true, scores, k)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        f"precision@{k}": float(p_at_k),
        f"recall@{k}": float(r_at_k),
        "cost": expected_cost(y_true, scores, threshold, amounts),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def aggregate(records):
    """Mean ± std across seeds, grouped by (mode, strategy, model).
    Returns (raw_df, grouped_df)."""
    import pandas as pd
    df = pd.DataFrame(records)
    keys = ["mode", "strategy", "model"]
    num = list(df.select_dtypes("number").columns)
    grouped = df.groupby(keys)[num].agg(["mean", "std"])
    return df, grouped
