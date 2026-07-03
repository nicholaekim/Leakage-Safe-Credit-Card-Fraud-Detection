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


def money_cost(y_true, y_pred, amounts, c_alert=None):
    """Example-dependent money cost (Bahnsen et al.): every alert (TP or FP)
    costs a fixed investigation fee c_alert; every missed fraud (FN) costs
    that transaction's own Amount. TN costs nothing."""
    c_alert = config.C_ALERT if c_alert is None else c_alert
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    n_alerts = int((y_pred == 1).sum())
    missed = (y_pred == 0) & (y_true == 1)
    return float(c_alert * n_alerts + amounts[missed].sum())


def savings(y_true, y_pred, amounts, c_alert=None):
    """Fraction of do-nothing fraud losses the model prevents, net of alert
    costs. 1 = perfect, 0 = no better than doing nothing, negative = the
    model's alerts cost more than the fraud it stops.

    Undefined (NaN) on slices with no fraud value at risk — possible in small
    temporal windows, or because the dataset contains zero-Amount frauds."""
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    cost_base = float(amounts[y_true == 1].sum())   # all fraud succeeds
    if cost_base <= 0:
        return float("nan")
    return 1.0 - money_cost(y_true, y_pred, amounts, c_alert) / cost_base


def pick_threshold(y_val, scores_val, objective="cost", amounts=None, grid=None):
    """Choose the decision threshold on VALIDATION data — never on test.
    objective: 'cost' (minimise expected_cost), 'f1' (maximise F1), or
    'savings' (maximise money saved; requires amounts)."""
    scores_val = np.asarray(scores_val)
    if grid is None:
        grid = np.unique(scores_val)
        if len(grid) > 2000:                   # cap for speed, but keep the
            base = np.quantile(scores_val, np.linspace(0, 1, 2000))
            # exact top tail dense: at 0.17% prevalence the positives (and
            # hence the money) live entirely above the ~99.5th percentile,
            # where uniform quantiles would leave only a few grid points.
            tail = grid[grid >= np.quantile(scores_val, 0.995)]
            grid = np.unique(np.concatenate([base, tail]))
    else:
        grid = np.asarray(grid)
    # Sentinel above every score so "flag nothing" is a selectable policy —
    # without it a hopeless model is forced to alert at a loss.
    grid = np.append(grid, np.max(grid) + 1.0)
    if objective == "savings" and amounts is None:
        raise ValueError("objective='savings' requires amounts")
    if objective not in ("cost", "f1", "savings"):
        raise ValueError(f"Unknown objective: {objective!r}")
    maximize = objective in ("f1", "savings")

    best_t, best_obj = 0.5, None
    for t in grid:
        if objective == "cost":
            val = expected_cost(y_val, scores_val, t, amounts)
        elif objective == "f1":
            val = f1_score(y_val, (scores_val >= t).astype(int), zero_division=0)
        else:
            val = savings(y_val, (scores_val >= t).astype(int), amounts)
        if np.isnan(val):           # degenerate slice (e.g. no fraud value)
            continue
        if best_obj is None or (val > best_obj if maximize else val < best_obj):
            best_obj, best_t = val, float(t)
    if best_obj is None:
        raise ValueError(
            f"objective {objective!r} undefined on every candidate threshold "
            "(degenerate validation slice?)")
    return best_t


def evaluate_predictions(y_true, scores, threshold, amounts=None, k=100) -> dict:
    """Full metric row for one fitted model on one split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    yhat = (scores >= threshold).astype(int)
    p_at_k, r_at_k = precision_recall_at_k(y_true, scores, k)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        # accuracy is reported ONLY to demonstrate how misleading it is at
        # 0.17% prevalence — never use it for model selection here.
        "accuracy": float((yhat == np.asarray(y_true)).mean()),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        f"precision@{k}": float(p_at_k),
        f"recall@{k}": float(r_at_k),
        "cost": expected_cost(y_true, scores, threshold, amounts),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if amounts is not None:
        out["money_cost"] = money_cost(y_true, yhat, amounts)
        out["savings"] = savings(y_true, yhat, amounts)
        out["fraud_value_missed"] = float(
            np.asarray(amounts)[(yhat == 0) & (np.asarray(y_true) == 1)].sum())
    return out


def evaluate_decisions(y_true, y_pred, amounts=None):
    """Metric row for explicit binary decisions. Unlike evaluate_predictions
    this needs no scores/threshold — used for decision policies (e.g. Bayes
    minimum risk: flag iff p*Amount > C_ALERT) that are not a single score
    cutoff."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_alerts": int(y_pred.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if amounts is not None:
        out["money_cost"] = money_cost(y_true, y_pred, amounts)
        out["savings"] = savings(y_true, y_pred, amounts)
        out["fraud_value_missed"] = float(
            np.asarray(amounts)[(y_pred == 0) & (y_true == 1)].sum())
        # value_recall = fraction of fraud VALUE caught. Report it alongside
        # count-recall: policies that skip economically-null frauds (e.g. the
        # bayes rule never flags Amount<=C_ALERT) look worse on count-recall
        # while losing no money.
        cost_base = float(np.asarray(amounts)[y_true == 1].sum())
        out["value_recall"] = (
            1.0 - out["fraud_value_missed"] / cost_base
            if cost_base > 0 else float("nan"))
    return out


def aggregate(records):
    """Mean ± std across seeds, grouped by (mode, strategy, model).
    Returns (raw_df, grouped_df)."""
    import pandas as pd
    df = pd.DataFrame(records)
    keys = ["mode", "strategy", "model"]
    num = list(df.select_dtypes("number").columns)
    grouped = df.groupby(keys)[num].agg(["mean", "std"])
    return df, grouped
