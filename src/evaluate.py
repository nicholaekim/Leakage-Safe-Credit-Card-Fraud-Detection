"""metrics, money costs and threshold picking.

accuracy is useless at a 0.17% fraud rate (always-legit gets 99.83%), so
pr-auc is the main score. roc-auc gets reported too but its optimistic under
this much imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from . import config


def precision_recall_at_k(y_true, scores, k):
    """if analysts can only review the top k flagged transactions, how many
    frauds do we catch. returns (precision@k, recall@k)"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(y_true[idx].sum())
    return tp / max(k, 1), tp / max(int(y_true.sum()), 1)


def expected_cost(y_true, scores, threshold, amounts=None, c_fp=None, c_fn=None):
    """total cost at a threshold. with config.AMOUNT_AWARE a missed fraud
    costs its own amount, otherwise a flat c_fn"""
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
    """money cost of a set of decisions: every alert (tp or fp) costs the
    flat c_alert fee, every missed fraud costs its own amount, tn is free"""
    c_alert = config.C_ALERT if c_alert is None else c_alert
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    n_alerts = int((y_pred == 1).sum())
    missed = (y_pred == 0) & (y_true == 1)
    return float(c_alert * n_alerts + amounts[missed].sum())


def savings(y_true, y_pred, amounts, c_alert=None):
    """share of do-nothing fraud losses the model prevents, net of alert
    fees. 1 = perfect, 0 = useless, negative = worse than doing nothing.
    returns nan when a slice has no fraud value at risk (can happen on small
    temporal windows, plus the data has some zero amount frauds)"""
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    cost_base = float(amounts[y_true == 1].sum())   # all fraud succeeds
    if cost_base <= 0:
        return float("nan")
    return 1.0 - money_cost(y_true, y_pred, amounts, c_alert) / cost_base


def pick_threshold(y_val, scores_val, objective="cost", amounts=None, grid=None):
    """pick the decision threshold on validation, never on test.
    objective is 'cost' (minimise expected_cost), 'f1', or 'savings'
    (maximise money saved, needs amounts)."""
    scores_val = np.asarray(scores_val)
    if grid is None:
        grid = np.unique(scores_val)
        if len(grid) > 2000:  # cap the grid for speed
            base = np.quantile(scores_val, np.linspace(0, 1, 2000))
            # keep the exact top tail dense. at 0.17% prevalence all the
            # positives (and the money) sit above the ~99.5th percentile
            # where uniform quantiles would only leave a few points
            tail = grid[grid >= np.quantile(scores_val, 0.995)]
            grid = np.unique(np.concatenate([base, tail]))
    else:
        grid = np.asarray(grid)
    # sentinel above every score so "flag nothing" is a pickable option,
    # otherwise a hopeless model is forced to alert at a loss
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
        if np.isnan(val):  # degenerate slice, eg no fraud value at risk
            continue
        if best_obj is None or (val > best_obj if maximize else val < best_obj):
            best_obj, best_t = val, float(t)
    if best_obj is None:
        raise ValueError(
            f"objective {objective!r} undefined on every candidate threshold "
            "(degenerate validation slice?)")
    return best_t


def evaluate_predictions(y_true, scores, threshold, amounts=None, k=100) -> dict:
    """full metric row for one fitted model on one split"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    yhat = (scores >= threshold).astype(int)
    p_at_k, r_at_k = precision_recall_at_k(y_true, scores, k)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        # accuracy is only here to show how misleading it is at 0.17%
        # prevalence, dont select models with it
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
    """metric row for plain binary decisions. no scores/threshold needed,
    used for policies like the bayes rule (flag if p*amount > fee) that
    arent a single score cutoff."""
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
        # value_recall = fraction of fraud VALUE caught. worth having next to
        # normal recall because policies that skip tiny frauds (the bayes rule
        # never flags amount <= fee) look worse on count recall while losing
        # no money
        cost_base = float(np.asarray(amounts)[y_true == 1].sum())
        out["value_recall"] = (
            1.0 - out["fraud_value_missed"] / cost_base
            if cost_base > 0 else float("nan"))
    return out


def aggregate(records):
    """mean and std across seeds grouped by (mode, strategy, model).
    returns (raw_df, grouped_df)"""
    import pandas as pd
    df = pd.DataFrame(records)
    keys = ["mode", "strategy", "model"]
    num = list(df.select_dtypes("number").columns)
    grouped = df.groupby(keys)[num].agg(["mean", "std"])
    return df, grouped
