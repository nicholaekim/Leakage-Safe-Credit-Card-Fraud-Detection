"""Paired bootstrap comparison of models sharing one test set.

Why PAIRED: all models are scored on the same test rows, so we resample the
rows once per bootstrap draw and evaluate every model on that identical
resample. Between-sample luck then cancels inside each draw, and the
confidence interval lands on the model DIFFERENCE — much tighter, and the
honest object of interest.

Why it matters here: the test set holds only ~95 frauds. Single-number
PR-AUC differences of a few hundredths — the kind every leaderboard crowns a
winner with — are routinely inside the paired 95% interval.

Cost trick: metrics are computed once per (model, draw) and cached, so a
tournament of K pairwise comparisons costs O(models x draws), not O(K x draws).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from . import evaluate


def bootstrap_indices(n, n_boot, seed):
    """Shared row resamples: one (n_boot, n) integer matrix per test set.
    int32 halves the footprint (~230 MB at B=1000, n=57k)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int32)


def boot_pr_auc(y, scores, idx):
    """PR-AUC (average precision) per bootstrap draw."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    return np.array([average_precision_score(y[ix], scores[ix]) for ix in idx])


def boot_savings(y, yhat, amounts, idx):
    """Savings per bootstrap draw for FIXED decisions (the model's tuned
    threshold is part of the artifact, so decisions resample with the rows)."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    amounts = np.asarray(amounts)
    return np.array([evaluate.savings(y[ix], yhat[ix], amounts[ix])
                     for ix in idx])


def paired_delta(boot_a, boot_b, alpha=0.05):
    """CI on (A - B) from paired per-draw metrics. Returns dict with the mean
    delta, the (alpha/2, 1-alpha/2) percentile interval, and the fraction of
    draws where A beats B."""
    d = np.asarray(boot_a) - np.asarray(boot_b)
    d = d[~np.isnan(d)]                    # savings is NaN on fraud-free draws
    if d.size == 0:
        raise ValueError("every bootstrap draw was undefined for this pair")
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_a_beats_b": float((d > 0).mean())}


def verdict(cis):
    """Cross-seed call for one comparison: 'real' if every seed's CI excludes
    zero with a consistent sign; 'suggestive' if most do; else
    'indistinguishable'. A single seed cannot earn 'real' — there is no
    cross-seed guard to pass."""
    if not cis:
        raise ValueError("verdict() needs at least one CI")
    if len(cis) == 1:
        c = cis[0]
        return ("unconfirmed (1 seed)" if c["lo"] > 0 or c["hi"] < 0
                else "indistinguishable")
    signs = []
    for c in cis:
        if c["lo"] > 0:
            signs.append(1)
        elif c["hi"] < 0:
            signs.append(-1)
        else:
            signs.append(0)
    n = len(signs)
    pos, neg = signs.count(1), signs.count(-1)
    if pos == n or neg == n:
        return "real"
    # 'suggestive' needs a real majority pointing one way — a lone
    # zero-straddling CI (or all-straddling CIs) is just indistinguishable.
    if (pos >= n - 1 and neg == 0 and pos > 0) or \
       (neg >= n - 1 and pos == 0 and neg > 0):
        return "suggestive"
    return "indistinguishable"
