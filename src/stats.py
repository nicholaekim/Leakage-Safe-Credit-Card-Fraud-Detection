"""paired bootstrap comparison of models that share one test set.

paired means: resample the rows once per draw and score every model on that
same resample. sample luck cancels inside each draw so the interval lands on
the model difference, which is what we actually care about.

matters here because the test set only has ~95 frauds, so pr-auc differences
of a few hundredths are usually inside the interval.

metrics are computed once per (model, draw) and reused, so k pairwise
comparisons cost o(models x draws) not o(k x draws).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from . import evaluate


def bootstrap_indices(n, n_boot, seed):
    """shared row resamples, one (n_boot, n) int matrix per test set.
    int32 halves the memory (~230mb at b=1000, n=57k)"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int32)


def boot_pr_auc(y, scores, idx):
    """pr-auc per bootstrap draw"""
    y = np.asarray(y)
    scores = np.asarray(scores)
    return np.array([average_precision_score(y[ix], scores[ix]) for ix in idx])


def boot_savings(y, yhat, amounts, idx):
    """savings per draw for fixed decisions. the tuned threshold is part of
    the model artifact so the decisions just resample with the rows"""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    amounts = np.asarray(amounts)
    return np.array([evaluate.savings(y[ix], yhat[ix], amounts[ix])
                     for ix in idx])


def paired_delta(boot_a, boot_b, alpha=0.05):
    """ci on (a - b) from the paired per draw metrics. returns the mean
    delta, the percentile interval and how often a beats b."""
    d = np.asarray(boot_a) - np.asarray(boot_b)
    d = d[~np.isnan(d)]  # savings can be nan on fraud free draws
    if d.size == 0:
        raise ValueError("every bootstrap draw was undefined for this pair")
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_a_beats_b": float((d > 0).mean())}


def verdict(cis):
    """cross seed call for one comparison. 'real' only if every seeds ci
    excludes zero with the same sign, 'suggestive' if a strict majority do
    (so 1-of-2 never qualifies), otherwise 'indistinguishable'. one seed
    alone can never earn 'real'."""
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
    # suggestive needs an actual strict majority pointing one way (pos*2 > n,
    # so 1-of-2 doesn't qualify) with at most one straddling ci and none
    # opposed. a lone straddling ci (or all straddling) is indistinguishable
    if (pos >= n - 1 and neg == 0 and pos * 2 > n) or \
       (neg >= n - 1 and pos == 0 and neg * 2 > n):
        return "suggestive"
    return "indistinguishable"
