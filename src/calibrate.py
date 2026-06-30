"""Probability calibration and reliability diagnostics.

A fraud score of 0.9 should mean "~90% of such transactions are fraud". Raw
tree/ensemble scores rarely satisfy that, and SMOTE makes it worse by
distorting the base rate. We measure calibration (Brier, ECE, reliability
curve) and fix it with a post-hoc map fit on a HELD-OUT calibration set.

The two methods:
  * 'sigmoid'  (Platt) — fit a logistic map score -> probability.
  * 'isotonic'         — fit a monotonic step function (more flexible, needs
                         more data; we have ~470 frauds so use with care).

NB: the dataset's own authors (Dal Pozzolo et al., 2015) studied calibration
under undersampling — a natural citation and a reproducible result here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config


def expected_calibration_error(y_true, prob, n_bins=None) -> float:
    """ECE = weighted mean gap between confidence and observed frequency."""
    n_bins = config.N_CALIB_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    n = len(prob)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def reliability_curve(y_true, prob, n_bins=None):
    """Returns (confidence, observed_frequency, count) per non-empty bin."""
    n_bins = config.N_CALIB_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf.append(prob[m].mean())
            acc.append(y_true[m].mean())
            count.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(count)


def brier(y_true, prob) -> float:
    return float(brier_score_loss(y_true, prob))


def calibration_report(y_true, prob, n_bins=None) -> dict:
    return {
        "brier": brier(y_true, prob),
        "ece": expected_calibration_error(y_true, prob, n_bins),
    }


class PrefitCalibrator:
    """Post-hoc calibrator fit on held-out (uncalibrated score, label) pairs.

    Usage:
        cal = PrefitCalibrator("isotonic").fit(scores_val, y_val)
        p_test = cal.predict(scores_test)

    Version-stable (no dependence on CalibratedClassifierCV's changing API)
    and transparent enough to explain in the report.
    """

    def __init__(self, method="isotonic"):
        self.method = method
        self.model = None

    def fit(self, scores, y):
        scores = np.asarray(scores).reshape(-1)
        y = np.asarray(y)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
        elif self.method in ("sigmoid", "platt"):
            self.model = LogisticRegression().fit(scores.reshape(-1, 1), y)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        return self

    def predict(self, scores):
        scores = np.asarray(scores).reshape(-1)
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(scores)
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
