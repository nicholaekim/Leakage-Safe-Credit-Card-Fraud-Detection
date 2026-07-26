"""probability calibration + reliability measurements.

a score of 0.9 should mean roughly 90% of those transactions are fraud. raw
model scores usually dont satisfy that, and resampling makes it worse by
shifting the base rate. we measure it (brier, ece, reliability curve) and fix
it with a post hoc map fit on validation only.

two methods: 'sigmoid' (platt, logistic map on log odds) and 'isotonic'
(monotone step function, more flexible but needs more data than ~470 frauds
really gives it).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expected_calibration_error(y_true, prob, n_bins=None, adaptive=False) -> float:
    """ece = weighted mean gap between predicted prob and observed rate.

    adaptive=True uses quantile bins instead of equal width ones. at 0.17%
    prevalence over 99% of scores land in [0, 0.1) so uniform bins barely see
    the top of the ranking where the decisions actually happen."""
    n_bins = config.N_CALIB_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    if adaptive:
        edges = np.unique(np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:  # near constant scores, fall back to uniform
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, len(edges) - 2)
    n = len(prob)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def reliability_curve(y_true, prob, n_bins=None):
    """returns (confidence, observed frequency, count) per non empty bin"""
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
    """post hoc calibrator fit on held out (score, label) pairs.

    cal = PrefitCalibrator("isotonic").fit(scores_val, y_val)
    p_test = cal.predict(scores_test)

    hand rolled instead of CalibratedClassifierCV so the behaviour is easy to
    explain and doesnt depend on sklearn version quirks.
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
            # proper platt fits the logistic map on log odds, basically
            # unregularized (same as sklearns internal sigmoid calibration).
            # fitting on raw p in [0,1] would leave the map nearly linear and
            # under correct right where all the scores actually live
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(
                _logit(scores).reshape(-1, 1), y)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        return self

    def predict(self, scores):
        scores = np.asarray(scores).reshape(-1)
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(scores)
        return self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
