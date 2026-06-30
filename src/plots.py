"""Minimal matplotlib helpers. Notebooks call these and compose figures; each
returns the Axes so you can overlay multiple models."""
from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from . import config, calibrate


def plot_pr_curve(y_true, scores, ax=None, label=None):
    ax = ax or plt.gca()
    p, r, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax.plot(r, p, label=f"{label or 'model'} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(loc="upper right")
    return ax


def plot_reliability(y_true, prob, n_bins=10, ax=None, label=None):
    ax = ax or plt.gca()
    conf, acc, _ = calibrate.reliability_curve(y_true, prob, n_bins)
    rep = calibrate.calibration_report(y_true, prob, n_bins)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(conf, acc, "o-", label=label or "model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability  (ECE={rep['ece']:.3f}, Brier={rep['brier']:.4f})")
    ax.legend(loc="upper left")
    return ax


def savefig(fig, name):
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
