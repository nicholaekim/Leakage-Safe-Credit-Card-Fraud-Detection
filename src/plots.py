"""small matplotlib helpers. each returns the axes so you can overlay
multiple models on one plot"""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import config, calibrate


def plot_reliability(y_true, prob, n_bins=10, ax=None, label=None):
    ax = ax or plt.gca()
    conf, acc, _ = calibrate.reliability_curve(y_true, prob, n_bins)
    rep = calibrate.calibration_report(y_true, prob, n_bins)
    if not ax.lines:  # only draw the diagonal once when overlaying
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
