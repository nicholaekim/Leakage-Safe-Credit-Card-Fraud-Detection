"""Leakage-safe preprocessing + resampling pipelines.

THE core rule of this project: anything that *learns* from data — scaling,
resampling — must be fit on the training fold ONLY. We enforce that
structurally by wrapping those steps in an imbalanced-learn Pipeline. The
sampler runs only during `fit`; at `predict` time it is bypassed, so the
validation/test sets are always scored on their real, imbalanced distribution.
"""
from __future__ import annotations

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_resampler(name, seed):
    """`name` in {'none', 'class_weight', 'smote', 'undersample'}.
    'none' and 'class_weight' use no resampler (class_weight is handled in the
    model itself)."""
    name = (name or "none").lower()
    if name in ("none", "class_weight"):
        return None
    if name == "smote":
        return SMOTE(random_state=seed)
    if name == "undersample":
        return RandomUnderSampler(random_state=seed)
    raise ValueError(f"Unknown resampler strategy: {name!r}")


def make_safe_pipeline(clf, resampler=None, scale=True):
    """Build the leakage-safe estimator: StandardScaler -> [resampler] -> clf.

    Use this everywhere models are fit. Because it is a single estimator,
    cross-validation, calibration, and permutation-importance all inherit the
    leakage-safe behaviour for free.
    """
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)
