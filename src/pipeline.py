"""leakage safe preprocessing + resampling pipelines.

main rule: anything that learns from data (scaler, smote) has to be fit on
the training fold only. wrapping them in an imblearn pipeline enforces that.
the sampler only runs during fit, so val/test always keep their real
imbalanced distribution.
"""
from __future__ import annotations

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_resampler(name, seed):
    """name is one of none / class_weight / smote / undersample.
    none and class_weight return no resampler (class_weight is a model arg)."""
    name = (name or "none").lower()
    if name in ("none", "class_weight"):
        return None
    if name == "smote":
        return SMOTE(random_state=seed)
    if name == "undersample":
        return RandomUnderSampler(random_state=seed)
    raise ValueError(f"Unknown resampler strategy: {name!r}")


def make_safe_pipeline(clf, resampler=None, scale=True):
    """scaler -> optional resampler -> clf, as one estimator.

    use this everywhere models get fit. since its a single estimator, cv,
    calibration and permutation importance all stay leakage safe for free.
    """
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)
