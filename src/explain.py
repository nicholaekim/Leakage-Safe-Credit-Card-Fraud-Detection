"""Model interpretation: permutation importance + SHAP.

Global view  : which features drive the model? (permutation importance is
               more honest than impurity-based importance, which is biased
               toward high-cardinality features.)
Local view   : why was THIS transaction flagged / missed? (SHAP per-sample.)

Limitation to discuss in the report: V1..V28 are anonymised PCA components, so
SHAP tells you *which component* matters, not a human-readable feature story.
This is a genuine interpretability ceiling of the ULB dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importances(fitted_pipe, X, y, scoring="average_precision",
                            n_repeats=10, seed=0) -> pd.DataFrame:
    """Permute each feature in the (raw) test frame and measure the drop in
    PR-AUC. Pass the leakage-safe pipeline as `fitted_pipe`."""
    r = permutation_importance(
        fitted_pipe, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=seed, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_shap(fitted_pipe, X, max_samples=2000, seed=0):
    """SHAP values for the tree model inside a fitted leakage-safe pipeline.

    Returns (shap_values, X_sample, feature_names). We apply the *fitted*
    scaler first so SHAP sees the same feature space the model was trained on.
    Requires `shap` and a tree-based final estimator.
    """
    import shap

    clf = fitted_pipe.named_steps["clf"]
    Xs = X.sample(max_samples, random_state=seed) if len(X) > max_samples else X
    if "scale" in fitted_pipe.named_steps:
        Xt = fitted_pipe.named_steps["scale"].transform(Xs)
    else:
        Xt = Xs.to_numpy()

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):       # some versions return [neg, pos]
        sv = sv[1]
    return sv, Xs, list(X.columns)
