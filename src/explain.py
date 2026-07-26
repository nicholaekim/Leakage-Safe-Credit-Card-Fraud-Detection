"""model interpretation, permutation importance + shap.

global: which features drive the model (permutation importance, since
impurity importance is biased). local: why was this one transaction flagged
or missed (shap per row).

caveat for the report: v1..v28 are anonymised pca components so this tells
you which component matters, not a human readable story.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importances(fitted_pipe, X, y, scoring="average_precision",
                            n_repeats=10, seed=0) -> pd.DataFrame:
    """shuffle each feature and measure the pr-auc drop. pass the fitted
    pipeline so scaling stays inside"""
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
    """shap values for the tree model inside a fitted pipeline.

    returns (shap_values, X_sample, feature_names). applies the fitted scaler
    first so shap sees the same feature space the model trained on. needs the
    shap package and a tree based final estimator.
    """
    import shap

    clf = fitted_pipe.named_steps["clf"]
    Xs = X.sample(max_samples, random_state=seed) if len(X) > max_samples else X
    if "scale" in fitted_pipe.named_steps:
        Xt = fitted_pipe.named_steps["scale"].transform(Xs)
    else:
        Xt = Xs.to_numpy()

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # newer shap returns (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xs, list(X.columns)
