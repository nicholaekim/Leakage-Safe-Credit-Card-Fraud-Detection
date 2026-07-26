"""the model zoo, baselines through boosted trees.

xgboost and lightgbm are optional. if they fail to load (eg libomp missing on
mac) they just get skipped so the rest still runs. brew install libomp fixes it.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,  # class_weight requires scikit-learn>=1.3
)

# remember what we already warned about so it prints once, not every loop
_UNAVAILABLE_WARNED = set()


def _note_unavailable(name, exc):
    if name not in _UNAVAILABLE_WARNED:
        _UNAVAILABLE_WARNED.add(name)
        print(f"[skip] {name} could not be loaded ({type(exc).__name__}); "
              f"continuing without it. macOS fix: `brew install libomp`.")


def pos_weight(y) -> float:
    """neg/pos ratio, used as scale_pos_weight for xgboost"""
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def get_models(seed, use_class_weight=True, spw=None) -> dict:
    """returns {name: estimator}.

    use_class_weight should be off when a resampler (smote/undersample) is
    used, otherwise the minority class gets counted twice. spw is
    scale_pos_weight for xgboost, compute it from the train y.
    """
    cw = "balanced" if use_class_weight else None
    models = {
        # linear baseline the ensembles have to beat
        "logreg": LogisticRegression(max_iter=2000, class_weight=cw),
        "random_forest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=seed,
            class_weight=("balanced_subsample" if use_class_weight else None),
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            random_state=seed, class_weight=cw,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="aucpr", random_state=seed, n_jobs=-1,
            scale_pos_weight=(spw if (use_class_weight and spw) else 1.0),
        )
    except Exception as exc:  # missing native libs raise more than ImportError
        _note_unavailable("xgboost", exc)
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1,
            class_weight=("balanced" if use_class_weight else None),
            verbose=-1,
        )
    except Exception as exc:  # same deal as xgboost
        _note_unavailable("lightgbm", exc)
    return models
