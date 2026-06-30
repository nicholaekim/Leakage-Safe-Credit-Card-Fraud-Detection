"""Model zoo: baselines -> ensembles.

Optional heavy deps (xgboost, lightgbm) are imported lazily and skipped if they
cannot load (e.g. a missing native library such as libomp on macOS), so the
core pipeline always runs. To enable the full ensemble roster, install them and
-- on macOS -- run `brew install libomp`.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,  # class_weight requires scikit-learn>=1.3
)

# Track which optional models we have already warned about, so a skipped model
# prints one clear note instead of spamming the loop.
_UNAVAILABLE_WARNED = set()


def _note_unavailable(name, exc):
    if name not in _UNAVAILABLE_WARNED:
        _UNAVAILABLE_WARNED.add(name)
        print(f"[skip] {name} could not be loaded ({type(exc).__name__}); "
              f"continuing without it. macOS fix: `brew install libomp`.")


def pos_weight(y) -> float:
    """neg/pos ratio — the scale_pos_weight for boosted trees."""
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def get_models(seed, use_class_weight=True, spw=None) -> dict:
    """Return {name: estimator}.

    use_class_weight : turn on cost-sensitive learning. Set False when an
                       explicit resampler (SMOTE/undersampling) is used, to
                       avoid double-counting the minority class.
    spw              : scale_pos_weight for boosters; compute from the TRAIN y
                       in the caller (models.pos_weight(y_train)).
    """
    cw = "balanced" if use_class_weight else None
    models = {
        # Baseline #1 — linear, the reference every ensemble must beat.
        "logreg": LogisticRegression(max_iter=2000, class_weight=cw),
        # Bagging ensemble.
        "random_forest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=seed,
            class_weight=("balanced_subsample" if use_class_weight else None),
        ),
        # Boosting ensemble (fast, native).
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
    except Exception as exc:  # native libs (e.g. libomp) raise non-ImportError
        _note_unavailable("xgboost", exc)
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1,
            class_weight=("balanced" if use_class_weight else None),
            verbose=-1,
        )
    except Exception as exc:  # native libs (e.g. libomp) raise non-ImportError
        _note_unavailable("lightgbm", exc)
    return models
