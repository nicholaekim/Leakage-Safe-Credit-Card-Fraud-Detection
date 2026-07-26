"""leakage safe hyperparameter search: does tuning even help here?

randomized search over lightgbm inside the safe pipeline, stratified 3 fold
cv on the training split only (every transform refits per fold so no
leakage). tuned model then gets compared to the default config on the
untouched val/test splits, in pr-auc and savings.

heads up: with ~280 training frauds the cv signal is noisy, so expect a
small or null improvement and report it either way. "tuning didnt matter"
is a finding, not a failure.

run:  python -m experiments.run_tuning            # ~5-8 min
      python -m experiments.run_tuning --quick    # smoke test
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import config, data, pipeline, models, evaluate

PARAM_DIST = {
    "clf__num_leaves": [15, 31, 63, 127],
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__n_estimators": [200, 400, 600, 800],
    "clf__min_child_samples": [10, 20, 50, 100],
    "clf__subsample": [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
}


def evaluate_artifact(pipe, s, label):
    """threshold on validation, report on test"""
    s_va = pipe.predict_proba(s["val"]["X"])[:, 1]
    thr = evaluate.pick_threshold(s["val"]["y"], s_va, "savings",
                                  amounts=s["val"]["amount"])
    s_te = pipe.predict_proba(s["test"]["X"])[:, 1]
    m = evaluate.evaluate_predictions(s["test"]["y"], s_te, thr,
                                      amounts=s["test"]["amount"])
    return {"config": label, "pr_auc": m["pr_auc"], "savings": m["savings"],
            "recall": m["recall"], "precision": m["precision"],
            "threshold": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    df = data.remove_duplicates(data.load_raw())
    if args.quick:
        df = pd.concat([
            df[df[config.TARGET] == 1],
            df[df[config.TARGET] == 0].sample(20000, random_state=0),
        ]).reset_index(drop=True)
        args.n_iter = 3

    zoo = models.get_models(args.seed)
    if "lightgbm" not in zoo:
        raise SystemExit("lightgbm unavailable - install it (plus libomp on "
                         "macOS) to run the tuning experiment.")

    s = data.stratified_split(df, seed=args.seed)
    Xtr, ytr = s["train"]["X"], s["train"]["y"]
    print(f"Search space: {args.n_iter} candidates x 3 folds on "
          f"{len(Xtr)} training rows ({int(ytr.sum())} frauds)")

    default_pipe = pipeline.make_safe_pipeline(zoo["lightgbm"])
    default_pipe.fit(Xtr, ytr)

    search = RandomizedSearchCV(
        pipeline.make_safe_pipeline(models.get_models(args.seed)["lightgbm"]),
        PARAM_DIST, n_iter=args.n_iter,
        cv=StratifiedKFold(3, shuffle=True, random_state=args.seed),
        scoring="average_precision", random_state=args.seed,
        n_jobs=1, refit=True, verbose=1,
    )
    search.fit(Xtr, ytr)

    cv = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep = [c for c in cv.columns
            if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "rank_test_score")]
    cv[keep].to_csv(config.TABLES_DIR / "tuning_cv_results.csv", index=False)
    print("\nTop CV candidates (average precision, 3-fold on train):")
    print(cv[keep].head(5).to_string(index=False))

    rows = [evaluate_artifact(default_pipe, s, "lightgbm_default"),
            evaluate_artifact(search.best_estimator_, s, "lightgbm_tuned")]
    out = pd.DataFrame(rows).round(4)
    out.to_csv(config.TABLES_DIR / "tuning_default_vs_tuned.csv", index=False)
    print("\nDefault vs tuned on the untouched test split:")
    print(out.to_string(index=False))
    print(f"\nBest params: { {k: v for k, v in search.best_params_.items()} }")
    print(f"CAVEAT: single split/seed - read the delta against the "
          f"bootstrap-measured noise floor (~+/-0.02 PR-AUC).")


if __name__ == "__main__":
    main()
