"""Headline experiment: leakage-SAFE vs leaky pipelines.

Runs every ensemble model under several imbalance strategies and seeds, the
correct way, and contrasts them against a deliberately leaky baseline that
commits the three classic sins. The leaky numbers come out *higher* — that is
the whole point, and the spine of the report.

Run from the repo root:
    python -m experiments.run_benchmark             # full run (slow)
    python -m experiments.run_benchmark --quick     # fast smoke test
    python -m experiments.run_benchmark --models logreg random_forest
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src import config, data, pipeline, models, evaluate


def run_safe(df, seed, strategies, model_filter=None):
    """The correct workflow: split first; scaling + resampling live inside the
    pipeline (train-fold only); threshold tuned on validation; evaluate on the
    real, imbalanced test set."""
    records = []
    s = data.stratified_split(df, seed=seed)
    Xtr, ytr = s["train"]["X"], s["train"]["y"]
    Xva, yva = s["val"]["X"], s["val"]["y"]
    Xte, yte = s["test"]["X"], s["test"]["y"]
    amt_va, amt_te = s["val"]["amount"], s["test"]["amount"]
    spw = models.pos_weight(ytr)

    for strat in strategies:
        use_cw = strat == "class_weight"
        resampler = pipeline.get_resampler(strat, seed)
        for name, clf in models.get_models(seed, use_cw, spw).items():
            if model_filter and name not in model_filter:
                continue
            pipe = pipeline.make_safe_pipeline(clf, resampler=resampler)
            pipe.fit(Xtr, ytr)
            s_va = pipe.predict_proba(Xva)[:, 1]
            thr = evaluate.pick_threshold(yva, s_va, "cost", amounts=amt_va)
            s_te = pipe.predict_proba(Xte)[:, 1]
            m = evaluate.evaluate_predictions(yte, s_te, thr, amounts=amt_te)
            m.update(mode="safe", strategy=strat, model=name, seed=seed)
            records.append(m)
    return records


def run_leaky(df, seed, model_filter=None):
    """The WRONG workflow, on purpose:
      leak #1: StandardScaler fit on the full dataset (train+test);
      leak #2: SMOTE applied to everything BEFORE the split, so synthetic
               neighbours of test frauds end up in train and the test set is
               artificially balanced;
      leak #3: the decision threshold is tuned on the test set itself.
    The reported metrics look great and mean nothing."""
    records = []
    X, y = data.xy(df)
    Xs = StandardScaler().fit_transform(X)                 # leak #1
    Xr, yr = SMOTE(random_state=seed).fit_resample(Xs, y)  # leak #2
    Xtr, Xte, ytr, yte = train_test_split(
        Xr, yr, test_size=0.2, random_state=seed, stratify=yr)
    spw = models.pos_weight(ytr)

    for name, clf in models.get_models(seed, use_class_weight=False, spw=spw).items():
        if model_filter and name not in model_filter:
            continue
        clf.fit(Xtr, ytr)
        s_te = clf.predict_proba(Xte)[:, 1]
        thr = evaluate.pick_threshold(yte, s_te, "f1")     # leak #3
        m = evaluate.evaluate_predictions(yte, s_te, thr)
        m.update(mode="leaky", strategy="smote", model=name, seed=seed)
        records.append(m)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS)
    ap.add_argument("--strategies", nargs="*",
                    default=["class_weight", "smote", "undersample", "none"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names, e.g. logreg random_forest")
    ap.add_argument("--no-leaky", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="subsample + 1 seed + 2 models for a fast smoke test")
    args = ap.parse_args()

    config.ensure_dirs()
    df = data.load_raw()
    print("Integrity:", data.integrity_report(df))
    df = data.remove_duplicates(df)
    print(f"Dropped {df.attrs.get('n_dropped_duplicates', 0)} duplicate rows; "
          f"{len(df)} remain.")

    seeds, strategies, model_filter = args.seeds, args.strategies, args.models
    if args.quick:
        df = pd.concat([
            df[df[config.TARGET] == 1],
            df[df[config.TARGET] == 0].sample(20000, random_state=0),
        ]).reset_index(drop=True)
        seeds, strategies = [0], ["class_weight", "smote"]
        model_filter = model_filter or ["logreg", "random_forest"]

    records = []
    for seed in seeds:
        print(f"[seed {seed}] safe pipelines ...")
        records += run_safe(df, seed, strategies, model_filter)
        if not args.no_leaky:
            print(f"[seed {seed}] leaky baseline ...")
            records += run_leaky(df, seed, model_filter)

    raw, grouped = evaluate.aggregate(records)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.TABLES_DIR / "benchmark_raw.csv", index=False)
    grouped.to_csv(config.TABLES_DIR / "benchmark_summary.csv")

    print("\n=== PR-AUC by mode (mean over seeds) ===")
    print(raw.pivot_table(index=["strategy", "model"], columns="mode",
                          values="pr_auc", aggfunc="mean").round(4))
    print(f"\nSaved tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
