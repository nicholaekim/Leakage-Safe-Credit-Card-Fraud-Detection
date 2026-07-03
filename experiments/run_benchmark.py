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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src import config, data, pipeline, models, evaluate


def get_split(df, seed, split="stratified"):
    """stratified: random class-balanced split (seed-varying).
    temporal: train on the earliest transactions, test on the latest —
    deterministic, so under multi-seed runs only model randomness varies."""
    if split == "temporal":
        return data.temporal_split(df)
    return data.stratified_split(df, seed=seed)


def run_safe(df, seed, strategies, model_filter=None, objective="savings",
             score_path=None, split="stratified"):
    """The correct workflow: split first; scaling + resampling live inside the
    pipeline (train-fold only); threshold tuned on validation; evaluate on the
    real, imbalanced test set.

    score_path: if given, save the test labels/amounts plus every model's test
    scores and tuned threshold to an .npz — the input for the paired-bootstrap
    comparison (experiments/run_bootstrap.py)."""
    records = []
    s = get_split(df, seed, split)
    Xtr, ytr = s["train"]["X"], s["train"]["y"]
    Xva, yva = s["val"]["X"], s["val"]["y"]
    Xte, yte = s["test"]["X"], s["test"]["y"]
    amt_va, amt_te = s["val"]["amount"], s["test"]["amount"]
    spw = models.pos_weight(ytr)

    store = {"y": yte.to_numpy().astype(np.int8),
             "amount": np.asarray(amt_te, dtype=np.float64),
             "meta": np.array([f"objective={objective}"])}
    for strat in strategies:
        use_cw = strat == "class_weight"
        resampler = pipeline.get_resampler(strat, seed)
        for name, clf in models.get_models(seed, use_cw, spw).items():
            if model_filter and name not in model_filter:
                continue
            pipe = pipeline.make_safe_pipeline(clf, resampler=resampler)
            pipe.fit(Xtr, ytr)
            s_va = pipe.predict_proba(Xva)[:, 1]
            thr = evaluate.pick_threshold(yva, s_va, objective, amounts=amt_va)
            s_te = pipe.predict_proba(Xte)[:, 1]
            m = evaluate.evaluate_predictions(yte, s_te, thr, amounts=amt_te)
            m.update(mode="safe", strategy=strat, model=name, seed=seed)
            records.append(m)
            store[f"{strat}__{name}"] = s_te.astype(np.float64)
            store[f"thr__{strat}__{name}"] = np.array([thr])
    if score_path is not None:
        np.savez_compressed(score_path, **store)
    return records


def run_baselines(df, seed, split="stratified"):
    """Reference points every model must be judged against:
      flag_nothing        — approve everything (the do-nothing bank; savings=0)
      flag_all            — block everything (deeply negative: alert costs)
      oracle_all_fraud    — flag every fraud regardless of size. NOT the money
                            ceiling: ~45% of frauds are <= C_ALERT, so paying
                            the fee to stop them loses money.
      oracle_cost_optimal — flag a fraud only when its Amount exceeds the
                            alert fee: the true savings ceiling.
    """
    s = get_split(df, seed, split)
    yte, amt = s["test"]["y"], s["test"]["amount"]
    y_arr = yte.to_numpy()
    rows = []
    for name, scores in (
        ("flag_nothing", np.zeros(len(yte))),
        ("flag_all", np.ones(len(yte))),
        ("oracle_all_fraud", y_arr.astype(float)),
        ("oracle_cost_optimal",
         ((y_arr == 1) & (amt > config.C_ALERT)).astype(float)),
    ):
        m = evaluate.evaluate_predictions(yte, scores, 0.5, amounts=amt)
        if name in ("flag_nothing", "flag_all"):
            # constant scores make top-k an arbitrary tie-break — meaningless
            m["precision@100"] = m["recall@100"] = float("nan")
        m.update(mode="safe", strategy="baseline", model=name, seed=seed)
        rows.append(m)
    return rows


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
    ap.add_argument("--split", default="stratified",
                    choices=["stratified", "temporal"],
                    help="temporal = train on earliest, test on latest")
    ap.add_argument("--objective", default="savings",
                    choices=["savings", "cost", "f1"],
                    help="validation objective for threshold selection")
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

    # The leaky baseline is only an illustration of inflated metrics, so by
    # default we run it on a light pair of models (it retrains on SMOTE-doubled
    # data and is otherwise the second-heaviest part of the run).
    leaky_filter = model_filter or ["logreg", "random_forest"]

    records = []
    for seed in seeds:
        print(f"[seed {seed}] safe pipelines ...")
        prefix = ("quick_" if args.quick else "") + \
                 ("temporal_" if args.split == "temporal" else "")
        score_path = config.SCORES_DIR / f"scores_{prefix}seed{seed}.npz"
        records += run_safe(df, seed, strategies, model_filter,
                            args.objective, score_path=score_path,
                            split=args.split)
        records += run_baselines(df, seed, split=args.split)
        if not args.no_leaky:
            print(f"[seed {seed}] leaky baseline ...")
            records += run_leaky(df, seed, leaky_filter)

    raw, grouped = evaluate.aggregate(records)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_temporal" if args.split == "temporal" else ""
    raw.to_csv(config.TABLES_DIR / f"benchmark_raw{suffix}.csv", index=False)
    grouped.to_csv(config.TABLES_DIR / f"benchmark_summary{suffix}.csv")

    modeled = raw[(raw["mode"] == "safe") & (raw["strategy"] != "baseline")]

    print("\n=== PR-AUC by mode (mean over seeds) ===")
    print(pd.concat([modeled, raw[raw["mode"] == "leaky"]])
          .pivot_table(index=["strategy", "model"], columns="mode",
                       values="pr_auc", aggfunc="mean").round(4))

    print(f"\n=== Savings — fraction of fraud losses prevented, net of "
          f"€{config.C_ALERT:.0f}/alert (1=perfect, 0=do-nothing) ===")
    sav = raw[raw["mode"] == "safe"].pivot_table(
        index=["strategy", "model"], values=["savings", "money_cost"],
        aggfunc="mean").round(4)
    print(sav.sort_values("savings", ascending=False))

    by = modeled.groupby(["strategy", "model"])[["f1", "savings"]].mean()
    print("\n=== Rank check: score winner vs money winner ===")
    print("top 3 by F1:     ",
          [f"{s}/{m}" for s, m in by.sort_values("f1", ascending=False).head(3).index])
    print("top 3 by savings:",
          [f"{s}/{m}" for s, m in by.sort_values("savings", ascending=False).head(3).index])
    print(f"\nSaved tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
