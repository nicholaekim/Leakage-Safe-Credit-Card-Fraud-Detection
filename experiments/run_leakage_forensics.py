"""Leakage forensics: a 2^4 factorial ablation of the classic fraud-ML sins.

Instead of one "leaky vs safe" comparison, each sin is an independent on/off
factor and all 16 combinations run, so the score inflation can be attributed
sin by sin:

  smote_before_split — SMOTE the WHOLE dataset, then split (synthetic
                       neighbours of test frauds leak into train, and the
                       test set becomes ~50% synthetic fraud). Safe version:
                       SMOTE inside the pipeline, train folds only. Both arms
                       use SMOTE *in standardized space* (kNN geometry), so
                       the factor isolates PLACEMENT — not SMOTE itself, and
                       not the interpolation space.
  scaler_on_all      — StandardScaler fit on train+val+test vs train only.
  keep_duplicates    — leave duplicate rows in, letting copies straddle the
                       train/test split. Row identity is defined in the space
                       the model sees (V1-28 + Amount + Class, Time excluded):
                       the dataset has ~9,144 feature-identical copies, far
                       more than its 1,081 full-row duplicates. NOTE: dedup
                       also changes n and class composition, so this factor's
                       estimand is "the global dedup decision", and its sign
                       is not predictable a priori.
  threshold_on_test  — tune the F1-optimal decision threshold on the test set
                       itself vs on validation.

DUAL EVALUATION — every config is measured twice, and the columns carry
different estimands. Read them accordingly:
  reported_*   — the config's own test split: the number the flawed notebook
                 would publish (its test set may be poisoned).
  true_pr_auc  — the SAME fitted model on a clean holdout, threshold-free:
                 the model-change metric. If a sin inflates reported_* but
                 leaves true_pr_auc flat, it poisoned the test set rather
                 than changing the model.
  true_f1/recall/precision — the deployed ARTIFACT (model + threshold) on the
                 clean holdout at (deduplicated) real-world prevalence. These
                 conflate model quality with threshold-prevalence mismatch —
                 deliberately: shipping a threshold tuned at synthetic 50%
                 prevalence IS a real deployment harm, but attribute it to
                 the operating point, not the scorer.

The clean holdout is carved BEFORE any sin can act, grouped by model-visible
identity: one copy of each selected feature-unique row, no feature-identical
copy left in the pool, surplus copies dropped. Its carve RNG is decoupled
from the modeling seed; note that re-carving per seed means true_* variance
includes holdout re-draw variance (~95 holdout frauds).

Pre-registered expectations (phrased to be testable):
  1. smote_before_split dominates reported inflation while true_pr_auc barely
     moves (poisoned test set, not a better model). Its true_f1 collapse is a
     threshold-transfer failure — a deployment harm of the sin, not evidence
     the scorer degraded.
  2. threshold_on_test inflates reported F1. Its effect on reported PR-AUC is
     EXACTLY zero by construction (PR-AUC ignores the threshold) — printed as
     an internal-consistency check: nonzero means a bug, not a finding.
  3. scaler_on_all is indistinguishable from seed noise (equivalence claim —
     judge it against the reported +/- sd, not as a point estimate).
  4. keep_duplicates: magnitude comparable to seed noise or modestly above;
     sign not predicted.

Main effects are reported overall AND stratified by smote_before_split: in
smote-ON cells the reported metrics sit near the ceiling of a ~50%-prevalence
synthetic test set, which compresses every other sin's apparent effect. The
full per-cell CSV supports any interaction analysis.

Run from the repo root:
    python -m experiments.run_leakage_forensics            # ~15-25 min
    python -m experiments.run_leakage_forensics --quick    # smoke test
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src import config, data, pipeline, models, evaluate

SINS = ("smote_before_split", "scaler_on_all", "keep_duplicates",
        "threshold_on_test")
METRICS = ("pr_auc", "f1", "recall", "precision")
# Row identity in the space the model sees (Time excluded on purpose).
KEY = config.FEATURES + [config.TARGET]


def _feature_hashes(df):
    """64-bit row hashes over the model-visible columns, with a guard that
    hash-grouping agrees with pandas value-equality (0.0/-0.0, NaN payloads)."""
    key_df = df[KEY]
    h = pd.util.hash_pandas_object(key_df, index=False).to_numpy()
    hash_dup = pd.Series(h).duplicated().to_numpy()
    value_dup = key_df.duplicated().to_numpy()
    assert (hash_dup == value_dup).all(), \
        "hash grouping diverged from value equality — holdout would leak"
    return h, ~hash_dup                      # (hashes, first-copy mask)


def carve_clean_holdout(df, seed, frac=0.20):
    """Split raw data (duplicates included) into (pool, holdout) such that no
    row in the holdout has a feature-identical copy in the pool.

    Feature-unique rows are selected stratified-by-class; the holdout gets
    exactly one copy of each selected row; the pool keeps ALL copies of the
    others (so keep_duplicates still has duplicates to exploit); surplus
    copies of held-out rows are dropped from both sides. The carve RNG is
    offset from the modeling seed so the two streams are decoupled.
    """
    h, first = _feature_hashes(df)
    uniq = df[first]
    uh = h[first]
    idx = np.arange(len(uniq))
    _, ho_idx = train_test_split(
        idx, test_size=frac, stratify=uniq[config.TARGET],
        random_state=10_000 + seed)
    in_holdout = np.isin(h, uh[ho_idx])
    holdout = df[first & in_holdout]
    pool = df[~in_holdout]
    dropped = int(in_holdout.sum()) - len(holdout)     # surplus copies
    return (pool.reset_index(drop=True),
            holdout.reset_index(drop=True), dropped)


def run_config(pool, holdout, seed, sins, clf):
    """One cell of the factorial: run the (possibly flawed) workflow on the
    pool, then score the SAME artifact on the clean holdout."""
    if sins["keep_duplicates"]:
        work = pool
    else:                                   # dedup in model-visible space
        _, first = _feature_hashes(pool)
        work = pool[first]
    X, y = data.xy(work)
    X_ho, y_ho = data.xy(holdout)

    if sins["scaler_on_all"]:                      # sin: scaler sees test data
        scaler = StandardScaler().fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X_ho = pd.DataFrame(scaler.transform(X_ho), columns=X_ho.columns)

    if sins["smote_before_split"]:                 # sin: resample, THEN split
        # Interpolate in standardized space — same kNN geometry as the safe
        # in-pipeline arm — then map back, so the smote factor isolates
        # placement and the scaler_on_all factor keeps its meaning. (Raw-space
        # SMOTE would let Amount, std ~250 vs ~1 for the V's, dominate
        # neighbour distances: a hidden smote x scaler confound.)
        sm_scaler = StandardScaler().fit(X)
        X_res, y = SMOTE(random_state=seed).fit_resample(
            sm_scaler.transform(X), y)
        X = pd.DataFrame(sm_scaler.inverse_transform(X_res),
                         columns=X_ho.columns)
        resampler = None
    else:                                          # safe: SMOTE in-pipeline
        resampler = SMOTE(random_state=seed)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    pipe = pipeline.make_safe_pipeline(
        clf, resampler=resampler, scale=not sins["scaler_on_all"])
    pipe.fit(X_tr, y_tr)
    s_va = pipe.predict_proba(X_va)[:, 1]
    s_te = pipe.predict_proba(X_te)[:, 1]

    if sins["threshold_on_test"]:                  # sin: tune on the test set
        thr = evaluate.pick_threshold(y_te, s_te, "f1")
    else:
        thr = evaluate.pick_threshold(y_va, s_va, "f1")

    reported = evaluate.evaluate_predictions(y_te, s_te, thr)
    s_ho = pipe.predict_proba(X_ho)[:, 1]
    true = evaluate.evaluate_predictions(y_ho, s_ho, thr)

    row = {**{f"rep_{m}": reported[m] for m in METRICS},
           **{f"true_{m}": true[m] for m in METRICS},
           **{f"infl_{m}": reported[m] - true[m] for m in METRICS},
           "threshold": thr, "n_sins": sum(sins.values())}
    row.update(sins)
    return row


def main_effects(raw, metric, sins=SINS):
    """Per-sin main effect, computed per seed then aggregated to mean +/- sd:
    the sd is the honest yardstick for 'is this effect real?'."""
    seeds = sorted(raw["seed"].unique())
    rows = []
    for sin in sins:
        d_rep, d_true = [], []
        for s in seeds:
            g = raw[raw["seed"] == s]
            on, off = g[g[sin]], g[~g[sin]]
            d_rep.append(on[f"rep_{metric}"].mean() - off[f"rep_{metric}"].mean())
            d_true.append(on[f"true_{metric}"].mean() - off[f"true_{metric}"].mean())
        rows.append({
            "sin": sin,
            f"d_rep_{metric}": np.mean(d_rep),
            f"d_rep_{metric}_sd": np.std(d_rep),
            f"d_true_{metric}": np.mean(d_true),
            f"d_true_{metric}_sd": np.std(d_true),
        })
    return pd.DataFrame(rows).set_index("sin").round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["logreg", "hist_gbm"],
                    help="fast pair by default; random_forest is much slower")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    df = data.load_raw()
    seeds, model_filter = args.seeds, args.models
    if args.quick:
        df = pd.concat([
            df[df[config.TARGET] == 1],
            df[df[config.TARGET] == 0].sample(20000, random_state=0),
        ]).reset_index(drop=True)
        seeds, model_filter = [0], ["logreg"]

    configs = [dict(zip(SINS, bits))
               for bits in product([False, True], repeat=len(SINS))]
    rows = []
    for seed in seeds:
        pool, holdout, dropped = carve_clean_holdout(df, seed)
        print(f"[seed {seed}] pool={len(pool)} holdout={len(holdout)} "
              f"(holdout frauds={int(holdout[config.TARGET].sum())}, "
              f"surplus dup copies dropped={dropped})")
        for name in model_filter:
            for i, sins in enumerate(configs):
                zoo = models.get_models(seed, use_class_weight=False)
                row = run_config(pool, holdout, seed, sins, zoo[name])
                row.update(model=name, seed=seed)
                rows.append(row)
                print(f"  [{name} {i+1:2d}/16] sins={sum(sins.values())} "
                      f"rep_pr_auc={row['rep_pr_auc']:.3f} "
                      f"true_pr_auc={row['true_pr_auc']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(config.TABLES_DIR / "leakage_forensics_raw.csv", index=False)

    others = tuple(s for s in SINS if s != "smote_before_split")
    effs = []
    for name in model_filter:
        sub = raw[raw["model"] == name]
        print(f"\n================ {name} ================")

        print("--- main effects, averaged over ALL strata "
              "(smote-ON cells compress the others — see stratified) ---")
        overall = pd.concat([main_effects(sub, "pr_auc"),
                             main_effects(sub, "f1")], axis=1)
        print(overall)
        o = overall.reset_index(); o["model"], o["stratum"] = name, "overall"
        effs.append(o)

        for flag in (False, True):
            st = sub[sub["smote_before_split"] == flag]
            label = f"smote_before_split={'ON' if flag else 'OFF'}"
            print(f"\n--- conditional effects within the {label} stratum ---")
            cond = pd.concat([main_effects(st, "pr_auc", others),
                              main_effects(st, "f1", others)], axis=1)
            print(cond)
            c = cond.reset_index(); c["model"], c["stratum"] = name, label
            effs.append(c)

        # Unrounded on purpose: main_effects() rounds for display, which
        # would silently raise this check's detection floor to 5e-5.
        diffs = []
        for s in sub["seed"].unique():
            g = sub[sub["seed"] == s]
            diffs.append(
                g.loc[g["threshold_on_test"], "rep_pr_auc"].mean()
                - g.loc[~g["threshold_on_test"], "rep_pr_auc"].mean())
        chk = abs(float(np.mean(diffs)))
        print(f"\nconsistency check — threshold_on_test effect on reported "
              f"PR-AUC (must be ~0 by construction): {chk:.2e} "
              f"{'PASS' if chk < 1e-9 else 'FAIL: investigate'}")

        safe = sub[sub["n_sins"] == 0]
        dirty = sub[sub["n_sins"] == len(SINS)]
        print(f"all-safe : reported PR-AUC={safe['rep_pr_auc'].mean():.4f} "
              f"(sd {safe['rep_pr_auc'].std(ddof=0):.4f})  "
              f"true={safe['true_pr_auc'].mean():.4f}")
        print(f"all-sins : reported PR-AUC={dirty['rep_pr_auc'].mean():.4f} "
              f"(sd {dirty['rep_pr_auc'].std(ddof=0):.4f})  "
              f"true={dirty['true_pr_auc'].mean():.4f}")

    pd.concat(effs).to_csv(
        config.TABLES_DIR / "leakage_forensics_effects.csv", index=False)
    print(f"\nSaved tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
