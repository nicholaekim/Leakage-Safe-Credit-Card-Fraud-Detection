"""Decision-policy ladder: how much money do smarter flagging rules — and
honest probabilities — actually earn?

Four policies, evaluated per probability variant:
  naive       — flag iff p >= 0.5                  (the Kaggle default)
  tuned       — flag iff p >= t*, t* maximises savings on VALIDATION
  bayes       — flag iff p * Amount > C_ALERT      (per-transaction expected-
                loss rule from decision theory; no tuning at all, but only
                works if p is an honest probability)
  tuned_bayes — flag iff p * Amount >= c*, c* tuned on VALIDATION. This rung
                de-confounds the ladder: tuned vs tuned_bayes isolates
                amount-awareness; tuned_bayes vs bayes isolates calibration
                (a tuned cutoff can absorb global miscalibration; the fixed
                C_ALERT cutoff cannot).

Three probability variants: raw model scores, Platt-scaled, isotonic — both
calibrators fit on the validation split only (never test).

Headline numbers: the *price of miscalibration* under the bayes rule —
savings(bayes, platt) - savings(bayes, raw) and the isotonic analogue,
reported SEPARATELY per calibrator. (Selecting the better calibrator per seed
on test savings would harvest test noise and inflate the delta.)

A side finding worth reporting: the bayes rule never flags Amount <= C_ALERT
transactions (p <= 1 implies p*Amount <= Amount <= C_ALERT) — it independently
discovers that tiny frauds are not worth the investigation fee. This depresses
its count-recall while costing nothing; compare value_recall instead.

Methodology caveat (state in the report): the validation split is used both
to fit the calibrators and to pick tuned thresholds — in-sample for isotonic,
so calibrated-variant thresholds are picked on slightly optimistic scores.
Test never influences any fitted or tuned quantity. Built-in sanity check:
the tuned row should be ~identical across variants (calibration maps are
monotone); material differences would be threshold-grid artifacts, not
calibration signal.

Run from the repo root:
    python -m experiments.run_policy_ladder            # ~5-8 min
    python -m experiments.run_policy_ladder --quick    # smoke test
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")           # save figures headless, before pyplot loads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, data, pipeline, models, evaluate, calibrate, plots

POLICIES = ("naive", "tuned", "bayes", "tuned_bayes")
VARIANTS = ("raw", "platt", "isotonic")
# Amount bands for the error-money breakdown. First band starts below 0 so
# the dataset's zero-Amount frauds are included.
BANDS = ((-1.0, config.C_ALERT), (config.C_ALERT, 100.0), (100.0, float("inf")))
BAND_LABELS = (f"<= ${config.C_ALERT:.0f}", f"${config.C_ALERT:.0f}-100", "> $100")


def decide(policy, p_test, amt_test, threshold=None):
    """Binary flag decisions for one policy. p_test must be probabilities in
    [0, 1] (predict_proba output) — the bayes rule is meaningless on margins."""
    if policy in ("tuned", "tuned_bayes") and threshold is None:
        raise ValueError(f"policy {policy!r} requires a threshold")
    if policy == "naive":
        return (p_test >= 0.5).astype(int)
    if policy == "tuned":
        return (p_test >= threshold).astype(int)
    if policy == "bayes":
        return (p_test * amt_test > config.C_ALERT).astype(int)
    if policy == "tuned_bayes":
        return (p_test * amt_test >= threshold).astype(int)
    raise ValueError(policy)


def band_breakdown(y_true, y_pred, amounts):
    """Where does the money live? Fraud value caught/missed per amount band."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    rows = []
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        m = (amounts > lo) & (amounts <= hi)
        fraud = m & (y_true == 1)
        rows.append({
            "band": label,
            "n_frauds": int(fraud.sum()),
            "fraud_value": round(float(amounts[fraud].sum()), 2),
            "value_caught": round(float(amounts[fraud & (y_pred == 1)].sum()), 2),
            "value_missed": round(float(amounts[fraud & (y_pred == 0)].sum()), 2),
            "n_alerts": int((m & (y_pred == 1)).sum()),
        })
    return pd.DataFrame(rows)


def run_ladder(df, seed, strategies, model_filter):
    """All policy x variant cells for one seed. Returns (rows, extras) where
    extras carries per-combo test scores for figures/breakdowns."""
    rows, extras = [], {}
    s = data.stratified_split(df, seed=seed)
    Xtr, ytr = s["train"]["X"], s["train"]["y"]
    Xva, yva = s["val"]["X"], s["val"]["y"]
    Xte, yte = s["test"]["X"], s["test"]["y"]
    amt_va, amt_te = s["val"]["amount"], s["test"]["amount"]
    cost_base = float(np.asarray(amt_te)[np.asarray(yte) == 1].sum())
    spw = models.pos_weight(ytr)

    for strat in strategies:
        use_cw = strat == "class_weight"
        resampler = pipeline.get_resampler(strat, seed)
        for name, clf in models.get_models(seed, use_cw, spw).items():
            if model_filter and name not in model_filter:
                continue
            pipe = pipeline.make_safe_pipeline(clf, resampler=resampler)
            pipe.fit(Xtr, ytr)
            p_va_raw = pipe.predict_proba(Xva)[:, 1]
            p_te_raw = pipe.predict_proba(Xte)[:, 1]

            variants = {"raw": (p_va_raw, p_te_raw)}
            for method, key in (("sigmoid", "platt"), ("isotonic", "isotonic")):
                cal = calibrate.PrefitCalibrator(method).fit(p_va_raw, yva)
                variants[key] = (cal.predict(p_va_raw), cal.predict(p_te_raw))
            extras[(strat, name)] = {"yte": yte, "amt_te": amt_te,
                                     "p_te": {k: v[1] for k, v in variants.items()}}

            for vkey, (p_va, p_te) in variants.items():
                quality = calibrate.calibration_report(yte, p_te)
                ece_adaptive = calibrate.expected_calibration_error(
                    yte, p_te, adaptive=True)
                thr = evaluate.pick_threshold(yva, p_va, "savings",
                                              amounts=amt_va)
                thr_tb = evaluate.pick_threshold(
                    yva, p_va * np.asarray(amt_va), "savings", amounts=amt_va)
                thresholds = {"naive": 0.5, "tuned": thr,
                              "bayes": np.nan, "tuned_bayes": thr_tb}
                for pol in POLICIES:
                    t = thresholds[pol]
                    yhat = decide(pol, p_te, amt_te,
                                  threshold=None if np.isnan(t) else t)
                    row = evaluate.evaluate_decisions(yte, yhat, amt_te)
                    row.update(
                        strategy=strat, model=name, variant=vkey, policy=pol,
                        seed=seed, cost_base=cost_base, threshold=t,
                        brier=quality["brier"], ece=quality["ece"],
                        ece_adaptive=ece_adaptive,
                    )
                    rows.append(row)
    return rows, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--strategies", nargs="+",
                    default=["class_weight", "smote"])
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "random_forest", "lightgbm"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    df = data.remove_duplicates(data.load_raw())
    print(f"{len(df)} rows after dedup.")

    seeds, strategies, model_filter = args.seeds, args.strategies, args.models
    if args.quick:
        df = pd.concat([
            df[df[config.TARGET] == 1],
            df[df[config.TARGET] == 0].sample(20000, random_state=0),
        ]).reset_index(drop=True)
        seeds, strategies = [0], ["class_weight"]
        model_filter = ["logreg", "random_forest"]

    all_rows, extras = [], {}
    for seed in seeds:
        print(f"[seed {seed}] fitting {model_filter} x {strategies} ...")
        rows, ex = run_ladder(df, seed, strategies, model_filter)
        all_rows += rows
        if seed == seeds[0]:
            extras = ex          # keep first seed's scores for figures

    raw = pd.DataFrame(all_rows)
    raw.to_csv(config.TABLES_DIR / "policy_ladder_raw.csv", index=False)
    summary = (raw.groupby(["strategy", "model", "policy", "variant"])
               [["savings", "money_cost", "recall", "precision", "n_alerts"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(config.TABLES_DIR / "policy_ladder_summary.csv")

    print("\n=== Savings by policy x probability variant (mean over seeds) ===")
    piv = raw.pivot_table(index=["strategy", "model", "policy"],
                          columns="variant", values="savings", aggfunc="mean")
    print(piv[list(VARIANTS)].round(4))

    print("\n=== Calibration quality on test (mean over seeds; ece_adaptive "
          "uses quantile bins — sensitive to the decision-relevant tail) ===")
    print(raw.pivot_table(index=["strategy", "model"], columns="variant",
                          values=["brier", "ece", "ece_adaptive"],
                          aggfunc="mean").round(4))

    print("\n=== The price of miscalibration under the bayes rule "
          "(calibrated minus raw, per calibrator — no best-of-two "
          "selection) ===")
    bay = raw[raw["policy"] == "bayes"].pivot_table(
        index=["strategy", "model", "seed"], columns="variant",
        values="savings")
    cb = raw.groupby(["strategy", "model", "seed"])["cost_base"].first()
    for cal in ("platt", "isotonic"):
        bay[f"delta_{cal}"] = bay[cal] - bay["raw"]
        bay[f"eur_{cal}"] = bay[f"delta_{cal}"] * cb
    out = bay.groupby(["strategy", "model"])[
        ["delta_platt", "eur_platt", "delta_isotonic", "eur_isotonic"]].mean()
    print(out.round({"delta_platt": 4, "eur_platt": 0,
                     "delta_isotonic": 4, "eur_isotonic": 0})
          .sort_values("eur_platt", ascending=False))

    # ---- figure + band table for the showcase combo -------------------------
    # A-priori pick: the strongest model family in the earlier validation
    # benchmarks (NOT selected on this run's test results).
    target = ("class_weight", "random_forest")
    if target in extras:
        ex = extras[target]
        fig, ax = plt.subplots(figsize=(6, 5))
        for vkey in VARIANTS:
            plots.plot_reliability(ex["yte"], ex["p_te"][vkey], ax=ax, label=vkey)
        ax.set_title("Reliability: RF + class_weight (test, seed %d)" % seeds[0])
        path = plots.savefig(fig, "reliability_rf_class_weight.png")
        plt.close(fig)
        print(f"\nSaved reliability diagram -> {path}")

        print("\n=== Where the money lives: tuned/raw vs bayes/platt "
              "(RF + class_weight, seed %d) ===" % seeds[0])
        r = raw[(raw.strategy == target[0]) & (raw.model == target[1])
                & (raw.seed == seeds[0])]
        thr = float(r[(r.policy == "tuned") & (r.variant == "raw")]
                    ["threshold"].iloc[0])
        for pol, vkey in (("tuned", "raw"), ("bayes", "platt")):
            yhat = decide(pol, ex["p_te"][vkey], ex["amt_te"],
                          threshold=thr if pol == "tuned" else None)
            print(f"\n[{pol} / {vkey}]")
            print(band_breakdown(ex["yte"], yhat, ex["amt_te"]).to_string(index=False))

    print(f"\nSaved tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
