"""decision policy ladder: do smarter flagging rules and honest
probabilities actually earn more money?

four policies, each run per probability variant:
  naive        flag if p >= 0.5 (the default everyone uses)
  tuned        flag if p >= t*, t* picked to maximise savings on validation
  bayes        flag if p * amount > fee. no tuning at all, straight expected
               loss, but it only works if p is an honest probability
  tuned_bayes  flag if p * amount >= c*, c* tuned on validation. this one
               separates the two effects: tuned vs tuned_bayes shows what
               amount awareness buys, tuned_bayes vs bayes shows what
               calibration buys (a tuned cutoff can absorb miscalibration,
               the fixed fee cutoff cant)

three probability variants: raw scores, platt, isotonic. calibrators are fit
on validation only, never test.

the headline is the price of miscalibration under the bayes rule, reported
separately per calibrator (picking the better one per seed on test savings
would just harvest test noise).

side note: the bayes rule never flags transactions with amount <= fee, since
p <= 1 means p*amount <= amount <= fee. that tanks its count recall while
losing zero money, so compare value_recall instead.

caveat for the report: validation gets used twice (fit calibrators + pick
thresholds), in sample for isotonic. test never touches anything fitted or
tuned. sanity check: the tuned row should be near identical across variants
since the calibration maps are monotone.

run from the repo root:
    python -m experiments.run_policy_ladder            # 5 seeds, ~10-15 min
    python -m experiments.run_policy_ladder --quick    # smoke test
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")  # headless backend, has to be set before pyplot loads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, data, pipeline, models, evaluate, calibrate, plots

POLICIES = ("naive", "tuned", "bayes", "tuned_bayes")
VARIANTS = ("raw", "platt", "isotonic")
# amount bands for the where-does-the-money-go table. first band starts below
# 0 so the zero amount frauds are included
BANDS = ((-1.0, config.C_ALERT), (config.C_ALERT, 100.0), (100.0, float("inf")))
BAND_LABELS = (f"<= ${config.C_ALERT:.0f}", f"${config.C_ALERT:.0f}-100", "> $100")


def decide(policy, p_test, amt_test, threshold=None):
    """binary flag decisions for one policy. p_test has to be probabilities
    in [0,1], the bayes rule makes no sense on raw margins"""
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
    """fraud value caught/missed per amount band"""
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
    """all policy x variant cells for one seed. extras carries per combo test
    scores for the figure and band table later"""
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
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
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
            extras = ex  # keep the first seeds scores for the figure

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
          "uses quantile bins - sensitive to the decision-relevant tail) ===")
    print(raw.pivot_table(index=["strategy", "model"], columns="variant",
                          values=["brier", "ece", "ece_adaptive"],
                          aggfunc="mean").round(4))

    print("\n=== The price of miscalibration under the bayes rule "
          "(calibrated minus raw, per calibrator - no best-of-two "
          "selection) ===")
    bay = raw[raw["policy"] == "bayes"].pivot_table(
        index=["strategy", "model", "seed"], columns="variant",
        values="savings")
    cb = raw.groupby(["strategy", "model", "seed"])["cost_base"].first()
    for cal in ("platt", "isotonic"):
        bay[f"delta_{cal}"] = bay[cal] - bay["raw"]
        bay[f"dollars_{cal}"] = bay[f"delta_{cal}"] * cb
    out = bay.groupby(["strategy", "model"])[
        ["delta_platt", "dollars_platt", "delta_isotonic", "dollars_isotonic"]].mean()
    print(out.round({"delta_platt": 4, "dollars_platt": 0,
                     "delta_isotonic": 4, "dollars_isotonic": 0})
          .sort_values("dollars_platt", ascending=False))

    # reliability figures: the showcase combo (rf, picked ahead of time as the
    # strongest family from earlier validation runs, not off this runs test)
    # plus logreg, whose miscalibration drives the $48k headline - the figure
    # shown next to that discussion has to be the same model
    for strat_t, model_t, label_t, fname in (
            ("class_weight", "random_forest", "RF",
             "reliability_rf_class_weight.png"),
            ("class_weight", "logreg", "LogReg",
             "reliability_logreg_class_weight.png")):
        if (strat_t, model_t) not in extras:
            continue
        ex = extras[(strat_t, model_t)]
        fig, ax = plt.subplots(figsize=(6, 5))
        for vkey in VARIANTS:
            plots.plot_reliability(ex["yte"], ex["p_te"][vkey], ax=ax, label=vkey)
        ax.set_title("Reliability: %s + %s (test, seed %d)"
                     % (label_t, strat_t, seeds[0]))
        path = plots.savefig(fig, fname)
        plt.close(fig)
        print(f"\nSaved reliability diagram -> {path}")

    target = ("class_weight", "random_forest")
    if target in extras:
        ex = extras[target]
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
