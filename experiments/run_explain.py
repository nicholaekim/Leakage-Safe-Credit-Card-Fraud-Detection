"""what drives the fraud model and where does it fail.

showcase model (picked up front as the strongest family from validation):
random forest + class_weight, seed 0, savings tuned threshold.

produces:
  1. permutation importance on test (pr-auc drop per shuffled feature)
  2. shap if installed: global ranking + local explanations for three cases,
     the most confident catch, the most confident false alarm, and the most
     expensive missed fraud
  3. error economics: where the mistakes live by amount band

caveat: v1..v28 are anonymised pca components so importance points at
components, not human readable causes.

run:  python -m experiments.run_explain            # ~5-10 min
      python -m experiments.run_explain --quick    # smoke test
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, data, pipeline, models, evaluate, explain

BANDS = ((-1.0, config.C_ALERT), (config.C_ALERT, 100.0), (100.0, float("inf")))
BAND_LABELS = (f"<= ${config.C_ALERT:.0f}", f"${config.C_ALERT:.0f}-100", "> $100")


def error_economics(y, yhat, amounts):
    y = np.asarray(y); yhat = np.asarray(yhat); amounts = np.asarray(amounts)
    rows = []
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        m = (amounts > lo) & (amounts <= hi)
        fraud = m & (y == 1)
        rows.append({
            "band": label,
            "n_frauds": int(fraud.sum()),
            "caught": int((fraud & (yhat == 1)).sum()),
            "missed": int((fraud & (yhat == 0)).sum()),
            "value_missed": round(float(amounts[fraud & (yhat == 0)].sum()), 2),
            "false_alarms": int((m & (y == 0) & (yhat == 1)).sum()),
        })
    return pd.DataFrame(rows)


def shap_case_table(sv_row, x_row, features, top=6):
    order = np.argsort(-np.abs(sv_row))[:top]
    return pd.DataFrame({
        "feature": [features[i] for i in order],
        "value": [float(x_row.iloc[i]) for i in order],
        "shap": [float(sv_row[i]) for i in order],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    df = data.remove_duplicates(data.load_raw())
    if args.quick:
        df = pd.concat([
            df[df[config.TARGET] == 1],
            df[df[config.TARGET] == 0].sample(20000, random_state=0),
        ]).reset_index(drop=True)

    s = data.stratified_split(df, seed=args.seed)
    Xtr, ytr = s["train"]["X"], s["train"]["y"]
    Xva, yva = s["val"]["X"], s["val"]["y"]
    Xte, yte = s["test"]["X"], s["test"]["y"]
    amt_te = s["test"]["amount"]

    print("Fitting showcase artifact: RandomForest + class_weight ...")
    clf = models.get_models(args.seed)["random_forest"]
    pipe = pipeline.make_safe_pipeline(clf)
    pipe.fit(Xtr, ytr)
    s_va = pipe.predict_proba(Xva)[:, 1]
    thr = evaluate.pick_threshold(yva, s_va, "savings",
                                  amounts=s["val"]["amount"])
    s_te = pipe.predict_proba(Xte)[:, 1]
    yhat = (s_te >= thr).astype(int)

    # permutation importance
    print("Permutation importance (PR-AUC drop, 5 repeats) ...")
    imp = explain.permutation_importances(
        pipe, Xte, yte, n_repeats=(2 if args.quick else 5), seed=args.seed)
    imp.to_csv(config.TABLES_DIR / "permutation_importance.csv", index=False)
    top15 = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top15["feature"], top15["importance_mean"],
            xerr=top15["importance_std"])
    ax.set_xlabel("PR-AUC drop when feature is shuffled")
    ax.set_title("Permutation importance - RF + class_weight (test)")
    fig.savefig(config.FIGURES_DIR / "permutation_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(imp.head(8).to_string(index=False))

    # shap (optional dependency)
    try:
        import shap  # noqa: F401
        have_shap = True
    except ImportError:
        have_shap = False
        print("\n[skip] shap not installed - global/local SHAP skipped "
              "(pip install shap; included in requirements.txt).")

    if have_shap:
        print("\nSHAP global summary (2000-row sample) ...")
        sv, Xs, feats = explain.tree_shap(pipe, Xte, max_samples=2000,
                                          seed=args.seed)
        mean_abs = pd.DataFrame({
            "feature": feats,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        mean_abs.to_csv(config.TABLES_DIR / "shap_global.csv", index=False)
        t = mean_abs.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(t["feature"], t["mean_abs_shap"])
        ax.set_xlabel("mean |SHAP value| (fraud-class contribution)")
        ax.set_title("SHAP global importance - RF + class_weight")
        fig.savefig(config.FIGURES_DIR / "shap_global.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(mean_abs.head(8).to_string(index=False))

        # local cases: confident tp, confident fp, most expensive fn
        y_arr = yte.to_numpy()
        amt = np.asarray(amt_te)
        cases = {}
        tp = np.where((y_arr == 1) & (yhat == 1))[0]
        fp = np.where((y_arr == 0) & (yhat == 1))[0]
        fn = np.where((y_arr == 1) & (yhat == 0))[0]
        if len(tp):
            cases["TP_most_confident_catch"] = tp[np.argmax(s_te[tp])]
        if len(fp):
            cases["FP_most_confident_false_alarm"] = fp[np.argmax(s_te[fp])]
        if len(fn):
            cases["FN_most_expensive_miss"] = fn[np.argmax(amt[fn])]
        if cases:
            rows_idx = list(cases.values())
            X_cases = Xte.iloc[rows_idx]
            sv_cases, _, _ = explain.tree_shap(pipe, X_cases,
                                               max_samples=len(X_cases),
                                               seed=args.seed)
            print("\nLocal explanations (top-6 signed SHAP contributions):")
            local_frames = []
            for k, (name, idx) in enumerate(cases.items()):
                t = shap_case_table(sv_cases[k], Xte.iloc[idx], feats)
                t["case"] = name
                t["score"] = float(s_te[idx])
                t["amount"] = float(amt[idx])
                local_frames.append(t)
                print(f"\n[{name}] score={s_te[idx]:.3f} "
                      f"amount=${amt[idx]:.2f}")
                print(t[["feature", "value", "shap"]].round(4)
                      .to_string(index=False))
            pd.concat(local_frames).to_csv(
                config.TABLES_DIR / "shap_local_cases.csv", index=False)

    # error economics
    print("\nError economics by amount band (test, savings-tuned threshold):")
    econ = error_economics(yte, yhat, amt_te)
    econ.to_csv(config.TABLES_DIR / "error_economics.csv", index=False)
    print(econ.to_string(index=False))
    print(f"\nSaved tables to {config.TABLES_DIR} and figures to "
          f"{config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
