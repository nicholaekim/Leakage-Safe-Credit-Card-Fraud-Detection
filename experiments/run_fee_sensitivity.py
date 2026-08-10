"""fee robustness: does the $5/alert assumption drive the conclusions?

the money benchmark assumes a flat $5 fee per investigated alert. this
script re-scores the SAME deployed artifacts (per-seed test scores plus the
thresholds that were tuned on validation for the $5 fee) under a range of
fees, so it answers the deployment question "we built for $5 - how bad is
it if the real fee is $1 or $20?". it does not re-tune thresholds per fee,
which would need the validation scores the benchmark doesn't cache; that is
the honest caveat printed with the results.

reference policies at each fee: do-nothing (0 by definition), flag-all, and
the oracle that flags exactly the frauds worth more than the fee.

needs:  python -m experiments.run_benchmark  first (saves the score arrays)
run:    python -m experiments.run_fee_sensitivity
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, evaluate
from experiments.run_bootstrap import load_seed

FEES = [1.0, 2.0, 5.0, 10.0, 20.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--fees", type=float, nargs="+", default=FEES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--split", default="stratified",
                    choices=["stratified", "temporal"],
                    help="which benchmark score files to load")
    args = ap.parse_args()
    if any(f <= 0 for f in args.fees):
        raise SystemExit("--fees must all be positive (log-scale axis)")
    if args.quick:
        args.seeds = [0]
    temporal = args.split == "temporal"

    config.ensure_dirs()
    data = {s: load_seed(s, args.quick, temporal) for s in args.seeds}
    combos = sorted(set.intersection(*(set(d["scores"]) for d in data.values())))
    print(f"{len(combos)} model combos x {len(args.seeds)} seeds x "
          f"fees {args.fees}")
    print("NOTE: thresholds stay as tuned on validation for the "
          f"${config.C_ALERT:.0f} fee - this measures how the deployed "
          "policy holds up if the fee assumption is wrong, not re-optimised "
          "policies.")

    rows = []
    for s in args.seeds:
        d = data[s]
        y, amt = d["y"], d["amount"]
        for fee in args.fees:
            for c in combos:
                yhat = (d["scores"][c] >= d["thr"][c]).astype(int)
                rows.append({"seed": s, "fee": fee, "policy": c,
                             "savings": evaluate.savings(y, yhat, amt, fee)})
            oracle = ((y == 1) & (amt > fee)).astype(int)
            rows.append({"seed": s, "fee": fee, "policy": "oracle",
                         "savings": evaluate.savings(y, oracle, amt, fee)})
            rows.append({"seed": s, "fee": fee, "policy": "flag_all",
                         "savings": evaluate.savings(y, np.ones_like(y), amt,
                                                     fee)})

    out = pd.DataFrame(rows)
    suffix = ("_quick" if args.quick else "") + \
             ("_temporal" if temporal else "")
    out.to_csv(config.TABLES_DIR / f"fee_sensitivity{suffix}.csv", index=False)

    mean = (out.groupby(["policy", "fee"])["savings"].mean()
               .unstack("fee").round(4))
    models_only = mean.drop(index=["oracle", "flag_all"], errors="ignore")
    print("\n=== mean savings by fee (thresholds tuned at "
          f"${config.C_ALERT:.0f}) ===")
    print(pd.concat([models_only, mean.loc[["oracle", "flag_all"]]]))
    print("\n=== money winner per fee ===")
    for fee in args.fees:
        col = models_only[fee]
        print(f"  ${fee:>5.2f}: {col.idxmax()}  ({col.max():.4f})")

    # small figure: top models by $5 savings (best-anywhere when $5 isn't
    # in the sweep), plus the oracle ceiling
    top = models_only[config.C_ALERT].nlargest(4).index if \
        config.C_ALERT in models_only.columns else \
        models_only.max(axis=1).nlargest(4).index
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in top:
        ax.plot(models_only.columns, models_only.loc[c], "o-", label=c)
    ax.plot(mean.columns, mean.loc["oracle"], "k--", label="oracle ceiling")
    ax.set_xscale("log")
    ax.set_xticks(args.fees)
    ax.set_xticklabels([f"${f:g}" for f in args.fees])
    ax.set_xlabel("alert fee (thresholds tuned at "
                  f"${config.C_ALERT:.0f})")
    ax.set_ylabel("savings")
    ax.set_title("Fee sensitivity: savings of the deployed policies")
    ax.legend(fontsize=8)
    fig.savefig(config.FIGURES_DIR / f"fee_sensitivity{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fee_sensitivity{suffix}.csv and .png to results/")


if __name__ == "__main__":
    main()
