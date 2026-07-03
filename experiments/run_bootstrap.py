"""Paired-bootstrap model comparison: which 'wins' are real?

Loads the per-seed test scores saved by run_benchmark (results/scores/*.npz),
then for each metric compares the observed leader against the next --top
challengers with paired bootstrap CIs — plus the narrative pair from the
savings benchmark (class_weight/random_forest vs smote/random_forest, the
F1-star that lost the most money).

Reading the output: delta = leader - challenger (or A - B for extra pairs).
A comparison is called 'real' only if the 95% CI excludes zero WITH THE SAME
SIGN in every seed; 'suggestive' if all-but-one; else 'indistinguishable'.
Each seed is a different train/test split, so its CI is conditional on that
split; requiring agreement across seeds guards against split luck.

POST-SELECTION CAVEAT (state in the report): the leader is chosen on the same
test data the CIs resample, and the narrative extra pair was also picked
after seeing the benchmark. So these comparisons are exploratory:
'indistinguishable' is the trustworthy direction, while 'real' wins FOR THE
LEADER carry winner's-curse optimism, and seeds share ~20% of test rows
pairwise so they are not independent replicates. For a confirmatory call,
pin the pairs found here and rerun the benchmark + this script on fresh
seeds (e.g. --seeds 3 4), which this script supports.

Prerequisite:  python -m experiments.run_benchmark  (saves the score arrays)
Run:           python -m experiments.run_bootstrap             # ~5-10 min
               python -m experiments.run_bootstrap --quick     # on quick scores
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config, evaluate, stats

EXTRA_PAIRS = [("class_weight__random_forest", "smote__random_forest")]


def load_seed(seed, quick=False):
    prefix = "quick_" if quick else ""
    path = config.SCORES_DIR / f"scores_{prefix}seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m experiments.run_benchmark"
            f"{' --quick' if quick else ''}` first to save score arrays.")
    with np.load(path) as z:
        combos = sorted(k for k in z.files
                        if "__" in k and not k.startswith("thr__")
                        and not k.startswith("meta"))
        return {
            "y": z["y"], "amount": z["amount"],
            "scores": {c: z[c] for c in combos},
            "thr": {c: float(z[f"thr__{c}"][0]) for c in combos},
            "meta": str(z["meta"][0]) if "meta" in z.files else "unknown",
        }


def observed(d, combo, metric):
    if metric == "pr_auc":
        return average_precision_score(d["y"], d["scores"][combo])
    yhat = (d["scores"][combo] >= d["thr"][combo]).astype(int)
    return evaluate.savings(d["y"], yhat, d["amount"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: 0 1 2 (just 0 with --quick)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--top", type=int, default=5,
                    help="challengers compared against the leader")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [0] if args.quick else [0, 1, 2]

    config.ensure_dirs()
    data = {s: load_seed(s, args.quick) for s in args.seeds}
    if len({d["meta"] for d in data.values()}) > 1:
        print("WARNING: score files were produced with different objectives "
              f"({ {s: d['meta'] for s, d in data.items()} }) — thresholds "
              "are not comparable across seeds.")
    combos = sorted(set.intersection(
        *(set(d["scores"]) for d in data.values())))
    union = set.union(*(set(d["scores"]) for d in data.values()))
    if union - set(combos):
        print(f"NOTE: dropped combos missing in some seeds: "
              f"{sorted(union - set(combos))}")
    print(f"{len(combos)} model combos x {len(args.seeds)} seeds, "
          f"B={args.n_boot}")
    print("NOTE: exploratory post-hoc comparisons — 'indistinguishable' is "
          "the trustworthy direction; 'real' wins for the leader carry "
          "selection optimism (see module docstring).")

    # One shared index matrix per seed, reused by both metrics and all
    # models — that reuse IS the pairing.
    idx_by_seed = {s: stats.bootstrap_indices(len(data[s]["y"]), args.n_boot,
                                              seed=1000 + s)
                   for s in args.seeds}

    rows = []
    for metric in ("pr_auc", "savings"):
        obs_mean = {c: np.mean([observed(data[s], c, metric)
                                for s in args.seeds]) for c in combos}
        ranked = sorted(combos, key=lambda c: -obs_mean[c])
        leader = ranked[0]
        pairs = [(leader, c) for c in ranked[1:1 + args.top]]
        for a, b in EXTRA_PAIRS:
            if (a in combos and b in combos
                    and (a, b) not in pairs and (b, a) not in pairs):
                pairs.append((a, b))
        involved = sorted({c for p in pairs for c in p})

        # metric per (seed, model, draw) — computed once, reused across pairs
        boot = {}
        for s in args.seeds:
            d = data[s]
            idx = idx_by_seed[s]
            for c in involved:
                if metric == "pr_auc":
                    boot[(s, c)] = stats.boot_pr_auc(d["y"], d["scores"][c], idx)
                else:
                    yhat = (d["scores"][c] >= d["thr"][c]).astype(int)
                    boot[(s, c)] = stats.boot_savings(d["y"], yhat,
                                                      d["amount"], idx)
            print(f"  [{metric} seed {s}] bootstrapped {len(involved)} models")

        print(f"\n=== {metric}: leader = {leader} "
              f"(observed mean {obs_mean[leader]:.4f}) ===")
        for a, b in pairs:
            cis = [stats.paired_delta(boot[(s, a)], boot[(s, b)])
                   for s in args.seeds]
            # observed (non-bootstrap) delta per seed — if it drifts from the
            # bootstrap mean, the percentile CI is suspect (bias diagnostic)
            obs_d = [observed(data[s], a, metric) - observed(data[s], b, metric)
                     for s in args.seeds]
            v = stats.verdict(cis)
            spans = " ".join(f"[{c['lo']:+.4f},{c['hi']:+.4f}]" for c in cis)
            print(f"  {a} vs {b}: obs d={np.mean(obs_d):+.4f} "
                  f"boot d={np.mean([c['delta'] for c in cis]):+.4f}"
                  f"  CI95/seed {spans}  -> {v}")
            for s, c, od in zip(args.seeds, cis, obs_d):
                rows.append({"metric": metric, "a": a, "b": b, "seed": s,
                             "obs_delta": od, **c, "verdict": v})

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "bootstrap_comparisons.csv", index=False)

    # forest plot: one panel per metric, one CI line per seed per comparison
    for metric in ("pr_auc", "savings"):
        sub = out[out["metric"] == metric]
        pairs_m = list(dict.fromkeys(zip(sub["a"], sub["b"])))
        fig, ax = plt.subplots(
            figsize=(8, 0.6 * len(pairs_m) * len(args.seeds) / 2 + 2))
        ytick, ylab = [], []
        for i, (a, b) in enumerate(pairs_m):
            for j, s in enumerate(args.seeds):
                r = sub[(sub.a == a) & (sub.b == b) & (sub.seed == s)].iloc[0]
                ypos = i + (j - (len(args.seeds) - 1) / 2) * 0.22
                ax.plot([r.lo, r.hi], [ypos, ypos], "-", color="C0", lw=2)
                ax.plot(r.delta, ypos, "o", color="C0", ms=4)
            ytick.append(i)
            ylab.append(f"{a}\nvs {b}  [{sub[(sub.a==a)&(sub.b==b)].verdict.iloc[0]}]")
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_yticks(ytick)
        ax.set_yticklabels(ylab, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"paired delta in {metric} (per-seed 95% CI, B={args.n_boot})")
        ax.set_title(f"Which wins are real? Paired bootstrap — {metric}")
        fig.savefig(config.FIGURES_DIR / f"bootstrap_forest_{metric}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved tables to {config.TABLES_DIR} and forest plots to "
          f"{config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
