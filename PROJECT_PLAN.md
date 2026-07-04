# Project Plan — Leakage-Safe Credit Card Fraud Detection, Measured in Money

**Course:** Applied Machine Learning (EECS 3404) · **Team:** 5 · **Compute:** laptop/Colab
**Deliverables:** final report + presentation video (code & concept walkthrough)

> **Thesis:** We benchmark fraud models in **dollars saved** rather than F1,
> **decompose each classic leakage sin's individual score inflation** with a
> 2⁴ factorial ablation, show via **paired bootstrap** that most model "wins"
> at ~95 test frauds are statistical noise, and demonstrate that
> **probability calibration changes money** under a per-transaction decision
> rule — with every claim interval-qualified and reproducible from raw data.

---

## 1. Problem & motivation

Credit-card fraud is a needle-in-a-haystack, cost-asymmetric problem: frauds
are 0.17% of transactions and a missed fraud costs far more than a false
alarm. Naive ML is actively misleading here for two reasons: (a) extreme
imbalance breaks accuracy and flatters ROC-AUC; (b) data leakage is unusually
easy to commit and silently inflates results. The thousands of public
projects on this dataset overwhelmingly do both. We build the honest version
— and quantify exactly how much each dishonest shortcut lies.

## 2. Dataset

ULB / Worldline (Kaggle `mlg-ulb/creditcardfraud`): 284,807 transactions,
492 frauds (0.172%), features `Time`, `V1..V28` (PCA-anonymised), `Amount`,
`Class`; ~2 days of European card transactions. Not committed to the repo —
`bash scripts/download_data.sh`.

Data-quality findings we discovered and handle:
- 1,081 exact duplicate rows — **and 9,144 duplicate copies in the space the
  model actually sees** (V1–28 + Amount, `Time` excluded). Standard
  `drop_duplicates()` leaves ~5,700 model-identical rows able to straddle a
  split. The forensics experiment uses feature-space identity.
- ~45% of frauds have `Amount` ≤ $5 — economically not worth a $5
  investigation. (This reshapes what a "good" model even means.)
- `Time` is excluded from features (position-in-window giveaway) but drives
  the temporal split.

## 3. What is implemented (one command each)

| Experiment | Command | Closes |
|---|---|---|
| E1 Leaky-vs-safe benchmark, in dollars | `python -m experiments.run_benchmark` | baselines, metrics, money |
| E2 Imbalance ablation (class_weight/SMOTE/undersample/none) | (inside E1) | comparative analysis |
| E3 Decision-policy ladder × calibration | `python -m experiments.run_policy_ladder` | calibration/uncertainty |
| E4 Leakage forensics 2⁴ factorial | `python -m experiments.run_leakage_forensics` | ablation, leakage |
| E5 Paired-bootstrap model comparison | `python -m experiments.run_bootstrap` | experimental rigor |
| E6 Temporal (out-of-time) robustness | `python -m experiments.run_benchmark --split temporal` | drift/robustness |
| E7 Interpretability + error economics | `python -m experiments.run_explain` | interpretation, diagnostics |
| E8 Leakage-safe hyperparameter search | `python -m experiments.run_tuning` | optimization |

All results land in `results/tables/` (CSV) and `results/figures/` (PNG).
`--quick` on E1/E3/E4/E7/E8 gives a fast smoke test.

## 4. Headline findings (3 seeds, full data)

1. **The leaky workflow reports near-perfection; the honest one doesn't.**
   All-sins pipeline: reported PR-AUC ≈ 0.99. Clean-holdout truth: ≈ 0.74.
   The honest workflow's own reported number matches its clean-holdout truth
   to within 0.003 — honesty verified, not assumed.
2. **One sin does almost all the lying.** SMOTE-before-split ≈ the entire
   inflation (+0.25 PR-AUC reported, ±0.03); scaler-on-all ≈ 0;
   duplicates ≈ noise; threshold-on-test inflates F1 only (PR-AUC effect is
   zero *by construction* — used as a built-in consistency check).
3. **The same sin ships a broken product.** Its threshold, tuned at synthetic
   50% prevalence, collapses at real prevalence: true F1 0.07 vs 0.95
   reported (a 13.5× overstatement) — two distinct harms, separately
   quantified (poisoned test set vs broken operating point).
4. **Money disagrees with F1.** Ranked by savings vs a do-nothing baseline,
   the F1 runner-up (SMOTE+RF) finished last of 20 configs: it catches many
   frauds but the cheap ones (misses ~56% more fraud *value* than the money
   winner). Ranked by dollars: class_weight+RF ≈ 0.69 savings; oracle ceiling
   0.97; block-everything −24.
5. **Calibration is worth money in proportion to miscalibration.**
   Class-weighted LogReg's raw "probabilities" are inflated ~300×; under the
   per-transaction rule (*flag iff p×Amount > $5*) they lose 3.5× the fraud
   value (savings −3.56); Platt scaling repairs the same model to +0.67 —
   ≈ $48k per test slice. SMOTE+RF: +$4.8k. Already-calibrated LightGBM: $0.
6. **Most "wins" are noise — but our money gap survives confirmation.**
   Paired bootstrap (B=1000, shared draws): every PR-AUC pairwise comparison
   is indistinguishable at 95%, and the PR-AUC leader even flips between seed
   sets (class_weight/LightGBM on seeds 0–2, SMOTE/LightGBM on 3–4) — the
   leaderboard crown changes with the seed. Our headline money gap
   (class_weight/RF vs SMOTE/RF, finding 4) was only directional on the
   discovery seeds 0–2 (CI touched zero), so we pre-registered it and
   re-tested on fresh seeds 3–4: both CIs exclude zero → **confirmed on
   held-out data** (seed 3 [+0.084, +0.227], seed 4 [+0.058, +0.167]).
   Discovery → pre-register → confirm is exactly what 95 heavy-tailed test
   frauds demand, and we ran it.
7. **Even a perfect oracle shouldn't flag everything**: flagging all fraud
   yields 0.958 savings; skipping frauds worth less than the fee yields
   0.972. The Bayes rule discovers this on its own (flags 0 of the ≤$5
   frauds).

## 5. Methodology guardrails (the checklist the report defends)

- Split first; every learned transform (scaler, SMOTE) inside an
  `imblearn.Pipeline` → fit on train folds only.
- Feature-space deduplication before splitting; duplicate-aware clean holdout.
- Threshold + calibration fit on validation only; test touched once.
- PR-AUC primary (accuracy reported only as "the misleading metric");
  precision/recall@k; example-dependent money costs (Bahnsen-style savings).
- Multi-seed mean ± sd everywhere; paired bootstrap for comparisons; explicit
  post-selection caveat (leader chosen on the same data ⇒ exploratory;
  "indistinguishable" is the trustworthy direction).
- Adversarial review of our own code found and fixed real flaws (documented
  in the report's methodology section): a test-set calibrator selection, a
  misspecified Platt implementation, a contaminated holdout definition, an
  interpolation-space confound, an untestable "hypothesis".

## 6. Limitations (owned, not hidden)

- PCA-anonymised features → importances name components, not causes; no
  demographic fairness analysis is possible.
- 2-day window → the temporal split is a robustness check, **not** a concept-
  drift study; drift claims are out of scope.
- ~95 test frauds → wide intervals on threshold-dependent and heavy-tailed
  metrics; we quantify rather than hide this.
- Cost model ($5/alert; missed fraud = its Amount) is an assumption; savings
  conclusions should be read against it (config knob: `C_ALERT`).
- Comparisons are post-hoc (see §4.6); confirmatory path = pin pairs, rerun
  on fresh seeds (`--seeds 3 4`).

## 7. Team roles (5)

| # | Owner area | Presents |
|---|---|---|
| 1 | Data, splits, leakage forensics (E4) | findings 1–3 |
| 2 | Benchmark + money metrics (E1/E2) | finding 4 |
| 3 | Calibration + policy ladder (E3) | findings 5, 7 |
| 4 | Statistics: bootstrap + variance (E5) | finding 6 |
| 5 | Interpretability, temporal, tuning (E6–E8); report/video lead | diagnostics + limitations |

## 8. Report & video structure

Problem → Data (incl. the 9,144-duplicates discovery) → Methodology
guardrails → E1/E2 money benchmark → E4 forensics (centerpiece) → E3
calibration-in-dollars → E5 which-wins-are-real → E6–E8 diagnostics → Limitations
→ Reproducibility. Video: 8–12 min code-and-concept walkthrough; lead with the
leaky-vs-honest scoreboard, end with "we stress-tested our own headline".

## 9. Reproducibility

Pinned `requirements.txt`; every figure/table regenerated by a script; fixed
seeds; data via one download script; `make quick` smoke-tests the pipeline in
~40 s. Repo map in [README.md](README.md).
