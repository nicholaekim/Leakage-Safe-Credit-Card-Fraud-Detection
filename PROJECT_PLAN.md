# Project Plan — Leakage-Safe Credit Card Fraud Detection

**Course:** Applied Machine Learning · **Team:** 5 · **Compute:** Colab / local
**Deliverables:** final report + presentation video (code & concept walkthrough)

> One-line thesis: *We build an honest fraud detector and prove it is honest —
> by quantifying how much standard methodological shortcuts inflate the
> reported performance, then reporting the leakage-safe numbers with proper
> imbalance metrics, calibration, and interpretation.*

---

## 1. Problem & motivation

Credit-card fraud is a needle-in-a-haystack, cost-asymmetric problem: frauds
are ~0.17% of transactions, and a missed fraud costs far more than a false
alarm. Two things make naive ML actively misleading here:

1. **Extreme class imbalance** breaks accuracy and even ROC-AUC as metrics.
2. **Data leakage** is unusually easy to commit (resampling, scaling, temporal
   ordering, threshold tuning) and silently inflates results — exactly the
   failure mode this course asks us to detect and avoid.

Our goal is a detector that is *accurate where it counts* (top of the ranked
list), *calibrated* (a 0.9 score means ~90% fraud), *explainable*, and above
all *honestly evaluated*.

## 2. Dataset

ULB / Worldline (Kaggle `mlg-ulb/creditcardfraud`): 284,807 transactions, 492
frauds (**0.172%**), features `Time`, `V1..V28` (PCA-anonymised), `Amount`,
`Class`; ~2 days of European card transactions, Sept 2013.

Known properties we handle explicitly:
- Exact **duplicate rows** → dropped before splitting (`data.remove_duplicates`).
- `Time` is elapsed seconds → used for the **temporal split**, excluded from
  model features by default (see `config.FEATURES` rationale).
- PCA-anonymised features → a real **interpretability ceiling** we discuss, not
  hide.

## 3. Objectives & success criteria

| # | Objective | Evidence |
|---|-----------|----------|
| O1 | Quantify leakage impact | Δ(PR-AUC, recall) between leaky and safe pipelines |
| O2 | Compare ensembles fairly | PR-AUC ± std over 5 seeds, leakage-safe |
| O3 | Handle imbalance correctly | ablation: class_weight vs SMOTE vs undersample vs none |
| O4 | Calibrate probabilities | Brier / ECE before vs after, reliability diagrams |
| O5 | Explain the model | SHAP global + local, permutation importance, error analysis |
| O6 | Show robustness/drift | stratified vs temporal split; day-1 → day-2 degradation |

## 4. Methodology

### 4.1 Leakage taxonomy (the spine — Experiment E1)

We run identical models two ways and report the gap.

| Leak | Wrong way (leaky) | Our safeguard |
|------|-------------------|---------------|
| Resampling | SMOTE on full data before split | sampler **inside** `imblearn.Pipeline`, train-fold only |
| Scaling | `StandardScaler.fit` on all data | scaler inside the pipeline |
| Temporal | random k-fold mixes future→past | report a **temporal split** alongside stratified |
| Duplicates | duplicates straddle train/test | drop before splitting |
| Threshold | threshold/calibration chosen on test | tuned on a held-out **validation** set |

`experiments/run_benchmark.py` implements both `run_safe` and `run_leaky`.
Expected result: leaky reports look *better* and are meaningless.

### 4.2 Validation design

- **Splits:** stratified 60/20/20 train/val/test **and** a temporal 60/20/20.
- **Model selection:** stratified k-fold CV *within train*, every transform
  inside the pipeline (no leakage across folds).
- **Threshold & calibration:** fit on validation, report on test.
- **Uncertainty:** repeat over `SEEDS = [0..4]`; report **mean ± std** (error
  bars on every headline metric).

### 4.3 Models & baselines

Trivial baseline (predict majority / `Amount` rule) → **LogisticRegression**
(linear reference) → **RandomForest** (bagging) → **HistGradientBoosting** →
**XGBoost** → **LightGBM** (boosting). An ensemble must beat LogReg on PR-AUC
to justify itself.

### 4.4 Imbalance handling (Experiment E2 — ablation)

`class_weight` vs `SMOTE` vs `RandomUnderSampler` vs nothing, holding model and
split fixed. Hypothesis to test: SMOTE can lift recall but **degrades
calibration** (it changes the base rate) — we measure both.

### 4.5 Metrics (imbalance-correct)

- **Primary:** PR-AUC / average precision.
- **Secondary:** ROC-AUC (reported but flagged optimistic), precision/recall/F1
  at the chosen threshold, **precision@k / recall@k** (analyst review budget).
- **Operational:** expected **cost** under a cost matrix (FN ≫ FP, optionally
  Amount-aware); threshold chosen to minimise cost on validation.

### 4.6 Calibration & uncertainty (Experiment E3)

Raw scores vs **Platt** vs **Isotonic** (`calibrate.PrefitCalibrator`, fit on
validation). Report Brier, ECE, and reliability diagrams before/after. Tie to
Dal Pozzolo et al. (2015), the dataset authors, on calibration under
undersampling.

### 4.7 Explainability (Experiment E4)

- **Global:** permutation importance (PR-AUC drop) + SHAP summary; note the
  PCA-anonymisation limit on human interpretation.
- **Local:** SHAP waterfalls for a true positive, a false positive, and a
  **missed fraud (FN)** — explain *why* each was scored as it was.
- **Error analysis:** characterise FNs/FPs vs `Amount` and `Time`.

### 4.8 Drift & robustness (Experiment E5)

Temporal split = train on day 1, test on day 2. Measure performance
degradation vs the stratified split and discuss adversarial concept drift in
fraud (fraudsters adapt; yesterday's model decays).

### 4.9 Bias, limitations, threats to validity

PCA features prevent demographic fairness analysis (no protected attributes)
and limit interpretability — stated as a limitation. Single dataset, single
2-day window → limited drift evidence. Cost matrix values are assumptions →
report sensitivity. (Optional extension: re-run the methodology on IEEE-CIS for
real features and a fairness lens.)

## 5. Experiments → artifacts

| ID | Experiment | Output |
|----|-----------|--------|
| E1 | Leaky vs safe | PR-AUC/recall gap table + bar chart |
| E2 | Imbalance ablation | metric × strategy heatmap |
| E3 | Calibration | reliability diagrams, Brier/ECE table |
| E4 | Explainability | SHAP summary + 3 local explanations, error analysis |
| E5 | Stratified vs temporal (drift) | side-by-side metric table |

## 6. Repository layout

See [README.md](README.md#repository-map). Core rule: all learning transforms
live inside `pipeline.make_safe_pipeline`, so CV, calibration, and permutation
importance inherit leakage-safety for free.

## 7. Reproducibility

Fixed seeds; all config in `src/config.py`; pinned `requirements.txt`;
deterministic pipelines; results written to `results/`; data and artifacts
git-ignored with a documented download path. Every figure/table is regenerated
by a script or notebook — no manual numbers.

## 8. Team roles (5)

Each owner writes their module(s), notebook section, and the matching report
section. Everyone contributes to the video.

| # | Owner area | Code | Report / figures |
|---|------------|------|------------------|
| 1 | **Data & Leakage** | `data.py`, `pipeline.py`, `run_benchmark.run_leaky` | §4.1–4.2, E1 |
| 2 | **Modeling & Ensembles** | `models.py`, tuning, E2 ablation | §4.3–4.4, E2 |
| 3 | **Evaluation & Metrics** | `evaluate.py`, multi-seed aggregation | §4.5, E1/E5 tables |
| 4 | **Calibration & Uncertainty** | `calibrate.py` | §4.6, E3 |
| 5 | **Explainability & Report** | `explain.py`, `plots.py`, error analysis | §4.7, E4 + report/video lead |

## 9. Timeline (set the deadline, then back-fill)

| Phase | Work | Owner(s) |
|-------|------|----------|
| 1 | Setup, data download, EDA (`01_eda.ipynb`) | all |
| 2 | Leakage-safe pipeline + E1 leaky-vs-safe | 1, 3 |
| 3 | Model zoo + E2 imbalance ablation | 2, 3 |
| 4 | Calibration (E3) + drift (E5) | 4, 1 |
| 5 | Explainability (E4) + error analysis | 5 |
| 6 | Report writing + presentation video | all (5 leads) |

> **TODO:** drop the assignment due date here and assign calendar dates.

## 10. Report & video structure (maps to the grading pillars)

1. Problem & motivation → 2. Data & preprocessing (dedup, splits, leakage
   safeguards) → 3. Methodology (validation, models, metrics) → 4. Results
   (E1–E5 with error bars) → 5. Calibration & uncertainty → 6. Interpretation &
   error analysis → 7. Limitations, drift, bias → 8. Conclusion &
   reproducibility. Video: 8–12 min walking through the code modules and the
   leaky-vs-safe result as the headline.

## 11. Risks & mitigations

- *RandomForest/SMOTE slow on full data* → develop with `--quick`, run full on
  Colab; cache results to `results/`.
- *Isotonic overfits with ~470 frauds* → prefer Platt where data is thin; show
  both.
- *Scope creep* → ULB end-to-end first; IEEE-CIS only if E1–E5 are done.
