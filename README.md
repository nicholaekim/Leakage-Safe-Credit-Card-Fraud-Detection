# Leakage-Safe Credit Card Fraud Detection

**Team:** Nicholas Kim · Balaji S Kumar · Vivekkumar Chaudhari
(EECS 3404 Applied Machine Learning, York University, Summer 2026)

Ensemble learning, probability calibration, and explainable diagnostics on the
ULB credit-card fraud dataset — with a deliberate focus on **doing the
evaluation honestly**. The project's spine is a controlled comparison between a
*leakage-safe* pipeline and a *leaky* one, showing how the common shortcuts
(resampling before splitting, scaling on all data, tuning the threshold on the
test set) inflate the reported numbers.

## Presentation Video Link
https://youtu.be/A1Avqo15IjE 

---

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate

# or python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt          # or: make setup
#    to reproduce reported numbers exactly, use the pinned versions instead:
#    pip install -r requirements.lock     # (Python 3.11.3)

# 2. data  (needs a Kaggle API token at ~/.kaggle/kaggle.json)
bash scripts/download_data.sh            # or: make data
#    ...or download manually from
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    and drop creditcard.csv into data/raw/

# 3. sanity-check the pipeline end-to-end (seconds)
python -m experiments.run_benchmark --quick   # or: make quick

# 4. the full experiment suite in one command (E1-E8 in order, long run)
make suite

#    ...or run the experiments individually (each saves CSVs to results/tables/)
#    E1-E4 and E6-E8 are self-contained; E5 and the fee sweep read the
#    per-seed score files that run_benchmark caches, so run E1 first
python -m experiments.run_benchmark            # E1/E2 leaky-vs-safe, in dollars (5 seeds)
python -m experiments.run_policy_ladder        # E3 decision policies x calibration (5 seeds)
python -m experiments.run_leakage_forensics   # E4 sin-by-sin 2^4 ablation (5 seeds)
python -m experiments.run_bootstrap           # E5 discovery: which wins are real (seeds 0-2, NEEDS E1's score files)
python -m experiments.run_bootstrap --seeds 3 4 --tag confirm \
    --pairs class_weight__random_forest:smote__random_forest
                                              # E5 confirmation: pre-registered money-gap
                                              # pair re-tested on held-back seeds 3-4 (NEEDS E1)
python -m experiments.run_benchmark --split temporal   # E6 out-of-time check (5 seeds)
python -m experiments.run_explain             # E7 SHAP + error economics (showcase, seed 0)
python -m experiments.run_tuning              # E8 leakage-safe hyperparameter search (seed 0)
python -m experiments.run_fee_sensitivity     # extra: savings under $1-$20 alert fees (NEEDS E1)
```

Results land in `results/tables/` (CSV) and `results/figures/` (PNG).

---

## Repository map

```
src/                core library (import as `from src import ...`)
  config.py         paths, columns, seeds, cost model — all knobs in one place
  data.py           load, integrity checks, stratified + temporal splits
  pipeline.py       leakage-safe Scaler -> [resampler] -> classifier
  models.py         LogReg, RandomForest, HistGBM, XGBoost, LightGBM
  evaluate.py       PR-AUC, precision/recall@k, money costs, savings, thresholds
  calibrate.py      Platt/Isotonic, Brier, ECE (uniform + adaptive), reliability
  explain.py        permutation importance, SHAP
  stats.py          paired bootstrap: shared draws, delta CIs, verdicts
  plots.py          reliability diagram helpers
experiments/
  run_benchmark.py           E1/E2: leaky-vs-safe + money benchmark (+ --split temporal = E6)
  run_policy_ladder.py       E3: naive/tuned/bayes policies x raw/Platt/isotonic
  run_leakage_forensics.py   E4: 2^4 sin ablation with clean-holdout dual evaluation
  run_bootstrap.py           E5: paired-bootstrap "which wins are real" (discovery + confirm)
  run_explain.py             E7: SHAP + permutation importance + error economics
  run_tuning.py              E8: leakage-safe RandomizedSearchCV
  run_fee_sensitivity.py     extra: deployed-policy savings under $1-$20 alert fees
notebooks/
  01_eda.ipynb      exploratory data analysis (start here after download)
data/  results/  reports/   (git-ignored content)
```

## Dataset

ULB / Worldline credit-card transactions: 284,807 transactions, 492 frauds
(**0.172%**). Features are `Time`, `V1..V28` (PCA-anonymised), `Amount`,
`Class`. Collected over two days. Not redistributed here — download via the
script above.

Before splitting, the pipeline deduplicates on the **model-visible feature
space** (`V1..V28` + `Amount`): the data has 9,144 such duplicate rows —
roughly eight times the 1,081 exact-row duplicates that ordinary
`drop_duplicates()` removes — and any copy straddling the train/test boundary
is leakage. 275,663 transactions (473 frauds) remain.

## Team and contributions

| Member | Contributions |
|---|---|
| **Nicholas Kim** | Designed and wrote the core codebase: the leakage-safe pipeline (`src/`), the money/savings metrics, and the eight experiment scripts (E1–E8); ran the experiments across all seeds, debugged, and verified the results. |
| **Balaji S Kumar** | Sourced the ULB dataset; exploratory data analysis and the data-integrity audit; preprocessing decisions and the stratified + temporal train/validation/test splits used by every experiment. |
| **Vivekkumar Chaudhari** | Environment and dependency management (pinned requirements, Makefile targets), cross-platform build fixes, repository hygiene; independent reproducibility verification from a fresh clone. |

All members took part in the adversarial review loop described in the report
(§6): build → review → fix → verify.
