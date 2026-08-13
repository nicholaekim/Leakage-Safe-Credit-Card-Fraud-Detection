# Leakage-Safe Credit Card Fraud Detection

Ensemble learning, probability calibration, and explainable diagnostics on the
ULB credit-card fraud dataset — with a deliberate focus on **doing the
evaluation honestly**. The project's spine is a controlled comparison between a
*leakage-safe* pipeline and a *leaky* one, showing how the common shortcuts
(resampling before splitting, scaling on all data, tuning the threshold on the
test set) inflate the reported numbers.

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
python -m experiments.run_benchmark            # E1/E2 leaky-vs-safe, in dollars
python -m experiments.run_policy_ladder        # E3 decision policies x calibration
python -m experiments.run_leakage_forensics   # E4 sin-by-sin 2^4 ablation
python -m experiments.run_bootstrap           # E5 which wins are real (needs E1)
python -m experiments.run_benchmark --split temporal   # E6 out-of-time check
python -m experiments.run_explain             # E7 SHAP + error economics
python -m experiments.run_tuning              # E8 leakage-safe hyperparameter search
python -m experiments.run_fee_sensitivity     # extra: savings under $1-$20 alert fees
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
  run_bootstrap.py           E5: paired-bootstrap "which wins are real"
  run_explain.py             E7: SHAP + permutation importance + error economics
  run_tuning.py              E8: leakage-safe RandomizedSearchCV
notebooks/
  01_eda.ipynb      exploratory data analysis (start here after download)
data/  results/  reports/   (git-ignored content)
```

## Dataset

ULB / Worldline credit-card transactions: 284,807 transactions, 492 frauds
(**0.172%**). Features are `Time`, `V1..V28` (PCA-anonymised), `Amount`,
`Class`. Collected over two days. Not redistributed here — download via the
script above.
