# Leakage-Safe Credit Card Fraud Detection

Ensemble learning, probability calibration, and explainable diagnostics on the
ULB credit-card fraud dataset — with a deliberate focus on **doing the
evaluation honestly**. The project's spine is a controlled comparison between a
*leakage-safe* pipeline and a *leaky* one, showing how the common shortcuts
(resampling before splitting, scaling on all data, tuning the threshold on the
test set) inflate the reported numbers.

See **[PROJECT_PLAN.md](PROJECT_PLAN.md)** for the full methodology, experiment
list, work split, and report/video structure.

---

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: make setup

# 2. data  (needs a Kaggle API token at ~/.kaggle/kaggle.json)
bash scripts/download_data.sh            # or: make data
#    ...or download manually from
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    and drop creditcard.csv into data/raw/

# 3. sanity-check the pipeline end-to-end (seconds)
python -m experiments.run_benchmark --quick   # or: make quick

# 4. full benchmark (leaky vs safe, all models, all seeds)
python -m experiments.run_benchmark           # or: make bench
```

Results land in `results/tables/` (CSV) and `results/figures/` (PNG).

### Google Colab

```python
!git clone <your-repo-url> && cd <repo>
!pip install -r requirements.txt
from google.colab import files; files.upload()        # upload kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!bash scripts/download_data.sh
!python -m experiments.run_benchmark --quick
```

---

## Repository map

```
src/                core library (import as `from src import ...`)
  config.py         paths, columns, seeds, cost model — all knobs in one place
  data.py           load, integrity checks, stratified + temporal splits
  pipeline.py       leakage-safe Scaler -> [resampler] -> classifier
  models.py         LogReg, RandomForest, HistGBM, XGBoost, LightGBM
  evaluate.py       PR-AUC, precision/recall@k, cost-sensitive threshold
  calibrate.py      Platt/Isotonic, Brier, ECE, reliability curve
  explain.py        permutation importance, SHAP
  plots.py          PR curve, reliability diagram helpers
experiments/
  run_benchmark.py  the headline leaky-vs-safe experiment
notebooks/
  01_eda.ipynb      exploratory data analysis (start here after download)
data/  results/  reports/   (git-ignored content)
```

## Dataset

ULB / Worldline credit-card transactions: 284,807 transactions, 492 frauds
(**0.172%**). Features are `Time`, `V1..V28` (PCA-anonymised), `Amount`,
`Class`. Collected over two days. Not redistributed here — download via the
script above.
