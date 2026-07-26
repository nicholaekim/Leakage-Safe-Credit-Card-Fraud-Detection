"""all the settings in one place - paths, columns, seeds, cost model"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SCORES_DIR = RESULTS_DIR / "scores"     # per-seed test score arrays for the bootstrap

RAW_CSV = RAW_DIR / "creditcard.csv"

# columns
TARGET = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
V_COLS = [f"V{i}" for i in range(1, 29)]

# time is left out of the features on purpose. its just elapsed seconds in a
# 2 day window so it doesnt generalise, and it gives away the temporal split.
# still used for the temporal split itself and eda
FEATURES = V_COLS + [AMOUNT_COL]

# seeds for repeat runs
SEEDS = [0, 1, 2, 3, 4]

# split fractions (of the full dataset), train gets whats left
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# basic cost weights for the old cost objective. missing a fraud is way worse
# than a false alarm. amount_aware=True makes each fn cost its own amount
COST_FP = 1.0
COST_FN = 100.0
AMOUNT_AWARE = False

# money metric: every alert (tp or fp) costs a flat fee, every missed fraud
# costs its own amount. savings = 1 - cost/do_nothing_cost, so 1 is perfect,
# 0 is useless and negative means the alerts cost more than the fraud
C_ALERT = 5.0

# calibration bins
N_CALIB_BINS = 10


def ensure_dirs():
    for d in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SCORES_DIR):
        d.mkdir(parents=True, exist_ok=True)
