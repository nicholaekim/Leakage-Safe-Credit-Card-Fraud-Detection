"""Central configuration: paths, columns, seeds, and the cost model.

Keeping every knob here makes runs reproducible and keeps the team from
hard-coding magic numbers across notebooks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RAW_CSV = RAW_DIR / "creditcard.csv"

# --- Columns -----------------------------------------------------------------
TARGET = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
V_COLS = [f"V{i}" for i in range(1, 29)]

# Time is used for the *temporal split* and for EDA, but is excluded from the
# model features by default: the raw elapsed-seconds value encodes a
# transaction's position in the 2-day collection window. Under a temporal
# split every test `Time` is larger than any training `Time`, so the model
# cannot generalise on it (and it acts as a giveaway of the split). Flip this
# to include it and discuss the difference as an ablation.
FEATURES = V_COLS + [AMOUNT_COL]

# --- Reproducibility ---------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]

# --- Split fractions (of the full dataset) -----------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20   # train = 1 - TEST_SIZE - VAL_SIZE

# --- Cost model for threshold selection --------------------------------------
# Missing a fraud (FN) is far costlier than a false alarm (FP). With
# AMOUNT_AWARE=True the FN cost for each transaction is its own `Amount`
# (money lost), and the FP cost stays a fixed investigation cost.
COST_FP = 1.0
COST_FN = 100.0
AMOUNT_AWARE = False

# --- Calibration / reliability ----------------------------------------------
N_CALIB_BINS = 10


def ensure_dirs():
    for d in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
