"""data loading, sanity checks and splitting.

two splits on purpose: stratified_split is the usual random class balanced
one, temporal_split trains on the earliest transactions and tests on the
latest (closer to how this would actually get deployed).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download the ULB dataset first:\n"
            "    bash scripts/download_data.sh\n"
            "or place creditcard.csv in data/raw/ (see README.md)."
        )
    return pd.read_csv(path)


def integrity_report(df: pd.DataFrame) -> dict:
    """quick sanity numbers, printed at the top of every run"""
    n = len(df)
    pos = int(df[config.TARGET].sum())
    return {
        "n_rows": n,
        "n_fraud": pos,
        "fraud_rate": pos / n,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_features": len(config.FEATURES),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """drop exact duplicate rows before splitting, otherwise identical
    transactions can end up on both sides of the split"""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    out.attrs["n_dropped_duplicates"] = before - len(out)
    return out


def xy(df: pd.DataFrame):
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def _pack(train_df, val_df, test_df) -> dict:
    out = {}
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        X, y = xy(d)
        out[name] = {"X": X, "y": y, "amount": d[config.AMOUNT_COL].to_numpy()}
    return out


def stratified_split(df, seed, test_size=None, val_size=None) -> dict:
    """stratified train/val/test. returns a dict per split with X, y and the
    raw amount column (needed for the money metrics)."""
    test_size = config.TEST_SIZE if test_size is None else test_size
    val_size = config.VAL_SIZE if val_size is None else val_size

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=seed
    )
    # rescale so val ends up as val_size of the original df
    val_rel = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel, stratify=train_df[config.TARGET],
        random_state=seed,
    )
    return _pack(train_df, val_df, test_df)


def temporal_split(df, test_size=None, val_size=None) -> dict:
    """time ordered split, no shuffling. train = earliest, test = latest"""
    test_size = config.TEST_SIZE if test_size is None else test_size
    val_size = config.VAL_SIZE if val_size is None else val_size

    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    return _pack(
        df.iloc[:n_train],
        df.iloc[n_train:n_train + n_val],
        df.iloc[n_train + n_val:],
    )
