"""Shared helpers for observational (non-randomized) methods."""
import numpy as np
import pandas as pd


def check_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")


def binary_treatment(df: pd.DataFrame, col: str) -> np.ndarray:
    check_columns(df, [col])
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.isna().any() or not set(vals.unique()) <= {0, 1} or vals.nunique() != 2:
        raise ValueError(f"Treatment column '{col}' must contain only 0 and 1 (both present)")
    return vals.to_numpy(dtype=int)


def numeric_frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    check_columns(df, cols)
    out = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Columns {bad} contain missing or non-numeric values; clean them first")
    return out
