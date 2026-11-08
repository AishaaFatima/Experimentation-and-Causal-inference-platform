"""Helpers for turning (dataframe, group column, outcome column) into two arrays."""
import numpy as np
import pandas as pd


def split_groups(df: pd.DataFrame, group_col: str, outcome_col: str, control_label=None):
    for c in (group_col, outcome_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    sub = df[[group_col, outcome_col]].dropna()
    levels = sorted(sub[group_col].unique(), key=str)
    if len(levels) != 2:
        raise ValueError(f"'{group_col}' must have exactly 2 groups, found {len(levels)}: {[str(l) for l in levels][:10]}")
    if control_label is None:
        control = levels[0]
    else:
        control = next((l for l in levels if str(l) == str(control_label)), None)
        if control is None:
            raise ValueError(f"control_label '{control_label}' not in {[str(l) for l in levels]}")
    treat = next(l for l in levels if l != control)
    try:
        y = pd.to_numeric(sub[outcome_col], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"Outcome column '{outcome_col}' must be numeric")
    mask = (sub[group_col] == control).to_numpy()
    return y[mask], y[~mask], str(control), str(treat)
