"""Generate synthetic A/B data with a known true effect (used for validation)."""
import numpy as np
import pandas as pd


def simulate_ab(n_per_group, metric="binary", baseline_rate=0.10, lift=0.02,
                mean=100.0, std=20.0, seed=None) -> pd.DataFrame:
    """Rows are shuffled so they act like users arriving over time (needed for sequential tests)."""
    rng = np.random.default_rng(seed)
    if metric == "binary":
        if not 0 < baseline_rate + lift < 1:
            raise ValueError("baseline_rate + lift must stay between 0 and 1")
        a = rng.binomial(1, baseline_rate, n_per_group)
        b = rng.binomial(1, baseline_rate + lift, n_per_group)
    else:
        a = rng.normal(mean, std, n_per_group)
        b = rng.normal(mean + lift, std, n_per_group)
    df = pd.DataFrame({"group": ["A"] * n_per_group + ["B"] * n_per_group,
                       "outcome": np.concatenate([a, b])})
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)
