"""Multiple-testing corrections, implemented by hand (verified against statsmodels in tests)."""
import numpy as np


def bonferroni(p_values, alpha=0.05) -> dict:
    p = np.asarray(p_values, float)
    adj = np.minimum(p * len(p), 1.0)
    return {"adjusted_p": adj, "reject": adj <= alpha}


def benjamini_hochberg(p_values, alpha=0.05) -> dict:
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]  # enforce monotonicity
    adj = np.empty(m)
    adj[order] = np.minimum(scaled, 1.0)
    return {"adjusted_p": adj, "reject": adj <= alpha}


def correct(p_values, method="benjamini_hochberg", alpha=0.05, labels=None) -> dict:
    res = bonferroni(p_values, alpha) if method == "bonferroni" else benjamini_hochberg(p_values, alpha)
    labels = labels or [f"test_{i + 1}" for i in range(len(p_values))]
    if len(labels) != len(p_values):
        raise ValueError("labels must have the same length as p_values")
    rows = [{"label": l, "p_value": float(p), "adjusted_p": float(a), "reject_null": bool(r)}
            for l, p, a, r in zip(labels, p_values, res["adjusted_p"], res["reject"])]
    n_naive = int(sum(p < alpha for p in p_values))
    n_rej = int(res["reject"].sum())
    return {
        "method": method, "alpha": alpha, "results": rows,
        "summary": (f"Without correction {n_naive} of {len(rows)} tests would look significant; "
                    f"after {method.replace('_', '-')} correction {n_rej} remain."),
    }
