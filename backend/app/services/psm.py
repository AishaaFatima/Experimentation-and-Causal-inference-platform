"""Propensity score matching (nearest neighbour on the logit of the score, with caliper)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .causal_utils import binary_treatment, check_columns, numeric_frame


def _design(X: pd.DataFrame, squares: bool, zero_flags: bool) -> pd.DataFrame:
    """Flexible propensity model: add squares of continuous columns and 'is zero' flags."""
    D = X.copy()
    for c in X.columns:
        binary = set(X[c].unique()) <= {0, 1}
        if squares and not binary:
            D[f"{c}__sq"] = X[c] ** 2
        if zero_flags and not binary and (X[c] == 0).any() and (X[c] >= 0).all():
            D[f"{c}__zero"] = (X[c] == 0).astype(float)
    return D


def fit_propensity(X: pd.DataFrame, t: np.ndarray, squares=True, zero_flags=True) -> np.ndarray:
    D = _design(X, squares, zero_flags)
    Z = StandardScaler().fit_transform(D)
    model = LogisticRegression(C=1.0, tol=1e-9, max_iter=20000).fit(Z, t)   # fully converged, so results do not depend on the solver or machine
    return np.clip(model.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)


TIE_TOL = 1e-9   # propensity scores closer than this (on the logit scale) count as tied


def match_att(y, t, ps, caliper_sd=0.2, k=1):
    """Match each treated unit to its k nearest controls (with replacement) on logit(ps), with a caliper.
    Controls tied with the k-th nearest are ALL used and averaged. In data with many identical profiles, ties are
    common, and picking one by sort order would make the answer depend on the computer and library version.
    Returns ATT, per-treated effects, (treated index, matched control indices) pairs, and the number dropped."""
    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd * logit.std() if caliper_sd else np.inf
    ti, ci = np.where(t == 1)[0], np.where(t == 0)[0]
    effects, matches, dropped = [], [], 0
    for i in ti:
        d = np.abs(logit[ci] - logit[i])
        kth = np.partition(d, min(k, len(d)) - 1)[min(k, len(d)) - 1]
        if kth > caliper:
            dropped += 1
            continue
        near = d <= kth + TIE_TOL
        chosen = ci[near & (d <= caliper)]
        matches.append((i, chosen))
        effects.append(y[i] - y[chosen].mean())
    if not effects:
        raise ValueError("No treated units could be matched; widen the caliper or check overlap")
    return float(np.mean(effects)), np.array(effects), matches, dropped


def balance_table(X: pd.DataFrame, t: np.ndarray, matches) -> list[dict]:
    """Standardized mean differences before and after matching (|SMD| < 0.1 is the usual 'balanced' bar)."""
    treated_all = t == 1
    treated_idx = np.array([i for i, _ in matches])
    w = np.zeros(len(t))                       # control weights after matching
    for _, chosen in matches:
        w[chosen] += 1.0 / len(chosen)
    rows = []
    for c in X.columns:
        x = X[c].to_numpy(float)
        sd = np.sqrt((x[treated_all].var(ddof=1) + x[~treated_all].var(ddof=1)) / 2)
        sd = sd if sd > 0 else 1.0
        before = (x[treated_all].mean() - x[~treated_all].mean()) / sd
        after = (x[treated_idx].mean() - np.average(x[w > 0], weights=w[w > 0])) / sd
        rows.append({"covariate": c, "smd_before": float(before), "smd_after": float(after),
                     "mean_treated": float(x[treated_all].mean()),
                     "mean_control_before": float(x[~treated_all].mean()),
                     "mean_control_after": float(np.average(x[w > 0], weights=w[w > 0]))})
    return rows


def _histogram(ps, t, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    return [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]),
             "treated": int(((ps[t == 1] >= edges[i]) & (ps[t == 1] < edges[i + 1] + (i == bins - 1))).sum()),
             "control": int(((ps[t == 0] >= edges[i]) & (ps[t == 0] < edges[i + 1] + (i == bins - 1))).sum())}
            for i in range(bins)]


def psm_analysis(df, treat_col, outcome_col, covariates, caliper_sd=0.05, k=1, n_boot=200,
                 alpha=0.05, seed=0, squares=False, zero_flags=False) -> dict:
    if not covariates:
        raise ValueError("Pick at least one covariate to match on")
    check_columns(df, [outcome_col])
    t = binary_treatment(df, treat_col)
    X = numeric_frame(df, covariates)
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Outcome '{outcome_col}' has missing or non-numeric values")
    y = y.to_numpy(float)

    ps = fit_propensity(X, t, squares, zero_flags)
    att, effects, matches, dropped = match_att(y, t, ps, caliper_sd, k)
    balance = balance_table(X, t, matches)

    # bootstrap: re-estimate the propensity model and the matching on each resample
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        tb = t[idx]
        if tb.sum() < 5 or (1 - tb).sum() < 5:
            continue
        try:
            psb = fit_propensity(X.iloc[idx].reset_index(drop=True), tb, squares, zero_flags)
            boots.append(match_att(y[idx], tb, psb, caliper_sd, k)[0])
        except ValueError:
            continue
    boots = np.array(boots)
    if len(boots) < 20:
        raise ValueError("Bootstrap failed too often; check overlap between groups")
    se = float(boots.std(ddof=1))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = float(2 * stats.norm.sf(abs(att / se))) if se > 0 else None

    naive = float(y[t == 1].mean() - y[t == 0].mean())
    max_after = max(abs(r["smd_after"]) for r in balance)
    max_before = max(abs(r["smd_before"]) for r in balance)
    used = len({j for _, ch in matches for j in ch})
    sig = bool(p is not None and p < alpha)
    summary = (
        f"The naive comparison says treated units differ from the comparison group by {naive:+,.0f}. "
        f"After matching each treated unit to similar comparison units, the estimated effect on the treated is "
        f"{att:+,.0f} ({1 - alpha:.0%} CI {lo:+,.0f} to {hi:+,.0f}), which is "
        f"{'statistically significant' if sig else 'not statistically significant'}. "
        f"Matching shrank the largest covariate imbalance from {max_before:.2f} to {max_after:.2f} "
        f"(under 0.10 is considered balanced)"
        f"{'.' if max_after < 0.1 else ', so some imbalance remains and the estimate should be treated with caution.'}"
    )
    return {
        "method": "propensity_score_matching", "att": att, "naive_difference": naive,
        "std_error": se, "ci_lower": float(lo), "ci_upper": float(hi), "p_value": p, "alpha": alpha,
        "n_treated": int((t == 1).sum()), "n_treated_matched": len(matches), "n_treated_dropped": dropped,
        "n_controls_used": used, "n_boot_used": int(len(boots)),
        "max_abs_smd_before": float(max_before), "max_abs_smd_after": float(max_after),
        "balanced": bool(max_after < 0.1), "balance": balance, "ps_histogram": _histogram(ps, t),
        "settings": {"caliper_sd": caliper_sd, "n_neighbors": k, "squares": squares, "zero_flags": zero_flags},
        "summary": summary,
    }


def psm_sensitivity(df, treat_col, outcome_col, covariates, caliper_sd=0.05, k=1) -> dict:
    """Re-run matching under several propensity-model specifications.
    Observational estimates often depend heavily on this choice, so we report the spread."""
    t = binary_treatment(df, treat_col)
    X = numeric_frame(df, covariates)
    check_columns(df, [outcome_col])
    y = pd.to_numeric(df[outcome_col]).to_numpy(float)
    rows = []
    for name, sq, zf in [("linear terms only", False, False), ("+ zero-earnings flags", False, True),
                         ("+ squared terms", True, False), ("+ squares and zero flags", True, True)]:
        ps = fit_propensity(X, t, sq, zf)
        att, _, matches, dropped = match_att(y, t, ps, caliper_sd, k)
        bal = max(abs(r["smd_after"]) for r in balance_table(X, t, matches))
        rows.append({"specification": name, "squares": sq, "zero_flags": zf, "att": att,
                     "max_abs_smd_after": float(bal), "balanced": bool(bal < 0.1), "n_dropped": dropped})
    atts = [r["att"] for r in rows]
    nb = sum(r["balanced"] for r in rows)
    return {"specifications": rows, "att_min": min(atts), "att_max": max(atts), "n_balanced": nb,
            "summary": (f"Across {len(rows)} reasonable model choices the estimated effect ranges from "
                        f"{min(atts):+,.0f} to {max(atts):+,.0f}. Only {nb} of {len(rows)} made the groups comparable "
                        f"(worst imbalance under 0.10), so give the others less weight. A wide range means the result "
                        f"depends on modelling choices: report a range, not a single number.")}
