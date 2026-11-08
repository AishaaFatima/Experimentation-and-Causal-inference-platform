"""Two-sample tests: Welch t-test (means) and chi-square / Fisher (proportions)."""
import numpy as np
from scipy import stats

from .fmt import fmt_num


def welch_t_test(control, treatment, alpha=0.05) -> dict:
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    n1, n2 = len(c), len(t)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group needs at least 2 observations")
    m1, m2 = c.mean(), t.mean()
    v1, v2 = c.var(ddof=1), t.var(ddof=1)
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    if se == 0:
        raise ValueError("Both groups have zero variance; test is undefined")
    diff = float(m2 - m1)
    t_stat = diff / se
    df = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = float(2 * stats.t.sf(abs(t_stat), df))
    crit = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = diff - crit * se, diff + crit * se
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    sig = p < alpha
    ref = max(abs(m1), abs(m2), abs(lo), abs(hi), abs(diff)) or 1.0
    f = lambda v, **k: fmt_num(v, ref=ref, **k)
    summary = (
        f"The treatment group averaged {f(m2)} versus {f(m1)} for control, a difference of "
        f"{f(diff, signed=True)} ({1 - alpha:.0%} CI {f(lo)} to {f(hi)}). With p = {p:.4f}, this is "
        f"{'statistically significant' if sig else 'not statistically significant'} at the {alpha:.0%} level."
    )
    if not sig:
        summary += " That is not proof of no effect: the test may simply lack the power to detect one."
    return {
        "method": "welch_t_test", "n_control": n1, "n_treatment": n2,
        "mean_control": float(m1), "mean_treatment": float(m2),
        "difference": diff, "relative_lift": float(diff / m1) if m1 != 0 else None,
        "std_error": se, "t_statistic": float(t_stat), "df": float(df), "p_value": p,
        "ci_lower": float(lo), "ci_upper": float(hi), "alpha": alpha,
        "cohens_d": float(diff / pooled_sd) if pooled_sd > 0 else None,
        "significant": bool(sig), "summary": summary,
    }


def chi_square_test(control, treatment, alpha=0.05) -> dict:
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    for arr in (c, t):
        if not np.isin(arr, [0, 1]).all():
            raise ValueError("Chi-square needs a binary (0/1) outcome column")
    n1, n2 = len(c), len(t)
    if n1 < 1 or n2 < 1:
        raise ValueError("Each group needs at least 1 observation")
    s1, s2 = int(c.sum()), int(t.sum())
    table = np.array([[s1, n1 - s1], [s2, n2 - s2]])
    if (table.sum(axis=0) == 0).any():
        raise ValueError("Outcome has no variation across both groups; test is undefined")
    chi2, p, _, expected = stats.chi2_contingency(table, correction=False)
    warnings, test_used = [], "chi_square"
    if expected.min() < 5:
        p = float(stats.fisher_exact(table)[1])
        test_used = "fisher_exact"
        warnings.append("Some expected counts are below 5, so Fisher's exact test was used instead of chi-square.")
    p1, p2 = s1 / n1, s2 / n2
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = diff - z * se, diff + z * se
    sig = p < alpha
    summary = (
        f"The conversion rate was {p2:.2%} in treatment versus {p1:.2%} in control, a difference of "
        f"{diff * 100:+.2f} percentage points ({1 - alpha:.0%} CI {lo * 100:+.2f} to {hi * 100:+.2f}). "
        f"With p = {p:.4f}, this is {'statistically significant' if sig else 'not statistically significant'} "
        f"at the {alpha:.0%} level."
    )
    if not sig:
        summary += " That is not proof of no effect: the test may simply lack the power to detect one."
    return {
        "method": test_used, "n_control": n1, "n_treatment": n2,
        "rate_control": p1, "rate_treatment": p2, "difference": float(diff),
        "relative_lift": float(diff / p1) if p1 > 0 else None,
        "chi2": float(chi2), "p_value": float(p),
        "ci_lower": float(lo), "ci_upper": float(hi), "alpha": alpha,
        "contingency_table": table.tolist(), "significant": bool(sig),
        "warnings": warnings, "summary": summary,
    }
