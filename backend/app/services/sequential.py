"""Group-sequential testing with Lan-DeMets alpha spending.

Boundaries are computed exactly (not approximated) by numerically integrating the
joint distribution of the interim test statistics under H0, using the Brownian-motion
representation: B(t) ~ N(0, t) with independent increments, Z_k = B(t_k) / sqrt(t_k).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .frequentist import welch_t_test
from .groups import split_groups


def resolve_looks(n_looks, looks=None) -> np.ndarray:
    t = np.asarray(looks if looks else [(k + 1) / n_looks for k in range(n_looks)], float)
    if len(t) < 2 or np.any(np.diff(t) < 0.01) or t[0] < 0.01 or abs(t[-1] - 1.0) > 1e-9:
        raise ValueError("looks must be increasing information fractions (gaps >= 0.01) ending at 1.0")
    return t


def spending_function(kind, t, alpha):
    t = np.asarray(t, float)
    if kind == "obrien_fleming":
        # Lan-DeMets O'Brien-Fleming-type spending for a two-sided test: alpha/2 is spent in each tail.
        return 2 * (2 - 2 * norm.cdf(norm.ppf(1 - alpha / 4) / np.sqrt(t)))
    if kind == "pocock":
        return alpha * np.log(1 + (np.e - 1) * t)
    raise ValueError(f"Unknown spending function '{kind}'")


def compute_boundaries(looks, alpha=0.05, kind="obrien_fleming", grid_n=1601, half_width=10.0) -> dict:
    t = np.asarray(looks, float)
    cum = spending_function(kind, t, alpha)
    inc = np.diff(np.concatenate([[0.0], cum]))
    x = np.linspace(-half_width, half_width, grid_n)
    dx = x[1] - x[0]
    mass, t_prev, zs = None, 0.0, []
    for k, (tk, ak) in enumerate(zip(t, inc)):
        sigma = np.sqrt(tk - t_prev)
        if k == 0:
            tail = lambda b: 2 * norm.sf(b / sigma)
            density = norm.pdf(x, scale=sigma)
        else:
            tail = lambda b: float(np.sum(mass * (norm.cdf((-b - x) / sigma) + norm.sf((b - x) / sigma))))
            kernel = norm.pdf((x[:, None] - x[None, :]) / sigma) / sigma
            density = kernel @ mass
        b = brentq(lambda b: tail(b) - ak, 1e-6, half_width - 1)
        mass = density * (np.abs(x) < b) * dx  # probability mass still "in play"
        zs.append(b / np.sqrt(tk))
        t_prev = tk
    zs = np.array(zs)
    return {
        "looks": t.tolist(), "z_bounds": zs.tolist(),
        "nominal_p": (2 * norm.sf(zs)).tolist(), "cumulative_alpha_spent": cum.tolist(),
        "alpha": alpha, "spending": kind,
    }


def analyze_stream(df, group_col, outcome_col, control_label, planned_total_n, looks, alpha, kind) -> dict:
    """Walk through the planned looks, using rows in file order as arrival order."""
    bounds = compute_boundaries(looks, alpha, kind)
    rows, stopped_at = [], None
    for i, (frac, zb) in enumerate(zip(looks, bounds["z_bounds"]), start=1):
        n = int(round(planned_total_n * frac))
        if n > len(df):
            rows.append({"look": i, "info_fraction": float(frac), "n_rows": n, "status": "not_reached"})
            continue
        c, tr, _, _ = split_groups(df.iloc[:n], group_col, outcome_col, control_label)
        res = welch_t_test(c, tr, alpha)
        z = res["t_statistic"]
        crossed = abs(z) >= zb
        rows.append({"look": i, "info_fraction": float(frac), "n_rows": n, "status": "analyzed",
                     "z": z, "z_bound": zb, "difference": res["difference"],
                     "crossed": bool(crossed)})
        if crossed and stopped_at is None:
            stopped_at = i
    analyzed = [r for r in rows if r["status"] == "analyzed"]
    if stopped_at:
        r = rows[stopped_at - 1]
        summary = (f"Stopped at look {stopped_at} of {len(looks)}: |z| = {abs(r['z']):.2f} crossed the "
                   f"boundary of {r['z_bound']:.2f}. The estimated difference is {r['difference']:+,.4g}.")
    elif analyzed:
        r = analyzed[-1]
        summary = (f"No stopping boundary crossed through look {r['look']} of {len(looks)} "
                   f"(|z| = {abs(r['z']):.2f} vs boundary {r['z_bound']:.2f}). Keep collecting data.")
    else:
        summary = "Not enough data has arrived to reach the first planned look."
    return {"bounds": bounds, "looks": rows, "stopped_at_look": stopped_at, "summary": summary}


def peeking_simulation(n_looks, alpha, n_sims, seed) -> dict:
    """Under H0, how often does 'stop at the first significant look' give a false positive?"""
    t = resolve_looks(n_looks)
    rng = np.random.default_rng(seed)
    inc = rng.normal(size=(n_sims, n_looks)) * np.sqrt(np.diff(np.concatenate([[0], t])))
    z = np.cumsum(inc, axis=1) / np.sqrt(t)
    out = {"n_looks": n_looks, "alpha": alpha, "n_sims": n_sims,
           "naive_peeking": float((np.abs(z) > norm.ppf(1 - alpha / 2)).any(axis=1).mean())}
    for kind in ("obrien_fleming", "pocock"):
        zb = np.array(compute_boundaries(t, alpha, kind)["z_bounds"])
        out[kind] = float((np.abs(z) >= zb).any(axis=1).mean())
    out["summary"] = (f"Checking {n_looks} times and stopping at the first p < {alpha} gives a "
                      f"{out['naive_peeking']:.1%} false-positive rate, not {alpha:.0%}. Alpha spending keeps it near {alpha:.0%}.")
    return out
