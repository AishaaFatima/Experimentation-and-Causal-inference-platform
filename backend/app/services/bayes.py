"""Bayesian A/B analysis with PyMC: the probability that the treatment beats the control.

Instead of a p-value, the model gives a full probability distribution for each group's true rate (or mean).
Everything on the page, such as "98% chance B is better", is read directly off those distributions.
"""
import logging

import numpy as np
from scipy import integrate, stats

from .fmt import fmt_num
from .groups import split_groups

MAX_PYTHON_MODE_OBS = 300   # per-row models above this are impractically slow without a C++ compiler


def _hdi(x: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Highest-density interval: the narrowest range containing `mass` of the samples."""
    x = np.sort(np.asarray(x, float))
    k = int(np.ceil(mass * len(x)))
    widths = x[k - 1:] - x[: len(x) - k + 1]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k - 1])


def _hist(x, bins=40, rng=None):
    lo, hi = rng if rng else np.percentile(x, [0.5, 99.5])
    counts, edges = np.histogram(x, bins=bins, range=(float(lo), float(hi)), density=True)
    return [{"x_start": float(edges[i]), "x_end": float(edges[i + 1]), "density": float(counts[i])} for i in range(bins)]


def beta_exact_prob(a1, b1, a2, b2) -> float:
    """Exact P(p2 > p1) for independent Beta(a1,b1) and Beta(a2,b2) posteriors (used to check the MCMC)."""
    f = lambda x: stats.beta.pdf(x, a2, b2) * stats.beta.cdf(x, a1, b1)
    return float(integrate.quad(f, 0, 1, limit=200)[0])


def _pct(p: float) -> str:
    """Never print 100% or 0% for a sampled probability."""
    return ">99.9%" if p > 0.999 else "<0.1%" if p < 0.001 else f"{p:.1%}"


def _diagnostics(idata, names) -> dict:
    out = {"max_r_hat": None, "min_ess_bulk": None, "n_divergences": None, "converged": None}
    try:
        import arviz as az
        s = az.summary(idata, var_names=names)
        out["max_r_hat"] = float(s["r_hat"].max())
        out["min_ess_bulk"] = float(s["ess_bulk"].min())
    except Exception:
        pass
    try:
        out["n_divergences"] = int(np.asarray(idata.sample_stats["diverging"]).sum())
    except Exception:
        pass
    if out["max_r_hat"] is not None:
        out["converged"] = bool(out["max_r_hat"] < 1.01 and out["min_ess_bulk"] > 400 and not out["n_divergences"])
    return out


def bayesian_ab(df, group_col, outcome_col, control_label=None, metric="auto", prior_alpha=1.0, prior_beta=1.0,
                draws=2000, tune=1000, chains=4, min_effect=0.0, seed=0, likelihood="normal") -> dict:
    try:
        import pymc as pm
    except ImportError as e:
        raise ModuleNotFoundError("PyMC is not installed. Run: pip install pymc") from e
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    c, t, cl, tl = split_groups(df, group_col, outcome_col, control_label)
    if len(c) < 2 or len(t) < 2:
        raise ValueError("Each group needs at least 2 observations")
    is_binary_data = bool(np.isin(c, [0, 1]).all() and np.isin(t, [0, 1]).all())
    if metric == "auto":
        metric = "binary" if is_binary_data else "continuous"
    if metric == "binary" and not is_binary_data:
        raise ValueError("metric='binary' needs an outcome column containing only 0 and 1")
    if likelihood not in ("normal", "student_t"):
        raise ValueError("likelihood must be 'normal' or 'student_t'")

    if metric == "binary":
        sc, st, nc, nt = int(c.sum()), int(t.sum()), len(c), len(t)
        with pm.Model():
            pc = pm.Beta("p_control", prior_alpha, prior_beta)
            pt = pm.Beta("p_treat", prior_alpha, prior_beta)
            pm.Binomial("obs_control", n=nc, p=pc, observed=sc)
            pm.Binomial("obs_treat", n=nt, p=pt, observed=st)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, progressbar=False, random_seed=seed)
        a = np.asarray(idata.posterior["p_control"]).ravel()
        b = np.asarray(idata.posterior["p_treat"]).ravel()
        names = ["p_control", "p_treat"]
        observed = {"control": sc / nc, "treatment": st / nt}
        exact = beta_exact_prob(prior_alpha + sc, prior_beta + nc - sc, prior_alpha + st, prior_beta + nt - st)
        model_text = f"Beta({prior_alpha:g}, {prior_beta:g}) prior on each group's rate with a binomial likelihood."
    else:
        allv = np.concatenate([c, t])
        m, s = float(allv.mean()), float(allv.std())
        if s == 0:
            raise ValueError("The outcome has no variation; the model is undefined")
        zc, zt = (c - m) / s, (t - m) / s            # standardize so the sampler works on a friendly scale
        if likelihood == "student_t":
            import pytensor
            if not pytensor.config.cxx and len(zc) + len(zt) > MAX_PYTHON_MODE_OBS:
                raise ValueError(
                    f"likelihood='student_t' fits every data point individually and would take many minutes on "
                    f"{len(zc) + len(zt)} rows because no C++ compiler was found. Use likelihood='normal' (same "
                    f"answer for means, and fast), or install a C++ compiler.")
        with pm.Model():
            mu_c, mu_t = pm.Normal("mu_control", 0, 10), pm.Normal("mu_treat", 0, 10)
            sd_c, sd_t = pm.HalfNormal("sigma_control", 5), pm.HalfNormal("sigma_treat", 5)
            if likelihood == "normal":
                # Exact shortcut: for Normal data, (sample mean, sum of squares) are sufficient statistics, so modelling
                # them gives the SAME posterior as modelling every row, but with a handful of operations instead of n.
                for tag, z, mu, sd in (("control", zc, mu_c, sd_c), ("treat", zt, mu_t, sd_t)):
                    n = len(z)
                    pm.Normal(f"mean_{tag}", mu=mu, sigma=sd / np.sqrt(n), observed=float(z.mean()))
                    pm.Gamma(f"sumsq_{tag}", alpha=(n - 1) / 2, beta=1 / (2 * sd**2),
                             observed=float(((z - z.mean()) ** 2).sum()))
            else:
                nu = pm.Exponential("nu_minus_one", 1 / 29) + 1
                pm.StudentT("obs_control", nu=nu, mu=mu_c, sigma=sd_c, observed=zc)
                pm.StudentT("obs_treat", nu=nu, mu=mu_t, sigma=sd_t, observed=zt)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, progressbar=False, random_seed=seed)
        a = np.asarray(idata.posterior["mu_control"]).ravel() * s + m
        b = np.asarray(idata.posterior["mu_treat"]).ravel() * s + m
        names = ["mu_control", "mu_treat", "sigma_control", "sigma_treat"]
        observed = {"control": float(c.mean()), "treatment": float(t.mean())}
        exact = None
        if likelihood == "normal":
            model_text = ("Normal likelihood with weakly informative priors on each group's mean and spread. "
                          "The mean parameter is the true average, matching what a t-test compares. Fitted through the "
                          "sample mean and sum of squares, which gives the same posterior as using every row but runs fast.")
        else:
            model_text = ("Student-t likelihood (robust to outliers). For skewed data such as earnings, its centre is a "
                          "'typical value' that can differ from the average, so read results as typical-value differences.")

    diff = b - a
    p_better = float((diff > 0).mean())
    lo, hi = _hdi(diff)
    rel = None
    if (a > 0).all():
        r = diff / a
        rl, rh = _hdi(r)
        rel = {"mean": float(r.mean()), "ci_lower": rl, "ci_upper": rh}
    loss_treat = float(np.maximum(a - b, 0).mean())      # expected loss if we choose treatment
    loss_control = float(np.maximum(b - a, 0).mean())    # expected loss if we stay with control
    shared_range = tuple(np.percentile(np.concatenate([a, b]), [0.5, 99.5]))
    diag = _diagnostics(idata, names)
    warns = []
    if diag["converged"] is False:
        warns.append("The sampler did not converge cleanly (check max_r_hat, min_ess_bulk, n_divergences). "
                     "Increase draws/tune before trusting these numbers.")
    if exact is not None and abs(exact - p_better) > 0.01:
        warns.append(f"The sampled probability differs from the exact answer by {abs(exact - p_better):.3f}; use more draws.")

    if metric == "binary":
        fmt = lambda v: f"{v:.2%}"
        pts = lambda v: f"{v * 100:+.2f} percentage points"
        loss = lambda v: "essentially zero" if v < 5e-6 else f"{v * 100:.3f} percentage points"
        center = "rate"
    else:
        ref_obs = max(abs(observed["control"]), abs(observed["treatment"])) or 1.0
        ref_diff = max(abs(float(diff.mean())), abs(lo), abs(hi)) or 1.0
        ref_loss = max(loss_treat, loss_control) or 1.0
        fmt = lambda v: fmt_num(v, ref=ref_obs)
        pts = lambda v: fmt_num(v, signed=True, ref=ref_diff)
        loss = lambda v: "essentially zero" if v < 1e-5 * s else fmt_num(v, ref=ref_loss)
        center = "mean" if likelihood == "normal" else "typical value"
    summary = (
        f"There is a {_pct(p_better)} probability that the treatment group's {center} is higher "
        f"than the control's (observed {fmt(observed['treatment'])} vs {fmt(observed['control'])}). The most likely difference is "
        f"{pts(float(diff.mean()))} (95% credible interval {pts(lo)} to {pts(hi)}). Choosing the treatment risks an expected loss of "
        f"{loss(loss_treat)} if the control is actually better; staying with the control risks {loss(loss_control)}."
    )
    return {
        "method": "bayesian_ab_pymc", "metric": metric, "model": model_text,
        "control_label": cl, "treatment_label": tl, "n_control": len(c), "n_treatment": len(t),
        "observed": observed,
        "posterior": {
            "control": {"mean": float(a.mean()), "ci_lower": _hdi(a)[0], "ci_upper": _hdi(a)[1]},
            "treatment": {"mean": float(b.mean()), "ci_lower": _hdi(b)[0], "ci_upper": _hdi(b)[1]},
            "difference": {"mean": float(diff.mean()), "median": float(np.median(diff)), "ci_lower": lo, "ci_upper": hi},
            "relative_lift": rel,
        },
        "prob_treatment_better": p_better,
        "prob_difference_exceeds_min_effect": float((diff > min_effect).mean()), "min_effect": min_effect,
        "expected_loss_choose_treatment": loss_treat, "expected_loss_choose_control": loss_control,
        "exact_prob_treatment_better": exact,
        "diagnostics": diag, "warnings": warns,
        "histograms": {"difference": _hist(diff), "control": _hist(a, rng=shared_range), "treatment": _hist(b, rng=shared_range)},
        "likelihood": likelihood if metric == "continuous" else None,
        "settings": {"draws": draws, "tune": tune, "chains": chains, "seed": seed, "prior_alpha": prior_alpha, "prior_beta": prior_beta},
        "caveat": ("These probabilities describe the groups as observed. Read them as the effect of the treatment only if "
                   "people were assigned at random (a true A/B test), not for observational data."),
        "summary": summary,
    }
