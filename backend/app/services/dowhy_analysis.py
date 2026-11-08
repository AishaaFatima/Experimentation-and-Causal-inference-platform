"""Second opinion on the causal effect using DoWhy: explicit assumptions, several estimators, refutation tests.

DoWhy's workflow is: (1) MODEL the causal graph, (2) IDENTIFY what must be adjusted for,
(3) ESTIMATE with several methods, (4) REFUTE by trying to break the result.
"""
import logging
import warnings

import numpy as np
import pandas as pd

from .causal_utils import binary_treatment, check_columns, numeric_frame


ESTIMATORS = {
    "linear_regression": ("backdoor.linear_regression", {},
                          "Regression adjustment. Assumes the effect is the same for everyone and covariates act linearly."),
    "propensity_score_weighting": ("backdoor.propensity_score_weighting",
                                   {"method_params": {"weighting_scheme": "ips_stabilized_weight"}},
                                   "Re-weights comparison units to look like the treated group (stabilized inverse-probability weights)."),
    "propensity_score_matching": ("backdoor.propensity_score_matching", {},
                                  "DoWhy's own nearest-neighbour matching on the propensity score. It uses no caliper, so it can "
                                  "pair units that are not really comparable."),
}

REFUTERS = {
    "random_common_cause": ("Add a random fake confounder",
                            "Adds a random, meaningless variable as an extra confounder. A trustworthy estimate should not move."),
    "placebo_treatment_refuter": ("Replace treatment with a random placebo",
                                  "Shuffles who got treated. With a fake treatment the true effect is zero, so the estimate should drop to about 0."),
    "data_subset_refuter": ("Re-estimate on random subsets",
                            "Re-runs the estimate on random subsets of the data. A trustworthy estimate should stay about the same."),
}


def _graph(treat, outcome, covariates) -> dict:
    edges = [{"from": c, "to": treat} for c in covariates] + [{"from": c, "to": outcome} for c in covariates]
    edges.append({"from": treat, "to": outcome})
    dot = "digraph {" + "; ".join(f'"{e["from"]}" -> "{e["to"]}"' for e in edges) + "}"
    return {"nodes": [treat, outcome, *covariates], "edges": edges, "dot": dot}


def dowhy_analysis(df, treat_col, outcome_col, covariates, refute_with="linear_regression",
                   n_simulations=25, alpha=0.05, seed=0) -> dict:
    try:
        from dowhy import CausalModel
    except ImportError as e:
        raise ModuleNotFoundError("DoWhy is not installed. Run: pip install dowhy") from e
    if not covariates:
        raise ValueError("Pick at least one covariate (potential confounder) to adjust for")
    if refute_with not in ESTIMATORS:
        raise ValueError(f"refute_with must be one of {list(ESTIMATORS)}")
    check_columns(df, [outcome_col])
    t = binary_treatment(df, treat_col)
    X = numeric_frame(df, covariates)
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Outcome '{outcome_col}' has missing or non-numeric values")
    data = X.copy()
    data[treat_col], data[outcome_col] = t, y.to_numpy(float)
    data = data.reset_index(drop=True)

    logging.getLogger("dowhy").setLevel(logging.ERROR)
    model = CausalModel(data=data, treatment=treat_col, outcome=outcome_col, common_causes=list(covariates))
    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimates, fitted = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        for key, (method, kwargs, note) in ESTIMATORS.items():
            try:
                est = model.estimate_effect(estimand, method_name=method, target_units="att", **kwargs)
                fitted[key] = est
                estimates.append({"method": key, "att": float(est.value), "note": note, "error": None})
            except Exception as e:  # report the failure instead of killing the whole analysis
                estimates.append({"method": key, "att": None, "note": note, "error": f"{type(e).__name__}: {e}"[:200]})

        if refute_with not in fitted:
            raise ValueError(f"Could not fit '{refute_with}' to refute; try another estimator")
        base = fitted[refute_with]
        refutations = []
        for key, (title, desc) in REFUTERS.items():
            np.random.seed(seed)
            kwargs = {"placebo_type": "permute"} if key == "placebo_treatment_refuter" else {}
            try:
                r = model.refute_estimate(estimand, base, method_name=key, num_simulations=n_simulations, **kwargs)
                p = r.refutation_result.get("p_value") if getattr(r, "refutation_result", None) else None
                new = float(r.new_effect)
                passed = None if p is None else bool(p > alpha)
                refutations.append({"test": key, "title": title, "description": desc,
                                    "original_effect": float(r.estimated_effect), "new_effect": new,
                                    "p_value": None if p is None else float(p), "passed": passed, "error": None})
            except Exception as e:
                refutations.append({"test": key, "title": title, "description": desc, "original_effect": None,
                                    "new_effect": None, "p_value": None, "passed": None,
                                    "error": f"{type(e).__name__}: {e}"[:200]})

    good = [e["att"] for e in estimates if e["att"] is not None]
    naive = float(data.loc[t == 1, outcome_col].mean() - data.loc[t == 0, outcome_col].mean())
    n_pass = sum(1 for r in refutations if r["passed"])
    n_run = sum(1 for r in refutations if r["passed"] is not None)
    summary = (
        f"The naive difference is {naive:+,.0f}. Adjusting for {len(covariates)} covariates, DoWhy's estimators give "
        f"{min(good):+,.0f} to {max(good):+,.0f} depending on the method. "
        f"{n_pass} of {n_run} refutation tests passed for the {refute_with.replace('_', ' ')} estimate. "
        f"Passing means the result is not fragile to these particular checks; it does NOT prove there are no "
        f"unmeasured confounders."
    ) if good else "No estimator could be fitted."
    return {
        "method": "dowhy", "naive_difference": naive, "alpha": alpha, "refuted_estimator": refute_with,
        "graph": _graph(treat_col, outcome_col, covariates),
        "identification": {"strategy": "backdoor adjustment", "adjustment_set": list(covariates),
                           "assumption": "No unmeasured confounders: every variable that affects both treatment and "
                                         "outcome is in the adjustment set."},
        "estimates": estimates,
        "estimate_min": min(good) if good else None, "estimate_max": max(good) if good else None,
        "refutations": refutations, "n_simulations": n_simulations, "summary": summary,
    }
