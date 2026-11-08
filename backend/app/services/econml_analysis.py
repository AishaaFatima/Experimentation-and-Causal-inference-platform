"""Machine-learning causal estimators from EconML (double machine learning, causal forest, doubly robust).

Idea: use flexible ML models to predict the outcome and the treatment from covariates, then estimate the
treatment effect from what is LEFT OVER (the part of each that the covariates cannot explain). This removes
confounding without assuming a linear model, and a causal forest can also show who benefits most.
"""
import warnings

import numpy as np
import pandas as pd

from .causal_utils import binary_treatment, check_columns, numeric_frame
from .psm import fit_propensity

CLIP_GRID = [0.001, 0.02, 0.05, 0.1]


def _models(seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    reg = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1)
    clf = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1)
    return reg, clf


def _subgroups(Xtr, cate, names, min_n=10, max_vars=8):
    rows = []
    for j, name in enumerate(names[:max_vars]):
        col = Xtr[:, j]
        if set(np.unique(col)) <= {0, 1}:
            groups = [(f"{name} = {int(v)}", col == v) for v in (0, 1)]
        else:
            med = float(np.median(col))
            groups = [(f"{name} <= {med:,.4g}", col <= med), (f"{name} > {med:,.4g}", col > med)]
        for label, mask in groups:
            if mask.sum() >= min_n:
                rows.append({"group": label, "n_treated": int(mask.sum()), "mean_effect": float(cate[mask].mean())})
    return rows


def econml_analysis(df, treat_col, outcome_col, covariates, effect_modifiers=None, n_trees=500,
                    min_propensity=0.05, alpha=0.05, seed=0) -> dict:
    try:
        from econml.dml import CausalForestDML, LinearDML
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ModuleNotFoundError("EconML is not installed. Run: pip install econml") from e

    if not covariates:
        raise ValueError("Pick at least one covariate to adjust for")
    check_columns(df, [outcome_col])
    t = binary_treatment(df, treat_col)
    Xdf = numeric_frame(df, covariates)
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Outcome '{outcome_col}' has missing or non-numeric values")
    y = y.to_numpy(float)
    mods = list(effect_modifiers) if effect_modifiers else list(covariates)
    unknown = [m for m in mods if m not in covariates]
    if unknown:
        raise ValueError(f"effect_modifiers must be a subset of covariates; not found in covariates: {unknown}")
    rest = [c for c in covariates if c not in mods]
    X = Xdf[mods].to_numpy(float)
    W = Xdf[rest].to_numpy(float) if rest else None
    Wall = Xdf.to_numpy(float)
    tr = t == 1
    reg, clf = _models(seed)

    estimates, notes = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Linear DML: one constant effect, ML used only to strip out the covariates' influence
        lin = LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=5, random_state=seed)
        lin.fit(y, t, X=None, W=Wall)
        s = lin.ate_inference()
        lo, hi = (float(v) for v in s.conf_int_mean(alpha=alpha))
        estimates.append({"method": "linear_dml", "estimand": "Average effect (assumes the same effect for everyone)",
                          "effect": float(s.mean_point), "ci_lower": lo, "ci_upper": hi,
                          "note": "Double ML with forest nuisance models. The confidence interval is a standard one."})

        # 2. Causal forest: effect can differ from person to person
        cf = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=5,
                             n_estimators=n_trees, min_samples_leaf=5, random_state=seed)
        cf.fit(y, t, X=X, W=W)
        a = cf.ate_inference(X=X[tr])
        lo, hi = (float(v) for v in a.conf_int_mean(alpha=alpha))
        cate = np.asarray(cf.effect(X[tr])).ravel()
        estimates.append({"method": "causal_forest", "estimand": "Average effect on the treated (ATT)",
                          "effect": float(a.mean_point), "ci_lower": lo, "ci_upper": hi,
                          "note": "EconML describes this interval as a conservative upper bound, so it is wide."})
        imp = sorted(zip(mods, (float(v) for v in cf.feature_importances_)), key=lambda kv: -kv[1])
        heterogeneity = {
            "cate_quantiles": {f"p{q}": float(np.percentile(cate, q)) for q in (5, 25, 50, 75, 95)},
            "share_negative": float((cate < 0).mean()),
            "subgroups": _subgroups(X[tr], cate, mods),
            "importances": [{"variable": k, "importance": v} for k, v in imp],
            "caution": (f"Only {int(tr.sum())} treated units, so differences between people are noisy and part of the "
                        f"spread is estimation noise. Treat subgroup results as hints, not findings."),
        }

        # 3. Doubly robust learner: correct if EITHER the outcome model OR the propensity model is right
        dr_runs = {}
        for mp in sorted(set(CLIP_GRID + [min_propensity])):
            dr = DRLearner(model_propensity=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           model_regression=reg(), model_final=GradientBoostingRegressor(random_state=seed),
                           min_propensity=mp, cv=5, random_state=seed)
            dr.fit(y, t, X=X, W=W)
            dr_runs[mp] = float(np.asarray(dr.effect(X[tr])).mean())
        estimates.append({"method": "doubly_robust", "estimand": "Average effect on the treated (ATT)",
                          "effect": dr_runs[min_propensity], "ci_lower": None, "ci_upper": None,
                          "note": f"Propensity scores clipped to [{min_propensity}, {1 - min_propensity}]. "
                                  f"No confidence interval is reported; see dr_sensitivity for how much this estimate moves."})

    ps = fit_propensity(Xdf, t, False, False)
    extreme = (ps < min_propensity) | (ps > 1 - min_propensity)
    overlap = {"clip": min_propensity, "n_extreme": int(extreme.sum()), "share_extreme": float(extreme.mean()),
               "n_extreme_treated": int((extreme & tr).sum()), "n_extreme_controls": int((extreme & ~tr).sum())}
    if overlap["n_extreme_treated"] > 0:
        notes.append(f"{overlap['n_extreme_treated']} of {int(tr.sum())} treated units have an estimated treatment "
                     f"probability outside [{min_propensity}, {1 - min_propensity}], meaning few comparable untreated "
                     f"units. Doubly robust estimates are known to be sensitive to units like this.")
    if overlap["n_extreme_controls"] > 0.10 * (~tr).sum():
        notes.append(f"{overlap['n_extreme_controls']} of {int((~tr).sum())} comparison units look very unlike the treated "
                     f"(treatment probability under {min_propensity}). They add little information about the effect on "
                     f"the treated.")
    spread = max(dr_runs.values()) - min(dr_runs.values())
    if spread > 0.5 * abs(np.mean(list(dr_runs.values()))):
        notes.append("The doubly robust estimate changes a lot with the propensity clipping level (see dr_sensitivity). "
                     "Do not rely on a single value.")
    dr_sens = [{"min_propensity": k, "att": v} for k, v in sorted(dr_runs.items())]
    effects = [e["effect"] for e in estimates]
    naive = float(y[tr].mean() - y[~tr].mean())
    ci_zero = [e["method"] for e in estimates if e["ci_lower"] is not None and e["ci_lower"] <= 0 <= e["ci_upper"]]
    summary = (
        f"The naive difference is {naive:+,.0f}. EconML's three estimators give {min(effects):+,.0f} to "
        f"{max(effects):+,.0f}. The doubly robust estimate alone ranges from {min(dr_runs.values()):+,.0f} to "
        f"{max(dr_runs.values()):+,.0f} depending on how extreme propensity scores are clipped. "
        + (f"The confidence interval includes zero for {' and '.join(m.replace('_', ' ') for m in ci_zero)}, so the data "
           f"cannot rule out no effect. " if ci_zero else "")
        + "Report this as a range."
    )
    return {"method": "econml", "naive_difference": naive, "alpha": alpha, "estimates": estimates,
            "estimate_min": min(effects), "estimate_max": max(effects), "heterogeneity": heterogeneity,
            "dr_sensitivity": dr_sens, "overlap": overlap, "warnings": notes,
            "n_treated": int(tr.sum()), "n_control": int((~tr).sum()), "summary": summary}
