"""Difference-in-differences for two periods (pre and post), with an optional placebo check."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .causal_utils import binary_treatment, check_columns, numeric_frame


def _did(df, t, pre, post, covariates, alpha):
    delta = (pd.to_numeric(df[post], errors="coerce") - pd.to_numeric(df[pre], errors="coerce")).to_numpy(float)
    if np.isnan(delta).any():
        raise ValueError(f"'{pre}' and '{post}' must be numeric with no missing values")
    X = pd.DataFrame({"treat": t.astype(float)})
    if covariates:
        X = pd.concat([X, numeric_frame(df, covariates).reset_index(drop=True)], axis=1)
    # Regressing the change on treatment gives the same estimate as the two-period DiD regression
    res = sm.OLS(delta, sm.add_constant(X)).fit(cov_type="HC1")
    est, se = float(res.params["treat"]), float(res.bse["treat"])
    lo, hi = res.conf_int(alpha=alpha).loc["treat"]
    return {
        "estimate": est, "std_error": se, "p_value": float(res.pvalues["treat"]),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "change_treated": float(delta[t == 1].mean()), "change_control": float(delta[t == 0].mean()),
    }


def did_analysis(df, treat_col, pre_col, post_col, covariates=None, placebo_pre_col=None,
                 placebo_post_col=None, alpha=0.05) -> dict:
    check_columns(df, [pre_col, post_col])
    t = binary_treatment(df, treat_col)
    main = _did(df, t, pre_col, post_col, covariates, alpha)
    sig = main["p_value"] < alpha
    pre_mean = [float(pd.to_numeric(df[c])[t == g].mean()) for g in (1, 0) for c in (pre_col, post_col)]
    out = {
        "method": "difference_in_differences", "alpha": alpha, **main,
        "means": {"treated_pre": pre_mean[0], "treated_post": pre_mean[1],
                  "control_pre": pre_mean[2], "control_post": pre_mean[3]},
        "significant": bool(sig),
    }
    summary = (
        f"Treated units changed by {main['change_treated']:+,.0f} from {pre_col} to {post_col}, versus "
        f"{main['change_control']:+,.0f} for the comparison group. The difference-in-differences estimate is "
        f"{main['estimate']:+,.0f} ({1 - alpha:.0%} CI {main['ci_lower']:+,.0f} to {main['ci_upper']:+,.0f}), "
        f"{'statistically significant' if sig else 'not statistically significant'}."
    )
    if placebo_pre_col and placebo_post_col:
        check_columns(df, [placebo_pre_col, placebo_post_col])
        pl = _did(df, t, placebo_pre_col, placebo_post_col, covariates, alpha)
        out["placebo"] = {"pre_col": placebo_pre_col, "post_col": placebo_post_col, **pl,
                          "parallel_trends_plausible": bool(pl["p_value"] >= alpha)}
        if pl["p_value"] < alpha:
            summary += (f" Warning: the placebo check ({placebo_pre_col} to {placebo_post_col}, before any treatment) "
                        f"found a 'fake effect' of {pl['estimate']:+,.0f}. The groups were already on different "
                        f"trends, so the parallel-trends assumption looks violated and this estimate is unreliable.")
        else:
            summary += " The placebo check on earlier periods found no pre-existing difference in trends."
    out["summary"] = summary
    return out
