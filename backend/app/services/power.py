"""Power and sample-size calculations for two-group experiments."""
import math

import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


def _effect_size(metric, mde, baseline_rate=None, std=None) -> float:
    if metric == "binary":
        p2 = baseline_rate + mde
        if not 0 < p2 < 1:
            raise ValueError("baseline_rate + mde must stay between 0 and 1")
        return abs(float(proportion_effectsize(p2, baseline_rate)))  # Cohen's h
    return abs(mde) / std  # Cohen's d


def _solver(metric):
    return NormalIndPower() if metric == "binary" else TTestIndPower()


def sample_size(metric, mde, alpha, power, ratio, alternative, baseline_rate=None, std=None) -> dict:
    es = _effect_size(metric, mde, baseline_rate, std)
    n1 = _solver(metric).solve_power(effect_size=es, alpha=alpha, power=power,
                                     ratio=ratio, alternative=alternative)
    n1 = math.ceil(n1)
    n2 = math.ceil(n1 * ratio)
    return {"effect_size": es, "n_control": n1, "n_treatment": n2, "n_total": n1 + n2}


def achieved_power(metric, mde, alpha, n_per_group, ratio, alternative, baseline_rate=None, std=None) -> float:
    es = _effect_size(metric, mde, baseline_rate, std)
    return float(_solver(metric).power(effect_size=es, nobs1=n_per_group, alpha=alpha,
                                       ratio=ratio, alternative=alternative))


def power_curve(metric, mde, alpha, power, ratio, alternative, baseline_rate=None, std=None, points=25) -> dict:
    needed = sample_size(metric, mde, alpha, power, ratio, alternative, baseline_rate, std)["n_control"]
    ns = np.unique(np.linspace(max(5, needed * 0.1), needed * 2, points).astype(int))
    curve = [{"n_per_group": int(n),
              "power": achieved_power(metric, mde, alpha, int(n), ratio, alternative, baseline_rate, std)}
             for n in ns]
    return {"required_n_per_group": needed, "curve": curve}
