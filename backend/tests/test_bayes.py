from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import frequentist
from app.services.bayes import _hdi, bayesian_ab, beta_exact_prob
from app.services.simulator import simulate_ab

LALONDE = Path(__file__).resolve().parents[1] / "data" / "lalonde.csv"   # works from any folder

client = TestClient(app)
FAST = dict(draws=1000, tune=500, chains=2)


def counts_df(sa, na, sb, nb):
    return pd.DataFrame({"g": ["A"] * na + ["B"] * nb,
                         "y": [1] * sa + [0] * (na - sa) + [1] * sb + [0] * (nb - sb)})


def test_hdi_of_uniform_samples():
    lo, hi = _hdi(np.random.default_rng(0).uniform(size=100_000))
    assert hi - lo == pytest.approx(0.95, abs=0.01) and lo == pytest.approx(0.0, abs=0.02)


@pytest.mark.slow
def test_binary_mcmc_matches_exact_beta_answer():
    df = counts_df(100, 1000, 112, 1000)             # close race, so the probability is not trivially 0 or 1
    res = bayesian_ab(df, "g", "y", draws=3000, tune=500, chains=2)
    exact = beta_exact_prob(1 + 100, 1 + 900, 1 + 112, 1 + 888)
    assert 0.6 < exact < 0.95
    assert res["prob_treatment_better"] == pytest.approx(exact, abs=0.02)
    assert res["exact_prob_treatment_better"] == pytest.approx(exact)
    assert res["diagnostics"]["converged"] and not res["warnings"]


@pytest.mark.slow
def test_obvious_winner_and_obvious_tie():
    win = bayesian_ab(simulate_ab(3000, "binary", 0.10, 0.05, seed=1), "group", "outcome", **FAST)
    assert win["prob_treatment_better"] > 0.999 and ">99.9%" in win["summary"]
    assert win["expected_loss_choose_treatment"] < 1e-3 < win["expected_loss_choose_control"]
    tie = bayesian_ab(simulate_ab(3000, "binary", 0.10, 0.0, seed=1), "group", "outcome", **FAST)
    assert 0.02 < tie["prob_treatment_better"] < 0.98            # no strong claim when there is no effect
    assert tie["posterior"]["difference"]["ci_lower"] < 0 < tie["posterior"]["difference"]["ci_upper"]


@pytest.mark.slow
def test_continuous_recovers_known_difference():
    df = simulate_ab(3000, "continuous", lift=5.0, mean=100, std=20, seed=3)
    res = bayesian_ab(df, "group", "outcome", **FAST)
    assert res["metric"] == "continuous"
    assert res["posterior"]["difference"]["mean"] == pytest.approx(5.0, abs=1.8)
    assert res["posterior"]["relative_lift"] is not None


@pytest.mark.slow
def test_bayesian_agrees_with_the_frequentist_t_test_on_lalonde():
    """With a Normal likelihood and flat-ish priors, P(B>A) should match the t-test's one-sided p-value."""
    df = pd.read_csv(LALONDE)
    res = bayesian_ab(df, "treat", "re78", **FAST)
    c, t = df[df.treat == 0].re78, df[df.treat == 1].re78
    tt = frequentist.welch_t_test(c, t)
    one_sided = tt["p_value"] / 2                                  # t is negative, so P(B>A) ~ one-sided p
    assert res["prob_treatment_better"] == pytest.approx(one_sided, abs=0.03)
    assert res["posterior"]["difference"]["mean"] == pytest.approx(tt["difference"], abs=60)


@pytest.mark.slow
def test_strong_prior_pulls_small_samples_toward_the_prior():
    df = counts_df(5, 20, 1, 20)                                   # raw rates 25% and 5%
    flat = bayesian_ab(df, "g", "y", **FAST)
    strong = bayesian_ab(df, "g", "y", prior_alpha=100, prior_beta=900, **FAST)   # prior mean 10%
    assert abs(strong["posterior"]["treatment"]["mean"] - 0.10) < 0.02
    assert abs(flat["posterior"]["treatment"]["mean"] - 0.10) > abs(strong["posterior"]["treatment"]["mean"] - 0.10)


@pytest.mark.slow
def test_histograms_are_densities():
    res = bayesian_ab(simulate_ab(1000, "binary", 0.1, 0.02, seed=5), "group", "outcome", **FAST)
    h = res["histograms"]["difference"]
    area = sum(b["density"] * (b["x_end"] - b["x_start"]) for b in h)
    assert 0.97 < area < 1.01 and len(h) == 40


def test_validation_errors():
    cont = simulate_ab(100, "continuous", seed=1)
    with pytest.raises(ValueError):
        bayesian_ab(cont, "group", "outcome", metric="binary", **FAST)
    with pytest.raises(ValueError):
        bayesian_ab(cont, "group", "outcome", likelihood="cauchy", **FAST)
    with pytest.raises(ValueError):
        bayesian_ab(cont, "group", "nope", **FAST)


@pytest.mark.slow
def test_endpoint_and_caveat():
    ds = client.post("/data/simulate", json={"n_per_group": 1500, "metric": "binary", "baseline_rate": 0.1,
                                            "lift": 0.03, "seed": 4}).json()["dataset_id"]
    r = client.post("/bayes/ab", json={"dataset_id": ds, "group_col": "group", "outcome_col": "outcome",
                                      "draws": 500, "tune": 300, "chains": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["metric"] == "binary" and body["prob_treatment_better"] > 0.95 and "random" in body["caveat"]
    assert client.post("/bayes/ab", json={"dataset_id": ds, "group_col": "zzz", "outcome_col": "outcome"}).status_code == 422


@pytest.mark.slow
def test_sufficient_statistics_give_the_same_posterior_as_the_full_per_row_model():
    """The fast Normal model must match a model that evaluates every single row."""
    import pymc as pm
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"g": ["A"] * 60 + ["B"] * 60, "y": np.concatenate([rng.normal(50, 12, 60), rng.normal(56, 12, 60)])})
    fast = bayesian_ab(df, "g", "y", draws=2000, tune=800, chains=2, seed=1)

    y = df.y.to_numpy(float)
    m, s = y.mean(), y.std()
    zc, zt = (y[:60] - m) / s, (y[60:] - m) / s
    with pm.Model():
        mc, mt = pm.Normal("mc", 0, 10), pm.Normal("mt", 0, 10)
        sc, st = pm.HalfNormal("sc", 5), pm.HalfNormal("st", 5)
        pm.Normal("oc", mc, sc, observed=zc)
        pm.Normal("ot", mt, st, observed=zt)
        idata = pm.sample(draws=2000, tune=800, chains=2, cores=1, progressbar=False, random_seed=1)
    full_diff = (np.asarray(idata.posterior["mt"]).ravel() - np.asarray(idata.posterior["mc"]).ravel()) * s
    assert fast["posterior"]["difference"]["mean"] == pytest.approx(full_diff.mean(), abs=0.4)
    assert fast["prob_treatment_better"] == pytest.approx(float((full_diff > 0).mean()), abs=0.03)


def test_student_t_refuses_huge_python_mode_runs(monkeypatch):
    import pytensor
    monkeypatch.setattr(pytensor.config, "cxx", "", raising=False)
    big = simulate_ab(400, "continuous", seed=1)
    with pytest.raises(ValueError, match="C\\+\\+ compiler"):
        bayesian_ab(big, "group", "outcome", likelihood="student_t", **FAST)
