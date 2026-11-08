from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import did, psm

LALONDE = Path(__file__).resolve().parents[1] / "data" / "lalonde.csv"   # works from any folder

client = TestClient(app)
COV = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def confounded(n=4000, effect=2000.0, seed=0):
    """Richer people are more likely to be treated AND earn more anyway -> naive estimate is biased."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    y = 5000 + 3000 * x + effect * t + rng.normal(0, 1000, n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def test_naive_is_biased_but_psm_recovers_the_true_effect():
    df = confounded()
    naive = df[df.t == 1].y.mean() - df[df.t == 0].y.mean()
    res = psm.psm_analysis(df, "t", "y", ["x"], n_boot=50)
    assert naive > 3500                                  # badly biased upward (true effect = 2000)
    assert abs(res["att"] - 2000) < 350                  # matching gets close
    assert res["ci_lower"] < 2000 < res["ci_upper"] or abs(res["att"] - 2000) < 250
    assert res["max_abs_smd_after"] < res["max_abs_smd_before"]


def test_psm_balance_improves_on_lalonde():
    df = pd.read_csv(LALONDE).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    res = psm.psm_analysis(df, "treat", "re78", COV, n_boot=50)
    assert res["max_abs_smd_before"] > 1.0
    assert res["balanced"]
    assert res["att"] > res["naive_difference"] + 1000   # corrects the wrong-sign naive answer


def test_psm_needs_binary_treatment_and_covariates():
    df = confounded(200)
    df["t"] = df["t"] * 2
    with pytest.raises(ValueError):
        psm.psm_analysis(df, "t", "y", ["x"], n_boot=20)
    with pytest.raises(ValueError):
        psm.psm_analysis(confounded(200), "t", "y", [], n_boot=20)


def test_did_recovers_effect_when_trends_are_parallel():
    rng = np.random.default_rng(1)
    n = 3000
    t = rng.integers(0, 2, n)
    base = 100 + 30 * t + rng.normal(0, 10, n)         # groups differ in LEVEL, which DiD removes
    df = pd.DataFrame({"t": t, "p0": base - 5 + rng.normal(0, 5, n), "pre": base, "post": base + 7 + 12 * t + rng.normal(0, 5, n)})
    res = did.did_analysis(df, "t", "pre", "post", placebo_pre_col="p0", placebo_post_col="pre")
    assert res["estimate"] == pytest.approx(12, abs=1.0)
    assert res["placebo"]["parallel_trends_plausible"]


def test_did_placebo_flags_violation_on_lalonde():
    df = pd.read_csv(LALONDE)
    res = did.did_analysis(df, "treat", "re75", "re78", placebo_pre_col="re74", placebo_post_col="re75")
    assert not res["placebo"]["parallel_trends_plausible"]
    assert "Warning" in res["summary"]


def test_sensitivity_reports_a_range():
    df = pd.read_csv(LALONDE)
    res = psm.psm_sensitivity(df, "treat", "re78", COV)
    assert len(res["specifications"]) == 4 and res["att_max"] > res["att_min"]


def test_causal_endpoints():
    ds = client.post("/data/sample/lalonde").json()["dataset_id"]
    r = client.post("/causal/psm", json={"dataset_id": ds, "treat_col": "treat", "outcome_col": "re78",
                                        "covariates": COV, "n_boot": 30})
    assert r.status_code == 200 and r.json()["balanced"]
    r = client.post("/causal/did", json={"dataset_id": ds, "treat_col": "treat", "pre_col": "re75",
                                        "post_col": "re78", "placebo_pre_col": "re74", "placebo_post_col": "re75"})
    assert r.status_code == 200 and r.json()["placebo"]["parallel_trends_plausible"] is False
    r = client.post("/causal/psm-sensitivity", json={"dataset_id": ds, "treat_col": "treat",
                                                    "outcome_col": "re78", "covariates": COV})
    assert r.status_code == 200
    bad = client.post("/causal/psm", json={"dataset_id": ds, "treat_col": "nope", "outcome_col": "re78", "covariates": COV})
    assert bad.status_code == 422


# ---- stability: results must not depend on sort order, row order, or how far the solver iterated ----
def test_ties_are_averaged_not_picked_by_sort_order():
    from app.services.psm import match_att
    ps = np.array([0.50, 0.30, 0.30, 0.30, 0.90])          # one treated unit; three controls tie as nearest
    t = np.array([1, 0, 0, 0, 0])
    y = np.array([100.0, 10.0, 20.0, 60.0, 5.0])
    att, _, matches, _ = match_att(y, t, ps, caliper_sd=0, k=1)
    assert att == pytest.approx(100 - 30)                    # mean of the three tied controls
    perm = np.array([0, 3, 1, 4, 2])
    att2, *_ = match_att(y[perm], t[perm], ps[perm], caliper_sd=0, k=1)
    assert att2 == pytest.approx(att)


def test_psm_on_lalonde_is_identical_after_shuffling_rows():
    df = pd.read_csv(LALONDE).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    a = psm.psm_analysis(df, "treat", "re78", COV, n_boot=30)
    b = psm.psm_analysis(df.sample(frac=1, random_state=3).reset_index(drop=True), "treat", "re78", COV, n_boot=30)
    assert a["att"] == pytest.approx(b["att"], abs=1.0)
    assert a["n_treated_matched"] == b["n_treated_matched"]


def test_stricter_matching_trades_treated_people_for_balance():
    df = pd.read_csv(LALONDE).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    strict = psm.psm_analysis(df, "treat", "re78", COV, caliper_sd=0.05, n_boot=20)
    loose = psm.psm_analysis(df, "treat", "re78", COV, caliper_sd=0.2, n_boot=20)
    assert strict["n_treated_dropped"] > loose["n_treated_dropped"]
    assert strict["max_abs_smd_after"] < loose["max_abs_smd_after"]
    assert strict["balanced"] and not loose["balanced"]


def test_sensitivity_reports_how_many_specifications_balanced():
    df = pd.read_csv(LALONDE)
    res = psm.psm_sensitivity(df, "treat", "re78", COV)
    assert res["n_balanced"] == sum(r["balanced"] for r in res["specifications"]) < len(res["specifications"])
    assert "Only" in res["summary"]
