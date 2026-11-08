import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.econml_analysis import econml_analysis

client = TestClient(app)
COV = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def confounded(n=3000, seed=0, hetero=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.integers(0, 2, n)                      # a binary effect modifier
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.0 * x))).astype(int)
    effect = (1000 + 2000 * z) if hetero else 2000.0
    y = 5000 + 3000 * x + effect * t + rng.normal(0, 1000, n)
    return pd.DataFrame({"t": t, "x": x, "z": z, "y": y})


def get(res, method):
    return next(e for e in res["estimates"] if e["method"] == method)


def test_recovers_known_constant_effect():
    res = econml_analysis(confounded(2000), "t", "y", ["x", "z"], n_trees=100)
    # Forest nuisance models leave a small downward bias (a few %), so test closeness, not exact coverage.
    assert get(res, "linear_dml")["effect"] == pytest.approx(2000, abs=300)
    assert get(res, "causal_forest")["effect"] == pytest.approx(2000, abs=500)
    naive_error = abs(res["naive_difference"] - 2000)
    assert naive_error > 1000                      # confounding badly biases the naive estimate
    assert abs(get(res, "linear_dml")["effect"] - 2000) < naive_error / 5


def test_causal_forest_finds_heterogeneity():
    res = econml_analysis(confounded(3000, hetero=True), "t", "y", ["x", "z"], n_trees=200)
    sub = {g["group"]: g["mean_effect"] for g in res["heterogeneity"]["subgroups"]}
    assert sub["z = 1"] - sub["z = 0"] > 800       # truth: 3000 vs 1000


def test_dr_sensitivity_covers_the_grid_and_main_value():
    res = econml_analysis(confounded(1000), "t", "y", ["x", "z"], n_trees=100, min_propensity=0.07)
    clips = [r["min_propensity"] for r in res["dr_sensitivity"]]
    assert 0.07 in clips and 0.001 in clips and clips == sorted(clips)
    assert get(res, "doubly_robust")["effect"] == pytest.approx(
        next(r["att"] for r in res["dr_sensitivity"] if r["min_propensity"] == 0.07))


def test_input_validation():
    df = confounded(300)
    with pytest.raises(ValueError):
        econml_analysis(df, "t", "y", [], n_trees=100)
    with pytest.raises(ValueError):
        econml_analysis(df, "t", "y", ["x"], effect_modifiers=["z"], n_trees=100)   # modifier not in covariates
    with pytest.raises(ValueError):
        econml_analysis(df, "nope", "y", ["x"], n_trees=100)


def test_lalonde_endpoint():
    ds = client.post("/data/sample/lalonde").json()["dataset_id"]
    r = client.post("/causal/econml", json={"dataset_id": ds, "treat_col": "treat", "outcome_col": "re78",
                                           "covariates": COV, "n_trees": 100})
    assert r.status_code == 200
    body = r.json()
    assert [e["method"] for e in body["estimates"]] == ["linear_dml", "causal_forest", "doubly_robust"]
    assert body["naive_difference"] < 0 < body["estimate_min"]            # every ML estimator flips the sign
    assert len(body["dr_sensitivity"]) == 4 and body["overlap"]["n_extreme_controls"] > 100
    assert body["heterogeneity"]["caution"] and body["warnings"]
    bad = client.post("/causal/econml", json={"dataset_id": ds, "treat_col": "zzz", "outcome_col": "re78", "covariates": COV})
    assert bad.status_code == 422
