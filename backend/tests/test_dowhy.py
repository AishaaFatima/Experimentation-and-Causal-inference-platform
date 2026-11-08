import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.dowhy_analysis import dowhy_analysis

client = TestClient(app)
COV = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def confounded(n=3000, effect=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    y = 5000 + 3000 * x + effect * t + rng.normal(0, 1000, n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def test_dowhy_recovers_known_effect_and_passes_refutations():
    res = dowhy_analysis(confounded(), "t", "y", ["x"], n_simulations=10)
    reg = next(e for e in res["estimates"] if e["method"] == "linear_regression")
    assert reg["att"] == pytest.approx(2000, abs=150)
    assert res["naive_difference"] > 3500                      # confounding makes naive badly biased
    by = {r["test"]: r for r in res["refutations"]}
    assert abs(by["placebo_treatment_refuter"]["new_effect"]) < 150   # fake treatment -> about zero
    assert by["random_common_cause"]["passed"] and by["data_subset_refuter"]["passed"]


def test_graph_and_identification_are_reported():
    res = dowhy_analysis(confounded(500), "t", "y", ["x"], n_simulations=5)
    assert {"from": "x", "to": "t"} in res["graph"]["edges"] and {"from": "t", "to": "y"} in res["graph"]["edges"]
    assert res["identification"]["adjustment_set"] == ["x"]


def test_requires_covariates_and_valid_columns():
    with pytest.raises(ValueError):
        dowhy_analysis(confounded(200), "t", "y", [], n_simulations=5)
    with pytest.raises(ValueError):
        dowhy_analysis(confounded(200), "t", "y", ["nope"], n_simulations=5)


def test_lalonde_dowhy_endpoint():
    ds = client.post("/data/sample/lalonde").json()["dataset_id"]
    r = client.post("/causal/dowhy", json={"dataset_id": ds, "treat_col": "treat", "outcome_col": "re78",
                                          "covariates": COV, "n_simulations": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["estimates"]) == 3 and len(body["refutations"]) == 3
    assert body["naive_difference"] < 0 < body["estimate_max"]      # adjustment flips the sign
    reg = next(e for e in body["estimates"] if e["method"] == "linear_regression")
    assert 1000 < reg["att"] < 2200
    bad = client.post("/causal/dowhy", json={"dataset_id": ds, "treat_col": "zzz", "outcome_col": "re78", "covariates": COV})
    assert bad.status_code == 422
