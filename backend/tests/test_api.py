import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _lalonde():
    r = client.post("/data/sample/lalonde")
    assert r.status_code == 200
    return r.json()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_sample_size_endpoint():
    r = client.post("/power/sample-size", json={"metric": "binary", "baseline_rate": 0.1, "mde": 0.02})
    assert r.status_code == 200 and 3800 <= r.json()["n_control"] <= 3900


def test_missing_baseline_is_422():
    assert client.post("/power/sample-size", json={"metric": "binary", "mde": 0.02}).status_code == 422


def test_lalonde_loads_and_drops_index_column():
    info = _lalonde()
    assert info["n_rows"] == 614
    assert "Unnamed: 0" not in [c["name"] for c in info["columns"]]
    assert "re78" in [c["name"] for c in info["columns"]]


def test_naive_lalonde_comparison_is_biased():
    """The headline of the project: the naive estimate has the WRONG SIGN.
    The experimental benchmark for this program is about +$1,800; the naive PSID comparison says about -$635."""
    ds = _lalonde()["dataset_id"]
    r = client.post("/ab/t-test", json={"dataset_id": ds, "group_col": "treat", "outcome_col": "re78"})
    body = r.json()
    assert r.status_code == 200
    assert body["difference"] == pytest.approx(-635.03, abs=0.5)
    assert body["control_label"] == "0" and body["treatment_label"] == "1"


def test_simulated_chi_square_via_api():
    info = client.post("/data/simulate", json={"n_per_group": 20000, "metric": "binary",
                                               "baseline_rate": 0.10, "lift": 0.02, "seed": 1}).json()
    r = client.post("/ab/chi-square", json={"dataset_id": info["dataset_id"], "group_col": "group", "outcome_col": "outcome"})
    body = r.json()
    assert body["significant"] and body["ci_lower"] < info["true_effect"] < body["ci_upper"]


def test_chi_square_on_continuous_outcome_is_422():
    ds = _lalonde()["dataset_id"]
    r = client.post("/ab/chi-square", json={"dataset_id": ds, "group_col": "treat", "outcome_col": "re78"})
    assert r.status_code == 422


def test_unknown_dataset_is_404_and_bad_column_is_422():
    assert client.post("/ab/t-test", json={"dataset_id": "nope", "group_col": "a", "outcome_col": "b"}).status_code == 404
    ds = _lalonde()["dataset_id"]
    assert client.post("/ab/t-test", json={"dataset_id": ds, "group_col": "zzz", "outcome_col": "re78"}).status_code == 422


def test_upload_endpoint():
    r = client.post("/data/upload", files={"file": ("x.csv", b"g,y\nA,1\nA,2\nB,3\nB,5\n", "text/csv")})
    assert r.status_code == 200 and r.json()["n_rows"] == 4


def test_multiple_testing_endpoint():
    r = client.post("/ab/multiple-testing", json={"p_values": [0.001, 0.02, 0.04, 0.5], "method": "bonferroni"})
    assert r.status_code == 200
    assert [x["reject_null"] for x in r.json()["results"]] == [True, False, False, False]


def test_sequential_boundaries_endpoint():
    r = client.post("/sequential/boundaries", json={"n_looks": 5})
    assert r.status_code == 200 and len(r.json()["z_bounds"]) == 5
