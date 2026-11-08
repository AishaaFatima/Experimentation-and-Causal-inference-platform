import numpy as np
import pytest

from app.services import sequential as seq
from app.services.simulator import simulate_ab


def test_single_look_is_the_usual_critical_value():
    # One look at t=1 should reproduce z = 1.96 (fixed-sample test); use 2 looks to satisfy validation
    b = seq.compute_boundaries(np.array([0.5, 1.0]), 0.05, "obrien_fleming")
    assert b["z_bounds"][-1] < 2.02  # final look is close to 1.96, only slightly inflated


def test_obrien_fleming_five_looks_matches_published_values():
    z = seq.compute_boundaries(seq.resolve_looks(5), 0.05, "obrien_fleming")["z_bounds"]
    # Published Lan-DeMets O'Brien-Fleming-type boundaries (e.g. gsDesign sfLDOF, 5 equal looks)
    expected = [4.88, 3.36, 2.68, 2.29, 2.03]
    assert z == pytest.approx(expected, abs=0.05)


def test_pocock_boundaries_are_nearly_flat():
    z = seq.compute_boundaries(seq.resolve_looks(5), 0.05, "pocock")["z_bounds"]
    assert max(z) - min(z) < 0.35
    assert z[0] == pytest.approx(2.41, abs=0.06)  # published Pocock constant for 5 looks


def test_obrien_fleming_boundaries_decrease():
    z = seq.compute_boundaries(seq.resolve_looks(4), 0.05, "obrien_fleming")["z_bounds"]
    assert z == sorted(z, reverse=True)


def test_peeking_inflates_false_positives_and_spending_fixes_it():
    res = seq.peeking_simulation(n_looks=5, alpha=0.05, n_sims=40000, seed=1)
    assert 0.12 < res["naive_peeking"] < 0.17     # ~14% expected
    assert 0.035 < res["obrien_fleming"] < 0.065
    assert 0.035 < res["pocock"] < 0.065


def test_bad_looks_rejected():
    with pytest.raises(ValueError):
        seq.resolve_looks(3, [0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        seq.resolve_looks(3, [0.3, 0.6, 0.9])


def test_analyze_stream_detects_a_big_effect_early_and_skips_unreached_looks():
    df = simulate_ab(5000, "continuous", lift=15, mean=100, std=20, seed=3)
    out = seq.analyze_stream(df, "group", "outcome", None, 10000, seq.resolve_looks(5), 0.05, "obrien_fleming")
    assert out["stopped_at_look"] is not None and out["stopped_at_look"] <= 2
    short = seq.analyze_stream(df.iloc[:3000], "group", "outcome", None, 10000, seq.resolve_looks(5), 0.05, "obrien_fleming")
    assert any(r["status"] == "not_reached" for r in short["looks"])
