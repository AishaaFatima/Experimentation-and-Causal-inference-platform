import pytest
from statsmodels.stats.power import NormalIndPower

from app.services import power


def test_binary_sample_size_matches_known_value():
    # 10% -> 12% baseline, alpha .05, power .8: about 3,841 per group
    res = power.sample_size("binary", mde=0.02, alpha=0.05, power=0.8, ratio=1.0,
                            alternative="two-sided", baseline_rate=0.10)
    assert 3800 <= res["n_control"] <= 3900


def test_power_at_required_n_is_at_least_target():
    res = power.sample_size("binary", 0.02, 0.05, 0.8, 1.0, "two-sided", baseline_rate=0.10)
    p = power.achieved_power("binary", 0.02, 0.05, res["n_control"], 1.0, "two-sided", baseline_rate=0.10)
    assert 0.8 <= p < 0.81


def test_continuous_sample_size_cohens_d():
    # d = 0.5 -> about 64 per group
    res = power.sample_size("continuous", mde=10, alpha=0.05, power=0.8, ratio=1.0,
                            alternative="two-sided", std=20)
    assert 62 <= res["n_control"] <= 66


def test_smaller_effect_needs_more_data():
    a = power.sample_size("binary", 0.04, 0.05, 0.8, 1, "two-sided", baseline_rate=0.1)["n_control"]
    b = power.sample_size("binary", 0.02, 0.05, 0.8, 1, "two-sided", baseline_rate=0.1)["n_control"]
    assert b > a


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        power.sample_size("binary", 0.95, 0.05, 0.8, 1, "two-sided", baseline_rate=0.1)


def test_curve_is_monotonic():
    curve = power.power_curve("binary", 0.02, 0.05, 0.8, 1, "two-sided", baseline_rate=0.1)["curve"]
    powers = [c["power"] for c in curve]
    assert powers == sorted(powers)
