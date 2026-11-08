import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from app.services import corrections, frequentist
from app.services.simulator import simulate_ab


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 200), rng.normal(0.3, 2, 150)
    ours = frequentist.welch_t_test(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert ours["t_statistic"] == pytest.approx(ref.statistic)
    assert ours["p_value"] == pytest.approx(ref.pvalue)


def test_chi_square_matches_scipy():
    rng = np.random.default_rng(2)
    a, b = rng.binomial(1, 0.10, 3000), rng.binomial(1, 0.13, 3000)
    ours = frequentist.chi_square_test(a, b)
    table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
    assert ours["p_value"] == pytest.approx(stats.chi2_contingency(table, correction=False)[1])


def test_chi_square_rejects_non_binary():
    with pytest.raises(ValueError):
        frequentist.chi_square_test([1, 2, 3], [0, 1, 1])


def test_chi_square_falls_back_to_fisher_for_small_counts():
    res = frequentist.chi_square_test([0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0])
    assert res["method"] == "fisher_exact" and res["warnings"]


def test_t_test_recovers_known_effect():
    df = simulate_ab(20000, "continuous", lift=2.0, mean=100, std=20, seed=7)
    res = frequentist.welch_t_test(df[df.group == "A"].outcome, df[df.group == "B"].outcome)
    assert res["ci_lower"] < 2.0 < res["ci_upper"]
    assert res["significant"]


def test_aa_false_positive_rate_is_about_alpha():
    """A/A tests (no true effect) should be 'significant' about 5% of the time."""
    rng = np.random.default_rng(123)
    n_sims, hits = 2000, 0
    for _ in range(n_sims):
        a, b = rng.binomial(1, 0.1, 2000), rng.binomial(1, 0.1, 2000)
        if a.sum() + b.sum() == 0:
            continue
        hits += frequentist.chi_square_test(a, b)["p_value"] < 0.05
    assert 0.035 < hits / n_sims < 0.065


def test_corrections_match_statsmodels():
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(0, 0.01, 5), rng.uniform(0, 1, 25)])
    for ours, ref in ((corrections.bonferroni, "bonferroni"), (corrections.benjamini_hochberg, "fdr_bh")):
        r = ours(p, 0.05)
        sm_reject, sm_adj, _, _ = multipletests(p, alpha=0.05, method=ref)
        assert np.allclose(r["adjusted_p"], sm_adj)
        assert (r["reject"] == sm_reject).all()


def test_bh_controls_false_discovery_better_than_no_correction():
    rng = np.random.default_rng(9)
    false_pos_naive = false_pos_bh = 0
    for _ in range(500):
        p = rng.uniform(size=20)  # all nulls true
        false_pos_naive += (p < 0.05).any()
        false_pos_bh += corrections.benjamini_hochberg(p)["reject"].any()
    assert false_pos_naive / 500 > 0.5      # ~64% expected
    assert false_pos_bh / 500 < 0.08        # ~5% expected
