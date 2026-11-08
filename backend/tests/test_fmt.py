from app.services.fmt import fmt_num
from app.services.frequentist import welch_t_test


def test_fmt_num_uses_four_significant_figures():
    assert fmt_num(6349.14) == "6,349"
    assert fmt_num(-635.03, signed=True) == "-635.0"
    assert fmt_num(694.98) == "695.0"
    assert fmt_num(58.123) == "58.12"
    assert fmt_num(0.012346) == "0.01235"
    assert fmt_num(0) == "0"
    assert fmt_num(12.5, signed=True) == "+12.50"
    # numbers in one sentence share precision, set by the largest one
    assert [fmt_num(v, ref=6984.17) for v in (-635.03, 697.19, 6349.14)] == ["-635", "697", "6,349"]


def test_t_test_summary_has_no_silly_decimals():
    s = welch_t_test([6000, 7000, 8000, 6500], [5000, 5500, 6000, 7000])["summary"]
    assert "." not in s.split("CI")[0].replace("p = ", "")   # amounts in the thousands print as whole numbers
