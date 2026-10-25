import numpy as np

from tkgat.eval.stats import bootstrap_ci, holm, paired_ttest, paired_wilcoxon


def test_holm_matches_hand_computation():
    # sorted p: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 -> all rejected at 0.05
    out = holm([0.04, 0.01, 0.02])
    assert np.allclose(out["p_adjusted"], [0.04, 0.03, 0.04])
    assert out["reject"] == [True, True, True]
    out = holm([0.01, 0.03, 0.04])  # 0.03, 0.06, 0.06 -> only the first
    assert out["reject"] == [True, False, False]


def test_bootstrap_ci_contains_true_shift():
    rng = np.random.default_rng(1)
    b = rng.random(200)
    a = b + 0.1 + rng.normal(0, 0.05, 200)
    ci = bootstrap_ci(a, b)
    assert ci["ci_low"] < 0.1 < ci["ci_high"]
    assert ci["ci_low"] > 0


def test_identical_scores_are_not_significant():
    x = np.array([0.1, 0.5, 0.0, 1.0])
    assert paired_ttest(x, x)["p_value"] == 1.0
    assert paired_wilcoxon(x, x)["p_value"] == 1.0


def test_wilcoxon_reports_pairs_and_nonzero_differences():
    from scipy.stats import wilcoxon

    a = np.array([0.0, 0.0, 0.5, 0.2, 1.0, 0.3, 0.0, 0.7, np.nan])
    b = np.array([0.0, 0.0, 0.1, 0.4, 0.6, 0.3, 0.2, 0.1, 0.5])
    res = paired_wilcoxon(a, b)
    assert res["n_pairs"] == res["n"] == 8  # the NaN user is dropped
    assert res["n_nonzero"] == 5  # three users have a difference of exactly 0
    expected = wilcoxon(a[:8], b[:8], zero_method="wilcox", alternative="two-sided").pvalue
    assert res["p_value"] == expected
    same = paired_wilcoxon(a[:8], a[:8])
    assert same["n_pairs"] == 8 and same["n_nonzero"] == 0 and same["p_value"] == 1.0
