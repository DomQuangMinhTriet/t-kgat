"""Structural guard: training code must not be able to read the test set."""

import inspect

from tkgat.train import bpr


def test_fit_bpr_never_reads_test():
    src = inspect.getsource(bpr)
    assert "data.test" not in src and "mask_test" not in src


def test_fit_bpr_has_no_test_argument():
    assert "test" not in inspect.signature(bpr.fit_bpr).parameters
