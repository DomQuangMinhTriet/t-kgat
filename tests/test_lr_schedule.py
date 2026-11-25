"""The learning-rate schedule of decision 0016 item 5 (pure arithmetic, no torch)."""

import math

import pytest

from tkgat.train.schedule import lr_at, set_lr


def test_none_is_constant():
    assert all(lr_at(e, 1e-3, "none", 100) == 1e-3 for e in (1, 2, 50, 100, 500))


def test_cosine_starts_at_lr0_and_ends_at_the_final_ratio():
    assert lr_at(1, 1e-3, "cosine", 100) == pytest.approx(1e-3)
    assert lr_at(100, 1e-3, "cosine", 100) == pytest.approx(1e-5)  # lr0 / 100
    assert lr_at(100, 1e-3, "cosine", 100, final_ratio=0.1) == pytest.approx(1e-4)


def test_cosine_midpoint_and_monotone_decrease():
    horizon = 101  # (e - 1) / (horizon - 1) = 0.5 at epoch 51
    mid = lr_at(51, 1.0, "cosine", horizon, final_ratio=0.01)
    assert mid == pytest.approx(0.01 + 0.99 * 0.5)
    values = [lr_at(e, 1.0, "cosine", horizon) for e in range(1, horizon + 1)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    expected = 0.01 + 0.99 * (1 + math.cos(math.pi * 10 / 100)) / 2
    assert lr_at(11, 1.0, "cosine", horizon) == pytest.approx(expected)


def test_epochs_after_the_horizon_keep_the_final_rate_and_one_epoch_is_constant():
    assert lr_at(250, 1e-3, "cosine", 100) == pytest.approx(1e-5)
    assert lr_at(1, 1e-3, "cosine", 1) == 1e-3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schedule": "step"}, "lr_schedule"),
        ({"schedule": "cosine", "final_ratio": 0}, "lr_final_ratio"),
        ({"schedule": "cosine", "final_ratio": 1.5}, "lr_final_ratio"),
        ({"schedule": "cosine", "epoch": 0}, "1-based"),
    ],
)
def test_bad_input_is_refused(kwargs, match):
    args = {"epoch": 1, "base_lr": 1e-3, "schedule": "cosine", "horizon": 10, "final_ratio": 0.01, **kwargs}
    with pytest.raises(ValueError, match=match):
        lr_at(args["epoch"], args["base_lr"], args["schedule"], args["horizon"], args["final_ratio"])


def test_set_lr_writes_every_group_of_every_optimiser():
    class Opt:
        def __init__(self, n):
            self.param_groups = [{"lr": 1.0} for _ in range(n)]

    a, b = Opt(1), Opt(2)
    set_lr([a, b], 0.25)
    assert [g["lr"] for g in a.param_groups + b.param_groups] == [0.25, 0.25, 0.25]
