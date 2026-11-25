"""The schedule inside the BPR and KGAT trainers (CPU, tiny synthetic split)."""

from dataclasses import replace

import pytest

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.train.bpr import BPRConfig, fit_bpr
from tkgat.train.kgat import fit_kgat, split_params
from tkgat.train.schedule import lr_at

from .test_train_kgat import FIT, MODEL, make_ckg, stable

BPR = BPRConfig(dim=8, lr=0.01, batch_size=64, max_epochs=6, eval_every=1, patience=99, device="cpu")


@pytest.fixture
def enc(synthetic):
    return encode(temporal_split(synthetic, SplitConfig(kcore=2)))


def lrs(history):
    return [row["lr"] for row in history]


def losses(history, key):
    return [row[key] for row in history]


def test_bpr_none_logs_a_constant_rate(enc):
    fit = fit_bpr(enc, BPR, seed=0)
    assert lrs(fit.history) == [0.01] * 6


def test_bpr_cosine_follows_the_formula_and_changes_the_loss_after_epoch_one(enc):
    const = fit_bpr(enc, BPR, seed=0)
    cos = fit_bpr(enc, replace(BPR, lr_schedule="cosine"), seed=0)
    assert lrs(cos.history) == pytest.approx([lr_at(e, 0.01, "cosine", 6, 0.01) for e in range(1, 7)])
    assert cos.history[0]["lr"] == 0.01 and cos.history[-1]["lr"] == pytest.approx(1e-4)
    assert losses(cos.history, "loss")[0] == losses(const.history, "loss")[0]  # epoch 1 has the same rate
    assert losses(cos.history, "loss")[1:] != losses(const.history, "loss")[1:]


def test_bpr_cosine_with_ratio_one_is_the_constant_run(enc):
    const = fit_bpr(enc, BPR, seed=3)
    flat = fit_bpr(enc, replace(BPR, lr_schedule="cosine", lr_final_ratio=1.0), seed=3)
    assert losses(flat.history, "loss") == losses(const.history, "loss")
    assert flat.best_epoch == const.best_epoch


def test_bpr_refuses_an_unknown_schedule(enc):
    with pytest.raises(ValueError, match="lr_schedule"):
        fit_bpr(enc, replace(BPR, lr_schedule="step"), seed=0)


def test_kgat_cosine_follows_the_formula_on_both_optimisers(enc):
    cfg = replace(FIT, max_epochs=5, patience=99, lr_schedule="cosine")
    fit = fit_kgat(enc, make_ckg(enc), MODEL, cfg, seed=0)
    assert lrs(fit.history) == pytest.approx([lr_at(e, 0.01, "cosine", 5, 0.01) for e in range(1, 6)])
    const = fit_kgat(enc, make_ckg(enc), MODEL, replace(FIT, max_epochs=5, patience=99), seed=0)
    assert lrs(const.history) == [0.01] * 5
    assert losses(fit.history, "loss_cf")[0] == losses(const.history, "loss_cf")[0]
    assert losses(fit.history, "loss_kg")[1:] != losses(const.history, "loss_kg")[1:]


def test_kgat_cosine_with_ratio_one_is_the_constant_run(enc):
    ckg = make_ckg(enc)
    const = fit_kgat(enc, ckg, MODEL, FIT, seed=5)
    flat = fit_kgat(enc, ckg, MODEL, replace(FIT, lr_schedule="cosine", lr_final_ratio=1.0), seed=5)
    assert stable(flat.history) == stable(const.history)


def test_split_params_routes_the_schedule_to_the_trainer_config():
    _, fit = split_params({"lr": 0.001, "lr_schedule": "cosine", "lr_final_ratio": 0.05, "l2": 1e-5}, "cpu")
    assert fit.lr_schedule == "cosine" and fit.lr_final_ratio == 0.05
