"""Refit on train + validation inside the trainers and in scripts/run.py --refit (CPU, tiny synthetic split)."""

import importlib.util
import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from tkgat.data.encode import encode, encode_trainval
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline.build import save_split
from tkgat.train import bpr
from tkgat.train import kgat as kgat_train
from tkgat.train.bpr import BPRConfig, fit_bpr
from tkgat.train.kgat import fit_kgat
from tkgat.train.schedule import lr_at

from .conftest import ROOT
from .test_train_kgat import FIT, MODEL, make_ckg

BPR = BPRConfig(dim=8, lr=0.01, batch_size=64, max_epochs=6, eval_every=2, patience=99, device="cpu")


class Forbidden:
    """Stands for a split part the refit must never read: any use raises."""

    def __getattr__(self, name):
        raise AssertionError(f"the refit read {name!r} of a forbidden split part")

    def __iter__(self):
        raise AssertionError("the refit iterated over a forbidden split part")

    def __len__(self):
        raise AssertionError("the refit took the length of a forbidden split part")


@pytest.fixture
def plain(synthetic):
    """The normal encoding: train as the train part, validation available."""
    return encode(temporal_split(synthetic, SplitConfig(kcore=2)))


@pytest.fixture
def enc(synthetic):
    """The refit encoding: train + validation as the train part."""
    return encode_trainval(temporal_split(synthetic, SplitConfig(kcore=2)))


def sealed(enc):
    """The same training data with validation and test replaced by objects that fail on any use."""
    return replace(enc, val=Forbidden(), mask_val=Forbidden(), test=Forbidden(), mask_test=Forbidden())


def test_the_trainers_still_cannot_read_the_test_split():
    for module, fit in ((bpr, bpr.fit_bpr), (kgat_train, kgat_train.fit_kgat)):
        src = inspect.getsource(module)
        assert "data.test" not in src and "mask_test" not in src
        assert "fixed_epochs" in inspect.signature(fit).parameters


def test_bpr_refit_runs_exactly_e_epochs_without_reading_validation_or_test(enc):
    fit = fit_bpr(sealed(enc), BPR, seed=0, fixed_epochs=3)
    assert [row["epoch"] for row in fit.history] == [1, 2, 3] and fit.best_epoch == 3 and fit.best_val == {}
    assert all("recall@20" not in row for row in fit.history)  # no validation was computed


def test_bpr_refit_is_the_first_e_epochs_of_the_normal_run(plain):
    normal = fit_bpr(plain, BPR, seed=4)
    again = fit_bpr(sealed(plain), BPR, seed=4, fixed_epochs=4)
    assert [r["loss"] for r in again.history] == [r["loss"] for r in normal.history[:4]]


def test_bpr_refit_keeps_the_schedule_horizon_of_max_epochs(enc):
    cfg = replace(BPR, lr_schedule="cosine")
    fit = fit_bpr(sealed(enc), cfg, seed=0, fixed_epochs=3)
    assert [r["lr"] for r in fit.history] == pytest.approx([lr_at(e, 0.01, "cosine", 6, 0.01) for e in (1, 2, 3)])


@pytest.mark.parametrize("bad", [0, 7, -1])
def test_refit_epochs_outside_the_budget_are_refused(enc, bad):
    with pytest.raises(ValueError, match="fixed_epochs"):
        fit_bpr(enc, BPR, seed=0, fixed_epochs=bad)
    with pytest.raises(ValueError, match="fixed_epochs"):
        fit_kgat(enc, make_ckg(enc), MODEL, replace(FIT, max_epochs=6), seed=0, fixed_epochs=bad)


def test_kgat_refit_trains_e_epochs_and_never_evaluates(enc):
    cfg = replace(FIT, max_epochs=6, eval_every=2, patience=1)  # a patience of 1 would stop a normal run early
    fit = fit_kgat(sealed(enc), make_ckg(enc), MODEL, cfg, seed=0, fixed_epochs=5)
    assert [row["epoch"] for row in fit.history] == [1, 2, 3, 4, 5]
    assert fit.best_epoch == 5 and fit.best_val == {} and not fit.model.training
    assert all("recall@20" not in row for row in fit.history)
    assert np.isfinite(fit.model.A.to_dense().detach().numpy()).all()


# ---- scripts/run.py --refit --------------------------------------------------------------------------------------

BPRMF = {
    "name": "bprmf",
    "seeds": [1, 2, 3],
    "grid": [{"dim": 4, "lr": 0.05, "l2": 0.0001}, {"dim": 8, "lr": 0.05, "l2": 0.0001}],
    "fixed": {"batch_size": 64, "max_epochs": 6, "eval_every": 1, "patience": 3, "select_metric": "recall@20"},
}
MOSTPOP = {"name": "mostpop", "grid": [{"window_days": None}, {"window_days": 7}]}


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def setup(synthetic, tmp_path):
    cfg = {"name": "synthetic", "run_name": "synthetic-refit", "processed_dir": str(tmp_path / "processed")}
    save_split(temporal_split(synthetic, SplitConfig(kcore=2)), cfg, {})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    models = {}
    for name, body in (("bprmf", BPRMF), ("mostpop", MOSTPOP)):
        models[name] = tmp_path / f"{name}.yaml"
        models[name].write_text(yaml.safe_dump(body))
    return tmp_path, models


def run(setup, model, *extra, tag="a"):
    tmp, models = setup
    load_run_script().main(
        ["--data", str(tmp / "data.yaml"), "--model", str(models[model]), "--device", "cpu"]
        + ["--registry", str(tmp / f"{tag}.csv"), "--runs-dir", str(tmp / f"runs_{tag}"), *extra]
    )
    return pd.read_csv(tmp / f"{tag}.csv"), tmp / f"runs_{tag}" / "synthetic-refit"


def test_refit_flag_cannot_be_combined_with_val_only(setup):
    with pytest.raises(SystemExit):
        run(setup, "bprmf", "--refit", "--val-only")


def test_bprmf_refit_logs_test_rows_with_the_epoch_count_from_validation(setup):
    reg, out = run(setup, "bprmf", "--refit", tag="refit")
    val, test = reg[reg["phase"] == "val"], reg[reg["phase"] == "test"]
    assert len(val) == 6 and len(test) == 3 and set(test["seed"]) == {1, 2, 3}
    selected = json.loads((out / "bprmf_refit_summary.json").read_text())
    point_rows = val[val["config"] == json.dumps(selected["selected"], sort_keys=True)]
    expected = int(np.ceil(np.median(point_rows["best_epoch"])))
    assert selected["refit"]["epochs"] == expected
    for config in test["config"]:
        row = json.loads(config)
        assert row["refit"] is True and row["epochs"] == expected and {"dim", "lr", "l2"} <= set(row)
    assert sorted(p.name for p in out.glob("*.npz")) == [f"bprmf_refit_seed{s}.npz" for s in (1, 2, 3)]
    assert not (out / "bprmf_summary.json").exists()  # the old summary name is not used by a refit


def test_the_default_path_is_the_old_one_and_the_selection_does_not_depend_on_refit(setup):
    old, out_old = run(setup, "bprmf", tag="old")
    new, out_new = run(setup, "bprmf", "--refit", tag="new")
    cols = ["dataset", "split_sha256", "model", "config", "seed", "phase", "best_epoch", "recall@20", "ndcg@20"]
    a, b = old[old["phase"] == "val"][cols], new[new["phase"] == "val"][cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert (
        json.loads((out_old / "bprmf_summary.json").read_text())["selected"]
        == json.loads((out_new / "bprmf_refit_summary.json").read_text())["selected"]
    )
    assert all("refit" not in json.loads(c) for c in old["config"])


def test_mostpop_refit_counts_on_train_plus_validation(setup):
    reg, out = run(setup, "mostpop", "--refit", tag="mp")
    test = reg[reg["phase"] == "test"]
    assert len(test) == 1 and json.loads(test["config"].iloc[0])["refit"] is True
    assert "epochs" not in json.loads(test["config"].iloc[0])
    assert (out / "mostpop_refit_seed0.npz").exists()


def test_refit_does_not_change_the_validation_rows_of_the_registry_key(setup):
    reg, _ = run(setup, "bprmf", "--refit", tag="keys")
    assert all("refit" not in json.loads(c) for c in reg[reg["phase"] == "val"]["config"])
