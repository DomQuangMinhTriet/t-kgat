"""scripts/run.py --val-only: no test row, no test file opened, same selection, default behaviour unchanged."""

import importlib.util
import json

import pandas as pd
import pytest
import yaml

from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline.build import load_saved, save_split

from .conftest import ROOT

RUN_NAME = "synthetic-val"
BPRMF = {
    "name": "bprmf",
    "seeds": [1, 2],
    "grid": [{"dim": 4, "lr": 0.05, "l2": 0.0001}, {"dim": 8, "lr": 0.05, "l2": 0.0001}],
    "fixed": {"batch_size": 64, "max_epochs": 3, "eval_every": 1, "patience": 2, "select_metric": "recall@20"},
}
MOSTPOP = {"name": "mostpop", "grid": [{"window_days": None}, {"window_days": 7}]}


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def setup(synthetic, tmp_path):
    cfg = {"name": "synthetic", "run_name": RUN_NAME, "processed_dir": str(tmp_path / "processed")}
    save_split(temporal_split(synthetic, SplitConfig(kcore=2)), cfg, {})
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg))
    models = {}
    for name, body in (("bprmf", BPRMF), ("mostpop", MOSTPOP)):
        models[name] = tmp_path / f"{name}.yaml"
        models[name].write_text(yaml.safe_dump(body))
    return tmp_path, data_cfg, models


def run(setup, model, *extra, runs="runs", registry="registry.csv"):
    tmp, data_cfg, models = setup
    load_run_script().main(
        ["--data", str(data_cfg), "--model", str(models[model]), "--device", "cpu"]
        + ["--registry", str(tmp / registry), "--runs-dir", str(tmp / runs), *extra]
    )
    return tmp / registry, tmp / runs / RUN_NAME


@pytest.mark.parametrize("model", ["bprmf", "mostpop"])
def test_val_only_writes_no_test_row_and_never_opens_the_test_files(setup, model):
    tmp, *_ = setup
    # if the run opened them it would fail: they are gone
    (tmp / "processed" / "test.csv").unlink()
    (tmp / "processed" / "seen_before_test.csv").unlink()
    registry, out = run(setup, model, "--val-only")
    reg = pd.read_csv(registry)
    assert set(reg["phase"]) == {"val"} and set(reg["model"]) == {model}
    assert not list(out.glob("*.npz"))  # no per-user test file
    assert not (out / f"{model}_summary.json").exists()  # the summary that holds test scores
    summary = json.loads((out / f"{model}_val_summary.json").read_text())
    assert summary["val_only"] is True and "test_mean_std" not in summary
    assert len(summary["val_means_recall@20"]) == 2 and len(summary["val_stds_recall@20"]) == 2


def test_without_the_flag_the_test_files_are_required(setup):
    tmp, *_ = setup
    (tmp / "processed" / "test.csv").unlink()
    with pytest.raises(FileNotFoundError):
        run(setup, "mostpop")


@pytest.mark.parametrize("model", ["bprmf", "mostpop"])
def test_default_behaviour_is_unchanged_and_selection_is_the_same(setup, model):
    reg_default, out_default = run(setup, model, runs="runs_default", registry="default.csv")
    reg_val, out_val = run(setup, model, "--val-only", runs="runs_val", registry="val.csv")
    default = pd.read_csv(reg_default)
    assert set(default["phase"]) == {"val", "test"} and (out_default / f"{model}_summary.json").exists()
    assert not (out_default / f"{model}_val_summary.json").exists()
    full = json.loads((out_default / f"{model}_summary.json").read_text())
    val_only = json.loads((out_val / f"{model}_val_summary.json").read_text())
    assert val_only["selected"] == full["selected"]
    assert val_only["val_means_recall@20"] == pytest.approx(full["val_means_recall@20"])
    val_rows = default[default["phase"] == "val"].drop(columns=["run_at", "seconds", "git_commit"])
    only_rows = pd.read_csv(reg_val).drop(columns=["run_at", "seconds", "git_commit"])
    # the default registry also holds test rows, so best_epoch is float there: compare values, not dtypes
    pd.testing.assert_frame_equal(val_rows.reset_index(drop=True), only_rows.reset_index(drop=True), check_dtype=False)


def test_load_saved_without_test_gives_empty_test_parts(setup):
    tmp, data_cfg, _ = setup
    cfg = yaml.safe_load(data_cfg.read_text())
    (tmp / "processed" / "test.csv").unlink()
    (tmp / "processed" / "seen_before_test.csv").unlink()
    split, _ = load_saved(cfg, with_test=False)
    assert len(split.test) == 0 and len(split.seen_before_test) == 0 and len(split.val) > 0
    assert list(split.test.columns) == ["user", "item", "rating", "ts"]
