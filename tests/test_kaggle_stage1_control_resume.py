"""Control configuration inheritance and overnight scheduling without training."""

import json
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from . import test_kaggle_stage1_resume as stage1_tests
from .conftest import ROOT

NOTEBOOK = "09_kaggle_tkgat_stage1_control_resume.ipynb"


def test_control_requires_completed_stage1():
    with pytest.raises(RuntimeError, match="Stage 1 is incomplete"):
        stage1_tests.helpers(NOTEBOOK)["require_control_inputs"]({"complete": lambda tag: False})


@pytest.mark.parametrize("gamma", [1, 3, 10])
def test_control_inherits_selected_gamma_and_parameters(tmp_path, gamma):
    for name in ("kgat.yaml", "tkgat.yaml", "tkgat_perm.yaml"):
        destination = tmp_path / "configs/model" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "configs/model" / name, destination)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/run.py", scripts / "run.py")
    folder = tmp_path / "results/runs/fixture"
    folder.mkdir(parents=True)
    selected = {"lr": 0.001, "l2": 0.0001, "l2_kg": 1e-5, "mess_dropout": 0.1, "gamma": gamma}
    (folder / "tkgat_summary.json").write_text(json.dumps({"selected": selected}), encoding="utf-8")
    suite = {"ROOT": tmp_path, "RUN_NAME": "fixture", "complete": lambda tag: tag == "tkgat"}
    assert stage1_tests.helpers(NOTEBOOK)["require_control_inputs"](suite) == [selected]

    path = tmp_path / "configs/model/tkgat_perm.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["control"] = "permute_kg_ts"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="interaction-timestamp permutation"):
        stage1_tests.helpers(NOTEBOOK)["require_control_inputs"](suite)


@pytest.mark.parametrize("completed_seeds", [0, 3, 5])
def test_one_session_finishes_remaining_seeds_and_stops(monkeypatch, completed_seeds):
    calls, backups = [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake CUDA")))

    def complete(tag):
        assert tag == "tkgat_perm"
        return completed_seeds + len(calls) >= 5

    suite = {"complete": complete, "execute": calls.append, "backup": backups.append, "DATA_CONFIG": "fixture.yaml"}
    namespace = {"suite": suite, "require_control_inputs": lambda _: None}
    code = stage1_tests.cells(NOTEBOOK)
    exec(compile(code[0], "control-settings", "exec"), namespace)
    assert namespace["NEW_CONTROL_FITS_THIS_SESSION"] == 5
    exec(compile(code[-1], "control-runner", "exec"), namespace)
    assert len(calls) == 5 - completed_seeds
    for args in calls:
        assert args[args.index("--model") + 1] == "configs/model/tkgat_perm.yaml"
        assert "--resume" in args and args[args.index("--max-new-runs") + 1] == "1"
    assert len(backups) == len(calls) + 1


def test_failed_fit_still_saves_recoverable_progress(monkeypatch):
    backups = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake CUDA")))

    def execute(args):
        raise RuntimeError("fit failed")

    suite = {"complete": lambda tag: False, "execute": execute, "backup": backups.append,
             "DATA_CONFIG": "fixture.yaml"}
    namespace = {"suite": suite, "require_control_inputs": lambda _: None}
    code = stage1_tests.cells(NOTEBOOK)
    exec(compile(code[0], "control-settings", "exec"), namespace)
    with pytest.raises(RuntimeError, match="fit failed"):
        exec(compile(code[-1], "control-runner", "exec"), namespace)
    assert backups == [namespace["CHECKPOINT_ZIP"]]
