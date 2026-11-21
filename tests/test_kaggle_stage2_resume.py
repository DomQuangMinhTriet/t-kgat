"""Stage 2 can start from completed KGAT and preserves its full gamma/seed grid."""

import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from . import test_kaggle_stage1_resume as stage1_tests

stage2_suite = stage1_tests.suite
NOTEBOOK = "08_kaggle_tkgat_stage2_resume.ipynb"


def test_stage2_inherits_kgat_and_does_not_require_stage1(stage2_suite):
    queried = []

    def complete(tag):
        queried.append(tag)
        return tag == "kgat"

    stage2_suite["complete"] = complete
    points = stage1_tests.helpers(NOTEBOOK)["require_stage2_inputs"](stage2_suite)
    selected = json.loads((stage2_suite["ROOT"] / "results/runs/fixture/kgat_summary.json").read_text())["selected"]
    assert queried == ["kgat"]
    assert [point["gamma"] for point in points] == [1, 3, 10]
    assert all(point["time_on"] == "kg" for point in points)
    assert all({key: value for key, value in point.items() if key not in {"gamma", "time_on"}} == selected
               for point in points)


def test_stage2_requires_completed_kgat():
    with pytest.raises(RuntimeError, match="KGAT is incomplete"):
        stage1_tests.helpers(NOTEBOOK)["require_stage2_inputs"]({"complete": lambda tag: False})


@pytest.mark.parametrize("change", ["gamma", "seeds", "time_on", "fixed"])
def test_stage2_rejects_incomplete_grid_or_incompatible_configuration(stage2_suite, change):
    path = stage2_suite["ROOT"] / "configs/model/tkgat_kg.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if change == "gamma":
        config["grid"] = [{"gamma": 1}]
    elif change == "seeds":
        config["seeds"] = [2019]
    elif change == "time_on":
        config["time_on"] = "interact"
    else:
        config["fixed"]["dim"] = 32
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        stage1_tests.helpers(NOTEBOOK)["require_stage2_inputs"](stage2_suite)


@pytest.mark.parametrize("already_complete", [False, True])
def test_stage2_only_launches_its_own_model_and_saves_each_fit(monkeypatch, already_complete):
    calls, backups, verified = [], [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake CUDA")))

    def complete(tag):
        assert tag == "tkgat_kg"
        return already_complete or len(calls) == 2

    suite = {"complete": complete, "execute": calls.append, "backup": backups.append, "DATA_CONFIG": "fixture.yaml"}
    namespace = {"suite": suite, "require_stage2_inputs": verified.append, "RUN_TRAINING": True,
                 "NEW_TKGAT_FITS_THIS_SESSION": 2, "CHECKPOINT_ZIP": "stage2-fixture.zip"}
    exec(compile(stage1_tests.cells(NOTEBOOK)[-1], "stage2-runner", "exec"), namespace)
    assert verified == [suite]
    assert len(calls) == (0 if already_complete else 2)
    for args in calls:
        assert args[args.index("--model") + 1] == "configs/model/tkgat_kg.yaml"
        assert "--resume" in args and args[args.index("--max-new-runs") + 1] == "1"
    assert len(backups) == len(calls) + 1
