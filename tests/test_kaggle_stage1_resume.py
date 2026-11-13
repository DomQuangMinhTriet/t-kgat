"""Stage 1 launch guards and bounded scheduling without real training."""

import ast
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from .conftest import ROOT


def cells(notebook_name="07_kaggle_tkgat_stage1_resume.ipynb"):
    notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            code.append(source)
    return code


def helpers(notebook_name="07_kaggle_tkgat_stage1_resume.ipynb"):
    tree = ast.parse(cells(notebook_name)[1])
    definitions = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "stage1-helpers", "exec"), namespace)
    return namespace


@pytest.fixture
def suite(tmp_path):
    for name in (
        "configs/model/kgat.yaml", "configs/model/tkgat.yaml", "configs/model/tkgat_kg.yaml", "scripts/run.py",
    ):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = yaml.safe_load((tmp_path / "configs/model/kgat.yaml").read_text(encoding="utf-8"))
    folder = tmp_path / "results/runs/fixture"
    folder.mkdir(parents=True)
    (folder / "kgat_summary.json").write_text(json.dumps({"selected": config["grid"][1]}), encoding="utf-8")
    return {"ROOT": tmp_path, "RUN_NAME": "fixture", "complete": lambda tag: tag != "tkgat"}


def test_missing_completed_kgat_is_rejected():
    suite = {"complete": lambda tag: tag in {"mostpop", "bprmf"}}
    with pytest.raises(RuntimeError, match="KGAT is incomplete"):
        helpers()["require_stage1_inputs"](suite)


def test_gamma_grid_inherits_selected_kgat_parameters(suite):
    points = helpers()["require_stage1_inputs"](suite)
    selected = json.loads((suite["ROOT"] / "results/runs/fixture/kgat_summary.json").read_text())["selected"]
    assert [point["gamma"] for point in points] == [1, 3, 10]
    assert all({key: value for key, value in point.items() if key != "gamma"} == selected for point in points)


@pytest.mark.parametrize("change", ["gamma", "seeds", "stage2", "fixed"])
def test_changed_protocol_is_rejected_before_training(suite, change):
    path = suite["ROOT"] / "configs/model/tkgat.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if change == "gamma":
        config["grid"] = [{"gamma": 1}]
    elif change == "seeds":
        config["seeds"] = [2019]
    elif change == "stage2":
        config["time_on"] = "kg"
    else:
        config["fixed"]["dim"] = 32
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        helpers()["require_stage1_inputs"](suite)


@pytest.mark.parametrize("already_complete", [False, True])
def test_only_stage1_runs_and_finished_model_is_not_retested(monkeypatch, already_complete):
    calls, backups, verified = [], [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake CUDA")))

    def complete(tag):
        assert tag == "tkgat"
        return already_complete or len(calls) == 2

    suite = {"complete": complete, "execute": calls.append, "backup": backups.append, "DATA_CONFIG": "fixture.yaml"}
    namespace = {"suite": suite, "require_stage1_inputs": verified.append, "RUN_TRAINING": True,
                 "NEW_TKGAT_FITS_THIS_SESSION": 2, "CHECKPOINT_ZIP": "fixture.zip"}
    exec(compile(cells()[-1], "stage1-runner", "exec"), namespace)
    assert verified == [suite]
    assert len(calls) == (0 if already_complete else 2)
    for args in calls:
        assert args[args.index("--model") + 1] == "configs/model/tkgat.yaml"
        assert "--resume" in args and args[args.index("--max-new-runs") + 1] == "1"
    assert len(backups) == len(calls) + 1
