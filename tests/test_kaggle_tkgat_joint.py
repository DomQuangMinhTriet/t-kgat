"""Standalone T-KGAT: no baseline, bounded resume, and fail-fast full-data memory checks."""

import importlib.util
import json
from zipfile import ZipFile

import pytest
import torch
import yaml

from .conftest import ROOT
from .test_run_kgat import GRID, load_run_script
from .test_tkgat import edit
from .test_tkgat_stage2 import run_tag
from .test_tkgat_stage2 import s2setup as _s2setup

s2setup = _s2setup


def load_wrapper():
    spec = importlib.util.spec_from_file_location("kaggle_tkgat_joint", ROOT / "scripts/kaggle_tkgat_joint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standalone(s2setup):
    edit(s2setup, "tkgat_kg", standalone=True, from_kgat=None, time_on="both", base_point=GRID[1])
    (s2setup.models / "kgat.yaml").unlink()
    (s2setup.runs / "kgat_summary.json").unlink()


def test_only_tkgat_runs_without_any_baseline_config_or_summary_and_resumes(s2setup, monkeypatch):
    standalone(s2setup)
    script = load_run_script()
    real, calls = script.fit_kgat, []

    def spy(data, ckg, model_cfg, fit_cfg, seed):
        assert model_cfg.time_on == "both" and model_cfg.gamma > 0
        calls.append((model_cfg.gamma, seed))
        return real(data, ckg, model_cfg, fit_cfg, seed)

    monkeypatch.setattr(script, "fit_kgat", spy)
    for step in range(4):
        reg = run_tag(s2setup, "tkgat_kg", "--resume", "--max-new-runs", "1", script=script)
        assert set(reg.model) == {"tkgat_kg"}
        assert len(calls) == step + 1
        if step < 3:
            assert set(reg.phase) == {"val"}
            assert not (s2setup.runs / "tkgat_kg_summary.json").exists()
    assert set(calls) == {(1, 1), (1, 2), (3, 1), (3, 2)}
    assert len(reg[reg.phase == "test"]) == 2
    summary = json.loads((s2setup.runs / "tkgat_kg_summary.json").read_text())
    assert summary["tkgat"]["from"] == "standalone"
    before = len(reg)
    reg = run_tag(s2setup, "tkgat_kg", "--resume", script=script)
    assert len(calls) == 4 and len(reg) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"from_kgat": "kgat"},
        {"control": "permute_train_ts"},
        {"grid": [{"gamma": 0}]},
        {"time_on": "kg"},
        {"base_point": {"lr": 0.001}},
        {"fixed": {"gamma": 1}},
    ],
)
def test_standalone_rejects_dependencies_or_invalid_points(s2setup, changes):
    standalone(s2setup)
    edit(s2setup, "tkgat_kg", **changes)
    path = s2setup.models / "tkgat_kg.yaml"
    with pytest.raises(ValueError):
        load_run_script().tkgat_points(yaml.safe_load(path.read_text()), path, s2setup.runs)


def test_full_data_is_blocked_before_csv_loading_or_training(tmp_path, monkeypatch):
    wrapper = load_wrapper()
    processed = tmp_path / "processed"
    (processed / "kg/default").mkdir(parents=True)
    (processed / "meta.json").write_text(json.dumps({"train_pos": 225429559, "n_users": 117872}))
    (processed / "kg/default/kg_meta.json").write_text(
        json.dumps(
            {
                "n_entities": 16458190,
                "n_triples": {"t1": 482762},
            }
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (15 * 2**30, 16 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test 16 GB GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not launch on insufficient memory")

    monkeypatch.setattr(wrapper.subprocess, "Popen", forbidden)
    output = tmp_path / "output"
    args = ["--processed", str(processed), "--out", str(output)]
    wrapper.main([*args, "--check-only"])
    report = json.loads((output / "memory-report.json").read_text())
    assert report["train_pairs"] == 225429559 and report["device_memory_lower_bound_GiB"] > 37
    assert report["blockers"] and not (output / "registry.csv").exists()
    with pytest.raises(SystemExit, match="Training was not started"):
        wrapper.main(args)


def test_code_package_preserves_source_and_contains_no_processed_or_export_data(tmp_path):
    wrapper = load_wrapper()
    source = tmp_path / "repo"
    (source / "src").mkdir(parents=True)
    (source / "src/example.py").write_bytes(b"# Windows newline\r\nx = 1\r\n")
    (source / "pyproject.toml").write_text("[project]\nname='example'\n")
    (source / "data/processed").mkdir(parents=True)
    (source / "data/processed/train.csv").write_text("private data")
    archive = tmp_path / "code.zip"
    wrapper.package_code(source, archive)
    with ZipFile(archive) as zipped:
        assert not any(name.startswith("data/") for name in zipped.namelist())
        assert zipped.read("src/example.py") == (source / "src/example.py").read_bytes()
        assert json.loads(zipped.read("SOURCE_SHA256.json")) == wrapper.source_identity(source)


def test_restore_rejects_another_source_or_dataset(tmp_path):
    source, output = tmp_path / "previous", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "run-identity.json").write_text(json.dumps({"source": "old"}))
    with pytest.raises(ValueError, match="another source"):
        load_wrapper().restore_results(source, output, {"source": "new"})
    assert not (output / "registry.csv").exists()


def test_shipped_notebook_has_valid_python_and_only_uses_the_joint_runner():
    notebook = json.loads((ROOT / "notebooks/10_kaggle_tkgat_joint.ipynb").read_text(encoding="utf-8"))
    source = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            compile(code, "kaggle_tkgat_joint", "exec")
            source += code
    assert "kaggle_tkgat_joint.py" in source and "RUN_TRAINING = False" in source
    assert "kaggle_lastfm.py" not in source
    cfg = yaml.safe_load((ROOT / "configs/model/tkgat_joint.yaml").read_text())
    assert cfg["standalone"] and cfg["time_on"] == "both"
    assert cfg["seeds"] == [2019, 2020, 2021, 2022, 2023]
    assert [p["gamma"] for p in cfg["grid"]] == [1, 3, 10]
    assert not any(key in cfg for key in ("from_kgat", "from_tkgat", "control"))
