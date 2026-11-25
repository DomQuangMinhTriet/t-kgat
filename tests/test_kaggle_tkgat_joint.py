"""Standalone T-KGAT: no baseline, bounded resume, and fail-fast full-data memory checks."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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


@pytest.mark.parametrize(
    "extra, refit",
    [([], True), (["--no-refit"], False), (["--val-only"], False), (["--benchmark-batches", "2"], False)],
)
def test_kaggle_wrapper_enables_refit_except_for_validation_only(tmp_path, monkeypatch, extra, refit):
    wrapper = load_wrapper()
    processed = tmp_path / "lastfm-3m-k10"
    (processed / "kg/default").mkdir(parents=True)
    for name in ("meta.json", "kg/default/kg_meta.json"):
        (processed / name).write_text("{}")
    monkeypatch.setattr(wrapper, "memory_report", lambda *args, **kwargs: {"blockers": []})
    commands = []

    class FakeProcess:
        stdout = ()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    def launch(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(wrapper.subprocess, "Popen", launch)
    wrapper.main(["--processed", str(processed), "--out", str(tmp_path / "output"), *extra])
    assert len(commands) == 1 and ("--refit" in commands[0]) == refit
    assert ("--val-only" in commands[0]) == ("--val-only" in extra or "--benchmark-batches" in extra)
    assert "--epochs-per-fit" in commands[0]
    assert commands[0][commands[0].index("--max-new-runs") + 1] == "2"
    assert commands[0][commands[0].index("--epochs-per-fit") + 1] == "150"
    assert str(ROOT / "configs/model/tkgat_joint.yaml") in commands[0]
    identity = json.loads((tmp_path / "output/run-identity.json").read_text())
    assert identity["run_name"] == "lastfm-3m-k10-joint"


def test_restore_merges_saved_epoch_into_benchmark_only_output_but_never_overwrites_progress(tmp_path):
    source, output = tmp_path / "previous", tmp_path / "output"
    (source / "runs/states").mkdir(parents=True)
    (output / "runs").mkdir(parents=True)
    identity = {"source": "same", "split": "same"}
    (source / "run-identity.json").write_text(json.dumps(identity))
    (source / "runs/states/partial.epoch.pt").write_bytes(b"saved weights and optimizer")
    (output / "runs/benchmark.json").write_text("{}")
    wrapper = load_wrapper()
    wrapper.restore_results(source, output, identity)
    assert (output / "runs/states/partial.epoch.pt").read_bytes() == b"saved weights and optimizer"
    assert (output / "runs/benchmark.json").exists()
    with pytest.raises(ValueError, match="already has progress"):
        wrapper.restore_results(source, output, identity)


def test_shipped_notebook_has_valid_python_and_only_uses_the_joint_runner():
    notebook = json.loads((ROOT / "notebooks/10_kaggle_tkgat_joint.ipynb").read_text(encoding="utf-8"))
    source = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            compile(code, "kaggle_tkgat_joint", "exec")
            source += code
    assert "kaggle_tkgat_joint.py" in source and "RUN_TRAINING = False" in source
    assert "MAX_NEW_RUNS = 2" in source and "EPOCHS_PER_FIT = None" in source
    assert "kaggle_lastfm.py" not in source
    cfg = yaml.safe_load((ROOT / "configs/model/tkgat_joint.yaml").read_text())
    assert cfg["standalone"] and cfg["time_on"] == "both"
    assert cfg["seeds"] == [2019, 2020, 2021, 2022, 2023]
    assert [p["gamma"] for p in cfg["grid"]] == [1, 3, 10]
    assert not any(key in cfg for key in ("from_kgat", "from_tkgat", "control"))


@pytest.mark.parametrize("conflict", [False, True])
def test_notebook_can_mount_previous_output_with_the_same_extracted_code(tmp_path, conflict):
    notebook = json.loads((ROOT / "notebooks/10_kaggle_tkgat_joint.ipynb").read_text(encoding="utf-8"))
    setup = "".join(next(c for c in notebook["cells"] if c["id"] == "joint-1")["source"])
    inputs, code, output = tmp_path / "input", tmp_path / "code", tmp_path / "output"
    for index, root in enumerate((inputs / "source-dataset", inputs / "previous-output/tkgat-joint-code")):
        root.mkdir(parents=True)
        contents = b"x = 2\n" if conflict and index else b"x = 1\n"
        (root / "example.py").write_bytes(contents)
        (root / "SOURCE_SHA256.json").write_text(json.dumps({"example.py": hashlib.sha256(contents).hexdigest()}))
    processed = inputs / "processed-dataset/lastfm-3m-k10"
    (processed / "kg/default").mkdir(parents=True)
    for name in ("meta.json", "train.csv", "kg/default/kg_meta.json"):
        (processed / name).write_text("{}")
    setup = setup.replace('Path("/kaggle/input")', f"Path({str(inputs)!r})")
    setup = setup.replace('Path("/kaggle/working/tkgat-joint-code")', f"Path({str(code)!r})")
    setup = setup.replace('Path("/kaggle/working/tkgat-joint")', f"Path({str(output)!r})")
    scope = {}
    if conflict:
        with pytest.raises(RuntimeError, match="different code versions"):
            exec(compile(setup, "notebook-setup", "exec"), scope)
    else:
        exec(compile(setup, "notebook-setup", "exec"), scope)
        assert scope["PROCESSED"] == processed
        assert (code / "example.py").read_bytes() == b"x = 1\n"


@pytest.fixture
def notebook_runner(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / "notebooks/10_kaggle_tkgat_joint.ipynb").read_text(encoding="utf-8"))
    setup = "".join(next(c for c in notebook["cells"] if c["id"] == "joint-2")["source"])
    setup = setup.split('run_tkgat("--check-only")')[0]
    scope = {
        "CODE": ROOT,
        "PROCESSED": tmp_path,
        "OUTPUT": tmp_path / "output",
        "os": os,
        "sys": sys,
        "subprocess": subprocess,
        "json": json,
    }
    exec(compile(setup, "notebook-runner", "exec"), scope)

    def run(lines, returncode=0, *extra):
        class FakeProcess:
            stdout = lines

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def wait(self):
                return returncode

        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
        scope["run_tkgat"](*extra)

    return run


def test_notebook_shows_completed_and_next_fit_without_batch_losses_or_metadata(notebook_runner, capsys):
    notebook_runner(
        [
            '{"train_pairs": 2428891}\n',
            "[train:select seed=2019 gamma=1] Initializing model and samplers on cuda\n",
            "[train:select seed=2019 gamma=1] epoch 1 CF 25/2372 loss=0.69 elapsed=6.4s\n",
            "[train:select seed=2019 gamma=1] Saved epoch 1: checkpoint.pt\n",
            "[val] grid={'gamma': 1} seed=2019 recall@20=0.5\n",
            "[train:select seed=2020 gamma=1] Initializing model and samplers on cuda\n",
            "[val] grid={'gamma': 1} seed=2020 recall@20=0.6\n",
            "Partial grid saved. Re-run with --resume; selection/test wait until the grid is complete.\n",
        ]
    )
    output = capsys.readouterr().out
    assert "Running run 1/20" in output and "Result ready: run 2/20" in output
    assert "Next: run 3/20" in output
    assert all(hidden not in output for hidden in ("train_pairs", "CF 25", "loss=", "recall@20", "Saved epoch"))


def test_notebook_reports_refit_continuation_and_result_by_its_global_run_number(notebook_runner, capsys):
    notebook_runner(
        [
            "selected grid point {'gamma': 3} (mean val recall@20 = 0.5)\n",
            "[train:refit seed=2019 gamma=3] Initializing model and samplers on cuda\n",
            "[train:refit seed=2019 gamma=3] Resumed after epoch 20; next epoch 21\n",
            "[train:refit seed=2019 gamma=3] epoch 21 KG 25/2725 loss=0.02\n",
            "[test] seed=2019 recall@20=0.7\n",
        ]
    )
    output = capsys.readouterr().out
    assert "Continuing run 16/20 (refit, gamma=3, seed=2019)" in output
    assert "next epoch 21" in output and "Result ready: run 16/20" in output
    assert "KG 25" not in output and "recall@20" not in output


def test_notebook_preserves_error_details_even_in_quiet_preflight(notebook_runner, capsys):
    with pytest.raises(subprocess.CalledProcessError):
        notebook_runner(["Traceback: checkpoint failed\n", "RuntimeError: sample failure\n"], 1, "--check-only")
    assert "RuntimeError: sample failure" in capsys.readouterr().out
