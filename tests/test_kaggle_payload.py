"""Portable source/data snapshots and result restore without real-data training."""

import ast
import importlib.util
import json
import shutil
import subprocess
from zipfile import ZipFile

import pytest

from .conftest import ROOT


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def payload(tmp_path):
    packer = script("package_lastfm_kaggle")
    source = tmp_path / "upstream"
    source.mkdir()
    for name in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md", "docs/design-kgat.md", "docs/lastfm-kaggle.md"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (source / "docs/decisions").mkdir()
    data = source / packer.DATA
    data.mkdir(parents=True)
    for part in ("train", "val", "test", "seen_before_val", "seen_before_test"):
        (data / f"{part}.csv").write_text("user,item,rating,ts\n1,2,1,100\n", encoding="utf-8")
    meta = {"clean": {"lastfm_timestamps": {"timestamp_policy": "latest"}},
            "sha256": {part: packer.sha(data / f"{part}.csv") for part in (
                "train", "val", "test", "seen_before_val", "seen_before_test")}}
    (data / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    kg = data / "kg/default"
    kg.mkdir(parents=True)
    for name in ("kg_t1.tsv", "kg_t2.tsv", "kg_t1_dates.tsv", "kg_t2_dates.tsv", "entities.csv", "relations.csv",
                 "kg_meta.json", "dropped.csv"):
        (kg / name).write_text("fixture\n", encoding="utf-8")
    git(source, "init", "-q")
    git(source, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "--allow-empty", "-qm", "upstream fixture")
    original_head = git(source, "rev-parse", "HEAD")
    archive = tmp_path / "payload.zip"
    result = packer.package(source, archive)
    assert git(source, "rev-parse", "HEAD") == original_head
    unpacked = tmp_path / "payload"
    with ZipFile(archive) as zipped:
        assert not any(".git/" in name for name in zipped.namelist())
        zipped.extractall(unpacked)
    cloned = tmp_path / "cloned"
    subprocess.run(["git", "clone", "-q", str(unpacked / "source.bundle"), str(cloned)], check=True)
    return unpacked, cloned, result


def test_payload_freezes_code_without_putting_data_in_git_and_restores_progress(payload, tmp_path, monkeypatch):
    unpacked, cloned, result = payload
    assert git(cloned, "rev-parse", "HEAD") == result["source_commit"]
    tracked = git(cloned, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert not any(name.startswith("data/") for name in tracked)
    runtime = script("kaggle_lastfm")
    monkeypatch.setattr(runtime, "ROOT", cloned)
    runtime.prepare(unpacked)
    assert not git(cloned, "status", "--porcelain", "--untracked-files=no")
    folder = cloned / "results/runs/lastfm-u25-latest/states"
    folder.mkdir(parents=True)
    (folder / "kgat_0_seed2019.pt").write_bytes(b"synthetic-checkpoint")
    checkpoint = tmp_path / "checkpoint.zip"
    runtime.backup(checkpoint)
    with ZipFile(checkpoint) as zipped:
        assert not any(".git/" in name for name in zipped.namelist())
        assert zipped.read("results/runs/lastfm-u25-latest/states/kgat_0_seed2019.pt") == b"synthetic-checkpoint"
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", str(unpacked / "source.bundle"), str(fresh)], check=True)
    monkeypatch.setattr(runtime, "ROOT", fresh)
    runtime.prepare(unpacked, checkpoint)
    assert (fresh / folder.relative_to(cloned) / "kgat_0_seed2019.pt").read_bytes() == b"synthetic-checkpoint"
    with pytest.raises(ValueError, match="already contain progress"):
        runtime.extract_results(checkpoint)


def test_payload_refuses_changed_input_and_changed_source(payload, monkeypatch):
    unpacked, cloned, _ = payload
    runtime = script("kaggle_lastfm")
    monkeypatch.setattr(runtime, "ROOT", cloned)
    file = unpacked / "data/processed/lastfm-u25-latest/train.csv"
    old = file.read_bytes()
    file.write_bytes(old + b"changed")
    with pytest.raises(ValueError, match="Payload hash mismatch"):
        runtime.prepare(unpacked)
    file.write_bytes(old)
    runtime.prepare(unpacked)
    (cloned / "src/tkgat/models/kgat.py").write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        runtime.record_identity()


def test_suite_only_requests_bounded_fits_from_the_first_pending_model(monkeypatch, tmp_path):
    runtime = script("kaggle_lastfm")
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "record_identity", lambda: {})
    monkeypatch.setattr(runtime, "complete", lambda tag: tag in {"mostpop", "bprmf"})
    calls, saved = [], []
    monkeypatch.setattr(runtime, "execute", calls.append)
    monkeypatch.setattr(runtime, "backup", saved.append)
    target = tmp_path / "checkpoint.zip"
    runtime.train(2, 2, target, "cpu")
    assert len(calls) == len(saved) == 2
    for args in calls:
        assert "configs/model/kgat.yaml" in args and "--resume" in args
        assert args[args.index("--max-new-runs") + 1] == "2"


def test_notebook_cells_are_valid_python_and_training_is_disabled_by_default():
    notebook = json.loads((ROOT / "notebooks/04_kaggle_lastfm_latest.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            ast.parse(text)
            code.append(text)
            assert cell["outputs"] == [] and cell["execution_count"] is None
    source = "\n".join(code)
    assert "RUN_TRAINING = False" in source and "RUN_MEASUREMENT = False" in source
    assert "generate_lastfm_drive.py" not in source
