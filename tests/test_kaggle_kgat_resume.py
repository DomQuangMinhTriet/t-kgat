"""Resume notebook recovery and guards; never train a real model."""

import ast
import csv
import json
import shutil
from zipfile import ZipFile

import pytest

from . import test_kaggle_payload as payload_tests
from .conftest import ROOT

payload = payload_tests.payload


def notebook_helpers():
    notebook = json.loads((ROOT / "notebooks/05_kaggle_kgat_resume.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def recovery_archive(" in "".join(cell["source"]))
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "resume-helpers", "exec"), namespace)
    return namespace


@pytest.fixture
def recovery(tmp_path):
    root = tmp_path / "input/recovery/results"
    root.mkdir(parents=True)
    identity = {"source_commit": "frozen", "dataset": "lastfm-u25-latest", "split_meta_sha256": "same"}
    (root / "kaggle-run.json").write_text(json.dumps(identity), encoding="utf-8")
    (root / "registry.csv").write_text("keep-this-registry-byte-for-byte\n", encoding="utf-8")
    state = root / "runs/lastfm-u25-latest/states/kgat_0_seed2019.pt"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"completed-fit-fixture")
    return root, identity


@pytest.mark.parametrize("zipped", [False, True])
def test_resolve_zip_or_extracted_recovery_without_changing_registry(recovery, tmp_path, zipped):
    root, identity = recovery
    original = (root / "registry.csv").read_bytes()
    if zipped:
        package = root.parent / "arbitrary-recovery-name.zip"
        with ZipFile(package, "w") as archive:
            for file in root.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(root.parent))
        shutil.rmtree(root)
    archive_path = notebook_helpers()["recovery_archive"](tmp_path / "input", tmp_path, identity)
    with ZipFile(archive_path) as archive:
        assert archive.read("results/registry.csv") == original
        assert archive.read("results/runs/lastfm-u25-latest/states/kgat_0_seed2019.pt") == b"completed-fit-fixture"


def test_refuse_missing_or_wrong_source_recovery(recovery, tmp_path):
    _, identity = recovery
    helper = notebook_helpers()["recovery_archive"]
    with pytest.raises(RuntimeError, match="No matching Recovery"):
        helper(tmp_path / "input", tmp_path, {**identity, "source_commit": "different"})
    with pytest.raises(RuntimeError, match="No matching Recovery"):
        helper(tmp_path / "empty", tmp_path, identity)


def test_multiple_recoveries_require_explicit_selection(recovery, tmp_path):
    root, identity = recovery
    shutil.copytree(root, tmp_path / "input/another/results")
    helper = notebook_helpers()["recovery_archive"]
    with pytest.raises(RuntimeError, match="Multiple recoveries"):
        helper(tmp_path / "input", tmp_path, identity)
    selected = helper(tmp_path / "input", tmp_path, identity, root.parent)
    with ZipFile(selected) as archive:
        assert json.loads(archive.read("results/kaggle-run.json")) == identity


def test_missing_baseline_blocks_kgat():
    verified = []

    def complete(model):
        verified.append(model)
        return model == "mostpop"

    with pytest.raises(RuntimeError, match="bprmf: incomplete"):
        notebook_helpers()["require_baselines"]({"complete": complete})
    assert verified == ["mostpop", "bprmf"]


def test_extracted_recovery_uses_runtime_identity_and_completeness_checks(payload, tmp_path, monkeypatch):
    unpacked, cloned, _ = payload
    runtime = payload_tests.script("kaggle_lastfm")
    monkeypatch.setattr(runtime, "ROOT", cloned)
    runtime.prepare(unpacked)
    recovered = tmp_path / "input/recovery/results"
    recovered.mkdir(parents=True)
    (recovered / "kaggle-run.json").write_text(json.dumps(runtime.identity()), encoding="utf-8")
    folder = recovered / "runs/lastfm-u25-latest"
    folder.mkdir(parents=True)
    with (recovered / "registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_at", "dataset", "split_sha256", "model", "config",
                                                  "seed", "phase"])
        writer.writeheader()
        for model, seeds in (("mostpop", [0]), ("bprmf", range(2019, 2024))):
            (folder / f"{model}_summary.json").write_text("{}", encoding="utf-8")
            for seed in seeds:
                writer.writerow({"run_at": "2026-01-01T00:00:00Z", "dataset": runtime.RUN_NAME,
                                 "split_sha256": "fixture", "model": model, "config": "{}",
                                 "seed": seed, "phase": "test"})
                (folder / f"{model}_seed{seed}.npz").write_bytes(b"fixture")
    helpers = notebook_helpers()
    restored = helpers["recovery_archive"](tmp_path / "input", tmp_path, runtime.identity())
    runtime.extract_results(restored)
    helpers["require_baselines"]({"complete": runtime.complete})
    assert not runtime.complete("kgat")
    with pytest.raises(ValueError, match="already contain progress"):
        runtime.extract_results(restored)


def test_resume_notebook_only_trains_kgat_with_baseline_guards():
    notebook = json.loads((ROOT / "notebooks/05_kaggle_kgat_resume.ipynb").read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            code.append(source)
    training = code[-1]
    assert training.index("require_baselines(suite)") < training.index("if RUN_TRAINING:")
    assert "'--model', 'configs/model/kgat.yaml'" in training
    assert "'--resume', '--max-new-runs', '1'" in training
    assert "finally:" in training and "suite['backup'](CHECKPOINT_ZIP)" in training
    assert "NEW_KG_FITS_THIS_SESSION = 2" in code[0]
