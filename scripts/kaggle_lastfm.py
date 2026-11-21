"""Prepare, run bounded stages, report and checkpoint the Last-FM Kaggle suite.

This wrapper preserves model configs, all seeds and the full gamma grid. KGAT
fits resume only at completed grid/seed boundaries, not in the middle of an epoch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = "configs/data/lastfm_u25_latest_drive.yaml"
RUN_NAME = "lastfm-u25-latest"
MODELS = ("mostpop", "bprmf", "kgat", "kgat_nokg", "tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def identity() -> dict:
    snapshot = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    for name, expected in snapshot["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed since packaging: {name}; create a new payload instead")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("The exported code has tracked modifications; restore the packaged source")
    processed = ROOT / "data/processed" / RUN_NAME
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    for part, expected in meta["sha256"].items():
        if sha(processed / f"{part}.csv") != expected:
            raise ValueError(f"Processed data changed: {part}")
    return {
        "source_commit": git("rev-parse", "HEAD"), "dataset": RUN_NAME, "data_config": DATA_CONFIG,
        "split_meta_sha256": sha(ROOT / "data/processed" / RUN_NAME / "meta.json"),
    }


def record_identity() -> dict:
    current = identity()
    path = ROOT / "results/kaggle-run.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise ValueError("Existing results belong to another code snapshot or split")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def extract_results(path: Path) -> None:
    expected = identity()
    if (ROOT / "results/registry.csv").exists() or any((ROOT / "results/runs").rglob("*.pt")):
        raise ValueError("Current results already contain progress; do not overwrite them with an older archive")
    with ZipFile(path) as archive:
        if json.loads(archive.read("results/kaggle-run.json")) != expected:
            raise ValueError("Checkpoint belongs to another source snapshot or dataset")
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if "\\" in name or not parts or parts[0] not in {"results", "reports"} or ".." in parts:
                raise ValueError(f"Unexpected checkpoint member: {name}")
        archive.extractall(ROOT)


def prepare(payload: Path, restore: Path | None = None) -> None:
    manifest = json.loads((payload / "payload-manifest.json").read_text(encoding="utf-8"))
    if git("rev-parse", "HEAD") != manifest["source_commit"]:
        raise ValueError("The cloned source does not match the payload manifest")
    for name, expected in manifest["files_sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid payload path: {name}")
        source = payload / name
        if sha(source) != expected:
            raise ValueError(f"Payload hash mismatch: {name}")
        if name == "source.bundle":
            continue
        destination = ROOT / name
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError(f"Existing data differs from the packaged data: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    if restore:
        extract_results(restore)
    record_identity()
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    for tag in MODELS:
        cfg = yaml.safe_load((ROOT / "configs/model" / f"{tag}.yaml").read_text())
        if tag != "mostpop" and cfg["seeds"] != list(range(2019, 2024)):
            raise ValueError(f"Expected all five seeds in {tag}")
        if tag in {"tkgat", "tkgat_kg"} and [p["gamma"] for p in cfg["grid"]] != [1, 3, 10]:
            raise ValueError(f"Expected the full gamma grid in {tag}")
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__, "pandas": pd.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "upstream": json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text()),
    }
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (ROOT / "results" / f"environment-{stamp}.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"Prepared {RUN_NAME}; code and input hashes verified. CUDA: {torch.cuda.is_available()}")


def backup(path: Path) -> None:
    record_identity()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for folder in ("results", "reports"):
            for source in sorted((ROOT / folder).rglob("*")):
                if source.is_file() and source.suffix != ".tmp":
                    archive.write(source, source.relative_to(ROOT).as_posix())
    temporary.replace(path)
    print(f"Checkpoint archive: {path}")


def complete(tag: str) -> bool:
    from tkgat import registry

    folder = ROOT / "results/runs" / RUN_NAME
    summary = folder / f"{tag}_summary.json"
    if not summary.exists():
        return False
    expected = [0] if tag == "mostpop" else list(range(2019, 2024))
    rows = registry.load(ROOT / "results/registry.csv")
    rows = rows[(rows["dataset"] == RUN_NAME) & (rows["model"] == tag) & (rows["phase"] == "test")]
    if sorted(rows["seed"].tolist()) != expected:
        raise ValueError(f"Summary exists but test rows are incomplete for {tag}; restore the complete result archive")
    for seed in expected:
        if not (folder / f"{tag}_seed{seed}.npz").exists():
            raise ValueError(f"Missing per-user output for {tag}, seed {seed}; restore the complete result archive")
    return True


def execute(arguments: list[str]) -> None:
    logs = ROOT / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    with (logs / f"{stamp}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Command failed; see {logs / (stamp + '.log')}")


def train(steps: int, max_new_runs: int, checkpoint: Path, device: str = "cuda") -> None:
    record_identity()
    for _ in range(steps):
        pending = next((tag for tag in MODELS if not complete(tag)), None)
        if pending is None:
            print("All models finished; use the report command to build the comparison table.")
            break
        if pending in {"mostpop", "bprmf"} and (ROOT / "results/registry.csv").exists():
            from tkgat import registry

            rows = registry.load(ROOT / "results/registry.csv")
            if ((rows["dataset"] == RUN_NAME) & (rows["model"] == pending) & (rows["phase"] == "test")).any():
                raise ValueError(
                    f"{pending} has test rows but no complete summary; inspect/restore results before rerunning"
                )
        arguments = ["scripts/run.py", "--data", DATA_CONFIG, "--model", f"configs/model/{pending}.yaml",
                     "--device", device]
        if pending not in {"mostpop", "bprmf"}:
            arguments += ["--resume", "--max-new-runs", str(max_new_runs)]
        print(f"Running {pending}; KGAT/T-KGAT stop after at most {max_new_runs} new grid/seed fits.", flush=True)
        try:
            execute(arguments)
        finally:
            backup(checkpoint)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "measure", "train", "report", "checkpoint", "status"])
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-new-runs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backup", type=Path, default=ROOT.parent / "tkgat-lastfm-results.zip")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.max_new_runs < 1:
        parser.error("steps and max-new-runs must be positive")
    if args.action == "prepare":
        if args.payload is None:
            parser.error("prepare requires --payload")
        prepare(args.payload, args.restore)
    elif args.action == "train":
        train(args.steps, args.max_new_runs, args.backup, args.device)
    elif args.action == "measure":
        record_identity()
        try:
            execute(["scripts/measure_kgat.py", "--data", DATA_CONFIG, "--device", args.device])
        finally:
            backup(args.backup)
    elif args.action == "report":
        record_identity()
        try:
            execute(["scripts/10_rq3_table.py", "--datasets", RUN_NAME, "--out-md", "reports/lastfm-latest.md",
                     "--out-models", "reports/models.csv", "--out-pairs", "reports/pairs.csv",
                     "--figures-dir", "reports/figures"])
        finally:
            backup(args.backup)
    elif args.action == "checkpoint":
        backup(args.backup)
    else:
        record_identity()
        for tag in MODELS:
            print(f"{tag}: {'complete' if complete(tag) else 'pending'}")


if __name__ == "__main__":
    main()
