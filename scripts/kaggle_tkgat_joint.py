"""Package or run standalone joint T-KGAT, preserving the full processed split.

The existing full-graph trainer is retained. The metadata-only memory check
rejects an impossible run before loading the large CSVs; it is not a streaming
trainer and never samples users/items to make a dataset fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("configs/model/tkgat_joint.yaml")
NOTEBOOK = Path("notebooks/10_kaggle_tkgat_joint.ipynb")
GIB = 2**30


def source_files(root: Path) -> list[Path]:
    return sorted(
        {
            *[
                p.relative_to(root)
                for folder in ("src", "scripts")
                for p in (root / folder).rglob("*.py")
                if "__pycache__" not in p.parts
            ],
            *[p.relative_to(root) for p in (root / "configs").rglob("*.yaml")],
            Path("pyproject.toml"),
        }
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root: Path) -> dict:
    return {p.as_posix(): sha(root / p) for p in source_files(root)}


def package_code(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    if (root / NOTEBOOK).exists():
        files.append(NOTEBOOK)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in files:
            # Exact bytes: source identity survives packaging across Windows/Linux.
            archive.write(root / relative, relative.as_posix())
        archive.writestr("SOURCE_SHA256.json", json.dumps(source_identity(root), indent=2))
    print(f"Code only: {output} ({output.stat().st_size:,} bytes)")


def memory_report(processed: Path, model_cfg: dict, device: str, *, refit=False) -> dict:
    """Metadata-only lower bounds, not a peak-memory prediction."""
    import torch

    split = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    kg = json.loads((processed / "kg/default/kg_meta.json").read_text(encoding="utf-8"))
    train_pairs = int(split["train_pos"])
    entities, users = int(kg["n_entities"]), int(split["n_users"])
    fit_pairs = train_pairs + int(split.get("val_pos", 0)) if refit else train_pairs
    kg_pairs = int(kg["n_triples"].get("t2", kg["n_triples"]["t1"])) if refit else int(kg["n_triples"]["t1"])
    edges = 2 * (fit_pairs + kg_pairs)
    # Graph head/relation/tail int64 + edge_ts float64. Embeddings are shared by
    # two Adam optimizers: parameters + gradients + four moment tensors.
    graph = edges * 32
    embedding_states = (entities + users) * int(model_cfg["fixed"]["dim"]) * 4 * 6
    lower_bound = graph + embedding_states
    report = {
        "train_pairs": train_pairs,
        "refit": refit,
        "fit_pairs": fit_pairs,
        "entities": entities,
        "ckg_edges": edges,
        "graph_GiB": graph / GIB,
        "embedding_training_states_GiB": embedding_states / GIB,
        "device_memory_lower_bound_GiB": lower_bound / GIB,
        "note": "Excludes samplers, propagation, attention temporaries, ranking scores and CSV/encoding RAM.",
        "blockers": [],
    }
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            report["blockers"].append("CUDA is unavailable; enable a GPU in the notebook settings.")
        else:
            free, total = torch.cuda.mem_get_info(torch.device(device))
            report.update(
                gpu=torch.cuda.get_device_name(torch.device(device)),
                free_VRAM_GiB=free / GIB,
                total_VRAM_GiB=total / GIB,
            )
            if lower_bound >= free:
                report["blockers"].append(
                    f"Full-graph training needs more than {lower_bound / GIB:.1f} GiB VRAM; "
                    f"only {free / GIB:.1f} GiB is free on this GPU. "
                    "A streaming/offloaded trainer or larger hardware is required; batch size alone cannot fix this."
                )
    elif device != "cpu":
        raise ValueError("device must be cpu, cuda or cuda:N")
    if hasattr(os, "sysconf"):
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        # Merely one copy of the graph and four train CSV columns; actual encode()
        # uses substantially more and validation/test are additional.
        cpu_floor = graph + train_pairs * 32 + (lower_bound if device == "cpu" else 0)
        report.update(available_RAM_GiB=available / GIB, RAM_lower_bound_GiB=cpu_floor / GIB)
        if cpu_floor >= available:
            report["blockers"].append(f"CPU RAM also cannot fit even the {cpu_floor / GIB:.1f} GiB lower bound.")
    return report


def restore_results(source: Path, output: Path, identity: dict) -> None:
    previous = json.loads((source / "run-identity.json").read_text(encoding="utf-8"))
    if previous != identity:
        raise ValueError("Saved results belong to another source/configuration or processed split")
    if (output / "registry.csv").exists() or (output / "runs").exists():
        raise ValueError("Current output already has progress; do not overwrite it with a saved session")
    for name in ("registry.csv", "runs"):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, output / name)
        elif path.is_file():
            shutil.copy2(path, output / name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-code", action="store_true", help="create a small source ZIP locally, without data")
    parser.add_argument("--code-out", type=Path, default=ROOT / "data/export/kaggle/tkgat-joint-code.zip")
    parser.add_argument("--processed", type=Path, help="directory containing meta.json, train.csv and kg/default/")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/tkgat-joint"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-runs", type=int, default=1, help="new selection/refit fits per call")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--refit", dest="refit", action="store_true", default=True, help="refit on train+val before test (default)"
    )
    mode.add_argument("--no-refit", dest="refit", action="store_false", help="legacy best-checkpoint test scoring")
    parser.add_argument("--restore", type=Path, help="saved tkgat-joint output directory from a previous session")
    parser.add_argument("--check-only", action="store_true", help="only report metadata and memory constraints")
    parser.add_argument("--val-only", action="store_true", help="never open test/t2 files or score test")
    args = parser.parse_args(argv)
    if args.package_code:
        package_code(ROOT, args.code_out.resolve())
        return
    if args.processed is None:
        parser.error("--processed is required unless --package-code is used")
    if args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    import yaml

    processed, output = args.processed.resolve(), args.out.resolve()
    missing = [name for name in ("meta.json", "kg/default/kg_meta.json") if not (processed / name).is_file()]
    if missing:
        parser.error(
            f"--processed must point to the extracted processed folder; missing: {', '.join(missing)}. "
            "On Kaggle, the processed CSVs and metadata must be visible under /kaggle/input."
        )
    output.mkdir(parents=True, exist_ok=True)
    model_cfg = yaml.safe_load((ROOT / MODEL).read_text(encoding="utf-8"))
    report = memory_report(processed, model_cfg, args.device, refit=args.refit and not args.val_only)
    (output / "memory-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        return
    if report["blockers"]:
        raise SystemExit("Training was not started:\n- " + "\n- ".join(report["blockers"]))
    run_name = f"{processed.name}-joint"
    identity = {
        "source_sha256": source_identity(ROOT),
        "model": MODEL.as_posix(),
        "run_name": run_name,
        "refit": args.refit,
        "split_meta_sha256": sha(processed / "meta.json"),
        "kg_meta_sha256": sha(processed / "kg/default/kg_meta.json"),
    }
    identity_path = output / "run-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Output belongs to another code/config/split; choose a new --out directory")
    if args.restore:
        restore_results(args.restore.resolve(), output, identity)
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    data_path = output / "data-config.yaml"
    data_path.write_text(
        yaml.safe_dump({"name": "lastfm", "run_name": run_name, "processed_dir": str(processed)}), encoding="utf-8"
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run.py"),
        "--data",
        str(data_path),
        "--model",
        str(ROOT / MODEL),
        "--device",
        args.device,
        "--resume",
        "--max-new-runs",
        str(args.max_new_runs),
        "--registry",
        str(output / "registry.csv"),
        "--runs-dir",
        str(output / "runs"),
    ]
    if args.val_only:
        command.append("--val-only")
    elif args.refit:
        command.append("--refit")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with (output / "training.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        with subprocess.Popen(
            command,
            env=env,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
    if code:
        raise SystemExit(code)
    print(f"Progress saved to {output}. Rerun to continue; resume is at completed fit boundaries, not mid-epoch.")


if __name__ == "__main__":
    main()
