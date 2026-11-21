"""Rebuild Last-FM from an in-memory Drive ZIP, including KG and timestamp export.

    python scripts/generate_lastfm_drive.py

Run from the repository root. Only generated outputs are saved locally.
The small support ZIP is read once per process and is never extracted onto disk.
Latest timestamps are computed by streaming raw listening events from a separate
Drive ZIP. This does not reuse the earliest-only timestamp cache.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from tkgat.data.drive import drive_archive, verify_archive
from tkgat.pipeline.build import PARTS, build_split, load_config, save_split

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "knowledge_graph_attention_network" / "Data" / "last-fm"


def run_step(filename: str, arguments: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [filename, *arguments]
        runpy.run_path(str(ROOT / "scripts" / filename), run_name="__main__")
    finally:
        sys.argv = previous


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data/lastfm_u25_latest_drive.yaml")
    parser.add_argument("--author-data", default=str(AUTHOR))
    parser.add_argument("--variant", choices=["default", "drop_undated", "static", "all"], default="all")
    parser.add_argument("--export-dir", help="KGAT timestamp export directory")
    parser.add_argument("--overwrite", action="store_true", help="replace generated outputs in existing directories")
    args = parser.parse_args(argv)
    cfg = load_config(args.data)
    if cfg["name"] != "lastfm":
        raise ValueError("This command requires a Last-FM data config")
    file_id = cfg["drive_zip_id"]
    policy = cfg.get("timestamp_policy", "earliest")
    if policy not in {"earliest", "latest"}:
        raise ValueError(f"unknown timestamp policy: {policy}")
    required = ["kg_final.txt", "item_list.txt", "entity_list.txt", "relation_list.txt"]
    if policy == "latest":
        if not cfg.get("raw_interactions"):
            raise ValueError("Latest timestamps require raw_interactions; the earliest-only cache cannot be converted")
        if cfg.get("reference_meta"):
            raise ValueError("Latest timestamps cannot use an earliest-policy reference_meta")
        required += ["user_list.txt", "train.txt", "test.txt"]
    for filename in required:
        if not (Path(args.author_data) / filename).is_file():
            raise FileNotFoundError(f"Missing author KGAT input: {Path(args.author_data) / filename}")

    export_variant = "default" if args.variant == "all" else args.variant
    export_dir = args.export_dir or str(
        ROOT / "data" / "export" / "kgat" / f"{cfg.get('run_name', cfg['name'])}-{export_variant}-timestamp"
    )
    for directory in (Path(cfg["processed_dir"]), Path(export_dir)):
        if directory.exists() and any(directory.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output is not empty: {directory}; choose another config/output or use --overwrite")

    print("Reading Drive ZIP into RAM (no source files are saved)...", flush=True)
    verify_archive(file_id, cfg["drive_zip_sha256"])
    archive = drive_archive(file_id)
    reference = None
    if cfg.get("reference_meta"):
        with archive.open(cfg["reference_meta"]) as handle:
            reference = json.load(handle)

    if policy == "latest":
        scanner = runpy.run_path(str(ROOT / "scripts" / "generate_lastfm_temporal_kgat.py"))
        mappings = scanner["load_mappings"](Path(args.author_data))
        pairs, author_stats = scanner["load_author_pairs"](Path(args.author_data), mappings)
        print("Streaming raw listening events; keeping the maximum timestamp of each KGAT pair...", flush=True)
        frame, scan_stats = scanner["scan_interactions"](
            cfg["raw_interactions"], pairs, mappings, cfg.get("progress_every", 10_000_000), "latest"
        )
        split, clean_report = build_split(cfg, interactions=frame)
        clean_report["lastfm_timestamps"] = {
            **author_stats, **scan_stats, "raw_interactions": cfg["raw_interactions"],
        }
    else:
        split, clean_report = build_split(cfg)
    out = save_split(split, cfg, clean_report)
    if reference is not None:
        actual = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        mismatches = [part for part in PARTS if actual["sha256"][part] != reference["sha256"][part]]
        if mismatches:
            raise RuntimeError(f"Rebuilt splits differ from the Drive reference: {mismatches}; KG/export not run")
        print("All five split/history SHA-256 hashes match the Drive reference.", flush=True)

    run_step(
        "06_build_kg_lastfm.py", ["--data", args.data, "--author-data", args.author_data, "--variant", args.variant]
    )
    export_args = [
        "--data", args.data, "--author-data", args.author_data, "--variant", export_variant,
        "--timestamps", "--out", export_dir,
    ]
    if args.overwrite:
        export_args.append("--overwrite")
    run_step("08_export_kgat.py", export_args)
    print(f"Processed data: {out}\nKGAT export with timestamps: {export_dir}")


if __name__ == "__main__":
    main()
