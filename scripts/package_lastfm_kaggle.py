"""Create a private Kaggle payload from the current code and already processed Last-FM.

No model is trained. Git commits are made only in a disposable export repository;
the user's checkout, branch and history are never modified. Data never enters Git.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/processed/lastfm-u25-latest")
MODEL_TAGS = ("mostpop", "bprmf", "kgat", "kgat_nokg", "tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def package(root: Path, output: Path) -> dict:
    data_files = [DATA / name for name in (
        "train.csv", "val.csv", "test.csv", "seen_before_val.csv", "seen_before_test.csv", "meta.json",
        "kg/default/kg_t1.tsv", "kg/default/kg_t2.tsv", "kg/default/kg_t1_dates.tsv", "kg/default/kg_t2_dates.tsv",
        "kg/default/entities.csv", "kg/default/relations.csv", "kg/default/kg_meta.json", "kg/default/dropped.csv",
    )]
    for name in data_files:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    meta = json.loads((root / DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["clean"]["lastfm_timestamps"]["timestamp_policy"] != "latest":
        raise ValueError("The Kaggle payload requires the latest-listen dataset")
    for part, expected in meta["sha256"].items():
        if sha(root / DATA / f"{part}.csv") != expected:
            raise ValueError(f"Processed split/history hash mismatch: {part}")

    source_files = sorted({
        *[p.relative_to(root) for folder in ("src", "scripts") for p in (root / folder).rglob("*.py")
          if "__pycache__" not in p.parts],
        *[p.relative_to(root) for p in (root / "configs").rglob("*.yaml")],
        *[p.relative_to(root) for p in (root / "docs/decisions").glob("*.md")],
        Path("pyproject.toml"), Path("README.md"), Path("docs/design-kgat.md"), Path("docs/lastfm-kaggle.md"),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tkgat-kaggle-", dir=output.parent) as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        hashes = {}
        for relative in source_files:
            text = (root / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8", newline="\n")
            hashes[relative.as_posix()] = sha(destination)
        (stage / ".gitignore").write_text(
            "data/processed/\ndata/export/\nresults/\nreports/\n__pycache__/\n*.egg-info/\n.venv/\n",
            encoding="utf-8", newline="\n",
        )
        hashes[".gitignore"] = sha(stage / ".gitignore")
        (stage / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8", newline="\n")
        hashes[".gitattributes"] = sha(stage / ".gitattributes")
        snapshot = {
            "format": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "upstream_commit": git(root, "rev-parse", "HEAD"),
            "upstream_status": git(root, "status", "--short"),
            "note": "A new export commit freezes this source snapshot; it is not a claim of a clean upstream checkout.",
            "source_sha256": hashes,
        }
        (stage / "SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        empty = Path(directory) / "empty-template"
        empty.mkdir()
        git(stage, "init", "-q", "-b", "main", f"--template={empty}")
        git(stage, "config", "core.autocrlf", "false")
        git(stage, "add", ".")
        git(stage, "-c", "user.name=T-KGAT export", "-c", "user.email=export@example.invalid",
            "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={empty}",
            "commit", "-qm", "Freeze Kaggle source snapshot; see SOURCE_SNAPSHOT.json for upstream provenance")
        commit = git(stage, "rev-parse", "HEAD")
        bundle = Path(directory) / "source.bundle"
        git(stage, "bundle", "create", str(bundle), "--all")
        manifest = {
            "format": 1, "source_commit": commit, "dataset": "lastfm-u25-latest",
            "data_config": "configs/data/lastfm_u25_latest_drive.yaml", "models": MODEL_TAGS,
            "files_sha256": {"source.bundle": sha(bundle), **{p.as_posix(): sha(root / p) for p in data_files}},
        }
        temporary = output.with_suffix(".tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(bundle, "source.bundle")
            for relative in data_files:
                archive.write(root / relative, relative.as_posix())
            archive.writestr("payload-manifest.json", json.dumps(manifest, indent=2))
        temporary.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "source_commit": commit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/export/kaggle/tkgat-lastfm-latest-kaggle.zip")
    args = parser.parse_args()
    print(json.dumps(package(ROOT, args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
