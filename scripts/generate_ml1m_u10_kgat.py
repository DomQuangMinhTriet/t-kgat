"""Generate the official ML-1M 10%-user subset and two KGAT dataset variants.

Outputs by default:
  knowledge_graph_attention_network/Data/ml-1m-u10-chrono/
  knowledge_graph_attention_network/Data/ml-1m-u10-chrono-timestamp/

The pipeline uses configs/data/ml1m_u10.yaml, the frozen Wikidata facts snapshot and KB4Rec links. It builds the
official temporal split, the time-consistent KG (Wikidata plus MovieLens genre/release metadata), then exports the
same interactions and KG twice; the second folder additionally contains timestamp sidecars.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkgat.data.loaders import load  # noqa: E402
from tkgat.kg.linking import attach_kb_ids, load_kb4rec  # noqa: E402
from tkgat.pipeline.build import build_split, load_config, save_split  # noqa: E402


def default_raw_root() -> Path:
    canonical = ROOT / "data" / "raw"
    extracted = ROOT / "data-for-Phon (1)" / "data" / "raw"
    return canonical if (canonical / "ml-1m" / "ratings.dat").is_file() else extracted


def run_script(relative_path: str, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(ROOT / relative_path), *argv]
        runpy.run_path(str(ROOT / relative_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def build_ml1m_links(raw_root: Path) -> Path:
    _, catalog = load("ml-1m", raw_root / "ml-1m")
    freebase, yago = load_kb4rec(raw_root / "kb4rec")
    links = attach_kb_ids(catalog[["item", "title", "year"]], "item", freebase, yago)
    out = ROOT / "data" / "processed" / "links" / "ml1m_links.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    links.to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=default_raw_root())
    ap.add_argument("--output-root", type=Path, default=ROOT / "knowledge_graph_attention_network" / "Data")
    ap.add_argument("--variant", choices=("default", "drop_undated", "static"), default="default")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="reuse data/processed/ml-1m-u10 and only create the two KGAT folders",
    )
    args = ap.parse_args(argv)

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    plain = output_root / "ml-1m-u10-chrono"
    timestamped = output_root / "ml-1m-u10-chrono-timestamp"
    if not args.overwrite:
        occupied = [str(path) for path in (plain, timestamped) if path.exists() and any(path.iterdir())]
        if occupied:
            raise FileExistsError("output folder is not empty; pass --overwrite:\n- " + "\n- ".join(occupied))

    config_path = ROOT / "configs" / "data" / "ml1m_u10.yaml"
    facts_path = raw_root / "wikidata" / "facts.tsv"
    if not args.export_only:
        required = (
            raw_root / "ml-1m" / "ratings.dat",
            raw_root / "ml-1m" / "movies.dat",
            raw_root / "kb4rec" / "ml2fb.txt",
            raw_root / "kb4rec" / "ml2yago.txt",
            facts_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing required input files:\n- " + "\n- ".join(missing))

    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        if not args.export_only:
            cfg = load_config(config_path)
            cfg["raw_dir"] = str(raw_root / "ml-1m")
            split, clean_report = build_split(cfg)
            save_split(split, cfg, clean_report)
            links_path = build_ml1m_links(raw_root)
            run_script(
                "scripts/06_build_kg.py",
                [
                    "--data",
                    str(config_path),
                    "--variant",
                    args.variant,
                    "--raw-dir",
                    str(raw_root / "ml-1m"),
                    "--facts",
                    str(facts_path),
                    "--links",
                    str(links_path),
                ],
            )

        common = ["--data", str(config_path), "--variant", args.variant]
        overwrite = ["--overwrite"] if args.overwrite else []
        run_script("scripts/08_export_kgat.py", [*common, "--out", str(plain), *overwrite])
        run_script(
            "scripts/08_export_kgat.py",
            [*common, "--timestamps", "--out", str(timestamped), *overwrite],
        )
    finally:
        os.chdir(old_cwd)

    print(f"chrono: {plain}")
    print(f"chrono + timestamp: {timestamped}")


if __name__ == "__main__":
    main()
