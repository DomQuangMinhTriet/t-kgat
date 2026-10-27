"""Export an official split and one KG variant to the KGAT file format (decision 0008).

    python scripts/08_export_kgat.py --data configs/data/ml1m_u10.yaml --variant default
    python scripts/08_export_kgat.py --data configs/data/ml1m_u10.yaml --variant default --timestamps

Needs the saved split (script 02) and the KG (script 06) of the data config. Output folder:
data/export/kgat/<run_name>-<variant>[-timestamp]/ (ignored by git). It holds the official train/val/test, the masks
to exclude when ranking, the KG at t1 (``kg_final.txt``, train and val) and at t2 (``kg_test.txt``, test); see
``tkgat.export.kgat``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tkgat.data.encode import encode
from tkgat.export.kgat import export_kgat
from tkgat.pipeline.build import load_config, load_saved
from tkgat.registry import sha256_file

ROOT = Path(__file__).resolve().parents[1]
LINKS = {
    "ml-100k": ("data/processed/links/ml100k_links.csv", "item_100k"),
    "ml-1m": ("data/processed/links/ml1m_links.csv", "item"),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["default", "drop_undated", "static"], default="default")
    ap.add_argument("--timestamps", action="store_true", help="add the timestamp files and column")
    ap.add_argument("--out", help="output folder (default: data/export/kgat/<run_name>-<variant>[-timestamp])")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.data)
    run_name = cfg.get("run_name", cfg["name"])
    split, meta = load_saved({**cfg, "processed_dir": str(ROOT / cfg["processed_dir"])})
    enc = encode(split)
    kg_dir = ROOT / cfg["processed_dir"] / "kg" / args.variant
    link_path, item_col = LINKS[cfg["name"]]
    links = pd.read_csv(ROOT / link_path, dtype={"freebase_id": str}).dropna(subset=[item_col, "freebase_id"])
    freebase = dict(zip(links[item_col].astype(int), links["freebase_id"], strict=True))

    out = Path(args.out) if args.out else ROOT / "data" / "export" / "kgat"
    if not args.out:
        out = out / f"{run_name}-{args.variant}{'-timestamp' if args.timestamps else ''}"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to replace its files")

    info = {
        "run_name": run_name,
        "kg_variant": args.variant,
        "split_meta_sha256": sha256_file(ROOT / cfg["processed_dir"] / "meta.json"),
        "user_subsample": meta.get("clean", {}).get("user_subsample"),
    }
    result = export_kgat(
        out,
        split,
        enc,
        entities=pd.read_csv(kg_dir / "entities.csv"),
        relations=pd.read_csv(kg_dir / "relations.csv"),
        kg_t1=pd.read_csv(kg_dir / "kg_t1.tsv", sep="\t"),
        kg_t2=pd.read_csv(kg_dir / "kg_t2.tsv", sep="\t"),
        freebase=freebase,
        timestamps=args.timestamps,
        info=info,
    )
    print(f"{out}: {result['interactions']} interactions, KG {result['kg_triples']}")


if __name__ == "__main__":
    main()
