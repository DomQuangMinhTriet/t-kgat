"""The RQ3 table: baselines, KGAT, KGAT without KG, T-KGAT and its permutation control, from the registry.

    python scripts/10_rq3_table.py
    python scripts/10_rq3_table.py --datasets ml-100k ml-1m-u10 --out-md /tmp/rq3.md

Reads ``results/registry.csv`` through ``tkgat.registry`` (only ``phase = test`` rows that are not ``-dirty``) and the
per-user test files under ``results/runs/<run_name>/``; writes ``docs/rq3-table.md``, ``results/rq3_models.csv`` and
``results/rq3_pairs.csv``. A model with no row yet is printed as "chưa chạy", not as an error. The KGAT and T-KGAT
numbers are preliminary and not frozen until the tag ``v0.2-kgat`` exists; the output says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tkgat.analysis.rq3 import DEFAULT_DATASETS, build, markdown, models_frame


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="run_name of the data configs")
    ap.add_argument("--out-md", default="docs/rq3-table.md")
    ap.add_argument("--out-models", default="results/rq3_models.csv")
    ap.add_argument("--out-pairs", default="results/rq3_pairs.csv")
    args = ap.parse_args(argv)

    result = build(args.registry, args.runs_dir, args.datasets)
    text = markdown(result)
    for path in (args.out_md, args.out_models, args.out_pairs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(text, encoding="utf-8")
    models_frame(result).to_csv(args.out_models, index=False)
    pairs = pd.DataFrame([row for info in result["datasets"].values() for row in info["pairs"]])
    pairs.to_csv(args.out_pairs, index=False)
    print(text)


if __name__ == "__main__":
    main()
