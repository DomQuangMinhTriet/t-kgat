"""Sensitivity of the baseline order to the split, from validation only.

    python scripts/10_sensitivity.py

Reads the validation rows of ``results/registry.csv`` (through ``registry.load()``) for the splits of
``configs/data/ml100k*.yaml`` and writes ``results/sensitivity_val.csv`` and ``docs/sensitivity-val.md``.
The splits were run with ``scripts/run.py --val-only`` (the base split ``ml-100k`` has its normal run). No test file and
no test row is read: choosing or judging a split by its test scores would be leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tkgat import registry
from tkgat.analysis.sensitivity import build_table, markdown
from tkgat.pipeline.build import load_config

DEFAULT_DATA = [
    "configs/data/ml100k.yaml",
    "configs/data/ml100k_k10.yaml",
    "configs/data/ml100k_q70_85.yaml",
    "configs/data/ml100k_q85_95.yaml",
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA, help="data configs; the first one is the base split")
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--out-csv", default="results/sensitivity_val.csv")
    ap.add_argument("--out-md", default="docs/sensitivity-val.md")
    args = ap.parse_args(argv)

    configs = []
    for path in args.data:
        cfg = load_config(path)
        processed = Path(cfg["processed_dir"])
        configs.append((cfg.get("run_name", cfg["name"]), processed, registry.sha256_file(processed / "meta.json")))
    table = build_table(configs, registry.load(args.registry), Path(args.runs_dir))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    text = markdown(table, base=configs[0][0])
    Path(args.out_md).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
