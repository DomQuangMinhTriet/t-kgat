"""BPR-MF against the MostPop baselines (paired Wilcoxon + Holm), learning curves, and the training table.

    python scripts/09_significance.py

Writes, for every dataset of the registry that has BPR-MF, MostPop and MostPop recent test rows:

  results/significance.csv            n_pairs, n_nonzero, p_raw, p_holm per comparison and metric
  results/bprmf_curves.csv            per-epoch mean and sample std (ddof=1) over seeds of the selected grid point
  docs/figures/fig5_bprmf_curve_<run_name>.png
  docs/bprmf-training-table.md        batch size, epochs, learning rate, ... read from the config and the histories

Only the grid point selected on validation is used (its per-user test files are the only ones on disk). With few
evaluation users (ML-100K has 51) the ``warning`` column says so: report it, do not hide it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from tkgat import registry
from tkgat.analysis.significance import (
    compare_dataset,
    datasets_with_all_models,
    draw_curve,
    load_curve,
    seeds_of,
    training_table_markdown,
)

LABELS = {"ml-100k": "ML-100K", "ml-1m": "ML-1M"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--model", default="configs/model/bprmf.yaml")
    ap.add_argument("--out-csv", default="results/significance.csv")
    ap.add_argument("--out-curves", default="results/bprmf_curves.csv")
    ap.add_argument("--figures-dir", default="docs/figures")
    ap.add_argument("--out-table", default="docs/bprmf-training-table.md")
    args = ap.parse_args(argv)

    reg = registry.load(args.registry)
    model_cfg = yaml.safe_load(Path(args.model).read_text(encoding="utf-8"))
    datasets = datasets_with_all_models(reg)
    if not datasets:
        raise SystemExit("no dataset in the registry has test rows for bprmf, mostpop and mostpop_recent")

    runs_dir = Path(args.runs_dir)
    rows, curves, curve_frames = [], [], []
    for ds in datasets:
        rows += compare_dataset(ds, reg, runs_dir)
        curve = load_curve(ds, runs_dir, model_cfg, seeds_of(reg, ds))
        curves.append(curve)
        curve_frames.append(curve.table.assign(dataset=ds, grid_index=curve.grid_index))
        draw_curve(curve, Path(args.figures_dir) / f"fig5_bprmf_curve_{ds}.png", LABELS.get(ds, ds))

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    pd.concat(curve_frames).to_csv(args.out_curves, index=False)
    Path(args.out_table).write_text(training_table_markdown(curves, model_cfg, LABELS), encoding="utf-8")

    show = out[["dataset", "metric", "model_b", "n_pairs", "n_nonzero", "mean_diff", "p_raw", "p_holm", "warning"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"wrote {args.out_csv}, {args.out_curves}, {args.out_table} and one curve figure per dataset")


if __name__ == "__main__":
    main()
