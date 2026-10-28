"""Time and memory of KGAT training, to size the real runs (docs/design-kgat.md, section 3).

    python scripts/measure_kgat.py --data configs/data/ml1m_u10.yaml configs/data/ml100k.yaml

For every data config it trains a few epochs on the split's train and validation parts, once on the CKG with the KG
(t1, ``--variant``) and once on the CKG without KG, and prints, per epoch and as a table: the seconds of the CF phase,
of the KG phase, of ``update_attention`` and of the validation, the peak ``torch.cuda.max_memory_allocated`` and the
number of edges. Then it extrapolates the hours of 5 seeds x grid.

It writes nothing (no registry, no files) and never opens the test files: the split is loaded with
``load_saved(..., with_test=False)`` and the KG with ``load_kg(..., with_test=False)``. The numbers are timings of
untuned epochs, not results. ``--data-root`` is the folder that holds ``data/processed`` (default: this repository).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tkgat.data.encode import encode
from tkgat.kg.ckg import build_ckg, load_kg, no_kg
from tkgat.models.kgat import KGATConfig
from tkgat.pipeline.build import load_config, load_saved
from tkgat.train.kgat import KGATFitConfig, fit_kgat

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("sec_cf", "sec_kg", "sec_attention", "sec_eval")


def steady(history: list[dict], key: str) -> float:
    """Mean of a timing column over the epochs after the first one (the first pays the CUDA warm-up)."""
    rows = history[1:] or history
    return float(np.mean([r[key] for r in rows]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="data configs, e.g. configs/data/ml1m_u10.yaml")
    ap.add_argument("--data-root", default=str(ROOT), help="folder that holds data/processed")
    ap.add_argument("--variant", default="default", choices=["default", "drop_undated", "static"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2019)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--grid-points", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)

    model_cfg = KGATConfig()
    fit_cfg = KGATFitConfig(max_epochs=args.epochs, eval_every=1, patience=args.epochs + 1, device=args.device)
    rows = []
    for path in args.data:
        cfg = load_config(path)
        run_name = cfg.get("run_name", cfg["name"])
        processed = Path(args.data_root) / cfg["processed_dir"]
        split, meta = load_saved({**cfg, "processed_dir": str(processed)}, with_test=False)
        data = encode(split)
        kg = load_kg(processed, args.variant, "t1", data, meta, with_test=False)
        for label, graph in ((f"kg:{args.variant}", kg), ("no_kg", no_kg(data))):
            ckg = build_ckg(data, graph)
            fit = fit_kgat(data, ckg, model_cfg, fit_cfg, args.seed)
            print(f"[{run_name} / {label}] device={fit.device}, nodes={ckg.n_nodes}, edges={ckg.n_edges}")
            for r in fit.history:
                print(
                    f"  epoch {r['epoch']}: cf {r['sec_cf']:.2f}s kg {r['sec_kg']:.2f}s att {r['sec_attention']:.2f}s "
                    f"val {r['sec_eval']:.2f}s | loss_cf {r['loss_cf']:.4f} loss_kg {r['loss_kg']:.4f} "
                    f"recall@20 {r['recall@20']:.4f}"
                )
            train_s = sum(steady(fit.history, k) for k in PHASES[:3])
            rows.append(
                {
                    "dataset": run_name,
                    "graph": label,
                    "nodes": ckg.n_nodes,
                    "edges": ckg.n_edges,
                    "cf_steps": -(-len(data.train) // fit_cfg.batch_size),
                    "kg_steps": -(-ckg.n_edges // fit_cfg.batch_size_kg),
                    **{k: steady(fit.history, k) for k in PHASES},
                    "epoch_s": train_s + steady(fit.history, "sec_eval") / 5,  # validation every 5 epochs in real runs
                    "peak_mb": fit.peak_memory / 2**20,
                }
            )

    head = (
        f"{'dataset':<11}{'graph':<16}{'nodes':>8}{'edges':>9}{'cf steps':>9}{'kg steps':>9}"
        f"{'cf s':>8}{'kg s':>8}{'att s':>8}{'val s':>8}{'s/epoch':>9}{'peak MB':>9}"
    )
    print(f"\nsteady-state seconds per epoch (mean of epochs 2..{args.epochs}; s/epoch counts validation once per 5)")
    print(head)
    for r in rows:
        print(
            f"{r['dataset']:<11}{r['graph']:<16}{r['nodes']:>8}{r['edges']:>9}{r['cf_steps']:>9}{r['kg_steps']:>9}"
            f"{r['sec_cf']:>8.2f}{r['sec_kg']:>8.2f}{r['sec_attention']:>8.2f}{r['sec_eval']:>8.2f}"
            f"{r['epoch_s']:>9.2f}{r['peak_mb']:>9.0f}"
        )
    runs = args.grid_points * args.seeds
    print(f"\nestimated hours for {args.seeds} seeds x {args.grid_points} grid points = {runs} runs of E epochs each")
    print(f"{'dataset':<11}{'graph':<16}{'E=100':>9}{'E=200':>9}{'E=300':>9}")
    for r in rows:
        hours = [r["epoch_s"] * e * runs / 3600 for e in (100, 200, 300)]
        print(f"{r['dataset']:<11}{r['graph']:<16}" + "".join(f"{h:>9.2f}" for h in hours))


if __name__ == "__main__":
    main()
