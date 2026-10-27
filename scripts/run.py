"""Train, select on validation, then score the selected configuration on test once.

Usage:
    python scripts/run.py --data configs/data/ml100k.yaml --model configs/model/bprmf.yaml

Protocol
--------
1. Every grid point is trained (or fitted) with every seed and scored on val.
2. The grid point with the best *mean* validation Recall@20 is selected.
3. Only the selected grid point is scored on test.  Test numbers of the other
   grid points are never computed.
4. Every val and test score is appended to ``results/registry.csv``; per-user
   test metrics are saved under ``results/runs/`` for significance tests.
5. Every trained run (grid point x seed) also writes its per-epoch history
   (loss, and val metrics on evaluated epochs) with its config to
   ``results/runs/<run_name>/<tag>_<grid index>_seed<seed>_history.csv``.

With ``--val-only`` only steps 1, 2 and 5 happen: the test files are not opened, no test row reaches the registry and no
per-user test file is saved; the selection goes to ``<tag>_val_summary.json``. Use it for sensitivity checks, where
choosing a split by its test scores would be leakage.

``run_name`` (optional in the data config, default ``name``) identifies the split in the registry
``dataset`` column and in ``results/runs/``; ``name`` only selects the loader. Two configs of the same
dataset (e.g. 5-core and 10-core) must have different run names so their results never overwrite each other.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from tkgat import registry
from tkgat.data.encode import encode
from tkgat.eval.metrics import evaluate
from tkgat.models.baselines import MostPop
from tkgat.pipeline.build import load_config, load_saved
from tkgat.train.bpr import BPRConfig, fit_bpr

METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")


def seed_std(values: list[float]) -> float | None:
    """Sample standard deviation over seeds (ddof=1); undefined (None) for a single seed."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--device", default="auto", help='"auto" (CUDA if available), "cuda", "cuda:1" or "cpu"')
    ap.add_argument(
        "--val-only",
        action="store_true",
        help="train and select on validation only: the test files are not read, no test row is written to the "
        "registry and no per-user test file is saved (sensitivity checks: never look at test to choose a split)",
    )
    args = ap.parse_args(argv)

    dcfg = load_config(args.data)
    mcfg = yaml.safe_load(Path(args.model).read_text(encoding="utf-8"))
    split, meta = load_saved(dcfg, with_test=not args.val_only)
    data = encode(split)
    split_sha = registry.sha256_file(Path(dcfg["processed_dir"]) / "meta.json")
    seeds = mcfg.get("seeds", [0])
    tag = Path(args.model).stem
    run_name = dcfg.get("run_name", dcfg["name"])
    base = {"dataset": run_name, "split_sha256": split_sha[:12], "model": tag}
    out_dir = Path(args.runs_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted: dict[int, dict[int, object]] = {}
    devices: dict[int, dict[int, str]] = {}
    val_means, val_stds = [], []
    for g, point in enumerate(mcfg["grid"]):
        fitted[g], devices[g] = {}, {}
        scores = []
        for seed in seeds:
            if mcfg["name"] == "mostpop":
                model, epoch, secs, dev = MostPop(data, **point), 0, 0.0, "cpu"
                res = evaluate(model.score, data.val, data.mask_val)
                val = {k: res[k] for k in METRICS}
                n_val = res["n_users"]
            elif mcfg["name"] == "bprmf":
                cfg = BPRConfig(**point, **mcfg.get("fixed", {}), device=args.device)
                fit = fit_bpr(data, cfg, seed)
                model, epoch, secs, val, dev = fit.model, fit.best_epoch, fit.seconds, fit.best_val, fit.device
                n_val = len(data.val)
                registry.save_history(
                    out_dir / f"{tag}_{g}_seed{seed}_history.csv",
                    fit.history,
                    {
                        **base,
                        "grid_index": g,
                        "seed": seed,
                        "config": dataclasses.asdict(cfg),
                        "device": dev,
                        "best_epoch": epoch,
                        "git_commit": registry.git_commit(),
                    },
                )
            else:
                raise ValueError(mcfg["name"])
            fitted[g][seed] = model
            devices[g][seed] = dev
            scores.append(val["recall@20"])
            registry.append(
                args.registry,
                {
                    **base,
                    "config": point,
                    "seed": seed,
                    "phase": "val",
                    "best_epoch": epoch,
                    **val,
                    "n_users": n_val,
                    "seconds": round(secs, 1),
                    "device": dev,
                },
            )
            print(
                f"[val] grid={point} seed={seed} device={dev} epoch={epoch} {secs:.0f}s "
                + " ".join(f"{k}={val[k]:.4f}" for k in METRICS)
            )
        val_means.append(float(np.mean(scores)))
        val_stds.append(seed_std(scores))

    best = int(np.argmax(val_means))
    print(f"selected grid point {mcfg['grid'][best]} (mean val recall@20 = {val_means[best]:.4f})")

    if args.val_only:
        # a separate file: <tag>_summary.json (with test scores) of the same run_name is never overwritten
        (out_dir / f"{tag}_val_summary.json").write_text(
            json.dumps(
                {
                    "val_only": True,
                    "selected": mcfg["grid"][best],
                    "val_means_recall@20": val_means,
                    "val_stds_recall@20": val_stds,
                },
                indent=2,
            )
        )
        return

    summary = []
    for seed, model in fitted[best].items():
        dev = devices[best][seed]
        res = evaluate(model.score, data.test, data.mask_test)
        test = {k: res[k] for k in METRICS}
        registry.append(
            args.registry,
            {
                **base,
                "config": mcfg["grid"][best],
                "seed": seed,
                "phase": "test",
                **test,
                "n_users": res["n_users"],
                "device": dev,
            },
        )
        np.savez(out_dir / f"{tag}_seed{seed}.npz", users=res["users"], **{k: v for k, v in res["per_user"].items()})
        summary.append(test)
        print(f"[test] seed={seed} " + " ".join(f"{k}={test[k]:.4f}" for k in METRICS))
    agg = {k: (float(np.mean([s[k] for s in summary])), seed_std([s[k] for s in summary])) for k in METRICS}
    (out_dir / f"{tag}_summary.json").write_text(
        json.dumps({"selected": mcfg["grid"][best], "val_means_recall@20": val_means, "test_mean_std": agg}, indent=2)
    )
    print(
        "test mean ± sample std over seeds: "
        + " ".join(f"{k}={m:.4f}±{'n/a' if s is None else f'{s:.4f}'}" for k, (m, s) in agg.items())
    )


if __name__ == "__main__":
    main()
