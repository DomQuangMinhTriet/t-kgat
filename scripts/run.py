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

KGAT (``configs/model/kgat*.yaml``, ``name: kgat``, ``kg: {variant: default | drop_undated | static | none}``): train
and validation use the CKG of the train interactions and the KG at t1. Each (grid point, seed) keeps only its best
``state_dict`` (CPU tensors, also written to ``<runs-dir>/<run_name>/states/``, which git ignores). Test is scored once
per seed of the selected point on a *new* graph, ``set_graph`` with the same train edges and the KG at t2 (triples
whose tail is unseen at t1 dropped), attention recomputed from the frozen weights (design Q1, Q2). ``--resume`` reuses
the checkpoints of finished (grid point, seed) pairs and does not score a seed on test again when the registry
already holds its test row. The file name selects the KG variant in the registry, so a variant is never mixed with
another; ``static`` is a leakage reference, flagged ``time_consistent: false`` in the summary.

T-KGAT Stage 1 (decision 0009; ``configs/model/tkgat.yaml``, ``name: tkgat``): the same run with Delta-t in the
attention. ``from_kgat: kgat`` takes (lr, l2, l2_kg, mess_dropout) from the point that ``kgat`` selected on validation
for this same ``run_name`` (``kgat_summary.json`` or ``kgat_val_summary.json``), and the grid only holds ``gamma``;
``fixed``, ``kg`` and ``seeds`` must equal those of ``kgat.yaml``. Its registry rows hold gamma, ``t_ref`` and
``dt_den`` in ``config``. ``configs/model/tkgat_perm.yaml`` (``control: permute_train_ts``, ``from_tkgat: tkgat``)
re-runs the point that ``tkgat`` selected with ``train_ts`` permuted over all train edges, one permutation per seed
(``perm_seed`` in ``config``).

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
import torch
import yaml

from tkgat import registry
from tkgat.data.encode import encode
from tkgat.eval.metrics import evaluate
from tkgat.kg.ckg import build_ckg, load_kg, no_kg, permute_train_ts, used_tails
from tkgat.models.baselines import MostPop
from tkgat.models.kgat import KGAT
from tkgat.pipeline.build import load_config, load_saved
from tkgat.train.bpr import BPRConfig, fit_bpr
from tkgat.train.kgat import fit_kgat, split_params

METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
KG_VARIANTS = ("default", "drop_undated", "static", "none")


def seed_std(values: list[float]) -> float | None:
    """Sample standard deviation over seeds (ddof=1); undefined (None) for a single seed."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def kgat_train_graph(mcfg: dict, dcfg: dict, data, meta: dict, val_only: bool):
    """The KG (``none`` = no KG) and the CKG used for training and validation: train edges + the KG at t1."""
    variant = mcfg.get("kg", {}).get("variant", "default")
    if variant not in KG_VARIANTS:
        raise ValueError(f"kg.variant must be one of {KG_VARIANTS}, got {variant!r}")
    if variant == "none":
        kg = no_kg(data)
    else:
        kg = load_kg(Path(dcfg["processed_dir"]), variant, "t1", data, meta, with_test=not val_only)
    return kg, build_ckg(data, kg)


def kgat_test_graph(dcfg: dict, data, meta: dict, kg_t1, ckg_t1):
    """The graph of the test score: the same train edges, the KG at t2 without the triples of tails unseen at t1."""
    if kg_t1.variant == "none":
        return ckg_t1
    kg_t2 = load_kg(Path(dcfg["processed_dir"]), kg_t1.variant, "t2", data, meta)
    return build_ckg(data, kg_t2, known_tails=used_tails(kg_t1))


TKGAT_CONTROLS = ("permute_train_ts",)


def tkgat_points(mcfg: dict, model_path: Path, out_dir: Path) -> tuple[list[dict], dict]:
    """The grid points of a T-KGAT config (decision 0009, items 4 and 5) and the point they extend.

    A gamma grid (``from_kgat``): every point is the (lr, l2, l2_kg, mess_dropout) that KGAT selected on validation
    plus a ``gamma > 0`` (``gamma = 0`` is the KGAT run itself). A control (``control: permute_train_ts``,
    ``from_tkgat``): the single point that T-KGAT selected, gamma included. Either way ``fixed``, ``kg`` and ``seeds``
    must equal those of the config it extends, and the selected point is read from this run_name's summary: nothing
    is copied by hand.
    """
    control = mcfg.get("control")
    if control not in (None, *TKGAT_CONTROLS):
        raise ValueError(f"control must be one of {TKGAT_CONTROLS}, got {control!r}")
    source, expected = ("from_tkgat", "tkgat") if control else ("from_kgat", "kgat")
    other = "from_kgat" if control else "from_tkgat"
    if not mcfg.get(source) or mcfg.get(other):
        raise ValueError(f"a {'control' if control else 'T-KGAT'} config needs {source} (and not {other})")
    tag = mcfg[source]
    ref = yaml.safe_load(Path(model_path).with_name(f"{tag}.yaml").read_text(encoding="utf-8"))
    if ref["name"] != expected or (not control and ref.get("control")) or (control and not ref.get("from_kgat")):
        raise ValueError(f"{source}: {tag}.yaml is not a {expected} config")
    for key in ("fixed", "kg", "seeds"):
        if mcfg.get(key) != ref.get(key):
            raise ValueError(f"{key} differs from {tag}.yaml: the registry key holds the file name and the point only")
    for name in (f"{tag}_summary.json", f"{tag}_val_summary.json"):
        if (out_dir / name).exists():
            base = json.loads((out_dir / name).read_text(encoding="utf-8"))["selected"]
            break
    else:
        raise FileNotFoundError(f"no {tag}_summary.json or {tag}_val_summary.json in {out_dir}: run {tag} first")
    if control:
        if mcfg.get("grid"):
            raise ValueError("a control run takes its point from the T-KGAT run: remove grid")
        return [base], base
    if "gamma" in base:
        raise ValueError(f"the point selected by {tag} already has a gamma: it is not a KGAT point")
    for point in mcfg["grid"]:
        if set(point) != {"gamma"} or not point["gamma"] > 0:
            raise ValueError(f"a T-KGAT grid point holds gamma > 0 only, got {point}")
    return [{**base, **point} for point in mcfg["grid"]], base


def registry_config(point: dict, ckg, seed: int, perm: bool) -> dict:
    """``config`` of a T-KGAT registry row: the grid point (gamma inside) and what Delta-t was computed with."""
    cfg = {**point, "t_ref": round(ckg.t_ref), "dt_den": round(ckg.dt_den)}
    if perm:
        cfg["perm_seed"] = seed  # the permutation is drawn from numpy.random.default_rng(seed)
    return cfg


def kgat_checkpoint_path(out_dir: Path, tag: str, g: int, seed: int) -> Path:
    return out_dir / "states" / f"{tag}_{g}_seed{seed}.pt"


def kgat_fingerprint(base: dict, mcfg: dict, g: int, seed: int, ckg) -> str:
    """Everything a checkpoint depends on besides the code: a checkpoint is reused only if this is identical."""
    fields = {
        **base,
        "grid_index": g,
        "point": mcfg["grid"][g],
        "fixed": mcfg.get("fixed", {}),
        "kg": mcfg.get("kg", {}),
        "seed": seed,
        "n_edges": ckg.n_edges,
    }
    if mcfg["name"] == "tkgat":  # a KGAT fingerprint stays what it was, so older KGAT checkpoints can be resumed
        fields.update(t_ref=ckg.t_ref, dt_den=ckg.dt_den, control=mcfg.get("control"))
    return json.dumps(fields, sort_keys=True)


def save_checkpoint(path: Path, fingerprint: str, fit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "fingerprint": fingerprint,
        "state": fit.best_state,
        "best_epoch": fit.best_epoch,
        "best_val": fit.best_val,
        "seconds": fit.seconds,
        "device": fit.device,
    }
    torch.save(payload, tmp)
    tmp.replace(path)  # a crash never leaves a half-written checkpoint under the final name


def load_checkpoint(path: Path, fingerprint: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["fingerprint"] != fingerprint:
        raise ValueError(f"{path} was written for another configuration; delete it or run without --resume")
    return ck


def logged_row(path: str, base: dict, config: dict, seed: int, phase: str) -> dict | None:
    """The newest registry row of this run key, or None (used by ``--resume`` so that nothing is logged twice)."""
    if not Path(path).exists():
        return None
    df = registry.load(path)
    key = (
        (df["dataset"] == base["dataset"])
        & (df["split_sha256"] == base["split_sha256"])
        & (df["model"] == base["model"])
        & (df["config"] == json.dumps(config, sort_keys=True))
        & (df["seed"] == seed)
        & (df["phase"] == phase)
    )
    return df[key].iloc[-1].to_dict() if key.any() else None


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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="kgat only: reuse the checkpoints of finished (grid point, seed) pairs and do not score on test a seed "
        "that already has its test row in the registry",
    )
    args = ap.parse_args(argv)

    dcfg = load_config(args.data)
    mcfg = yaml.safe_load(Path(args.model).read_text(encoding="utf-8"))
    if args.resume and mcfg["name"] not in ("kgat", "tkgat"):
        ap.error("--resume is only implemented for kgat and tkgat")
    split, meta = load_saved(dcfg, with_test=not args.val_only)
    data = encode(split)
    split_sha = registry.sha256_file(Path(dcfg["processed_dir"]) / "meta.json")
    seeds = mcfg.get("seeds", [0])
    tag = Path(args.model).stem
    run_name = dcfg.get("run_name", dcfg["name"])
    base = {"dataset": run_name, "split_sha256": split_sha[:12], "model": tag}
    out_dir = Path(args.runs_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    kgat = mcfg["name"] in ("kgat", "tkgat")
    tkgat = mcfg["name"] == "tkgat"
    base_point = None
    if tkgat:
        grid, base_point = tkgat_points(mcfg, Path(args.model), out_dir)
        mcfg = {**mcfg, "grid": grid}
    perm = tkgat and mcfg.get("control") == "permute_train_ts"
    kg_t1 = ckg_t1 = None
    if kgat:
        kg_t1, ckg_t1 = kgat_train_graph(mcfg, dcfg, data, meta, args.val_only)
        print(
            f"[kgat] KG {kg_t1.variant} at t1: {len(kg_t1)} triples, CKG {ckg_t1.n_nodes} nodes {ckg_t1.n_edges} edges"
        )
    train_graphs, test_graphs = {None: (data, ckg_t1)}, {}

    def graph_for(seed: int):
        """(data, train graph) of a seed: the same for every seed, except that the control shuffles ``train_ts``."""
        key = seed if perm else None
        if key not in train_graphs:
            shuffled = permute_train_ts(data, seed)
            train_graphs[key] = (shuffled, build_ckg(shuffled, kg_t1))
        return train_graphs[key]

    def test_graph_for(seed: int):
        key = seed if perm else None
        if key not in test_graphs:
            test_graphs[key] = kgat_test_graph(dcfg, graph_for(seed)[0], meta, kg_t1, graph_for(seed)[1])
        return test_graphs[key]

    def tkgat_info() -> dict:
        ckg = graph_for(seeds[0])[1]
        return {
            "tkgat": {
                "from": mcfg.get("from_tkgat") or mcfg.get("from_kgat"),
                "base_point": base_point,
                "t_ref": round(ckg.t_ref),
                "dt_den": round(ckg.dt_den),
                "control": mcfg.get("control"),
            }
        }

    fitted: dict[int, dict[int, object]] = {}  # kgat: the best state_dict (CPU tensors), not the model
    devices: dict[int, dict[int, str]] = {}
    val_means, val_stds = [], []
    for g, point in enumerate(mcfg["grid"]):
        fitted[g], devices[g] = {}, {}
        scores = []
        for seed in seeds:
            resumed = False
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
            elif kgat:
                model_cfg, fit_cfg = split_params({**point, **mcfg.get("fixed", {})}, args.device)
                data_s, ckg_s = graph_for(seed)
                ck_path = kgat_checkpoint_path(out_dir, tag, g, seed)
                fingerprint = kgat_fingerprint(base, mcfg, g, seed, ckg_s)
                n_val = len(data.val)
                if args.resume and ck_path.exists():
                    ck = load_checkpoint(ck_path, fingerprint)
                    model, epoch, secs, val, dev = (
                        ck["state"],
                        ck["best_epoch"],
                        ck["seconds"],
                        ck["best_val"],
                        ck["device"],
                    )
                    resumed = True
                else:
                    fit = fit_kgat(data_s, ckg_s, model_cfg, fit_cfg, seed)
                    save_checkpoint(ck_path, fingerprint, fit)
                    registry.save_history(
                        out_dir / f"{tag}_{g}_seed{seed}_history.csv",
                        fit.history,
                        {
                            **base,
                            "grid_index": g,
                            "seed": seed,
                            "config": {"model": dataclasses.asdict(model_cfg), "fit": dataclasses.asdict(fit_cfg)},
                            "kg_variant": kg_t1.variant,
                            "device": fit.device,
                            "best_epoch": fit.best_epoch,
                            "peak_memory_mb": round(fit.peak_memory / 2**20),
                            **(
                                {
                                    "gamma": point["gamma"],
                                    "t_ref": round(ckg_s.t_ref),
                                    "dt_den": round(ckg_s.dt_den),
                                    "perm_seed": seed if perm else None,
                                }
                                if tkgat
                                else {}
                            ),
                            "git_commit": registry.git_commit(),
                        },
                    )
                    model, epoch, secs, val, dev = fit.best_state, fit.best_epoch, fit.seconds, fit.best_val, fit.device
                    del fit  # the GPU model goes; only the CPU state of the best epoch is kept
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                raise ValueError(mcfg["name"])
            fitted[g][seed] = model
            devices[g][seed] = dev
            scores.append(val["recall@20"])
            reg_point = registry_config(point, graph_for(seed)[1], seed, perm) if tkgat else point
            if not (resumed and logged_row(args.registry, base, reg_point, seed, "val") is not None):
                registry.append(
                    args.registry,
                    {
                        **base,
                        "config": reg_point,
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
                    **({"kg": {"t1": ckg_t1.summary()}} if kgat else {}),
                    **(tkgat_info() if tkgat else {}),
                },
                indent=2,
            )
        )
        return

    ckg_t2 = test_graph_for(seeds[0]) if kgat else None  # for the summary: the KG counts do not depend on the seed
    best_model_cfg = split_params({**mcfg["grid"][best], **mcfg.get("fixed", {})}, args.device)[0] if kgat else None
    summary = []
    for seed, model in fitted[best].items():
        dev = devices[best][seed]
        point = mcfg["grid"][best]
        reg_point = registry_config(point, graph_for(seed)[1], seed, perm) if tkgat else point
        done = logged_row(args.registry, base, reg_point, seed, "test") if args.resume else None
        if done is not None:  # scored before: test is not touched again
            test = {k: float(done[k]) for k in METRICS}
            summary.append(test)
            print(f"[test] seed={seed} (already in the registry) " + " ".join(f"{k}={test[k]:.4f}" for k in METRICS))
            continue
        if kgat:
            # frozen weights of the best epoch on the test graph; set_graph resets A, so recompute it
            net = KGAT(graph_for(seed)[1], best_model_cfg, device=args.device)
            net.load_state_dict(model)
            net.set_graph(test_graph_for(seed))
            net.update_attention()
            net.eval()
            score = net.scorer()
        else:
            score = model.score
        res = evaluate(score, data.test, data.mask_test)
        test = {k: res[k] for k in METRICS}
        registry.append(
            args.registry,
            {
                **base,
                "config": reg_point,
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
        json.dumps(
            {
                "selected": mcfg["grid"][best],
                "val_means_recall@20": val_means,
                "test_mean_std": agg,
                **({"kg": {"t1": ckg_t1.summary(), "t2": ckg_t2.summary()}} if kgat else {}),
                **(tkgat_info() if tkgat else {}),
            },
            indent=2,
        )
    )
    print(
        "test mean ± sample std over seeds: "
        + " ".join(f"{k}={m:.4f}±{'n/a' if s is None else f'{s:.4f}'}" for k, (m, s) in agg.items())
    )


if __name__ == "__main__":
    main()
