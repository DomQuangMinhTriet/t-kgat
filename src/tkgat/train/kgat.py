"""KGAT training: alternating CF and KG steps, early stopping on the validation set (docs/design-kgat.md, 2.5 and 4).

Like ``train/bpr.py``, ``fit_kgat`` only reads ``train``, ``val`` and ``mask_val`` of the encoded split; the graph it
receives is the one built from the train interactions and the KG at t1. The test split, its mask and the KG at t2 are
never passed in: the test score is the caller's job, once, after the grid point is selected (``scripts/run.py``).

One epoch:

1. CF phase: every train pair once (``randperm``), a negative item that is not a train positive of the user, BPR loss,
   Adam #1 on ``model.cf_parameters()``. The attention matrix ``A`` is a constant here (uniform in epoch 1).
2. KG phase: every CKG edge once (``randperm``), a corrupted tail drawn from the nodes that have an edge and rejected
   when ``(h, r, t')`` is an edge, TransR loss, Adam #2 on ``model.kg_parameters()``.
3. ``model.update_attention()`` under ``no_grad`` from the new embeddings.
4. Every ``eval_every`` epochs: validation through ``model.scorer()`` and ``tkgat.eval.evaluate``; the best state (by
   ``select_metric``) is kept on the CPU; training stops after ``patience`` evaluations without improvement.

Everything runs on ``device``: batches, negatives and rejection tests are tensor operations with a device generator.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tkgat.data.encode import EncodedSplit
from tkgat.device import describe, resolve_device, seed_all
from tkgat.eval.metrics import evaluate
from tkgat.kg.ckg import CKG
from tkgat.models.kgat import KGAT, KGATConfig
from tkgat.train.bpr import METRICS, NegativeSampler
from tkgat.train.progress import cpu_state, load_epoch, save_epoch
from tkgat.train.schedule import check_schedule, lr_at, set_lr

MAX_REDRAWS = 100


@dataclass
class KGATFitConfig:
    """Trainer settings (the model's own settings are ``KGATConfig``)."""

    lr: float = 1e-3
    batch_size: int = 1024
    batch_size_kg: int = 2048
    max_epochs: int = 300
    eval_every: int = 5
    patience: int = 10  # in evaluations: 10 x 5 epochs = the paper's 50 epochs
    select_metric: str = "recall@20"
    device: str = "auto"
    lr_schedule: str = "none"  # "none" (constant, decision 0013) or "cosine" (decision 0016 item 5)
    lr_final_ratio: float = 0.01  # cosine: the rate at ``max_epochs`` is ``lr * lr_final_ratio``


def split_params(params: dict, device: str = "auto") -> tuple[KGATConfig, KGATFitConfig]:
    """Split a flat dict (a grid point plus ``fixed``) into the model and trainer configs; unknown keys are an error."""
    model_keys = {f.name for f in dataclasses.fields(KGATConfig)}
    fit_keys = {f.name for f in dataclasses.fields(KGATFitConfig)} - {"device"}
    unknown = set(params) - model_keys - fit_keys
    if unknown:
        raise ValueError(f"unknown KGAT settings: {sorted(unknown)}")
    return (
        KGATConfig(**{k: v for k, v in params.items() if k in model_keys}),
        KGATFitConfig(**{k: v for k, v in params.items() if k in fit_keys}, device=device),
    )


@dataclass
class KGATFitResult:
    model: KGAT
    best_epoch: int
    best_val: dict
    best_state: dict  # state_dict of the best epoch, tensors on the CPU
    history: list = field(default_factory=list)
    seconds: float = 0.0
    device: str = "cpu"
    peak_memory: int = 0  # torch.cuda.max_memory_allocated in bytes (0 on the CPU)
    complete: bool = True  # False: an epoch budget paused this fit; no validation selection/test may use it yet


class KGNegativeSampler:
    """Corrupted tails for CKG edges: uniform over the nodes that have an edge, never a tail of ``(h, r)``.

    Nodes without any edge (entities that exist only at t2, and the like) are never drawn, so their embedding never
    gets a gradient (design L5). Edges are stored as sorted keys ``(h * n_rel + r) * n_nodes + t`` on the device and
    a candidate is rejected with one ``searchsorted``.
    """

    def __init__(self, edges: torch.Tensor, n_nodes: int, n_rel: int, generator: torch.Generator):
        self.n_nodes, self.n_rel, self.gen = n_nodes, n_rel, generator
        self.keys = torch.unique((edges[:, 0] * n_rel + edges[:, 1]) * n_nodes + edges[:, 2])
        self.active = torch.unique(edges[:, 0])  # every edge has its inverse, so heads cover all nodes with an edge

    def is_edge(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q = (h * self.n_rel + r) * self.n_nodes + t
        pos = torch.searchsorted(self.keys, q).clamp_max(len(self.keys) - 1)
        return self.keys[pos] == q

    def _draw(self, n: int) -> torch.Tensor:
        pick = torch.randint(0, len(self.active), (n,), device=self.active.device, generator=self.gen)
        return self.active[pick]

    def __call__(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        neg = self._draw(len(h))
        bad = self.is_edge(h, r, neg)
        for _ in range(MAX_REDRAWS):
            if not bool(bad.any()):
                return neg
            neg[bad] = self._draw(int(bad.sum()))
            bad = self.is_edge(h, r, neg)
        raise RuntimeError(f"no corrupted tail found for {int(bad.sum())} edges after {MAX_REDRAWS} redraws")


def _sync(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def fit_kgat(
    data: EncodedSplit,
    ckg: CKG,
    model_cfg: KGATConfig,
    cfg: KGATFitConfig,
    seed: int,
    fixed_epochs: int | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_key: str | None = None,
    max_new_epochs: int | None = None,
    benchmark_batches: int = 0,
    log_every_batches: int = 25,
) -> KGATFitResult:
    """Train KGAT on ``ckg`` (train interactions + KG at t1) and keep the best epoch by validation ``select_metric``.

    With ``fixed_epochs`` (the refit of decision 0016 item 4: ``ckg`` then holds train and validation interactions and
    the KG at t2) it trains exactly that many epochs, never evaluates and never stops early: the result is the state
    of the last epoch, ``best_epoch == fixed_epochs`` and ``best_val`` is empty. The schedule horizon stays
    ``cfg.max_epochs``. ``data.val`` and ``data.mask_val`` are not read in that mode.
    """
    check_schedule(cfg.lr_schedule, cfg.lr_final_ratio)
    if fixed_epochs is not None and not 1 <= fixed_epochs <= cfg.max_epochs:
        raise ValueError(f"fixed_epochs must be in [1, max_epochs={cfg.max_epochs}], got {fixed_epochs}")
    if benchmark_batches < 0 or log_every_batches < 1 or (max_new_epochs is not None and max_new_epochs < 1):
        raise ValueError("benchmark_batches must be nonnegative; epoch/log budgets must be positive")
    if checkpoint_path is not None and not checkpoint_key:
        raise ValueError("checkpoint_key is required for safe epoch resume")
    if max_new_epochs is not None and checkpoint_path is None:
        raise ValueError("max_new_epochs requires a checkpoint_path so paused training can resume")
    if benchmark_batches and (fixed_epochs is not None or checkpoint_path is not None):
        raise ValueError("benchmark cannot refit or write official epoch checkpoints")
    n_epochs = 1 if benchmark_batches else cfg.max_epochs if fixed_epochs is None else fixed_epochs
    phase = "benchmark" if benchmark_batches else "refit" if fixed_epochs is not None else "select"
    label = f"[train:{phase} seed={seed} gamma={model_cfg.gamma}]"
    device = resolve_device(cfg.device)
    print(f"{label} Initializing model and samplers on {device}", flush=True)
    seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = KGAT(ckg, model_cfg, device=device, **({"refit": True} if fixed_epochs is not None else {}))
    opt_cf = torch.optim.Adam(model.cf_parameters(), lr=cfg.lr)
    opt_kg = torch.optim.Adam(model.kg_parameters(), lr=cfg.lr)

    train = torch.as_tensor(data.train, dtype=torch.long, device=device)
    neg_items = NegativeSampler(train, data.n_items, gen)
    edges = torch.tensor(np.stack([ckg.head, ckg.relation, ckg.tail], axis=1), dtype=torch.long, device=device)
    neg_tails = KGNegativeSampler(edges, ckg.n_nodes, ckg.n_rel, gen)

    best, best_state, best_epoch, best_val, bad = -1.0, None, 0, {}, 0
    history = []
    last_epoch, previous_seconds = 0, 0.0
    signature = json.dumps(
        {
            "key": checkpoint_key,
            "model": dataclasses.asdict(model_cfg),
            "fit": dataclasses.asdict(cfg),
            "seed": seed,
            "fixed_epochs": fixed_epochs,
            "n_train": len(train),
            "n_nodes": ckg.n_nodes,
            "n_edges": ckg.n_edges,
        },
        sort_keys=True,
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.exists():
        saved = load_epoch(checkpoint, signature, device)
        model.load_state_dict(saved["state"])
        opt_cf.load_state_dict(saved["opt_cf"])
        opt_kg.load_state_dict(saved["opt_kg"])
        model.update_attention()
        best, best_state = saved["best"], saved["best_state"]
        best_epoch, best_val, bad = saved["best_epoch"], saved["best_val"], saved["bad"]
        history, last_epoch, previous_seconds = saved["history"], saved["epoch"], saved["seconds"]
        gen.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        print(f"{label} Resumed after epoch {last_epoch}; next epoch {last_epoch + 1}", flush=True)
    print(f"{label} Ready: {len(train):,} pairs / {len(edges):,} edges; budget {n_epochs} epochs", flush=True)
    t0 = time.perf_counter()
    epoch_range = range(last_epoch + 1, n_epochs + 1)
    if fixed_epochs is None and bad >= cfg.patience:
        epoch_range = range(0)
    complete = True
    for epoch in epoch_range:
        lr = lr_at(epoch, cfg.lr, cfg.lr_schedule, cfg.max_epochs, cfg.lr_final_ratio)
        set_lr([opt_cf, opt_kg], lr)  # both Adam optimisers follow the same schedule
        model.train()
        start = _sync(device)

        perm = torch.randperm(len(train), device=device, generator=gen)
        total_cf = torch.zeros((), device=device)
        cf_examples, cf_batches = 0, 0
        cf_total = math.ceil(len(train) / cfg.batch_size)
        print(f"{label} epoch {epoch}/{n_epochs}: CF 0/{cf_total}", flush=True)
        last_log = start
        for lo in range(0, len(perm), cfg.batch_size):
            batch = train[perm[lo : lo + cfg.batch_size]]
            users, pos = batch[:, 0], batch[:, 1]
            loss = model.cf_loss(users, pos, neg_items(users))
            opt_cf.zero_grad(set_to_none=True)
            loss.backward()
            opt_cf.step()
            total_cf += loss.detach() * len(batch)
            cf_examples += len(batch)
            cf_batches += 1
            if cf_batches == 1 or cf_batches % log_every_batches == 0 or time.perf_counter() - last_log >= 30:
                print(
                    f"{label} epoch {epoch} CF {cf_batches}/{cf_total} "
                    f"loss={float(total_cf) / cf_examples:.5f} elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
                last_log = time.perf_counter()
            if benchmark_batches and cf_batches >= benchmark_batches:
                break
        t_cf = _sync(device)

        perm = torch.randperm(len(edges), device=device, generator=gen)
        total_kg = torch.zeros((), device=device)
        kg_examples, kg_batches = 0, 0
        kg_total = math.ceil(len(edges) / cfg.batch_size_kg)
        print(f"{label} epoch {epoch}: KG 0/{kg_total}", flush=True)
        last_log = t_cf
        for lo in range(0, len(perm), cfg.batch_size_kg):
            batch = edges[perm[lo : lo + cfg.batch_size_kg]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            loss = model.kg_loss(h, r, t, neg_tails(h, r))
            opt_kg.zero_grad(set_to_none=True)
            loss.backward()
            opt_kg.step()
            total_kg += loss.detach() * len(batch)
            kg_examples += len(batch)
            kg_batches += 1
            if kg_batches == 1 or kg_batches % log_every_batches == 0 or time.perf_counter() - last_log >= 30:
                print(
                    f"{label} epoch {epoch} KG {kg_batches}/{kg_total} "
                    f"loss={float(total_kg) / kg_examples:.5f} elapsed={time.perf_counter() - t_cf:.1f}s",
                    flush=True,
                )
                last_log = time.perf_counter()
            if benchmark_batches and kg_batches >= benchmark_batches:
                break
        t_kg = _sync(device)

        print(f"{label} epoch {epoch}: updating attention", flush=True)
        model.update_attention()
        t_att = _sync(device)

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss_cf": float(total_cf) / cf_examples,
            "loss_kg": float(total_kg) / kg_examples,
            "cf_batches": cf_batches,
            "kg_batches": kg_batches,
        }
        t_eval = t_att
        if fixed_epochs is None and not benchmark_batches and epoch % cfg.eval_every == 0:
            print(f"{label} epoch {epoch}: validation", flush=True)
            res = evaluate(model.scorer(), data.val, data.mask_val)
            t_eval = _sync(device)
            row.update({k: res[k] for k in METRICS})
            if res[cfg.select_metric] > best:
                best, best_epoch, bad = res[cfg.select_metric], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val = {k: res[k] for k in METRICS}
            else:
                bad += 1
        row.update(
            seconds=t_eval - start,
            sec_cf=t_cf - start,
            sec_kg=t_kg - t_cf,
            sec_attention=t_att - t_kg,
            sec_eval=t_eval - t_att,
        )
        history.append(row)
        done = epoch >= n_epochs or (fixed_epochs is None and not benchmark_batches and bad >= cfg.patience)
        print(
            f"{label} epoch {epoch} complete in {row['seconds']:.1f}s "
            f"loss_cf={row['loss_cf']:.5f} loss_kg={row['loss_kg']:.5f} "
            f"val={row.get('recall@20', 'not evaluated')}",
            flush=True,
        )
        if checkpoint is not None:
            save_epoch(
                checkpoint,
                {
                    "signature": signature,
                    "state": cpu_state(model.state_dict()),
                    "opt_cf": cpu_state(opt_cf.state_dict()),
                    "opt_kg": cpu_state(opt_kg.state_dict()),
                    "epoch": epoch,
                    "best": best,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_val": best_val,
                    "bad": bad,
                    "history": history,
                    "seconds": previous_seconds + time.perf_counter() - t0,
                    "device_type": device.type,
                    "sampler_rng": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                },
            )
            print(f"{label} Saved epoch {epoch}: {checkpoint}", flush=True)
        if done:
            break
        if max_new_epochs is not None and epoch - last_epoch >= max_new_epochs:
            complete = False
            break
    if not complete:
        model.eval()
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        return KGATFitResult(
            model,
            best_epoch,
            best_val,
            best_state or {},
            history,
            previous_seconds + time.perf_counter() - t0,
            describe(device),
            peak,
            complete=False,
        )
    if fixed_epochs is not None or benchmark_batches:
        best_epoch, best_val = n_epochs, {}
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("no validation evaluation happened; max_epochs must be >= eval_every")
    model.load_state_dict(best_state)
    model.update_attention()  # A is a function of the parameters and the graph: recompute it for the best epoch
    model.eval()
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return KGATFitResult(
        model,
        best_epoch,
        best_val,
        best_state,
        history,
        previous_seconds + time.perf_counter() - t0,
        describe(device),
        peak,
    )
