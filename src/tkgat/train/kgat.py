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
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tkgat.data.encode import EncodedSplit
from tkgat.device import describe, resolve_device, seed_all
from tkgat.eval.metrics import evaluate
from tkgat.kg.ckg import CKG
from tkgat.models.kgat import KGAT, KGATConfig
from tkgat.train.bpr import METRICS, NegativeSampler

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


def fit_kgat(data: EncodedSplit, ckg: CKG, model_cfg: KGATConfig, cfg: KGATFitConfig, seed: int) -> KGATFitResult:
    """Train KGAT on ``ckg`` (train interactions + KG at t1) and keep the best epoch by validation ``select_metric``."""
    device = resolve_device(cfg.device)
    seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = KGAT(ckg, model_cfg, device=device)
    opt_cf = torch.optim.Adam(model.cf_parameters(), lr=cfg.lr)
    opt_kg = torch.optim.Adam(model.kg_parameters(), lr=cfg.lr)

    train = torch.as_tensor(data.train, dtype=torch.long, device=device)
    neg_items = NegativeSampler(train, data.n_items, gen)
    edges = torch.tensor(np.stack([ckg.head, ckg.relation, ckg.tail], axis=1), dtype=torch.long, device=device)
    neg_tails = KGNegativeSampler(edges, ckg.n_nodes, ckg.n_rel, gen)

    best, best_state, best_epoch, best_val, bad = -1.0, None, 0, {}, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start = _sync(device)

        perm = torch.randperm(len(train), device=device, generator=gen)
        total_cf = torch.zeros((), device=device)
        for lo in range(0, len(perm), cfg.batch_size):
            batch = train[perm[lo : lo + cfg.batch_size]]
            users, pos = batch[:, 0], batch[:, 1]
            loss = model.cf_loss(users, pos, neg_items(users))
            opt_cf.zero_grad(set_to_none=True)
            loss.backward()
            opt_cf.step()
            total_cf += loss.detach() * len(batch)
        t_cf = _sync(device)

        perm = torch.randperm(len(edges), device=device, generator=gen)
        total_kg = torch.zeros((), device=device)
        for lo in range(0, len(perm), cfg.batch_size_kg):
            batch = edges[perm[lo : lo + cfg.batch_size_kg]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            loss = model.kg_loss(h, r, t, neg_tails(h, r))
            opt_kg.zero_grad(set_to_none=True)
            loss.backward()
            opt_kg.step()
            total_kg += loss.detach() * len(batch)
        t_kg = _sync(device)

        model.update_attention()
        t_att = _sync(device)

        row = {
            "epoch": epoch,
            "loss_cf": float(total_cf) / len(train),
            "loss_kg": float(total_kg) / len(edges),
        }
        t_eval = t_att
        if epoch % cfg.eval_every == 0:
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
        if bad >= cfg.patience:
            break
    if best_state is None:
        raise ValueError("no validation evaluation happened; max_epochs must be >= eval_every")
    model.load_state_dict(best_state)
    model.update_attention()  # A is a function of the parameters and the graph: recompute it for the best epoch
    model.eval()
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return KGATFitResult(
        model, best_epoch, best_val, best_state, history, time.perf_counter() - t0, describe(device), peak
    )
