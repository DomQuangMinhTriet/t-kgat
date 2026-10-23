"""BPR training loop with early stopping on the validation set.

The function receives the encoded split but only ever reads ``train`` and
``val``/``mask_val``.  Test evaluation is done by the caller, once, after
model selection (see ``scripts/run.py``).

Everything in the loop runs on ``cfg.device`` (``"auto"`` = CUDA when available):
the train pairs live on the device, batches are drawn with a device-side
generator, and negatives are sampled and checked against the train positives
there as well, so a GPU run never waits on Python loops.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tkgat.data.encode import EncodedSplit
from tkgat.device import describe, resolve_device, seed_all
from tkgat.eval.metrics import evaluate
from tkgat.models.baselines import BPRMF

METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")


@dataclass
class BPRConfig:
    dim: int = 64
    lr: float = 1e-3
    l2: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 300
    eval_every: int = 5
    patience: int = 5  # in evaluations
    select_metric: str = "recall@20"
    device: str = "auto"


@dataclass
class FitResult:
    model: BPRMF
    best_epoch: int
    best_val: dict
    history: list = field(default_factory=list)
    seconds: float = 0.0
    device: str = "cpu"


class NegativeSampler:
    """Uniform negatives that are never a train positive of the same user.

    Train pairs are stored as sorted keys ``u * n_items + i`` on the device; a
    candidate is rejected with one ``searchsorted`` and redrawn until no batch
    element collides.  Only train positives are used, never val or test.
    """

    def __init__(self, train: torch.Tensor, n_items: int, generator: torch.Generator):
        self.n_items = n_items
        self.gen = generator
        self.keys = torch.unique(train[:, 0] * n_items + train[:, 1])  # sorted

    def is_positive(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        q = users * self.n_items + items
        pos = torch.searchsorted(self.keys, q).clamp_max(len(self.keys) - 1)
        return self.keys[pos] == q

    def __call__(self, users: torch.Tensor) -> torch.Tensor:
        neg = torch.randint(0, self.n_items, users.shape, device=users.device, generator=self.gen)
        bad = self.is_positive(users, neg)
        while bool(bad.any()):
            neg[bad] = torch.randint(0, self.n_items, (int(bad.sum()),), device=users.device, generator=self.gen)
            bad = self.is_positive(users, neg)
        return neg


def fit_bpr(data: EncodedSplit, cfg: BPRConfig, seed: int) -> FitResult:
    device = resolve_device(cfg.device)
    seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = BPRMF(data.n_users, data.n_items, cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train = torch.as_tensor(data.train, dtype=torch.long, device=device)
    sampler = NegativeSampler(train, data.n_items, gen)

    best, best_state, best_epoch, best_val, bad = -1.0, None, 0, {}, 0
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(train), device=device, generator=gen)
        total = torch.zeros((), device=device)
        for start in range(0, len(perm), cfg.batch_size):
            batch = train[perm[start : start + cfg.batch_size]]
            u, i = batch[:, 0], batch[:, 1]
            j = sampler(u)
            x_ui, x_uj, reg = model(u, i, j)
            loss = F.softplus(-(x_ui - x_uj)).mean() + cfg.l2 * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(batch)
        row = {"epoch": epoch, "loss": float(total) / len(perm)}
        if epoch % cfg.eval_every == 0:
            model.eval()
            res = evaluate(model.score, data.val, data.mask_val)
            row.update({k: res[k] for k in METRICS})
            if res[cfg.select_metric] > best:
                best, best_epoch, bad = res[cfg.select_metric], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                best_val = {k: res[k] for k in METRICS}
            else:
                bad += 1
        history.append(row)
        if bad >= cfg.patience:
            break
    if best_state is None:
        raise ValueError("no validation evaluation happened; max_epochs must be >= eval_every")
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, best_val, history, time.time() - t0, describe(device))
