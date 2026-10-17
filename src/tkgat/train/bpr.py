"""BPR training loop with early stopping on the validation set.

The function receives the encoded split but only ever reads ``train`` and
``val``/``mask_val``.  Test evaluation is done by the caller, once, after
model selection (see ``scripts/run.py``).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from tkgat.data.encode import EncodedSplit
from tkgat.eval.metrics import evaluate
from tkgat.models.baselines import BPRMF


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


@dataclass
class FitResult:
    model: BPRMF
    best_epoch: int
    best_val: dict
    history: list = field(default_factory=list)
    seconds: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_negatives(rng: np.random.Generator, users: np.ndarray, pos_sets: list[set], n_items: int) -> np.ndarray:
    neg = rng.integers(0, n_items, size=len(users))
    for k, u in enumerate(users):
        while neg[k] in pos_sets[u]:
            neg[k] = rng.integers(0, n_items)
    return neg


def fit_bpr(data: EncodedSplit, cfg: BPRConfig, seed: int) -> FitResult:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = BPRMF(data.n_users, data.n_items, cfg.dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pos_sets = [set() for _ in range(data.n_users)]
    for u, i in data.train:
        pos_sets[u].add(int(i))

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = rng.permutation(len(data.train))
        total = 0.0
        for start in range(0, len(perm), cfg.batch_size):
            batch = data.train[perm[start : start + cfg.batch_size]]
            u, i = batch[:, 0], batch[:, 1]
            j = _sample_negatives(rng, u, pos_sets, data.n_items)
            x_ui, x_uj, reg = model(torch.as_tensor(u), torch.as_tensor(i), torch.as_tensor(j))
            loss = F.softplus(-(x_ui - x_uj)).mean() + cfg.l2 * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
        row = {"epoch": epoch, "loss": total / len(perm)}
        if epoch % cfg.eval_every == 0:
            model.eval()
            res = evaluate(model.score, data.val, data.mask_val)
            row.update({k: res[k] for k in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")})
            if res[cfg.select_metric] > best:
                best, best_epoch, bad = res[cfg.select_metric], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                best_val = {k: res[k] for k in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")}
            else:
                bad += 1
        history.append(row)
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, best_val, history, time.time() - t0)
