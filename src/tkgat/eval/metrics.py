"""Full-ranking top-K metrics.

Every item in the train vocabulary is a candidate; items the user already
interacted with before the cut-off are masked out.  Metrics are returned per
user so that paired significance tests can be run on them.

``score_fn`` may return a NumPy array or a torch tensor.  Masking and the top-K
selection run on the tensor's device (so on the GPU for GPU models), in float64
so that tiny deterministic tie-breaks (MostPop) survive.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch

ScoreFn = Callable[[np.ndarray], "np.ndarray | torch.Tensor"]  # user indices -> (B, n_items) scores


def _as_scores(raw: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw.detach().to(torch.float64).clone()
    return torch.from_numpy(np.array(raw, dtype=np.float64))  # copy: score_fn may return a read-only view


def recall_ndcg_at_k(ranked: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float]:
    """``ranked``: item indices ordered by score; ``truth``: ground-truth items."""
    top = ranked[:k]
    hits = np.isin(top, truth)
    if len(truth) == 0:
        return float("nan"), float("nan")
    recall = hits.sum() / len(truth)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts[: len(top)]).sum())
    idcg = float(discounts[: min(len(truth), k)].sum())
    return float(recall), dcg / idcg


def evaluate(
    score_fn: ScoreFn,
    truth: dict[int, np.ndarray],
    mask: sp.csr_matrix,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 512,
) -> dict:
    users = np.array(sorted(truth), dtype=np.int64)
    kmax = max(ks)
    per_user = {f"{m}@{k}": np.empty(len(users)) for k in ks for m in ("recall", "ndcg")}
    for start in range(0, len(users), batch_size):
        b = users[start : start + batch_size]
        scores = _as_scores(score_fn(b))
        seen = torch.from_numpy(mask[b].toarray().astype(bool)).to(scores.device)
        scores.masked_fill_(seen, float("-inf"))
        top = torch.topk(scores, kmax, dim=1, sorted=True).indices.cpu().numpy()
        for j, u in enumerate(b):
            for k in ks:
                r, n = recall_ndcg_at_k(top[j], truth[int(u)], k)
                per_user[f"recall@{k}"][start + j] = r
                per_user[f"ndcg@{k}"][start + j] = n
    return {
        "n_users": int(len(users)),
        "users": users,
        "per_user": per_user,
        **{name: float(np.nanmean(vals)) for name, vals in per_user.items()},
    }
