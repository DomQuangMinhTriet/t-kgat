"""Non-personalised and matrix-factorisation baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tkgat.data.encode import EncodedSplit


class MostPop:
    """Rank items by their number of train positives.

    ``window_days`` restricts the count to the last N days before the train
    cut-off ("recent popularity"); ``None`` uses the whole train period.
    Both variants only read train interactions.
    """

    def __init__(self, data: EncodedSplit, window_days: float | None = None):
        items, ts = data.train[:, 1], data.train_ts
        if window_days is not None:
            keep = ts >= ts.max() - window_days * 86400
            items = items[keep]
        self.pop = np.bincount(items, minlength=data.n_items).astype(np.float64)
        # tiny deterministic tie-break so equal counts keep a stable order
        self.pop += np.linspace(0, 1e-6, data.n_items)[::-1]

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def forward(self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        eu, ei, ej = self.user(u), self.item(i), self.item(j)
        x_ui = (eu * ei).sum(-1)
        x_uj = (eu * ej).sum(-1)
        reg = (eu.pow(2).sum(-1) + ei.pow(2).sum(-1) + ej.pow(2).sum(-1)).mean()
        return x_ui, x_uj, reg

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        """Scores of every item for ``users``, as a tensor on the model's device."""
        idx = torch.as_tensor(users, device=self.user.weight.device)
        return self.user.weight[idx] @ self.item.weight.T
