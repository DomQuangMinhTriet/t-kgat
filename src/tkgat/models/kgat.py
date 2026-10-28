"""KGAT (Wang et al., KDD 2019) on the collaborative knowledge graph of ``tkgat.kg.ckg`` (docs/design-kgat.md).

Model only: the samplers, the alternating training loop and early stopping belong to ``tkgat.train.kgat``.

* TransR: ``e_x^r = W_r e_x``, ``g(h, r, t) = ||e_h^r + e_r - e_t^r||^2`` (section 2.1; the code stores ``W_r`` as a
  ``d x k`` matrix and multiplies row vectors, which is the same map as the paper's ``k x d`` matrix on column vectors).
* Attention ``pi(h, r, t) = (W_r e_t)^T tanh(W_r e_h + e_r)`` softmax-ed over **all** edges of the head (2.2). The
  scores are computed for the whole graph grouped by relation (``W_r`` is never gathered per edge) in chunks under
  ``no_grad``, so the sparse matrix ``A`` is a constant in the CF steps and attention is learned only through the KG
  loss (Q4). ``A`` starts uniform, ``1 / |N_h|`` (Q5).
* Propagation ``e_N = A e`` with ``torch.sparse.mm``; Bi-Interaction aggregator, LeakyReLU slope 0.2, biases ``b1``
  and ``b2``, message dropout, L2 normalisation of every layer >= 1; the prediction is the inner product of the
  concatenated ``e*`` of user and item (2.3).
* Losses and regularisers follow the TF1 code (Q9): ``l2`` on the propagated vectors of the CF batch, ``l2_kg`` on the
  projected vectors of the KG batch (2.1, 2.4).

Time interface (Q12, decisions 0005 and 0009 pending): the graph's ``edge_ts`` (parallel to the edges) and a
per-relation ``delta_r`` are kept, and ``gamma`` is a config field. Delta-t is **not implemented**: ``gamma`` must be 0,
which is plain KGAT, and any other value raises ``NotImplementedError``. When it is written, the term
``- gamma * delta_r[r] * dt`` is subtracted from ``pi`` before the softmax, and ``gamma = 0`` must keep giving the
tensors of plain KGAT.

Graph tensors, ``A`` and ``edge_ts`` are non-persistent buffers: ``state_dict`` holds the parameters only, so one set
of weights can be loaded and then scored on the graph of another cut-off (``set_graph``, which recomputes ``A``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tkgat.device import resolve_device
from tkgat.kg.ckg import CKG

LEAKY_SLOPE = 0.2  # tf.nn.leaky_relu default; torch's default is 0.01


@dataclass(frozen=True)
class KGATConfig:
    dim: int = 64  # entity / user embedding size d
    kge_dim: int = 64  # relation space k
    layers: Sequence[int] = (64, 32, 16)  # output sizes of the propagation layers
    mess_dropout: float = 0.1  # drop probability of the messages (not a keep probability)
    l2: float = 1e-5  # regulariser of the CF loss
    l2_kg: float = 1e-5  # regulariser of the KG loss
    gamma: float = 0.0  # time decay of the attention; 0 = plain KGAT (Delta-t not implemented yet)
    attention_chunk: int = 65536  # edges per block when the attention scores are computed

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(x) for x in self.layers))


class BiInteraction(nn.Module):
    """``drop(LeakyReLU(W1 (e + e_N) + b1) + LeakyReLU(W2 (e * e_N) + b2))`` for row-vector embeddings."""

    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_in, d_out))
        self.b1 = nn.Parameter(torch.empty(1, d_out))
        self.w2 = nn.Parameter(torch.empty(d_in, d_out))
        self.b2 = nn.Parameter(torch.empty(1, d_out))
        for p in self.parameters():
            nn.init.xavier_uniform_(p)  # biases too, as in the TF1 code
        self.drop = nn.Dropout(dropout)

    def forward(self, ego: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        total = F.leaky_relu((ego + side) @ self.w1 + self.b1, LEAKY_SLOPE)
        inter = F.leaky_relu((ego * side) @ self.w2 + self.b2, LEAKY_SLOPE)
        return self.drop(total + inter)


class KGAT(nn.Module):
    def __init__(self, ckg: CKG, cfg: KGATConfig | None = None, device: str | torch.device = "auto"):
        super().__init__()
        cfg = cfg or KGATConfig()
        if cfg.gamma != 0.0:
            raise NotImplementedError("Delta-t attention is not implemented (decision 0009 pending): use gamma = 0")
        self.cfg = cfg
        self.n_users, self.n_items, self.n_entities = ckg.n_users, ckg.n_items, ckg.n_entities
        self.n_relations, self.n_rel = ckg.n_relations, ckg.n_rel
        self.n_nodes = ckg.n_nodes

        d, k = cfg.dim, cfg.kge_dim
        self.user_embed = nn.Parameter(torch.empty(self.n_users, d))
        self.entity_embed = nn.Parameter(torch.empty(self.n_entities, d))
        self.relation_embed = nn.Parameter(torch.empty(self.n_rel, k))
        self.trans_W = nn.Parameter(torch.empty(self.n_rel, d, k))
        for p in (self.user_embed, self.entity_embed, self.relation_embed):
            nn.init.xavier_uniform_(p)
        bound = (6.0 / (d + k)) ** 0.5  # Xavier on each d x k matrix
        nn.init.uniform_(self.trans_W, -bound, bound)
        sizes = [d, *cfg.layers]
        self.aggregators = nn.ModuleList(
            BiInteraction(sizes[i], sizes[i + 1], cfg.mess_dropout) for i in range(len(cfg.layers))
        )
        self.to(resolve_device(device))
        self.register_buffer("delta_r", torch.zeros(self.n_rel, device=self.user_embed.device), persistent=False)
        self.set_graph(ckg)

    # ------------------------------------------------------------------ graph and attention

    @property
    def device(self) -> torch.device:
        return self.user_embed.device

    def set_graph(self, ckg: CKG) -> None:
        """Use another graph over the same id spaces (e.g. the t2 KG for the test score) and recompute ``A``."""
        same = (ckg.n_users, ckg.n_items, ckg.n_entities, ckg.n_relations) == (
            self.n_users,
            self.n_items,
            self.n_entities,
            self.n_relations,
        )
        if not same:
            raise ValueError("the graph has other id spaces than the model's embedding tables")
        dev = self.device
        # torch.tensor copies: on the CPU as_tensor would share memory with the CKG's arrays
        head = torch.tensor(ckg.head, device=dev)
        if len(head) > 1 and bool((head[1:] < head[:-1]).any()):
            raise ValueError("edges must be sorted by head")
        relation = torch.tensor(ckg.relation, device=dev)
        self._buffer("head", head)
        self._buffer("relation", relation)
        self._buffer("tail", torch.tensor(ckg.tail, device=dev))
        self._buffer("edge_ts", torch.tensor(ckg.edge_ts, device=dev))  # reserved for Delta-t, not read yet
        _, counts = torch.unique_consecutive(head, return_counts=True)
        self._buffer("_segments", counts)  # edges per non-empty head; the edges of a head are contiguous
        self._buffer("_degree", torch.bincount(head, minlength=self.n_nodes))
        self._buffer("_by_relation", torch.argsort(relation, stable=True))
        per_rel = torch.bincount(relation, minlength=self.n_rel).cpu().tolist()
        bounds = np.concatenate([[0], np.cumsum(per_rel)]).tolist()
        self._relation_slices = [(r, bounds[r], bounds[r + 1]) for r in range(self.n_rel) if per_rel[r]]
        self.reset_attention()

    def _buffer(self, name: str, value: torch.Tensor) -> None:
        self.register_buffer(name, value, persistent=False)

    def _set_attention(self, values: torch.Tensor) -> None:
        index = torch.stack([self.head, self.tail])
        size = (self.n_nodes, self.n_nodes)
        # edges of two relations between the same (h, t) add up, as the triple sum of the paper's eq. 3
        with torch.sparse.check_sparse_tensor_invariants(False):  # indices come from the CKG, already validated
            self._buffer("A", torch.sparse_coo_tensor(index, values, size).coalesce())

    @torch.no_grad()
    def reset_attention(self) -> None:
        """Uniform attention ``1 / |N_h|`` (Q5), the value of ``A`` before the first update."""
        values = (1.0 / self._degree[self.head]).to(self.user_embed.dtype)
        self._set_attention(values)

    @torch.no_grad()
    def update_attention(self) -> None:
        """Recompute ``A`` from the current embeddings (no gradient: the CF steps treat it as a constant)."""
        self._set_attention(self._segment_softmax(self._edge_scores()))

    def _edge_scores(self) -> torch.Tensor:
        """``pi(h, r, t)`` of every edge, relation by relation and in chunks (never one ``W_r`` per edge)."""
        emb = self.node_embed()
        pi = torch.empty(len(self.head), dtype=emb.dtype, device=emb.device)
        for r, lo, hi in self._relation_slices:
            w, e_r = self.trans_W[r], self.relation_embed[r]
            for start in range(lo, hi, self.cfg.attention_chunk):
                idx = self._by_relation[start : min(start + self.cfg.attention_chunk, hi)]
                e_h = emb[self.head[idx]] @ w
                e_t = emb[self.tail[idx]] @ w
                pi[idx] = (e_t * torch.tanh(e_h + e_r)).sum(-1)
        # T-KGAT (decisions 0005, 0009): pi -= gamma * delta_r[relation] * dt goes here, before the softmax
        return pi

    def _segment_softmax(self, pi: torch.Tensor) -> torch.Tensor:
        """Softmax of ``pi`` inside each head's contiguous block of edges (shifted by the block maximum)."""
        counts, n = self._segments, len(pi)
        top = torch.segment_reduce(pi, "max", lengths=counts)
        weight = (pi - torch.repeat_interleave(top, counts, output_size=n)).exp()
        total = torch.segment_reduce(weight, "sum", lengths=counts)
        return weight / torch.repeat_interleave(total, counts, output_size=n)

    # ------------------------------------------------------------------ propagation and prediction

    def node_embed(self) -> torch.Tensor:
        """``e^(0)``: users first, then entities (items are the first entities), shape (n_nodes, d)."""
        return torch.cat([self.user_embed, self.entity_embed], dim=0)

    def propagate(self) -> torch.Tensor:
        """``e* = e^(0) || e^(1) || ... `` (n_nodes, d + sum(layers)); layers >= 1 enter L2-normalised."""
        ego = self.node_embed()
        parts = [ego]
        for layer in self.aggregators:
            ego = layer(ego, torch.sparse.mm(self.A, ego))
            parts.append(F.normalize(ego, p=2, dim=1))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def eval_embeddings(self) -> torch.Tensor:
        """``e*`` without dropout, whatever the training mode (the mode is restored)."""
        was_training = self.training
        self.eval()
        try:
            return self.propagate()
        finally:
            self.train(was_training)

    def scorer(self):
        """A score function for ``tkgat.eval.evaluate``: ``e*`` is computed once, now, with the current ``A``."""
        star = self.eval_embeddings()
        items = star[self.n_users : self.n_users + self.n_items]

        def score(users: np.ndarray) -> torch.Tensor:
            return star[torch.as_tensor(users, device=star.device)] @ items.T

        return score

    def score(self, users: np.ndarray) -> torch.Tensor:
        """Scores of every item for ``users`` (recomputes ``e*``; use ``scorer()`` to rank many batches)."""
        return self.scorer()(users)

    # ------------------------------------------------------------------ losses and parameter groups

    def cf_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """BPR loss plus ``l2`` times half the mean squared norm of the propagated vectors (items as 0..I-1)."""
        star = self.propagate()
        e_u, e_i, e_j = star[users], star[self.n_users + pos], star[self.n_users + neg]
        bpr = F.softplus(-((e_u * e_i).sum(-1) - (e_u * e_j).sum(-1))).mean()
        reg = 0.5 * (e_u.pow(2).sum(-1) + e_i.pow(2).sum(-1) + e_j.pow(2).sum(-1)).mean()
        return bpr + self.cfg.l2 * reg

    def kg_loss(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, t_neg: torch.Tensor) -> torch.Tensor:
        """TransR ranking loss of a batch of CKG edges (node ids, relation ids) and corrupted tails, plus ``l2_kg``.

        ``W_r`` is gathered per *sample* here (a batch of a few thousand), never per edge of the whole graph.
        """
        emb = self.node_embed()
        w = self.trans_W[r]

        def project(nodes: torch.Tensor) -> torch.Tensor:
            return torch.bmm(emb[nodes].unsqueeze(1), w).squeeze(1)

        e_h, e_t, e_n, e_r = project(h), project(t), project(t_neg), self.relation_embed[r]
        g_pos = (e_h + e_r - e_t).pow(2).sum(-1)
        g_neg = (e_h + e_r - e_n).pow(2).sum(-1)
        reg = 0.5 * (e_h.pow(2).sum(-1) + e_r.pow(2).sum(-1) + e_t.pow(2).sum(-1) + e_n.pow(2).sum(-1)).mean()
        return F.softplus(g_pos - g_neg).mean() + self.cfg.l2_kg * reg

    def cf_parameters(self) -> list[nn.Parameter]:
        """Parameters of the CF optimiser: the two embedding tables and the aggregators."""
        return [self.user_embed, self.entity_embed, *self.aggregators.parameters()]

    def kg_parameters(self) -> list[nn.Parameter]:
        """Parameters of the KG optimiser: the two embedding tables, the relation embeddings and ``W_r``."""
        return [self.user_embed, self.entity_embed, self.relation_embed, self.trans_W]
