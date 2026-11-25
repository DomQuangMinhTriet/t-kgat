"""Map raw ids to contiguous indices using the *train* vocabulary only."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tkgat.data.split import TemporalSplit


@dataclass
class EncodedSplit:
    n_users: int
    n_items: int
    user_ids: np.ndarray  # index -> raw user id
    item_ids: np.ndarray  # index -> raw item id
    train: np.ndarray  # (n, 2) int64 [u, i]
    train_ts: np.ndarray
    val: dict[int, np.ndarray]  # u -> ground-truth items
    test: dict[int, np.ndarray]
    mask_val: sp.csr_matrix  # items to exclude when ranking for val
    mask_test: sp.csr_matrix

    def train_csr(self) -> sp.csr_matrix:
        u, i = self.train[:, 0], self.train[:, 1]
        return sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(self.n_users, self.n_items))


def encode(s: TemporalSplit) -> EncodedSplit:
    user_ids, item_ids = s.users, s.items
    umap = pd.Series(np.arange(len(user_ids)), index=user_ids)
    imap = pd.Series(np.arange(len(item_ids)), index=item_ids)

    def pairs(df: pd.DataFrame) -> np.ndarray:
        d = df[df["user"].isin(umap.index) & df["item"].isin(imap.index)]
        return np.stack([umap[d["user"]].to_numpy(), imap[d["item"]].to_numpy()], axis=1).astype(np.int64)

    def grouped(df: pd.DataFrame) -> dict[int, np.ndarray]:
        p = pairs(df)
        out: dict[int, list[int]] = {}
        for u, i in p:
            out.setdefault(int(u), []).append(int(i))
        return {u: np.asarray(v, dtype=np.int64) for u, v in out.items()}

    def mask(df: pd.DataFrame) -> sp.csr_matrix:
        p = pairs(df)
        m = sp.csr_matrix((np.ones(len(p), dtype=np.float32), (p[:, 0], p[:, 1])), shape=(len(user_ids), len(item_ids)))
        m.data[:] = 1.0
        return m

    tr = s.train.sort_values(["user", "ts"])
    return EncodedSplit(
        n_users=len(user_ids),
        n_items=len(item_ids),
        user_ids=user_ids,
        item_ids=item_ids,
        train=pairs(tr),
        train_ts=tr["ts"].to_numpy(),
        val=grouped(s.val),
        test=grouped(s.test),
        mask_val=mask(s.seen_before_val),
        mask_test=mask(s.seen_before_test),
    )


def encode_trainval(s: TemporalSplit) -> EncodedSplit:
    """The split for the refit (decision 0016 item 4): train and validation interactions together as the train part.

    The vocabulary is the train vocabulary of ``encode(s)`` (users, items and their order): validation keeps only users
    and items that survive in train, so adding it changes no id, and the KG entity ids stay valid. ``val`` is empty and
    ``mask_val`` is not used by a refit. ``test`` and ``mask_test`` are exactly those of ``encode(s)``: the mask is
    every interaction before t2, which is train, validation and the pairs the 5-core or the cold filters dropped.
    ``train_ts`` holds the timestamps of the merged part, so the Delta-t anchor ``t_ref`` is recomputed on it.
    """
    merged = pd.concat([s.train, s.val], ignore_index=True)
    if merged.duplicated(["user", "item"]).any():
        raise ValueError("a (user, item) pair is in both train and val")
    out = encode(replace(s, train=merged, val=s.val.iloc[0:0]))
    base = encode(s)
    if not (np.array_equal(out.user_ids, base.user_ids) and np.array_equal(out.item_ids, base.item_ids)):
        raise ValueError("validation holds a user or item that is not in train: the vocabulary would change")
    return out
