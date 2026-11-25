"""Refit on train and validation (decision 0016 item 4): the number of epochs and the graph.

After a grid point is selected on validation, every seed is trained again from scratch on train ∪ validation for ``E``
epochs, without early stopping, and scored on test once. ``E`` is the median over the seeds of the best epoch of the
selected point, a number that comes from validation histories only (:func:`refit_epochs`). The graph of the refit has
the train and validation interactions as edges and the KG at t2 (:func:`refit_graph`); the Delta-t anchors of
``build_ckg`` are recomputed from the merged timestamps there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tkgat.data.encode import EncodedSplit
from tkgat.kg.ckg import (
    CKG,
    KGTriples,
    build_ckg,
    load_kg,
    load_kg_dates,
    permute_kg_ts,
    permute_train_ts,
)


def refit_epochs(best_epochs: list[int]) -> int:
    """Median over the seeds of the validation best epoch, rounded up when the number of seeds is even."""
    if not len(best_epochs) or min(best_epochs) < 1:
        raise ValueError(f"need the best epoch (>= 1) of every seed, got {best_epochs}")
    return int(np.ceil(np.median(best_epochs)))


def refit_graph(
    dcfg: dict,
    data: EncodedSplit,
    meta: dict,
    kg_t1: KGTriples,
    *,
    control: str | None,
    seed: int,
    time_kg: bool,
) -> tuple[EncodedSplit, CKG]:
    """The training graph of a refit: ``data`` is ``encode_trainval(split)``, the KG is the one at t2.

    ``kg_t1`` only says which KG variant to use (``none`` gives a graph without KG). Every t2 entity is trainable here
    because the model is trained from scratch, so no tail is dropped. The controls of T-KGAT shuffle what they
    shuffled before, now over the merged edges (``permute_train_ts``) or the t2 fact dates (``permute_kg_ts``), one
    permutation per seed. Returns the (possibly permuted) data and the graph.
    """
    if control in ("permute_train_ts", "permute_both_ts"):
        data = permute_train_ts(data, seed)
    if kg_t1.variant == "none":
        return data, build_ckg(data, kg_t1)
    processed = Path(dcfg["processed_dir"])
    kg_t2 = load_kg(processed, kg_t1.variant, "t2", data, meta)
    kg_ts = None
    if time_kg:
        kg_ts = load_kg_dates(processed, kg_t2)
        if control in ("permute_kg_ts", "permute_both_ts"):
            kg_ts = permute_kg_ts(kg_ts, seed)
    return data, build_ckg(data, kg_t2, kg_ts=kg_ts)
