"""Collaborative knowledge graph (CKG) of KGAT: the train interactions plus the item-attribute KG (docs/design-kgat.md).

Only data structures and guards, no model. What goes in, and what is decided:

* Interaction edges come from ``EncodedSplit.train`` only (positives before t1). Val and test interactions never become
  edges. At test time the graph keeps the *train* edges and only the KG changes from t1 to t2 (design Q1): this follows
  decision 0002, item 7 ("no retraining on train+val"), so no edge exists at test that training did not see.
* The KG comes from ``scripts/06_build_kg.py`` (``kg_t1.tsv`` for train and val, ``kg_t2.tsv`` for test). Entity and
  relation ids are shared by both cut-offs and by the three variants, so one embedding table serves t1 and t2.
* At test, a triple whose tail never occurs in ``kg_t1`` is dropped and the number is reported (design Q2): the
  embedding of an entity the model never trained on must not enter a test score. The set of "known" tails is computed
  from ``kg_t1`` alone (:func:`used_tails`).

Node ids: user ``u`` is ``u``; entity ``e`` is ``n_users + e``, so item ``i`` is node ``n_users + i``.
Relation ids (``R`` KG relations, ``n_rel = 2R + 2``): ``0`` Interact (user -> item), ``r + 1`` KG relation ``r``,
``R + 1`` Interact^-1 (item -> user), ``r + R + 2`` inverse of KG relation ``r``. ``R`` comes from ``relations.csv``,
never from the triples seen, so it does not shrink when a relation has no triple at a cut-off.
Edges are sorted by (head, tail, relation) so that the edges of a head are contiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.data.encode import EncodedSplit
from tkgat.registry import sha256_file

INTERACT = 0
CUTOFFS = ("t1", "t2")


def relation_id(r: int) -> int:
    """Relation id of KG relation ``r`` in the CKG."""
    return r + 1


def inverse_relation_id(rid: int, n_relations: int) -> int:
    """Inverse of a relation id in ``0..R`` (Interact or a KG relation) for ``R = n_relations`` KG relations."""
    return rid + n_relations + 1


@dataclass(frozen=True)
class KGTriples:
    """Integer triples ``(h, r, t)`` of one KG at one cut-off, without inverse relations.

    Heads are items (``h < n_items``) and tails are other entities (``n_items <= t < n_entities``): the KG is 1-hop.
    """

    variant: str
    cutoff: str
    triples: np.ndarray = field(repr=False)  # (T, 3) int64
    n_items: int
    n_entities: int
    relations: tuple[str, ...]
    time_consistent: bool
    wikidata_snapshot: str | None

    def __post_init__(self) -> None:
        if self.cutoff not in CUTOFFS:
            raise ValueError(f"cutoff must be one of {CUTOFFS}, got {self.cutoff!r}")
        tri = self.triples
        if tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype != np.int64:
            raise ValueError("triples must be an int64 array of shape (T, 3)")
        problems = []
        if self.n_entities < self.n_items:
            problems.append("fewer entities than items")
        if len(tri):
            h, r, t = tri[:, 0], tri[:, 1], tri[:, 2]
            if h.min() < 0 or h.max() >= self.n_items:
                problems.append("head outside the item range")
            if t.min() < self.n_items or t.max() >= self.n_entities:
                problems.append("tail is an item or outside the entity range")
            if r.min() < 0 or r.max() >= self.n_relations:
                problems.append("relation id outside relations.csv")
            if len(np.unique(tri, axis=0)) != len(tri):
                problems.append("duplicate triples")
        if problems:
            raise ValueError("invalid KG: " + "; ".join(problems))

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    def __len__(self) -> int:
        return len(self.triples)


def no_kg(enc: EncodedSplit, cutoff: str = "t1") -> KGTriples:
    """The empty KG ("KGAT without KG", the reference of RQ2): the CKG then only has Interact and Interact^-1."""
    return KGTriples(
        variant="none",
        cutoff=cutoff,
        triples=np.empty((0, 3), dtype=np.int64),
        n_items=enc.n_items,
        n_entities=enc.n_items,
        relations=(),
        time_consistent=True,
        wikidata_snapshot=None,
    )


def used_tails(kg: KGTriples) -> np.ndarray:
    """Sorted unique tail entities of a KG; applied to the t1 KG it is the set of tails seen in training."""
    return np.unique(kg.triples[:, 2])


def load_kg(processed_dir: str | Path, variant: str, cutoff: str, enc: EncodedSplit, split_meta: dict) -> KGTriples:
    """Read ``<processed_dir>/kg/<variant>/kg_<cutoff>.tsv`` and check that it belongs to this split.

    This is the only function that reads the KG triple files. ``enc`` and ``split_meta`` must be the encoded split and
    the ``meta.json`` content loaded from ``processed_dir`` (``tkgat.pipeline.build.load_saved``). Raises ``ValueError``
    listing every problem: the KG was built from another split (hashes of the split files in ``kg_meta.json`` differ
    from ``meta.json`` or from the files on disk), the item order differs, the id spaces are not contiguous, or the
    triple count differs from ``kg_meta.json``.
    """
    if cutoff not in CUTOFFS:
        raise ValueError(f"cutoff must be one of {CUTOFFS}, got {cutoff!r}")
    processed_dir = Path(processed_dir)
    kg_dir = processed_dir / "kg" / variant
    info = json.loads((kg_dir / "kg_meta.json").read_text(encoding="utf-8"))
    entities = pd.read_csv(kg_dir / "entities.csv", dtype={"key": str})
    relations = pd.read_csv(kg_dir / "relations.csv")
    tri = pd.read_csv(kg_dir / f"kg_{cutoff}.tsv", sep="\t")

    problems = []
    if info.get("variant") != variant:
        problems.append(f"kg_meta.json is for variant {info.get('variant')!r}, not {variant!r}")
    built_from = info["inputs"]["split_sha256"]
    if built_from != split_meta["sha256"]:
        problems.append("the KG was built from another split (split hashes in kg_meta.json differ from meta.json)")
    for part, digest in split_meta["sha256"].items():
        if sha256_file(processed_dir / f"{part}.csv") != digest:
            problems.append(f"{part}.csv changed after the split was saved")
    if info["n_items"] != enc.n_items:
        problems.append(f"the KG has {info['n_items']} items, the split has {enc.n_items}")
    if list(entities["entity"]) != list(range(len(entities))) or len(entities) != info["n_entities"]:
        problems.append("entity ids are not contiguous or differ from kg_meta.json")
    elif len(entities) < enc.n_items or not (entities["kind"].iloc[: enc.n_items] == "item").all():
        problems.append("the first entities are not the items")
    elif not np.array_equal(entities["key"].iloc[: enc.n_items].astype(int).to_numpy(), enc.item_ids):
        problems.append("entity keys of the items differ from EncodedSplit.item_ids (other item order)")
    elif (entities["kind"].iloc[enc.n_items :] == "item").any():
        problems.append("an item entity after the first n_items entities")
    if list(relations["relation"]) != list(range(len(relations))) or len(relations) != info["n_relations"]:
        problems.append("relation ids are not contiguous or differ from kg_meta.json")
    if list(tri.columns) != ["h", "r", "t"]:
        problems.append(f"kg_{cutoff}.tsv has columns {list(tri.columns)}, expected ['h', 'r', 't']")
    elif len(tri) != info["n_triples"][cutoff]:
        problems.append(f"kg_{cutoff}.tsv has {len(tri)} triples, kg_meta.json says {info['n_triples'][cutoff]}")
    if problems:
        raise ValueError(f"KG {variant}/{cutoff} does not match the split:\n- " + "\n- ".join(problems))

    return KGTriples(
        variant=variant,
        cutoff=cutoff,
        triples=tri[["h", "r", "t"]].to_numpy(dtype=np.int64),
        n_items=enc.n_items,
        n_entities=len(entities),
        relations=tuple(relations["name"]),
        time_consistent=bool(info["time_consistent"]),
        wikidata_snapshot=info["wikidata_snapshot"],
    )


@dataclass(frozen=True)
class CKG:
    """Edge list of the collaborative knowledge graph, sorted by (head, tail, relation).

    ``edge_ts`` holds, for Interact and Interact^-1 edges, the train timestamp of the (user, item) pair (float64: a
    float32 would round Unix timestamps to about a minute) and NaN for KG edges. Plain KGAT does not read it.
    """

    n_users: int
    n_items: int
    n_entities: int
    n_relations: int  # KG relations R (without Interact and inverses)
    head: np.ndarray = field(repr=False)
    relation: np.ndarray = field(repr=False)
    tail: np.ndarray = field(repr=False)
    edge_ts: np.ndarray = field(repr=False)
    kg_variant: str
    cutoff: str
    time_consistent: bool
    wikidata_snapshot: str | None
    n_interactions: int
    n_triples: int  # KG triples used (after the Q2 rule)
    n_triples_dropped_unseen_tail: int

    @property
    def n_rel(self) -> int:
        return 2 * self.n_relations + 2

    @property
    def n_nodes(self) -> int:
        return self.n_users + self.n_entities

    @property
    def n_edges(self) -> int:
        return len(self.head)

    def degree(self) -> np.ndarray:
        """Number of edges per head, length ``n_nodes``; every edge has its inverse, so this is also the in-degree."""
        return np.bincount(self.head, minlength=self.n_nodes)

    def n_isolated_entities(self) -> int:
        """Entities without any edge (their embedding gets no gradient and is never a KG negative)."""
        return int((self.degree()[self.n_users :] == 0).sum())

    def summary(self) -> dict:
        """Counts and provenance for ``kgat_summary.json``."""
        return {
            "kg_variant": self.kg_variant,
            "cutoff": self.cutoff,
            "time_consistent": self.time_consistent,
            "wikidata_snapshot": self.wikidata_snapshot,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "n_rel": self.n_rel,
            "n_nodes": self.n_nodes,
            "n_interactions": self.n_interactions,
            "n_triples": self.n_triples,
            "n_triples_dropped_unseen_tail": self.n_triples_dropped_unseen_tail,
            "n_edges": self.n_edges,
            "n_isolated_entities": self.n_isolated_entities(),
        }


def build_ckg(enc: EncodedSplit, kg: KGTriples, known_tails: np.ndarray | None = None) -> CKG:
    """Build the CKG from the train interactions of ``enc`` and the KG ``kg``.

    Only ``enc.train`` and ``enc.train_ts`` are read; val and test never enter. With ``known_tails`` (``used_tails``
    of the t1 KG) every triple whose tail is not in that set is dropped and counted (design Q2): pass it for the test
    graph, leave it ``None`` for the training graph.
    """
    if kg.n_items != enc.n_items:
        raise ValueError(f"the KG has {kg.n_items} items, the split has {enc.n_items}")
    pairs = enc.train
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("duplicate (user, item) pairs in the train interactions")
    if len(enc.train_ts) != len(pairs):
        raise ValueError("train_ts is not aligned with train")
    n_users, n_items, n_rel_kg = enc.n_users, enc.n_items, kg.n_relations

    tri = kg.triples
    dropped = 0
    if known_tails is not None:
        keep = np.isin(tri[:, 2], known_tails)
        dropped = int((~keep).sum())
        tri = tri[keep]

    user = pairs[:, 0]
    item = n_users + pairs[:, 1]
    h_kg, r_kg, t_kg = n_users + tri[:, 0], tri[:, 1], n_users + tri[:, 2]
    head = np.concatenate([user, item, h_kg, t_kg])
    tail = np.concatenate([item, user, t_kg, h_kg])
    relation = np.concatenate(
        [
            np.full(len(pairs), INTERACT),
            np.full(len(pairs), inverse_relation_id(INTERACT, n_rel_kg)),
            relation_id(r_kg),
            inverse_relation_id(relation_id(r_kg), n_rel_kg),
        ]
    ).astype(np.int64)
    ts = np.asarray(enc.train_ts, dtype=np.float64)
    edge_ts = np.concatenate([ts, ts, np.full(2 * len(tri), np.nan)])

    order = np.lexsort((relation, tail, head))  # last key is the primary one: head, then tail, then relation
    return CKG(
        n_users=n_users,
        n_items=n_items,
        n_entities=kg.n_entities,
        n_relations=n_rel_kg,
        head=head[order].astype(np.int64),
        relation=relation[order],
        tail=tail[order].astype(np.int64),
        edge_ts=edge_ts[order],
        kg_variant=kg.variant,
        cutoff=kg.cutoff,
        time_consistent=kg.time_consistent,
        wikidata_snapshot=kg.wikidata_snapshot,
        n_interactions=len(pairs),
        n_triples=len(tri),
        n_triples_dropped_unseen_tail=dropped,
    )
