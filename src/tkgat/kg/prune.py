"""Relation report and the ``pruned`` KG variant (decision 0016 item 8).

The report says, for every relation of a built KG variant, how many triples it has, how many items it reaches, how many
of its triples carry a date and how many distinct tails it uses. The rule of 0016 item 8 drops a relation that covers
fewer than 5% of the items **or** has fewer than 1,000 triples (at t1, the graph that training and validation use).
``write_pruned`` writes the KG without those relations as a new variant folder; it reads files of the pipeline only (the
KG triple and date files, ``relations.csv``, ``kg_meta.json``), never an interaction file, so no score can influence it.

The variant ``pruned`` is used only if validation Recall@20 on the small set is not lower than ``default`` and it has
fewer triples (the gate of item 9, ``docs/small-set-gate.md``).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

MIN_ITEM_SHARE = 0.05
MIN_TRIPLES = 1000
CUTS = ("t1", "t2")


def read_kg(kg_dir: str | Path, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(triples, dates)`` of one cut-off of a built variant: ``h, r, t`` and ``h, r, t, ts`` in the same order."""
    kg_dir = Path(kg_dir)
    tri = pd.read_csv(kg_dir / f"kg_{cut}.tsv", sep="\t")
    dates = pd.read_csv(kg_dir / f"kg_{cut}_dates.tsv", sep="\t")
    if not tri.equals(dates[["h", "r", "t"]]):
        raise ValueError(f"kg_{cut}_dates.tsv does not list the triples of kg_{cut}.tsv in the same order")
    return tri, dates


def relation_table(kg_dir: str | Path, cut: str = "t1") -> pd.DataFrame:
    """One row per relation at ``cut``: triples, items reached, share of items, dated triples and share, tails."""
    kg_dir = Path(kg_dir)
    info = json.loads((kg_dir / "kg_meta.json").read_text(encoding="utf-8"))
    relations = pd.read_csv(kg_dir / "relations.csv")
    _, dates = read_kg(kg_dir, cut)
    n_items = int(info["n_items"])
    grouped = dates.groupby("r").agg(
        triples=("t", "size"),
        items=("h", "nunique"),
        tails=("t", "nunique"),
        dated=("ts", lambda s: int(s.notna().sum())),
    )
    table = relations.set_index("relation").join(grouped).fillna(0)
    for col in ("triples", "items", "tails", "dated"):
        table[col] = table[col].astype(int)
    table["item_share"] = table["items"] / n_items
    table["dated_share"] = (table["dated"] / table["triples"].where(table["triples"] > 0)).fillna(0.0)
    return table.reset_index()


def prune_rule(
    table: pd.DataFrame, min_item_share: float = MIN_ITEM_SHARE, min_triples: int = MIN_TRIPLES
) -> pd.Series:
    """True for the relations to keep: at least ``min_item_share`` of the items **and** at least ``min_triples``."""
    return (table["item_share"] >= min_item_share) & (table["triples"] >= min_triples)


def write_pruned(src: str | Path, dst: str | Path, drop: list[str]) -> dict:
    """Write the variant at ``dst`` from the built variant at ``src`` without the relations named in ``drop``.

    Relation ids are renumbered 0..R'-1 in the old order, entity ids and the entity table stay as they are (a tail that
    only the dropped relations used keeps its row and simply has no edge, which the model handles: design L5), and
    ``kg_meta.json`` is rewritten with the new counts and ``variant: pruned``. Returns the new ``kg_meta`` content.
    """
    src, dst = Path(src), Path(dst)
    info = json.loads((src / "kg_meta.json").read_text(encoding="utf-8"))
    relations = pd.read_csv(src / "relations.csv")
    unknown = sorted(set(drop) - set(relations["name"]))
    if unknown:
        raise ValueError(f"not relations of {src}: {unknown}")
    kept = relations[~relations["name"].isin(drop)].reset_index(drop=True)
    if kept.empty:
        raise ValueError("every relation would be dropped")
    new_id = dict(zip(relations.loc[~relations["name"].isin(drop), "relation"], range(len(kept)), strict=True))
    dst.mkdir(parents=True, exist_ok=True)

    n_triples, n_dated, tails, heads = {}, {}, {}, {}
    for cut in CUTS:
        tri, dates = read_kg(src, cut)
        keep = tri["r"].isin(new_id)
        tri, dates = tri[keep].reset_index(drop=True), dates[keep].reset_index(drop=True)
        tri["r"] = tri["r"].map(new_id)
        dates["r"] = dates["r"].map(new_id)
        tri.to_csv(dst / f"kg_{cut}.tsv", sep="\t", index=False)
        dates.to_csv(dst / f"kg_{cut}_dates.tsv", sep="\t", index=False)
        n_triples[cut], n_dated[cut] = len(tri), int(dates["ts"].notna().sum())
        tails[cut], heads[cut] = set(tri["t"]), int(tri["h"].nunique())
    pd.DataFrame({"relation": range(len(kept)), "name": kept["name"]}).to_csv(dst / "relations.csv", index=False)
    shutil.copyfile(src / "entities.csv", dst / "entities.csv")

    out = copy.deepcopy(info)
    names = set(kept["name"])
    out.update(
        variant="pruned",
        pruned_from=info["variant"],
        dropped_relations=sorted(drop),
        prune_rule=f"item share >= {MIN_ITEM_SHARE} and triples >= {MIN_TRIPLES} at t1 (decision 0016 item 8)",
        n_relations=len(kept),
        n_triples=n_triples,
        n_dated_triples=n_dated,
        n_tail_entities_used={c: len(t) for c, t in tails.items()},
        n_entities_only_in_t2=len(tails["t2"] - tails["t1"]),
        n_items_with_facts=heads,
    )
    out["entity_ids"] = f"those of {info['variant']} (the entity table is copied, dropped tails keep their row)"
    for key in ("date_kind_per_relation", "per_relation"):
        if key in out:
            out[key] = [row for row in out[key] if row["relation"] in names]
    out.pop("n_dropped", None)  # the per-reason counts describe the source variant, not this one
    (dst / "kg_meta.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
