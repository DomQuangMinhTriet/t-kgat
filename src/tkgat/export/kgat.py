"""Write an official split and KG in the file format of the KGAT code (decision 0008).

Nothing is decided here: the interactions, the cut-offs, the entity and relation ids and the KG come from the
pipeline (``save_split``, ``06_build_kg.py``); this module only re-labels ids and writes text files. The layout
follows the KGAT dataset folders and the exporter of PR #20, plus the two things the project protocol needs:

``train.txt``, ``val.txt``, ``test.txt``    ``user item item ...`` (positives, items in time order)
``val_mask.txt``, ``test_mask.txt``         every interaction before t1 / t2 (any rating) to exclude when ranking
``<split>_rating.txt``                      ``user item:rating ...``
``<split>_timestamp.txt``                   ``user item:ts ...``, only in the timestamp variant
``kg_final.txt``                            KG at t1 (train and val), ``h r t`` per line
``kg_test.txt``                             KG at t2 (test)
``user_list.txt``, ``item_list.txt``, ``entity_list.txt``, ``entity_metadata.tsv``, ``relation_list.txt``
``interactions_chronological.tsv``, ``dataset_meta.json``

Users are numbered in the order of ``EncodedSplit.user_ids`` and items in the order of ``EncodedSplit.item_ids``;
items are the first entities, as in the KGAT code, so item ``i`` is also entity ``i``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from tkgat.data.encode import EncodedSplit
from tkgat.data.split import TemporalSplit

SPLITS = ("train", "val", "test")
NO_FREEBASE = "NA"


def _grouped(df: pd.DataFrame, user_map: dict[int, int], item_map: dict[int, int]) -> dict[int, list[tuple]]:
    """``{user: [(ts, item, rating)]}`` in time order (ties by item id), in remapped ids; unknown ids are skipped."""
    d = df[df["user"].isin(user_map) & df["item"].isin(item_map)]
    ratings = d["rating"] if "rating" in d else pd.Series(0, index=d.index)
    rows = sorted(zip(d["ts"], d["user"], d["item"], ratings, strict=True), key=lambda r: (r[1], r[0], r[2]))
    out: dict[int, list[tuple]] = {}
    for ts, user, item, rating in rows:
        out.setdefault(user_map[int(user)], []).append((int(ts), item_map[int(item)], int(rating)))
    return out


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


def _write_items(path: Path, grouped: dict[int, list[tuple]]) -> None:
    _write_lines(path, [" ".join(str(v) for v in (u, *[item for _, item, _ in grouped[u]])) for u in sorted(grouped)])


def _write_labels(path: Path, grouped: dict[int, list[tuple]], field: int) -> None:
    lines = [" ".join([str(u), *[f"{row[1]}:{row[field]}" for row in grouped[u]]]) for u in sorted(grouped)]
    _write_lines(path, lines)


def _write_kg(path: Path, kg: pd.DataFrame) -> None:
    _write_lines(path, [f"{int(h)} {int(r)} {int(t)}" for h, r, t in zip(kg["h"], kg["r"], kg["t"], strict=True)])


def export_kgat(
    out: Path,
    split: TemporalSplit,
    enc: EncodedSplit,
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    kg_t1: pd.DataFrame,
    kg_t2: pd.DataFrame,
    freebase: dict[int, str],
    timestamps: bool,
    info: dict | None = None,
) -> dict:
    """Write the KGAT folder ``out`` and return its ``dataset_meta.json`` content.

    ``entities`` (``entity, kind, key, label``), ``relations`` (``relation, name``) and the integer triples
    ``kg_t1`` / ``kg_t2`` (``h, r, t``) are the files of ``data/processed/<run>/kg/<variant>/``; ``freebase`` maps a
    raw item id to its Freebase id. ``timestamps`` adds the timestamp files and column; everything else is identical.
    """
    if list(entities["entity"]) != list(range(len(entities))):
        raise ValueError("entity ids must be contiguous from 0")
    n_items = enc.n_items
    if not (entities["kind"].iloc[:n_items] == "item").all() or list(
        entities["key"].iloc[:n_items].astype(int)
    ) != list(enc.item_ids):
        raise ValueError("the first entities must be the train items in EncodedSplit order")
    out.mkdir(parents=True, exist_ok=True)
    user_map = {int(raw): i for i, raw in enumerate(enc.user_ids)}
    item_map = {int(raw): i for i, raw in enumerate(enc.item_ids)}

    parts = {"train": split.train, "val": split.val, "test": split.test}
    grouped = {name: _grouped(df, user_map, item_map) for name, df in parts.items()}
    for name in SPLITS:
        _write_items(out / f"{name}.txt", grouped[name])
        _write_labels(out / f"{name}_rating.txt", grouped[name], field=2)
        if timestamps:
            _write_labels(out / f"{name}_timestamp.txt", grouped[name], field=0)
    masks = {"val": split.seen_before_val, "test": split.seen_before_test}
    for name, df in masks.items():
        _write_items(out / f"{name}_mask.txt", _grouped(df, user_map, item_map))

    header = ["ordinal", "split", "user", "item", "rating", *(["timestamp"] if timestamps else [])]
    with (out / "interactions_chronological.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        ordinal = 0
        for name in SPLITS:
            events = sorted((row[0], u, row[1], row[2]) for u, rows in grouped[name].items() for row in rows)
            for ts, u, item, rating in events:
                writer.writerow([ordinal, name, u, item, rating, *([ts] if timestamps else [])])
                ordinal += 1

    _write_lines(out / "user_list.txt", ["org_id remap_id", *[f"{raw} {i}" for raw, i in user_map.items()]])
    _write_lines(
        out / "item_list.txt",
        [
            "org_id remap_id freebase_id",
            *[f"{raw} {i} {freebase.get(raw, NO_FREEBASE)}" for raw, i in item_map.items()],
        ],
    )
    org = [
        f"movielens.item.{k}" if kind == "item" else str(k)
        for kind, k in zip(entities["kind"], entities["key"], strict=True)
    ]
    _write_lines(out / "entity_list.txt", ["org_id remap_id", *[f"{o} {i}" for i, o in enumerate(org)]])
    meta_rows = zip(org, entities["entity"], entities["kind"], entities["label"].fillna(""), strict=True)
    with (out / "entity_metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["org_id", "remap_id", "kind", "label"])
        writer.writerows(meta_rows)
    _write_lines(
        out / "relation_list.txt",
        [
            "org_id remap_id",
            *[f"wikidata.{n} {int(r)}" for r, n in zip(relations["relation"], relations["name"], strict=True)],
        ],
    )
    _write_kg(out / "kg_final.txt", kg_t1)
    _write_kg(out / "kg_test.txt", kg_t2)

    meta = {
        "protocol": "official split of the project (decisions 0002, 0004, 0006); see docs/data-dictionary.md",
        "timestamp_variant": timestamps,
        "cutoffs": {"t1": int(split.t1), "t2": int(split.t2)},
        "n_users": enc.n_users,
        "n_items": n_items,
        "n_entities": int(len(entities)),
        "n_relations": int(len(relations)),
        "interactions": {name: int(len(df)) for name, df in parts.items()},
        "kg_triples": {"kg_final.txt (t1)": int(len(kg_t1)), "kg_test.txt (t2)": int(len(kg_t2))},
        "mask_rows": {name: int(len(df)) for name, df in masks.items()},
        **(info or {}),
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
