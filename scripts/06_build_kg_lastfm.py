"""Build the time-consistent Last-FM KG at t1 (train/val) and t2 (test), decisions 0004 and 0006.

    python scripts/06_build_kg_lastfm.py --data configs/data/lastfm.yaml
    python scripts/06_build_kg_lastfm.py --data configs/data/lastfm_u10.yaml --variant default

The Last-FM counterpart of ``scripts/06_build_kg.py``: the protocol and the output format are identical,
only the sources differ. Needs the saved split (script 02), the knowledge graph published with KGAT
(``knowledge_graph_attention_network/Data/last-fm``) and ``data-last-fm/freebase-subset.tsv``, a dump of
Freebase of 2015-08-09 whose subjects are the ``org_id`` of ``entity_list.txt`` (so no linking step).

A fact is dated only through the existence date of its tail (the published KG has no statement dates):
the earliest of ``music.release.release_date``, ``music.artist.active_start`` and
``people.person.date_of_birth``. A head is a train item, so its own release never drops its facts
(decision 0006 B).

Output, one folder per variant (default, drop_undated, static) in <processed_dir>/kg/<variant>/, in the
format ``tkgat.kg.ckg.load_kg`` checks:
  kg_t1.tsv, kg_t2.tsv   integer triples h, r, t (no inverse relations; the model adds them)
  kg_t1_dates.tsv, kg_t2_dates.tsv
                         the same triples in the same order plus ts, the Unix time (UTC midnight) from which the
                         fact holds, empty when undated (T-KGAT Stage 2, decision 0009 item 9)
  entities.csv           entity, kind (item | freebase), key, label
                         entities 0..n_items-1 are the train items in tkgat.data.encode order
  relations.csv          relation, name
  dropped.csv            facts removed at each cut-off, with the reason
  kg_meta.json           counts per relation, limitations, md5 of the inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tkgat.data.encode import encode
from tkgat.kg.lastfm import (
    DATE_COLUMNS,
    FREEBASE_SNAPSHOT,
    HEAD_NOT_ITEM,
    KNOWN_LIMITATIONS,
    check_kg,
    item_entities,
    lastfm_facts,
    read_author_kg,
    read_freebase_subset,
    relation_stats,
    seeded_triples,
)
from tkgat.kg.temporal import STATIC_NOTE, VARIANTS, assign_ids, cutoff_day, cutoff_facts, triple_dates
from tkgat.pipeline.build import load_config, load_saved

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_DEFAULT = ROOT / "knowledge_graph_attention_network" / "Data" / "last-fm"
FREEBASE_DEFAULT = ROOT / "data-last-fm" / "freebase-subset.tsv"
DROPPED_COLUMNS = [
    "h",
    "item",
    "author_head",
    "author_tail",
    "head_freebase_id",
    "relation",
    "tail",
    "tail_label",
    *DATE_COLUMNS,
    "reason",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_relation(kept: dict[str, pd.DataFrame], dropped: dict[str, pd.DataFrame]) -> list[dict]:
    """Kept and dropped triples per relation and cut-off, as in ``scripts/06_build_kg.py``."""
    rows = []
    for cut in kept:
        k = kept[cut]["relation"].value_counts().rename("kept")
        d = (
            dropped[cut].groupby(["relation", "reason"]).size().unstack(fill_value=0)
            if len(dropped[cut])
            else pd.DataFrame(index=k.index)
        )
        t = pd.concat([k, d], axis=1).fillna(0).astype(int)
        for reason in ("future_tail", "undated", HEAD_NOT_ITEM):
            if reason not in t:
                t[reason] = 0
        t.insert(0, "cutoff", cut)
        rows.append(t.rename_axis("relation").reset_index())
    return pd.concat(rows).to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--author-data", default=str(AUTHOR_DEFAULT), help="the published KGAT Data/last-fm folder")
    ap.add_argument("--freebase", default=str(FREEBASE_DEFAULT), help="Freebase facts of the KG entities")
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = ap.parse_args()

    cfg = load_config(args.data)
    s, meta = load_saved(cfg)
    enc = encode(s)
    author = Path(args.author_data)
    freebase_path = Path(args.freebase)

    items = item_entities(enc.item_ids, author / "item_list.txt")
    triples, entity_ids, relations = read_author_kg(author)
    seeded = seeded_triples(items, triples)
    tail_facts = read_freebase_subset(freebase_path, set(entity_ids.loc[seeded["t"].unique()]))
    facts, format_dropped = lastfm_facts(items, triples, entity_ids, relations, tail_facts)
    print(
        f"[{cfg.get('run_name', cfg['name'])}] author triples {len(triples)}, item-seeded {len(seeded)}, "
        f"head-seeded (1-hop, usable) {len(facts)}, dropped as {HEAD_NOT_ITEM} {len(format_dropped)}"
    )

    cutoffs = {"t1": meta["t1"], "t2": meta["t2"]}
    # every variant shares the entity ids of the static KG (all facts of the train items)
    universe, _ = cutoff_facts(facts, cutoffs["t1"], "static")

    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        kept, dropped = {}, {}
        for cut, ts in cutoffs.items():
            kept[cut], temporal_dropped = cutoff_facts(facts, ts, variant)
            dropped[cut] = pd.concat([temporal_dropped, format_dropped], ignore_index=True)
            problems = check_kg(kept[cut], ts, variant, enc.n_items)
            if problems:
                raise RuntimeError(f"KG check failed at {cut} ({variant}):\n- " + "\n- ".join(problems))
        key = ["h", "relation", "tail"]
        if kept["t1"].merge(kept["t2"], on=key).shape[0] != len(kept["t1"]):
            raise RuntimeError(f"KG at t1 is not contained in KG at t2 ({variant})")
        ent, rel, tri = assign_ids(enc.n_items, items, kept, relations, universe=universe)

        out = Path(cfg["processed_dir"]) / "kg" / variant
        out.mkdir(parents=True, exist_ok=True)
        n_dated = {}
        for cut, t in tri.items():
            t.to_csv(out / f"kg_{cut}.tsv", sep="\t", index=False)
            dates = triple_dates(kept[cut], ent, rel)
            if not dates[["h", "r", "t"]].equals(t[["h", "r", "t"]]):
                raise RuntimeError(f"kg_{cut}_dates.tsv would not follow the triples of kg_{cut}.tsv ({variant})")
            dates.to_csv(out / f"kg_{cut}_dates.tsv", sep="\t", index=False)
            n_dated[cut] = int(dates["ts"].notna().sum())
        ent.to_csv(out / "entities.csv", index=False)
        rel.to_csv(out / "relations.csv", index=False)
        pd.concat([d.reindex(columns=DROPPED_COLUMNS).assign(cutoff=c) for c, d in dropped.items()]).to_csv(
            out / "dropped.csv", index=False
        )

        tails = {c: set(k["tail"]) for c, k in kept.items()}
        info = {
            "dataset": cfg["name"],
            "run_name": cfg.get("run_name", cfg["name"]),
            "variant": variant,
            "decisions": ["0004", "0006"],
            "wikidata_snapshot": None,  # Last-FM uses Freebase, not Wikidata
            "kg_snapshot": f"Freebase dump {FREEBASE_SNAPSHOT} ({freebase_path.as_posix()})",
            "kg_source": "knowledge graph published with KGAT (Data/last-fm), no statement dates",
            "time_consistent": variant != "static",
            "note": STATIC_NOTE if variant == "static" else "",
            "known_limitations": KNOWN_LIMITATIONS,
            "entity_ids": "shared by default, drop_undated and static (vocabulary of the static KG)",
            "cutoffs": {c: {"ts": int(ts), "day_utc": str(cutoff_day(ts))} for c, ts in cutoffs.items()},
            "inputs": {
                "author_kg": (author / "kg_final.txt").as_posix(),
                "author_kg_md5": md5(author / "kg_final.txt"),
                "author_entities": (author / "entity_list.txt").as_posix(),
                "author_entities_md5": md5(author / "entity_list.txt"),
                "author_items": (author / "item_list.txt").as_posix(),
                "author_items_md5": md5(author / "item_list.txt"),
                "freebase": freebase_path.as_posix(),
                "freebase_md5": md5(freebase_path),
                "split_sha256": meta.get("sha256", {}),
            },
            "n_items": enc.n_items,
            "n_items_linked": int(items["freebase_id"].notna().sum()),
            "n_items_with_facts": {c: int(k["h"].nunique()) for c, k in kept.items()},
            "n_author_triples": len(triples),
            "n_item_seeded_triples": len(seeded),
            "n_head_seeded_triples": len(facts),
            "n_dropped_head_not_train_item": len(format_dropped),
            "n_entities": len(ent),
            "n_relations": len(rel),
            "n_triples": {c: len(t) for c, t in tri.items()},
            "n_dated_triples": n_dated,
            "n_tail_entities_used": {c: len(t) for c, t in tails.items()},
            "n_entities_only_in_t2": len(tails["t2"] - tails["t1"]),
            "n_dropped": {c: d["reason"].value_counts().to_dict() for c, d in dropped.items()},
            "date_kind_per_relation": relation_stats(facts).reset_index().to_dict(orient="records"),
            "per_relation": per_relation(kept, dropped),
        }
        (out / "kg_meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{info['run_name']} / {variant}] triples t1={len(tri['t1'])} t2={len(tri['t2'])} "
            f"dated t1={n_dated['t1']} entities={len(ent)} only-in-t2={info['n_entities_only_in_t2']} -> {out}"
        )


if __name__ == "__main__":
    main()
