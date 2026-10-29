"""Build the time-consistent KG at t1 (train/val) and t2 (test), decisions 0004 and 0006.

    python scripts/06_build_kg.py --data configs/data/ml100k.yaml
    python scripts/06_build_kg.py --data configs/data/ml1m.yaml --variant default

Needs the saved split (script 02), the links (script 03) and data/raw/wikidata/facts.tsv (script 04).
Output, one folder per variant (default, drop_undated, static) in <processed_dir>/kg/<variant>/:
  kg_t1.tsv, kg_t2.tsv   integer triples h, r, t (no inverse relations; the model adds them)
  kg_t1_dates.tsv, kg_t2_dates.tsv
                         the same triples in the same order plus ts, the Unix time (UTC midnight) from which the
                         fact holds, empty when undated (T-KGAT Stage 2, decision 0009 item 9); kg_t*.tsv are unchanged
  entities.csv           entity, kind (item | wikidata | movielens_metadata), key, label
                         entities 0..n_items-1 are the train items in tkgat.data.encode order
  relations.csv          relation, name
  dropped.csv            facts removed at each cut-off, with the reason
  kg_meta.json           counts per relation, limitations, md5 of the inputs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from tkgat.data.encode import encode
from tkgat.data.loaders import load
from tkgat.kg.temporal import (
    KNOWN_LIMITATIONS,
    STATIC_NOTE,
    VARIANTS,
    annotate,
    assign_ids,
    build_kg,
    check_kg,
    cutoff_day,
    cutoff_facts,
    item_entities,
    movielens_facts,
    relation_stats,
    triple_dates,
)
from tkgat.pipeline.build import load_config, load_saved

LINKS = {
    "ml-100k": ("data/processed/links/ml100k_links.csv", "item_100k"),
    "ml-1m": ("data/processed/links/ml1m_links.csv", "item"),
}
ITEM_METADATA_FILES = {"ml-100k": "u.item", "ml-1m": "movies.dat"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_relation(kept: dict[str, pd.DataFrame], dropped: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for cut in kept:
        k = kept[cut]["relation"].value_counts().rename("kept")
        d = (
            dropped[cut].groupby(["relation", "reason"]).size().unstack(fill_value=0)
            if len(dropped[cut])
            else pd.DataFrame(index=k.index)
        )  # static drops nothing
        t = pd.concat([k, d], axis=1).fillna(0).astype(int)
        for reason in ("future_statement", "future_tail", "undated"):
            if reason not in t:
                t[reason] = 0
        t.insert(0, "cutoff", cut)
        rows.append(t.rename_axis("relation").reset_index())
    return pd.concat(rows).to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--facts", default="data/raw/wikidata/facts.tsv")
    ap.add_argument("--links", help="override the links file for this dataset")
    ap.add_argument("--raw-dir", help="override raw_dir from the data config (needed for MovieLens metadata)")
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = ap.parse_args()

    cfg = load_config(args.data)
    s, meta = load_saved(cfg)
    enc = encode(s)
    link_path, item_col = LINKS[cfg["name"]]
    link_path = Path(args.links or link_path)
    links = pd.read_csv(link_path, dtype={"freebase_id": str})
    items = item_entities(enc.item_ids, links, item_col)
    raw_dir = Path(args.raw_dir or cfg["raw_dir"])
    catalog_path = raw_dir / ITEM_METADATA_FILES[cfg["name"]]
    _, catalog = load(cfg["name"], raw_dir)
    ml_facts = movielens_facts(enc.item_ids, catalog)

    facts_path = Path(args.facts)
    raw = pd.read_csv(facts_path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    facts = annotate(raw[raw["freebase_id"].isin(set(items["freebase_id"].dropna()))])
    relations = sorted(set(raw["relation"]) | set(ml_facts["relation"]))
    cutoffs = {"t1": meta["t1"], "t2": meta["t2"]}
    film_facts = items[["h", "freebase_id"]].merge(facts, on="freebase_id")
    source_facts = pd.concat([film_facts, ml_facts], ignore_index=True)
    # every variant shares the entity ids of the static KG (all facts of the train items)
    wiki_universe, _ = build_kg(facts, items, cutoffs["t1"], "static")
    universe = pd.concat([wiki_universe, ml_facts], ignore_index=True)

    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        kept, dropped = {}, {}
        wikidata_kept = {}
        for cut, ts in cutoffs.items():
            wikidata_kept[cut], wiki_dropped = build_kg(facts, items, ts, variant)
            ml_kept, ml_dropped = cutoff_facts(ml_facts, ts, variant)
            kept[cut] = pd.concat([wikidata_kept[cut], ml_kept], ignore_index=True)
            dropped[cut] = pd.concat([wiki_dropped, ml_dropped], ignore_index=True)
            problems = check_kg(kept[cut], ts, variant, enc.n_items)
            if problems:
                raise RuntimeError(f"KG check failed at {cut} ({variant}):\n- " + "\n- ".join(problems))
        key = ["h", "relation", "tail"]
        if not kept["t1"].merge(kept["t2"], on=key).shape[0] == len(kept["t1"]):
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
        cols = [
            "freebase_id",
            "item",
            "relation",
            "tail",
            "tail_label",
            "point_in_time",
            "start_time",
            "tail_publication",
            "tail_inception",
            "tail_birth",
            "reason",
        ]
        pd.concat([d[cols].assign(cutoff=c) for c, d in dropped.items()]).to_csv(out / "dropped.csv", index=False)

        tails = {c: set(k["tail"]) for c, k in kept.items()}
        items_t = {c: int(k["h"].nunique()) for c, k in kept.items()}
        info = {
            "dataset": cfg["name"],
            "variant": variant,
            "decisions": ["0004", "0006"],
            "wikidata_snapshot": "2026-10-17",
            "time_consistent": variant != "static",
            "note": STATIC_NOTE if variant == "static" else "",
            "known_limitations": KNOWN_LIMITATIONS,
            "entity_ids": "shared by default, drop_undated and static (vocabulary of the static KG)",
            "cutoffs": {c: {"ts": int(ts), "day_utc": str(cutoff_day(ts))} for c, ts in cutoffs.items()},
            "inputs": {
                "facts": str(facts_path),
                "facts_md5": md5(facts_path),
                "links": str(link_path),
                "links_md5": md5(link_path),
                "movielens_items": str(catalog_path),
                "movielens_items_md5": md5(catalog_path),
                "split_sha256": meta.get("sha256", {}),
            },
            "n_items": enc.n_items,
            "n_items_linked": int(items["freebase_id"].notna().sum()),
            "n_items_with_facts": items_t,
            "n_items_with_wikidata_facts": {c: int(k["h"].nunique()) for c, k in wikidata_kept.items()},
            "movielens_metadata_relations": sorted(set(ml_facts["relation"])),
            "movielens_metadata_note": (
                "genres and release year/decade come from the MovieLens item catalog; ML-1M release years are "
                "parsed from the title"
            ),
            "n_movielens_metadata_triples": {
                c: int(kept[c]["relation"].str.startswith("movielens.").sum()) for c in kept
            },
            "n_entities": len(ent),
            "n_relations": len(rel),
            "n_triples": {c: len(t) for c, t in tri.items()},
            "n_dated_triples": n_dated,
            "n_tail_entities_used": {c: len(t) for c, t in tails.items()},
            "n_entities_only_in_t2": len(tails["t2"] - tails["t1"]),
            "n_dropped": {c: d["reason"].value_counts().to_dict() for c, d in dropped.items()},
            "date_kind_per_relation": relation_stats(source_facts).reset_index().to_dict(orient="records"),
            "per_relation": per_relation(kept, dropped),
        }
        (out / "kg_meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{cfg['name']} / {variant}] triples t1={len(tri['t1'])} t2={len(tri['t2'])} "
            f"entities={len(ent)} only-in-t2={info['n_entities_only_in_t2']} -> {out}"
        )


if __name__ == "__main__":
    main()
