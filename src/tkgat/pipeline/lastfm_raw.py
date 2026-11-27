"""Temporal splits and KGAT-format export of raw LFM-1b pairs.

Interactions come from the user's LFM-1b Drive file; facts and date evidence come
from the supplied Freebase TSV. KGAT supplies only the nine-relation schema.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.data.encode import encode
from tkgat.data.lastfm_audit import audit_raw_split
from tkgat.data.lastfm_raw import load_raw_pairs, local_source_path, open_raw_text, source_sha256, write_json
from tkgat.data.leakage import check_split
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.export.kgat import export_kgat
from tkgat.kg.lastfm import (
    DATE_COLUMNS,
    FREEBASE_SNAPSHOT,
    annotate_lastfm,
    check_kg,
    relation_stats,
)
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS, freebase_id, read_freebase_source
from tkgat.kg.temporal import assign_ids, cutoff_facts, triple_dates
from tkgat.pipeline.build import save_split
from tkgat.registry import sha256_file

LIMITATIONS = [
    "When used, Freebase 2015-08-09 is a later snapshot, not knowledge observed at each listening timestamp.",
    "KG timestamps are tail existence proxies, not statement creation or validity dates.",
    "Missing fact timestamps remain null; no interaction date or snapshot date is substituted.",
    "The supplied track-to-MID mapping has paper lineage and covers only part of LFM-1b.",
    "Only the nine KGAT relations are used; missing source facts are never filled from the paper KG.",
    "Literal values are preserved as stable literal nodes; their contents are not assumed to be fact validity dates.",
    "The supplied Freebase TSV is an extraction of selected entities/predicates, not the complete Freebase dump.",
    "Tracks without Freebase links or KG edges are retained in interactions; their KG coverage is reported.",
    "Only train items seed KG heads; the processed one-hop format keeps separate tail entities.",
    "The inherited conservative date rule reads dates ending in -01-01 as year-only (31 December).",
    "Unparseable dates, including BC years, are undated. Multiple usable values use the earliest tail date.",
]

def read_item_links(path: str | Path) -> dict[int, str]:
    """Accept KGAT item_list.txt or TSV with raw ``item,freebase_id`` columns."""
    with open_raw_text(path) as handle:
        first = handle.readline()
        delimiter = "\t" if "\t" in first else None
        columns = first.split()
        # RecBole link files may use typed token headers rather than KGAT names.
        columns = [name.split(":", 1)[0] for name in columns]
        aliases = {"item_id": "item", "track_id": "item", "tracks_id": "item", "entity_id": "freebase_id"}
        columns = [aliases.get(name, name) for name in columns]
        raw_column = "org_id" if "org_id" in columns else "item"
        if raw_column not in columns or "freebase_id" not in columns:
            raise ValueError("Item links require raw item/org_id and freebase_id columns")
        index, fb_index = columns.index(raw_column), columns.index("freebase_id")
        links = {}
        for number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            row = line.rstrip("\r\n").split(delimiter)
            if len(row) != len(columns):
                raise ValueError(f"{path}:{number}: invalid item link row")
            raw, fb = int(row[index]), freebase_id(row[fb_index])
            if fb in {"", "NA"}:
                continue
            if raw < 0 or not fb.startswith(("m.", "g.")):
                raise ValueError(f"{path}:{number}: expected raw numeric item and Freebase MID")
            if raw in links and links[raw] != fb:
                raise ValueError(f"Conflicting Freebase links for raw track {raw}")
            links[raw] = fb
    return links


def make_facts(items: pd.DataFrame, graph: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    linked = items.dropna(subset=["freebase_id"])
    joined = linked.merge(graph, left_on="freebase_id", right_on="head", how="inner")
    facts = joined[[
        "h", "item", "relation", "tail", "tail_kind", "tail_label", "raw_object", "fact_source", "source_path",
    ]].copy()
    facts = facts.join(evidence.rename(columns={"tail_label": "evidence_label"}), on="tail")
    facts["tail_label"] = facts.tail_label.where(facts.tail_label.ne(""), facts.evidence_label.fillna(""))
    for column in DATE_COLUMNS:
        facts[column] = facts[column].fillna("")
    return annotate_lastfm(facts.drop_duplicates(["h", "relation", "tail"]).reset_index(drop=True))


def kgat_namespace(entities, triples, links, item_ids):
    """Use KGAT's shared item/entity namespace, merging tails that are also items."""
    n = len(item_ids)
    original, used, by_fb = [], set(), {}
    for raw in item_ids:
        fb = links.get(int(raw))
        org = fb if fb and fb not in used else f"lfm1b.track.{raw}"
        original.append(org)
        if fb:
            used.add(fb)
            by_fb.setdefault(fb, len(original) - 1)
    # Only tail ids are remapped. Avoid Python dictionaries for millions of item ids.
    extra, mapping = [], {}
    for old, key in zip(entities.entity.iloc[n:], entities.key.iloc[n:], strict=True):
        if key in by_fb:
            mapping[int(old)] = by_fb[key]
        else:
            mapping[int(old)] = n + len(extra)
            extra.append(int(old))
            original.append(str(key))
    keep = np.concatenate([np.arange(n, dtype="int64"), np.asarray(extra, dtype="int64")])
    exported_entities = entities.iloc[keep].copy().reset_index(drop=True)
    exported_entities["entity"] = np.arange(len(keep))
    exported_triples = {
        cut: frame.assign(t=frame.t.map(mapping).astype("int64")) for cut, frame in triples.items()
    }
    return exported_entities, exported_triples, original, mapping


def check_snapshot_containment(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    """Preserve the old builder's t1-subset-t2 guard in semantic fact ids."""
    keys = ["h", "relation", "tail"]
    if len(earlier.merge(later[keys], on=keys, validate="one_to_one")) != len(earlier):
        raise ValueError("KG at t1 is not contained in KG at t2")


def build_raw_dataset(
    pairs: str | Path,
    output: str | Path,
    export: str | Path,
    freebase: str | Path,
    *,
    item_links: str | Path,
    q_val: float = 0.8,
    q_test: float = 0.9,
    kcore: int = 10,
    overwrite: bool = False,
    kg_mode: str = "two-hop",
    kg_entity_min: int = 1,
    kg_relation_min: int = 1,
    engine: str = "pandas",
    memory_limit: str = "1GB",
    scratch_dir: str | Path | None = None,
    target_interactions: int | None = None,
    user_seed: int = 2019,
    compact_entities: bool = False,
    protocol_v2: bool = False,
) -> dict:
    started = time.perf_counter()
    output, export = Path(output), Path(export)
    if output.resolve() == export.resolve() or output.resolve() in export.resolve().parents:
        raise ValueError("Processed and export directories must be separate, non-nested directories")
    if export.resolve() in output.resolve().parents:
        raise ValueError("Processed and export directories must be separate, non-nested directories")
    sources = [pairs, str(pairs) + ".meta.json", freebase, item_links]
    for source in sources:
        if source is None:
            continue
        path = local_source_path(source)
        if path is not None and any(path.is_relative_to(folder.resolve()) for folder in (output, export)):
            raise ValueError(f"Output directory contains input source {source}; source files must be preserved")
    for folder in (output, export):
        if folder.exists() and any(folder.iterdir()) and not overwrite:
            raise FileExistsError(f"Output is not empty: {folder}; use another directory or --overwrite")
    if kg_mode not in {"one-hop", "two-hop"}:
        raise ValueError("Unknown KG mode")
    protocol = None
    if kg_mode == "two-hop":
        from tkgat.kg.freebase_dump import validate_extract

        extraction = validate_extract(freebase)
    if protocol_v2:
        from tkgat.data.protocol_v2 import KCORE, Q_TEST, Q_VAL, check_build_inputs

        if (q_val, q_test, kcore, kg_mode, kg_entity_min, kg_relation_min, target_interactions) != (
            Q_VAL, Q_TEST, KCORE, "two-hop", 1, 1, None,
        ):
            raise ValueError("0021 requires quantiles 0.8/0.9, train 10-core, two-hop, "
                             "no KG frequency pruning or sampling")
        protocol = check_build_inputs(pairs, item_links, extraction, read_item_links(item_links))
    if target_interactions is not None and (target_interactions < 1 or user_seed < 0):
        raise ValueError("target_interactions must be positive and user_seed nonnegative")
    kg_options = {"kg_mode": kg_mode, "kg_entity_min": kg_entity_min, "kg_relation_min": kg_relation_min,
                  "compact_entities": compact_entities or target_interactions is not None, "protocol": protocol,
                  "build_started": started}
    if engine == "auto":
        manifest = json.loads(Path(str(pairs) + ".meta.json").read_text(encoding="utf-8"))
        engine = "duckdb" if target_interactions is not None or manifest["pairs"] >= 2_000_000 else "pandas"
    config = SplitConfig(q_val=q_val, q_test=q_test, min_rating=1, kcore=kcore)
    if engine == "duckdb":
        from tkgat.data.lastfm_disk import disk_split

        with disk_split(pairs, config, scratch_dir or output.parent, memory_limit,
                        target_interactions=target_interactions, user_seed=user_seed) as disk:
            return _finish_raw_dataset(
                pairs, output, export, freebase, item_links, disk.split, disk.enc,
                disk.cut_meta, disk.audit, None, disk=disk, **kg_options,
            )
    if engine != "pandas":
        raise ValueError(f"Unknown processing engine: {engine}")
    if target_interactions is not None:
        raise ValueError("Train-only size sampling requires --engine auto or duckdb")
    interactions, cut_meta = load_raw_pairs(pairs)
    split = temporal_split(interactions, config)
    problems = check_split(split)
    if problems or split.train.empty:
        raise ValueError(f"Invalid temporal split: {problems or ['empty training set']}")
    audit, excluded_interactions = audit_raw_split(interactions, split)
    enc = encode(split)
    return _finish_raw_dataset(
        pairs, output, export, freebase, item_links, split, enc, cut_meta, audit, excluded_interactions, **kg_options,
    )


def _finish_raw_dataset(pairs, output, export, freebase, item_links, split, enc,
                        cut_meta, audit, excluded_interactions, *, disk=None, kg_mode="one-hop",
                        kg_entity_min=1, kg_relation_min=1, compact_entities=False, protocol=None, build_started=None):
    """Shared KG construction and provenance for both interaction-processing engines."""
    kg_started = time.perf_counter()
    build_started = kg_started if build_started is None else build_started
    if disk is not None:
        disk.log("Reading Drive mapping/Freebase and building the temporal KG")
    links_path = item_links
    links = read_item_links(links_path)
    items = pd.DataFrame({"h": np.arange(enc.n_items), "item": enc.item_ids})
    items["freebase_id"] = items.item.map(links).astype("object")
    primary = read_freebase_source(freebase, set(items.freebase_id.dropna()), all_heads=kg_mode == "two-hop")
    graph, evidence = primary.graph, primary.evidence
    bundle, extraction = None, None
    if kg_mode == "two-hop":
        from tkgat.kg.lastfm_twohop import build_twohop

        extraction = json.loads(Path(str(freebase) + ".meta.json").read_text(encoding="utf-8"))
        seed_mids = (set(items.freebase_id.dropna()) if extraction.get("seed_scope") == "protocol-v2 train items"
                     else set(links.values()))
        seed_hash = hashlib.sha256("\n".join(sorted(seed_mids)).encode()).hexdigest()
        if seed_hash != extraction["seed_mids_sha256"]:
            raise ValueError("Mapping MID set differs from the 01c extraction; rerun 01c with this mapping")
        if extraction.get("item_links_sha256") != source_sha256(links_path):
            raise ValueError("Mapping SHA-256 differs from 01c extraction; rerun 01c with this mapping")
        bundle = build_twohop(items, graph, evidence, {"t1": split.t1, "t2": split.t2},
                              entity_min=kg_entity_min, relation_min=kg_relation_min,
                              compact_entities=compact_entities)
        facts, kept, dropped = bundle.facts, bundle.kept, bundle.dropped
        entities, relations, triples, dates = bundle.entities, bundle.relations, bundle.triples, bundle.dates
        relation_names = list(PAPER_RELATIONS)
    else:
        facts = make_facts(items, graph, evidence)
        relation_names = list(PAPER_RELATIONS)
        kept, dropped = {}, {}
        for cut, ts in (("t1", split.t1), ("t2", split.t2)):
            kept[cut], dropped[cut] = cutoff_facts(facts, ts, "default")
            problems = check_kg(kept[cut], ts, "default", enc.n_items)
            if problems:
                raise ValueError(f"KG leakage at {cut}: {problems}")
        check_snapshot_containment(kept["t1"], kept["t2"])
        entities, relations, triples = assign_ids(enc.n_items, items, kept, relation_names, universe=facts)
        item_labels = items.freebase_id.map(evidence.tail_label).replace("", pd.NA)
        entities.loc[:enc.n_items - 1, "label"] = item_labels.fillna(entities.loc[:enc.n_items - 1, "label"])
        dates = {cut: triple_dates(frame, entities, relations) for cut, frame in kept.items()}
        for cut in dates:
            if not dates[cut][["h", "r", "t"]].equals(triples[cut]):
                raise ValueError(f"Fact date alignment failed at {cut}")
    timing_seconds = {"kg_construction": time.perf_counter() - kg_started}
    print(f"[02b] KG construction complete: elapsed={timing_seconds['kg_construction']:.2f}s", flush=True)
    source_info = {
        "pairs": str(pairs), "pairs_sha256": cut_meta["sha256"], "cut_manifest": cut_meta,
        "item_links": str(links_path) if links_path is not None else None,
        "freebase_dates": str(freebase), "kg_source": str(freebase),
        "freebase_source_report": primary.report,
        "uses_published_interactions": False, "uses_published_user_mapping": False,
        "uses_published_kg": False,
        "graph_scope": "two-hop" if bundle is not None else "item_attributes",
        "kg_construction": bundle.report if bundle is not None else {"method": "legacy outgoing one-hop"},
        "freebase_extraction": extraction,
        "protocol_v2": protocol,
        "freebase_sha256": source_sha256(freebase),
        "relation_schema": "KGAT Last-FM: fixed nine relations, original order",
        "selected_relations": relation_names,
        "item_links_lineage": "RecBole/RecSysDatasets; exact upstream path pending confirmation" if protocol else
        "supplied explicit raw-track-to-MID map; old remap ids ignored",
        "item_links_sha256": source_sha256(links_path),
    }
    if disk is not None and disk.sampling is not None:
        source_info["user_sampling"] = disk.sampling
    cfg = {"name": "lastfm", "processed_dir": str(output), "csv_lineterminator": "\n"}
    if disk is None:
        save_split(split, cfg, {"raw_lfm1b": source_info})
        excluded_interactions.to_csv(output / "dropped_interactions.csv", index=False, lineterminator="\n")
    else:
        source_info["execution"] = {"engine": "duckdb", "memory_limit": disk.db.execute(
            "SELECT current_setting('memory_limit')",
        ).fetchone()[0], "source_zip_saved_locally": False}
        disk.save(output, source_info)
        disk.log("Writing processed KG files and provenance")
    split_meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    write_json(output / "validation_report.json", audit)
    write_json(output / "freebase_source_report.json", primary.report)
    kg_dir = output / "kg" / "default"
    kg_dir.mkdir(parents=True, exist_ok=True)
    entities.to_csv(kg_dir / "entities.csv", index=False, lineterminator="\n")
    relations.to_csv(kg_dir / "relations.csv", index=False, lineterminator="\n")
    for cut in kept:
        triples[cut].to_csv(kg_dir / f"kg_{cut}.tsv", sep="\t", index=False, lineterminator="\n")
        dates[cut].ts = dates[cut].ts.astype("Int64")
        dates[cut].to_csv(kg_dir / f"kg_{cut}_dates.tsv", sep="\t", index=False, lineterminator="\n")
    pd.concat([frame.assign(cutoff=cut) for cut, frame in dropped.items()]).to_csv(
        kg_dir / "dropped.csv", index=False, lineterminator="\n",
    )
    def covered_items(frame):
        if bundle is None:
            return int(frame.h.nunique())
        endpoints = np.union1d(frame.h.dropna().to_numpy(dtype="int64"), frame.t.dropna().to_numpy(dtype="int64"))
        return int((endpoints < enc.n_items).sum())

    coverage = {
        "train_items": enc.n_items, "items_linked": int(items.freebase_id.notna().sum()),
        "items_without_links": int(items.freebase_id.isna().sum()),
        "items_with_facts_before_time_filter": covered_items(facts),
        "items_with_facts": {cut: covered_items(frame) for cut, frame in kept.items()},
        "triples": {cut: len(frame) for cut, frame in triples.items()},
        "dated_triples": {cut: int(frame.ts.notna().sum()) for cut, frame in dates.items()},
        "source_graph_triples": len(graph),
        "source_rows_excluded_head": primary.report["outside_linked_train_heads"],
        "source_rows_excluded_relation": primary.report["relation_not_selected"],
        "time_exclusions": {cut: len(frame) for cut, frame in dropped.items()},
        "fact_sources": facts.fact_source.value_counts().to_dict(),
    }
    coverage["item_mid_link_rate"] = coverage["items_linked"] / enc.n_items
    coverage["item_fact_rate"] = {cut: count / enc.n_items for cut, count in coverage["items_with_facts"].items()}
    coverage["evaluation_retention"] = {
        part: {
            "kept": audit["kept"][part],
            "dropped": sum(audit["dropped_by_reason"][part].values()),
            "kept_rate": audit["kept"][part] / max(
                1, audit["kept"][part] + sum(audit["dropped_by_reason"][part].values()),
            ),
        } for part in ("val", "test")
    }
    info = {
        **source_info, "coverage": coverage, "known_limitations": LIMITATIONS if bundle is None else [
            "Source snapshot date is archive metadata, not verified as KGAT's snapshot.",
            "The explicit track-to-MID mapping covers only part of LFM-1b; unlinked interactions are retained.",
            "Fact timestamps are endpoint-existence proxies, not statement creation/validity dates.",
            "Frequency pruning is disabled by default under 0021; optional thresholds use t1 only.",
            "Only the previously requested nine relations are traversed, with original RDF directions.",
        ],
        "kg_variant": "default", "time_consistent": True,
        "kg_snapshot": f"Freebase mirror metadata date {extraction['source_snapshot']}" if extraction
        else f"Freebase {FREEBASE_SNAPSHOT}",
        "date_semantics": bundle.report["date_semantics"] if bundle else
        "tail-existence proxies from the supplied Freebase TSV; undated stays null",
        "cutoffs": {"t1": split.t1, "t2": split.t2},
        "interaction_audit": audit,
        "timing_seconds": timing_seconds,
    }
    tails = {cut: set(frame["tail"]) for cut, frame in kept.items()}
    kg_info = {
        **info, "dataset": "lastfm", "run_name": "lastfm-raw", "variant": "default",
        "time_consistent": True, "wikidata_snapshot": None,
        "inputs": {"split_sha256": split_meta["sha256"], **source_info},
        "n_items": enc.n_items, "n_entities": len(entities), "n_relations": len(relations),
        "n_items_linked": coverage["items_linked"], "n_items_with_facts": coverage["items_with_facts"],
        "n_triples": coverage["triples"], "n_dated_triples": coverage["dated_triples"],
        "n_tail_entities_used": {cut: len(values) for cut, values in tails.items()},
        "n_entities_only_in_t2": len(tails["t2"] - tails["t1"]),
        "n_dropped": {cut: frame.reason.value_counts().to_dict() for cut, frame in dropped.items()},
        "date_kind_per_relation": relation_stats(facts).reset_index().to_dict(orient="records")
        if bundle is None else bundle.report,
        "entity_ids": "train items first; shared Freebase entities at both endpoints" if bundle else
        "processed KG: train items first, then separate Freebase tail entities",
    }
    write_json(kg_dir / "kg_meta.json", kg_info)
    # Preserve all raw tracks, including those without a link; never invent a MID.
    pd.DataFrame({"item": enc.item_ids, "freebase_id": items.freebase_id}).to_csv(
        output / "item_links.tsv", sep="\t", index=False, lineterminator="\n",
    )
    if bundle is None:
        exp_entities, exp_triples, original, tail_mapping = kgat_namespace(entities, triples, links, enc.item_ids)
    else:
        exp_entities, exp_triples, original, tail_mapping = entities, triples, bundle.original, None
    exporter = export_kgat if disk is None else disk.export
    meta = exporter(
        export, split, enc, exp_entities, relations, exp_triples["t1"], exp_triples["t2"], links,
        timestamps=True, info={**info, "protocol": "raw LFM-1b; temporal quantiles; train-only k-core"},
        item_namespace="lfm1b", relation_namespace="freebase",
    )
    with (export / "entity_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("org_id remap_id\n")
        for index, org in enumerate(original):
            handle.write(f"{org} {index}\n")
    with (export / "relation_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("org_id remap_id\n")
        for index, name in enumerate(relation_names):
            uri = name if name.startswith(("http://", "https://")) else f"http://rdf.freebase.com/ns/{name}"
            handle.write(f"{uri} {index}\n")
    entity_meta = pd.DataFrame({
        "org_id": original, "remap_id": exp_entities.entity,
        "kind": exp_entities.kind, "label": exp_entities.label.fillna(""),
    })
    entity_meta.to_csv(export / "entity_metadata.tsv", sep="\t", index=False, lineterminator="\n")
    for cut, label in (("t1", "final"), ("t2", "test")):
        dated = dates[cut].copy()
        if tail_mapping is not None:
            dated["t"] = dated.t.map(tail_mapping).astype("int64")
        dated["ts"] = dated.ts.astype("Int64")
        dated["date_source"] = np.where(
            dated.ts.notna(), "endpoint_existence_proxy" if bundle else "tail_existence_proxy", "undated",
        )
        dated.to_csv(export / f"kg_{label}_timestamp.tsv", sep="\t", index=False, lineterminator="\n")
    provenance = []
    tail_ids = entities[entities.kind.ne("item")].set_index("key").entity.to_dict() if bundle is None else {}
    relation_ids = relations.set_index("name").relation.to_dict()
    for cut in ("t1", "t2"):
        frame = kept[cut][["h", "relation", "tail", "fact_source", "source_path", "raw_object"]].copy()
        frame["r"] = frame.relation.map(relation_ids).astype("int64")
        frame["t"] = frame["tail"].map(tail_ids).map(tail_mapping).astype("int64") if bundle is None else kept[cut].t
        provenance.append(frame[["h", "r", "t", "fact_source", "source_path", "raw_object"]].assign(cutoff=cut))
    pd.concat(provenance, ignore_index=True).sort_values(["cutoff", "h", "r", "t"]).to_csv(
        export / "fact_sources.tsv", sep="\t", index=False, lineterminator="\n",
    )
    # Export the evidence too: each date can be audited against the original literals.
    evidence_out = evidence.rename_axis("freebase_id").reset_index()
    evidence_out.to_csv(export / "fact_date_evidence.tsv", sep="\t", index=False, lineterminator="\n")
    unlinked = items[items.freebase_id.isna()][["item"]]
    unlinked.to_csv(export / "unlinked_items.tsv", sep="\t", index=False, lineterminator="\n")
    if disk is not None:
        disk.log("Hashing generated exports")
    meta["sha256"] = {
        path.name: sha256_file(path) for path in sorted(export.iterdir())
        if path.is_file() and path.name != "dataset_meta.json"
    }
    meta["built_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    timing_seconds["total_before_metadata_finalization"] = time.perf_counter() - build_started
    write_json(kg_dir / "kg_meta.json", kg_info)
    # Persist hashes of every processed intermediate, including both KG snapshots.
    processed_meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    processed_meta["artifact_sha256"] = {
        path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "meta.json"
    }
    write_json(output / "meta.json", processed_meta)
    write_json(export / "dataset_meta.json", meta)
    print(f"[02b] Build complete: elapsed={timing_seconds['total_before_metadata_finalization']:.2f}s", flush=True)
    return meta
