"""Independent pre-upload checks for the protocol-v2 data artifacts (no scores)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.data.lastfm_disk import disk_split, literal
from tkgat.data.protocol_v2 import KCORE, Q_TEST, Q_VAL, check_build_inputs
from tkgat.data.split import SplitConfig
from tkgat.kg.freebase_dump import validate_extract
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS
from tkgat.pipeline.lastfm_raw import read_item_links
from tkgat.registry import sha256_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hashes(root, hashes):
    root = Path(root).resolve()
    if not hashes:
        raise ValueError(f"Missing artifact hashes: {root}")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch or unsafe path: {name}")


def check_grouped(path, expected, *, label=None):
    """Compare a grouped KGAT interaction file with independently rebuilt rows."""
    expected = iter(expected)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) < 2:
                raise ValueError(f"Invalid grouped interaction line: {path}")
            user = int(values[0])
            for value in values[1:]:
                row = next(expected, None)
                if row is None:
                    raise ValueError(f"Extra exported interactions: {path}")
                uid, item, ts = (int(v) for v in row)
                token = str(item) if label is None else f"{item}:{ts if label == 'ts' else 1}"
                if (user, value) != (uid, token):
                    raise ValueError(f"Exported interactions differ from the source: {path}")
    if next(expected, None) is not None:
        raise ValueError(f"Missing exported interactions: {path}")


def check_clean_dataset(processed, export, *, scratch_dir=None, memory_limit="1GB"):
    """Recompute source quantiles/core/masks on disk and compare every processed row.

    Existing pipeline audit output is evidence, not a substitute for this check.
    This reads test *data* to validate construction, never a model or test score.
    """
    processed, export = Path(processed), Path(export)
    meta = read_json(processed / "meta.json")
    exported = read_json(export / "dataset_meta.json")
    source = meta["clean"]["raw_lfm1b"]
    pairs, mapping, freebase = source["pairs"], source["item_links"], source["kg_source"]
    extraction = validate_extract(freebase)
    links = read_item_links(mapping)
    protocol = check_build_inputs(pairs, mapping, extraction, links)
    if source.get("protocol_v2") != protocol or exported.get("protocol_v2") != protocol:
        raise ValueError("Missing or inconsistent protocol-v2 provenance")
    config = SplitConfig(Q_VAL, Q_TEST, 1, KCORE)
    if meta["config"] != vars(config):
        raise ValueError("Processed split parameters differ from 0021")
    check_hashes(processed, meta["artifact_sha256"])
    check_hashes(export, exported["sha256"])
    if sha256_file(freebase) != source["freebase_sha256"]:
        raise ValueError("Source extraction changed since the build")
    checked_rows = {}
    with disk_split(pairs, config, scratch_dir or processed.parent, memory_limit) as disk:
        n_items, n_users = disk.enc.n_items, disk.enc.n_users
        linked_rate = sum(int(raw) in links for raw in disk.enc.item_ids) / n_items
        if (exported["n_items"], exported["n_users"]) != (n_items, n_users):
            raise ValueError("Exported vocabulary sizes differ from train")
        item_lines = (f"{raw} {index} {links.get(int(raw), 'NA')}\n"
                      for index, raw in enumerate(disk.enc.item_ids))
        with (export / "item_list.txt").open(encoding="utf-8") as handle:
            if handle.readline() != "org_id remap_id freebase_id\n":
                raise ValueError("Invalid exported item mapping header")
            if any(actual != expected for actual, expected in zip(handle, item_lines, strict=True)):
                raise ValueError("Exported item mapping differs from train and the source mapping")
        if (disk.t1, disk.t2) != (meta["t1"], meta["t2"]):
            raise ValueError("Saved cutoffs differ from source quantiles")
        for part in ("train", "val", "test", "seen_before_val", "seen_before_test"):
            path = literal(processed / f"{part}.csv")
            disk.db.execute(f'CREATE OR REPLACE VIEW saved AS SELECT "user" AS u,item AS i,ts '
                            f"FROM read_csv_auto({path})")
            changed = disk.one(f"SELECT count(*) FROM ((SELECT u,i,ts FROM saved EXCEPT ALL "
                               f"SELECT u,i,ts FROM {part}) UNION ALL "
                               f"(SELECT u,i,ts FROM {part} EXCEPT ALL SELECT u,i,ts FROM saved))")
            if changed:
                raise ValueError(f"{part} differs from the raw-source temporal split/core/masks")
            if part in ("train", "val", "test"):
                if disk.one(f"SELECT count(*) FROM read_csv_auto({path}) WHERE rating <> 1 OR rating IS NULL"):
                    raise ValueError(f"{part} contains non-implicit ratings")
            checked_rows[part] = disk.count(part)
            query = f"SELECT uid,iid,ts FROM {part} JOIN users USING(u) JOIN items USING(i) ORDER BY uid,ts,iid"
            if part in ("train", "val", "test"):
                for suffix, label in (("", None), ("_timestamp", "ts"), ("_rating", "rating")):
                    check_grouped(export / f"{part}{suffix}.txt", disk.rows(query), label=label)
            else:
                name = "val_mask" if part == "seen_before_val" else "test_mask"
                check_grouped(export / f"{name}.txt", disk.rows(query))
    kg_dir = processed / "kg" / "default"
    relations = pd.read_csv(kg_dir / "relations.csv")
    if tuple(relations.name) != PAPER_RELATIONS:
        raise ValueError("Expected all nine relation IDs, including empty relations")
    entities = pd.read_csv(kg_dir / "entities.csv")
    if not np.array_equal(entities.entity, np.arange(len(entities))):
        raise ValueError("Non-contiguous entity IDs")
    graphs = {}
    coverage = {"item_mid_link_rate": linked_rate, "item_fact_rate": {}}
    for cut, label in (("t1", "final"), ("t2", "test")):
        graph = pd.read_csv(kg_dir / f"kg_{cut}.tsv", sep="\t", dtype="int64")
        ids_dtype = {"h": "int64", "r": "int64", "t": "int64"}
        dates = pd.read_csv(kg_dir / f"kg_{cut}_dates.tsv", sep="\t", dtype=ids_dtype)
        if not graph.equals(dates[["h", "r", "t"]]) or graph.duplicated().any():
            raise ValueError(f"Duplicate or misaligned KG at {cut}")
        if len(graph) and (
            graph[["h", "t"]].min().min() < 0 or graph[["h", "t"]].max().max() >= len(entities)
            or graph.r.min() < 0 or graph.r.max() >= len(PAPER_RELATIONS)
        ):
            raise ValueError(f"KG IDs outside vocabulary at {cut}")
        if (dates.ts.dropna() >= meta[cut]).any():
            raise ValueError(f"Future KG dates at {cut}")
        exported_graph = export / f"kg_{label}.txt"
        text = exported_graph.read_text(encoding="utf-8")
        expected = "".join(f"{h} {r} {t}\n" for h, r, t in graph.itertuples(index=False, name=None))
        if text != expected:
            raise ValueError(f"KG export differs from processed KG at {cut}")
        exported_dates = pd.read_csv(export / f"kg_{label}_timestamp.tsv", sep="\t", dtype=ids_dtype)
        if not dates[["h", "r", "t", "ts"]].equals(exported_dates[["h", "r", "t", "ts"]]):
            raise ValueError(f"KG date export differs at {cut}")
        graphs[cut] = pd.MultiIndex.from_frame(graph)
        endpoints = np.union1d(graph.h, graph.t)
        coverage["item_fact_rate"][cut] = int((endpoints < exported["n_items"]).sum()) / exported["n_items"]
    if len(graphs["t1"].difference(graphs["t2"])):
        raise ValueError("KG at t1 is not contained in t2")
    if coverage != {key: exported["coverage"][key] for key in coverage}:
        raise ValueError("KG coverage report does not match artifacts")
    return {
        "passed": True, "protocol": protocol, "checked_rows": checked_rows, "coverage": coverage,
        "processed_meta_sha256": sha256_file(processed / "meta.json"),
        "export_meta_sha256": sha256_file(export / "dataset_meta.json"),
        "checks": ["source_hashes", "artifact_hashes", "global_quantiles", "train_only_10core",
                   "known_users_items", "complete_masks", "implicit_feedback", "kg_ids_and_dates",
                   "nine_relations", "kg_export_alignment", "kg_t1_subset_t2", "two_coverage_rates"],
    }
