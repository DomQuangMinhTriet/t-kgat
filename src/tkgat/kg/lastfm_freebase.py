"""Read facts and date evidence from Freebase TSV sources, never from a prebuilt paper KG."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tkgat.data.lastfm_raw import open_raw_text
from tkgat.kg.lastfm import DATE_COLUMNS, DATE_PREDICATES, literal_value, parse_freebase_date

# KGAT Last-FM's schema and original relation ids; no paper triples are inputs.
PAPER_RELATIONS = (
    "common.topic.notable_types",
    "music.recording.releases",
    "music.recording.artist",
    "music.recording.engineer",
    "music.recording.producer",
    "music.recording.canonical_version",
    "music.recording.song",
    "music.single.versions",
    "music.recording.featured_artists",
)


def freebase_id(value: str) -> str:
    value = value.strip().removeprefix("<").removesuffix(">")
    value = value.removeprefix("http://rdf.freebase.com/ns/")
    if value.startswith(("/m/", "/g/")):
        value = value[1:].replace("/", ".")
    return value


@dataclass
class FreebaseSource:
    graph: pd.DataFrame
    evidence: pd.DataFrame
    report: dict


def read_freebase_source(source: str | Path, heads: set[str], *, all_heads: bool = False) -> FreebaseSource:
    """Read Freebase facts/evidence, restricted to KGAT's nine relations.

    Legacy mode keeps outgoing linked-item edges. With ``all_heads``, preserve
    extracted resource heads so the two-hop builder can apply cutoff-aware
    traversal from train seeds later.

    The paper supplies only the relation schema, never fallback facts. Predicates
    must match exactly; missing relations are reported rather than reconstructed
    from other predicates. Resource objects remain resources; literal objects
    receive stable nodes based on the exact RDF term.
    Evidence for resource tails is collected throughout the file, including rows
    whose head is not a recommendation item. A literal's contents are not assumed
    to be the fact's validity date.
    """
    heads = {freebase_id(head) for head in heads}
    selected = set(PAPER_RELATIONS)
    wanted = dict(zip(DATE_PREDICATES, DATE_COLUMNS, strict=True))
    evidence_rows, facts = {}, {}
    report = {
        "source": str(source), "rows_scanned": 0, "outside_linked_train_heads": 0,
        "resource_fact_rows": 0, "literal_fact_rows": 0, "duplicate_fact_rows": 0,
        "relation_not_selected": 0,
        "selected_relation_source_rows": dict.fromkeys(PAPER_RELATIONS, 0),
    }
    with open_raw_text(source) as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader, None) != ["subject", "predicate", "object"]:
            raise ValueError("Freebase source must have the TSV header subject,predicate,object")
        for number, row in enumerate(reader, 2):
            if len(row) != 3 or not all(value.strip() for value in row):
                raise ValueError(f"{source}:{number}: expected three nonempty tab-separated fields")
            subject, predicate, raw_object = row
            subject, predicate = freebase_id(subject), freebase_id(predicate)
            report["rows_scanned"] += 1
            if predicate in selected:
                report["selected_relation_source_rows"][predicate] += 1
            column = wanted.get(predicate)
            if column is not None or predicate == "type.object.name":
                evidence = evidence_rows.setdefault(subject, {})
                if column is None:
                    label = literal_value(raw_object)
                    evidence["tail_label"] = min(evidence.get("tail_label", label), label)
                else:
                    parsed = parse_freebase_date(raw_object)
                    old = evidence.get(column)
                    old_date = parse_freebase_date(old) if old else None
                    earlier = parsed is not None and (old_date is None or (parsed, raw_object) < (old_date, old))
                    if old is None or earlier:
                        evidence[column] = raw_object
                    elif parsed is None and old_date is None:
                        evidence[column] = min(old, raw_object)
            if subject not in heads and not all_heads:
                report["outside_linked_train_heads"] += 1
                continue
            if predicate not in selected:
                report["relation_not_selected"] += 1
                continue
            is_literal = raw_object.lstrip().startswith('"')
            if is_literal:
                tail = "literal.sha256." + hashlib.sha256(raw_object.encode("utf-8")).hexdigest()
                kind, label = "freebase_literal", literal_value(raw_object)
                report["literal_fact_rows"] += 1
            else:
                tail, kind, label = freebase_id(raw_object), "freebase", ""
                report["resource_fact_rows"] += 1
            key = (subject, predicate, tail)
            if key in facts:
                report["duplicate_fact_rows"] += 1
                facts[key]["raw_object"] = min(facts[key]["raw_object"], raw_object)
            else:
                facts[key] = {
                    "head": subject, "relation": predicate, "tail": tail,
                    "tail_kind": kind, "tail_label": label, "raw_object": raw_object,
                    "fact_source": "freebase", "source_path": str(source),
                }
    graph = pd.DataFrame(facts.values(), columns=[
        "head", "relation", "tail", "tail_kind", "tail_label", "raw_object", "fact_source", "source_path",
    ]).sort_values(["head", "relation", "tail"]).reset_index(drop=True)
    evidence = pd.DataFrame.from_dict(evidence_rows, orient="index").reindex(columns=[*DATE_COLUMNS, "tail_label"])
    evidence.index = evidence.index.astype("object")
    evidence = evidence.astype("object").fillna("").sort_index()
    report["unique_facts"] = len(graph)
    report["entity_heads_with_facts"] = int(graph["head"].nunique())
    report["relations"] = sorted(graph.relation.unique())
    counts = graph.relation.value_counts().to_dict()
    report["selected_relation_counts"] = {name: counts.get(name, 0) for name in PAPER_RELATIONS}
    report["relations_without_linked_train_facts"] = [name for name in PAPER_RELATIONS if name not in counts]
    if report["rows_scanned"] != (
        report["outside_linked_train_heads"] + report["relation_not_selected"]
        + len(graph) + report["duplicate_fact_rows"]
    ):
        raise RuntimeError("Freebase row accounting does not balance")
    return FreebaseSource(graph, evidence, report)
