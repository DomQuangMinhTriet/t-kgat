"""Train-seeded, bidirectional two-hop Freebase KG with cutoff-safe quality selection."""

from __future__ import annotations

import datetime as dt
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tkgat.kg.lastfm import DATE_COLUMNS, DATE_PREDICATES, parse_freebase_date
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS
from tkgat.kg.temporal import cutoff_day


def graph_stats(heads, tails):
    """Count triple incidences at each endpoint; self-loops contribute twice."""
    degrees = Counter(heads)
    degrees.update(tails)
    return {"triples": len(heads), "entities": len(degrees), "max_degree": max(degrees.values(), default=0)}


def reachable(graph, seeds, *, hop_stats=None):
    """Keep original edges within two undirected traversal steps; do not invent inverses."""
    frontier, keep = set(seeds), pd.Series(False, index=graph.index)
    for hop in (1, 2):
        incident = graph["head"].isin(frontier) | graph["tail"].isin(frontier)
        keep |= incident
        frontier.update(graph.loc[incident, "head"])
        frontier.update(graph.loc[incident, "tail"])
        if hop_stats is not None:
            stats = graph_stats(graph.loc[keep, "head"], graph.loc[keep, "tail"])
            hop_stats.append({"hop": hop, **stats})
            print(f"[02b] Train-seeded hop {hop}, before time filtering: "
                  f"triples={stats['triples']:,} entities={stats['entities']:,} "
                  f"max_degree={stats['max_degree']:,}", flush=True)
    return graph.loc[keep].copy()


def endpoint_dates(evidence):
    result = {}
    for key, row in evidence.iterrows():
        dates = [parse_freebase_date(row[column]) for column in DATE_COLUMNS if row[column]]
        result[key] = min((day for day in dates if day is not None), default=None)
    return result


@dataclass
class TwoHopKG:
    facts: pd.DataFrame
    kept: dict
    dropped: dict
    entities: pd.DataFrame
    relations: pd.DataFrame
    triples: dict
    dates: dict
    original: list
    report: dict


def build_twohop(items, graph, evidence, cutoffs, *, entity_min=1, relation_min=1, compact_entities=False):
    """Use the fixed nine-relation schema, without unapproved frequency pruning.

    Optional frequency eligibility is learned once from t1 only. Both endpoint dates matter because an
    incoming edge can have a future head. Date-less edges are retained. The cached
    source may include other seeds; only surviving train-item MIDs seed this KG.
    """
    if entity_min < 1 or relation_min < 1:
        raise ValueError("KG frequency thresholds must be positive")
    started = time.perf_counter()
    hop_stats = []
    seeds = set(items.freebase_id.dropna())
    resource_graph = graph[graph.tail_kind.eq("freebase")].copy()
    base = reachable(resource_graph, seeds, hop_stats=hop_stats)
    base = base.sort_values(["head", "relation", "tail"]).reset_index(drop=True)
    base["source_index"] = np.arange(len(base))
    dates = endpoint_dates(evidence)
    base["head_date"] = base["head"].map(dates)
    base["tail_date"] = base["tail"].map(dates)
    base["edge_date"] = [max((d for d in (a, b) if pd.notna(d)), default=None)
                         for a, b in zip(base.head_date, base.tail_date, strict=True)]

    def temporal(cut):
        day = cutoff_day(cutoffs[cut])
        return base.loc[base.edge_date.map(lambda value: pd.isna(value) or value < day).to_numpy(bool)]

    train_candidates = reachable(temporal("t1"), seeds)
    counts = Counter(train_candidates["head"])
    counts.update(train_candidates["tail"])
    # Threshold 1 means no frequency filter, including entities/relations first
    # available at t2. Old 10/50 pruning is not a rule of decision 0021.
    good_entities = (set(base["head"]) | set(base["tail"])) if entity_min == 1 else {
        key for key, count in counts.items() if count >= entity_min
    }
    entity_filtered = train_candidates[
        train_candidates["head"].isin(good_entities) & train_candidates["tail"].isin(good_entities)
    ]
    relation_counts = entity_filtered.relation.value_counts().to_dict()
    good_relations = set(PAPER_RELATIONS) if relation_min == 1 else {
        key for key, count in relation_counts.items() if count >= relation_min
    }
    selected, reasons = {}, {}
    for cut in ("t1", "t2"):
        available = reachable(temporal(cut), seeds)
        quality = available[available["head"].isin(good_entities) & available["tail"].isin(good_entities)
                            & available.relation.isin(good_relations)]
        selected[cut] = set(reachable(quality, seeds).source_index)
        day = cutoff_day(cutoffs[cut])
        reasons[cut] = pd.Series(np.select([
            base.head_date.map(lambda d, day=day: pd.notna(d) and d >= day).to_numpy(bool),
            base.tail_date.map(lambda d, day=day: pd.notna(d) and d >= day).to_numpy(bool),
            ~base.source_index.isin(available.source_index),
            ~(base["head"].isin(good_entities) & base["tail"].isin(good_entities)),
            ~base.relation.isin(good_relations),
            ~base.source_index.isin(selected[cut]),
        ], ["future_head", "future_tail", "outside_cutoff_neighborhood", "below_train_entity_frequency",
            "below_train_relation_frequency", "disconnected_after_quality"], default=""), index=base.index)
    if not selected["t1"] <= selected["t2"]:
        raise ValueError("Two-hop KG at t1 is not contained in t2")

    # Shared item/entity ids; aliases preserve distinct raw tracks with the same MID.
    by_mid, original, used = {}, [], set()
    for row in items.itertuples(index=False):
        mid = row.freebase_id if pd.notna(row.freebase_id) else None
        original.append(mid if mid and mid not in used else f"lfm1b.track.{row.item}")
        if mid:
            used.add(mid)
            by_mid.setdefault(mid, []).append(int(row.h))
    namespace = base[base.source_index.isin(selected["t2"])] if compact_entities else base
    nodes = sorted((set(namespace["head"]) | set(namespace["tail"])) - set(by_mid))
    n_items = len(items)
    node_ids = {node: [n_items + i] for i, node in enumerate(nodes)}
    node_ids.update(by_mid)
    original.extend(nodes)
    labels = evidence.tail_label.to_dict()
    entities = pd.concat([
        pd.DataFrame({"entity": np.arange(n_items), "kind": "item", "key": items.item.astype(str),
                      "label": [labels.get(mid, "") or (mid if pd.notna(mid) else "")
                                for mid in items.freebase_id]}),
        pd.DataFrame({"entity": np.arange(n_items, n_items + len(nodes)), "kind": "freebase",
                      "key": nodes, "label": [labels.get(node, "") for node in nodes]}),
    ], ignore_index=True)
    relations = pd.DataFrame({"relation": np.arange(len(PAPER_RELATIONS)), "name": PAPER_RELATIONS})
    relation_ids = relations.set_index("name").relation.to_dict()
    facts = base.assign(h=base["head"].map(node_ids), t=base["tail"].map(node_ids)).explode("h").explode("t")
    id_dtype = "Int64" if compact_entities else "int64"
    facts["h"], facts["t"] = facts.h.astype(id_dtype), facts.t.astype(id_dtype)
    facts["r"] = facts.relation.map(relation_ids).astype("int64")
    facts["ts"] = [int(dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc).timestamp())
                   if pd.notna(day) else None for day in facts.edge_date]
    facts["ts"] = facts.ts.astype("Int64")
    facts = facts.sort_values(["h", "r", "t"]).reset_index(drop=True)
    kept, dropped, triples, dated, cutoff_stats = {}, {}, {}, {}, {}
    for cut in ("t1", "t2"):
        reason = facts.source_index.map(reasons[cut])
        kept[cut] = facts[reason.eq("")].copy().reset_index(drop=True)
        kept[cut][["h", "t"]] = kept[cut][["h", "t"]].astype("int64")
        dropped[cut] = facts[reason.ne("")].assign(reason=reason[reason.ne("")]).reset_index(drop=True)
        triples[cut] = kept[cut][["h", "r", "t"]].copy()
        dated[cut] = kept[cut][["h", "r", "t", "ts"]].copy()
        # Independent evidence reparse, including the head of incoming facts.
        checked_dates = endpoint_dates(evidence)
        for row in kept[cut].itertuples(index=False):
            known = [checked_dates.get(node) for node in (row.head, row.tail)]
            day = max((value for value in known if value is not None), default=None)
            expected = None if day is None else int(dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp())
            if day is not None and day >= cutoff_day(cutoffs[cut]):
                raise ValueError(f"Future endpoint in two-hop KG at {cut}")
            if (expected is None) != pd.isna(row.ts) or (expected is not None and row.ts != expected):
                raise ValueError("Two-hop fact timestamp does not match source evidence")
        if triples[cut].duplicated().any():
            raise ValueError("Duplicate two-hop triples")
        cutoff_stats[cut] = graph_stats(triples[cut].h, triples[cut].t)
        stats = cutoff_stats[cut]
        print(f"[02b] KG at {cut}, retained entity IDs: triples={stats['triples']:,} "
              f"entities={stats['entities']:,} max_degree={stats['max_degree']:,}", flush=True)
    report = {
        "method": "two-hop, either endpoint; original RDF direction", "graph_scope": "two-hop",
        "source_resource_edges": len(resource_graph), "train_seed_neighborhood_edges": len(base),
        "outside_train_seed_neighborhood": len(resource_graph) - len(base),
        "literal_edges_excluded": len(graph) - len(resource_graph),
        "seed_mids": len(seeds), "entity_min": entity_min, "relation_min": relation_min,
        "frequency_reference": "t1 only, entity pass then relation pass, no iterative quality pruning",
        "relation_counts_after_entity_filter_t1": {key: relation_counts.get(key, 0) for key in PAPER_RELATIONS},
        "source_edge_exclusions": {cut: reasons[cut][reasons[cut].ne("")].value_counts().to_dict() for cut in reasons},
        "hop_stats": hop_stats,
        "hop_stats_scope": "cumulative train-seeded RDF neighborhood before time filtering and ID expansion",
        "cutoff_graph_stats": cutoff_stats,
        "degree_definition": "in_degree + out_degree over distinct triples; self-loops count twice",
        "elapsed_seconds": time.perf_counter() - started,
        "date_semantics": {
            "endpoint_date": "earliest_valid_release_active_start_or_birth",
            "date_predicates": dict(zip(DATE_COLUMNS, DATE_PREDICATES, strict=True)),
            "fact_date": "max_available_endpoint_dates",
            "one_endpoint_dated": "use_available_date",
            "both_endpoints_undated": "keep_fact_with_null_timestamp",
            "unparseable_dates": "treated_as_undated",
            "date_precision": {
                "year": "last_day_of_year", "month": "last_day_of_month",
                "full_date_ending_01_01": "treated_as_year_only_last_day_of_year",
            },
            "timestamp_encoding": "UTC_midnight_Unix_seconds",
            "cutoff": "fact_date_strictly_before_cutoff_UTC_calendar_day",
            "relation_scope": "same_endpoint_rule_for_all_nine_relations",
            "interpretation": "endpoint_existence_proxy_not_statement_creation_or_validity_date",
        },
    }
    if compact_entities:
        report["entity_namespace"] = {
            "method": "all train items plus endpoints used by the retained t1/t2 KG",
            "removed_unused_entities": len((set(base["head"]) | set(base["tail"])) - set(by_mid)) - len(nodes),
            "dropped_fact_ids": "blank for excluded endpoints; original head/tail MIDs remain auditable",
        }
    return TwoHopKG(facts, kept, dropped, entities, relations, triples, dated, original, report)
