"""CKG: ids, inverse edges, the train-only interaction edges, the test-time KG rule (Q2) and the KG/split guards."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.ckg import (
    INTERACT,
    KGTriples,
    build_ckg,
    inverse_relation_id,
    load_kg,
    no_kg,
    used_tails,
)
from tkgat.pipeline.build import load_saved, save_split

from .conftest import ROOT

RELATIONS = ["genre", "director", "award"]  # "award" has no triple at t1


@pytest.fixture
def world(synthetic, tmp_path):
    """A saved synthetic split, its encoding and a KG written the way scripts/06_build_kg.py writes it."""
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    cfg = {"name": "synthetic", "processed_dir": str(tmp_path / "processed")}
    save_split(split, cfg, {})
    split, meta = load_saved(cfg)
    enc = encode(split)
    n = enc.n_items
    assert n >= 6
    t1 = pd.DataFrame({"h": [0, 1, 2, 3], "r": [0, 1, 0, 1], "t": [n, n + 1, n + 1, n]})
    # t2 adds a fact of the relation unused at t1 on a known tail, and a fact on a tail that t1 never uses
    t2 = pd.concat([t1, pd.DataFrame({"h": [4, 5], "r": [2, 0], "t": [n, n + 2]})], ignore_index=True)
    kg_dir = tmp_path / "processed" / "kg" / "default"
    kg_dir.mkdir(parents=True)
    t1.to_csv(kg_dir / "kg_t1.tsv", sep="\t", index=False)
    t2.to_csv(kg_dir / "kg_t2.tsv", sep="\t", index=False)
    entities = pd.DataFrame(
        {
            "entity": range(n + 3),
            "kind": ["item"] * n + ["wikidata"] * 3,
            "key": [*[str(i) for i in enc.item_ids], "Q1", "Q2", "Q3"],
            "label": [""] * (n + 3),
        }
    )
    entities.to_csv(kg_dir / "entities.csv", index=False)
    pd.DataFrame({"relation": range(3), "name": RELATIONS}).to_csv(kg_dir / "relations.csv", index=False)
    info = {
        "variant": "default",
        "time_consistent": True,
        "wikidata_snapshot": "2026-10-17",
        "inputs": {"split_sha256": meta["sha256"]},
        "n_items": n,
        "n_entities": n + 3,
        "n_relations": 3,
        "n_triples": {"t1": len(t1), "t2": len(t2)},
    }
    (kg_dir / "kg_meta.json").write_text(json.dumps(info), encoding="utf-8")
    return SimpleNamespace(split=split, enc=enc, meta=meta, dir=tmp_path / "processed", kg_dir=kg_dir, info=info)


def kg_at(w, cutoff):
    return load_kg(w.dir, "default", cutoff, w.enc, w.meta)


def triple_set(ckg):
    return set(zip(ckg.head.tolist(), ckg.relation.tolist(), ckg.tail.tolist(), strict=True))


def test_ids_and_edge_count(world):
    w = world
    kg1 = kg_at(w, "t1")
    ckg = build_ckg(w.enc, kg1)
    p, t = len(w.enc.train), len(kg1)
    assert (ckg.n_users, ckg.n_items, ckg.n_entities) == (w.enc.n_users, w.enc.n_items, w.enc.n_items + 3)
    assert ckg.n_relations == 3 and ckg.n_rel == 8 and ckg.n_nodes == w.enc.n_users + w.enc.n_items + 3
    assert ckg.n_edges == 2 * (p + t) == len(triple_set(ckg))  # no duplicate edge
    assert ckg.head.min() >= 0 and ckg.head.max() < ckg.n_nodes and ckg.tail.max() < ckg.n_nodes
    assert ckg.relation.min() == INTERACT and ckg.relation.max() < ckg.n_rel
    assert not (ckg.head == ckg.tail).any()


def test_every_edge_has_its_inverse(world):
    ckg = build_ckg(world.enc, kg_at(world, "t1"))
    r = ckg.n_relations
    edges = triple_set(ckg)
    for h, rel, t in edges:
        inv = inverse_relation_id(rel, r) if rel <= r else rel - r - 1
        assert (t, inv, h) in edges


def test_relation_ids_follow_the_tf1_layout(world):
    w = world
    kg1 = kg_at(w, "t1")
    ckg = build_ckg(w.enc, kg1)
    n_users, r = ckg.n_users, ckg.n_relations
    edges = triple_set(ckg)
    h, rel, t = kg1.triples[0]  # (0, 0, n)
    assert (n_users + h, rel + 1, n_users + t) in edges
    assert (n_users + t, rel + r + 2, n_users + h) in edges
    u, i = w.enc.train[0]
    assert (u, INTERACT, n_users + i) in edges and (n_users + i, r + 1, u) in edges


def test_interaction_edges_are_exactly_the_train_pairs(world):
    w = world
    ckg = build_ckg(w.enc, kg_at(w, "t1"))
    users = ckg.head < ckg.n_users
    assert (ckg.relation[users] == INTERACT).all()
    got = set(zip(ckg.head[users].tolist(), (ckg.tail[users] - ckg.n_users).tolist(), strict=True))
    train = set(map(tuple, w.enc.train.tolist()))
    assert got == train and len(got) == ckg.n_interactions
    for part in (w.enc.val, w.enc.test):
        later = {(u, int(i)) for u, items in part.items() for i in items}
        assert later and not (got & later)  # no val/test pair is an edge


def test_edge_timestamps_come_from_train_and_kg_edges_have_none(world):
    w = world
    ckg = build_ckg(w.enc, kg_at(w, "t1"))
    ts = {tuple(p): float(t) for p, t in zip(w.enc.train.tolist(), w.enc.train_ts, strict=True)}
    interact = (ckg.relation == INTERACT) | (ckg.relation == ckg.n_relations + 1)
    assert not np.isnan(ckg.edge_ts[interact]).any() and np.isnan(ckg.edge_ts[~interact]).all()
    fwd = ckg.relation == INTERACT
    for h, t, stamp in zip(ckg.head[fwd], ckg.tail[fwd], ckg.edge_ts[fwd], strict=True):
        assert ts[(int(h), int(t) - ckg.n_users)] == stamp
    assert ckg.edge_ts.dtype == np.float64


def test_edges_are_sorted_by_head_tail_relation(world):
    ckg = build_ckg(world.enc, kg_at(world, "t1"))
    key = np.stack([ckg.head, ckg.tail, ckg.relation], axis=1)
    assert (key == key[np.lexsort((key[:, 2], key[:, 1], key[:, 0]))]).all()


def test_relation_count_does_not_shrink_when_a_relation_has_no_triple(world):
    ckg = build_ckg(world.enc, kg_at(world, "t1"))
    unused = 2  # "award": no triple at t1
    assert ckg.n_relations == 3 and ckg.n_rel == 8
    assert not np.isin(ckg.relation, [unused + 1, unused + ckg.n_relations + 2]).any()


def test_test_graph_drops_triples_with_a_tail_unseen_at_t1(world):
    w = world
    kg1, kg2 = kg_at(w, "t1"), kg_at(w, "t2")
    n = w.enc.n_items
    known = used_tails(kg1)
    assert list(known) == [n, n + 1]
    ckg1 = build_ckg(w.enc, kg1)
    ckg2 = build_ckg(w.enc, kg2, known_tails=known)
    assert (ckg1.n_triples, ckg1.n_triples_dropped_unseen_tail) == (4, 0)
    assert (ckg2.n_triples, ckg2.n_triples_dropped_unseen_tail) == (5, 1)  # the fact on tail n + 2 is gone
    assert ckg2.n_triples + ckg2.n_triples_dropped_unseen_tail == len(kg2)
    forward = (ckg2.relation >= 1) & (ckg2.relation <= ckg2.n_relations)  # KG relations, not inverses
    assert set((ckg2.tail[forward] - ckg2.n_users).tolist()) <= {n, n + 1}
    assert ckg2.n_isolated_entities() == 1  # entity n + 2 stays isolated
    # without the rule the triple would stay
    assert build_ckg(w.enc, kg2).n_triples == len(kg2)


def test_test_graph_has_the_same_interaction_edges_as_the_training_graph(world):
    w = world
    kg1, kg2 = kg_at(w, "t1"), kg_at(w, "t2")
    ckg1 = build_ckg(w.enc, kg1)
    ckg2 = build_ckg(w.enc, kg2, known_tails=used_tails(kg1))

    def interaction_edges(c):
        return {e for e in triple_set(c) if e[1] in (INTERACT, c.n_relations + 1)}

    assert interaction_edges(ckg1) == interaction_edges(ckg2) and len(interaction_edges(ckg1)) == 2 * len(w.enc.train)
    assert triple_set(ckg1) < triple_set(ckg2)  # KG t1 is contained in KG t2


def test_val_and_test_do_not_enter_the_graph(world):
    w = world
    kg1 = kg_at(w, "t1")
    junk = {0: np.arange(5)}
    blind = dataclasses.replace(w.enc, val=junk, test=junk, mask_val=w.enc.mask_test, mask_test=w.enc.mask_val)
    a, b = build_ckg(w.enc, kg1), build_ckg(blind, kg1)
    for name in ("head", "relation", "tail"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert np.array_equal(a.edge_ts, b.edge_ts, equal_nan=True)


def test_no_kg_keeps_only_interaction_edges(world):
    w = world
    ckg = build_ckg(w.enc, no_kg(w.enc))
    assert ckg.n_edges == 2 * len(w.enc.train) and ckg.n_rel == 2 and ckg.n_entities == w.enc.n_items
    assert set(ckg.relation.tolist()) == {0, 1}


def test_summary_reports_the_provenance(world):
    kg1 = kg_at(world, "t1")
    s = build_ckg(world.enc, kg1).summary()
    assert s["kg_variant"] == "default" and s["cutoff"] == "t1" and s["time_consistent"] is True
    assert s["wikidata_snapshot"] == "2026-10-17" and s["n_triples"] == len(kg1)
    assert s["n_edges"] == 2 * (s["n_interactions"] + s["n_triples"])
    json.dumps(s)  # plain numbers only


def test_load_kg_reads_the_requested_cutoff(world):
    kg1, kg2 = kg_at(world, "t1"), kg_at(world, "t2")
    assert (len(kg1), len(kg2)) == (4, 6) and kg1.cutoff == "t1" and kg2.cutoff == "t2"
    assert kg1.relations == tuple(RELATIONS) and kg1.n_entities == world.enc.n_items + 3


def test_guard_rejects_a_kg_built_from_another_split(world):
    w = world
    info = {**w.info, "inputs": {"split_sha256": {**w.meta["sha256"], "train": "0" * 64}}}
    (w.kg_dir / "kg_meta.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="another split"):
        kg_at(w, "t1")


def test_guard_rejects_split_files_changed_after_they_were_saved(world):
    w = world
    with (w.dir / "train.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match=r"train\.csv changed"):
        kg_at(w, "t1")


def test_without_test_the_test_files_are_not_needed(world):
    w = world
    (w.dir / "test.csv").unlink()
    (w.dir / "seen_before_test.csv").unlink()
    kg1 = load_kg(w.dir, "default", "t1", w.enc, w.meta, with_test=False)
    assert len(kg1) == 4 and kg1.cutoff == "t1"
    with pytest.raises(ValueError, match=r"test\.csv is missing"):  # a report, not a FileNotFoundError
        load_kg(w.dir, "default", "t1", w.enc, w.meta)


def test_without_test_the_test_files_are_not_hashed(world):
    w = world
    with (w.dir / "test.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    load_kg(w.dir, "default", "t1", w.enc, w.meta, with_test=False)
    with pytest.raises(ValueError, match=r"test\.csv changed"):
        load_kg(w.dir, "default", "t1", w.enc, w.meta)


def test_without_test_the_other_parts_are_still_checked(world):
    w = world
    (w.dir / "test.csv").unlink()
    (w.dir / "seen_before_test.csv").unlink()
    with (w.dir / "train.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match=r"train\.csv changed"):
        load_kg(w.dir, "default", "t1", w.enc, w.meta, with_test=False)
    (w.dir / "val.csv").unlink()
    with pytest.raises(ValueError, match=r"val\.csv is missing"):
        load_kg(w.dir, "default", "t1", w.enc, w.meta, with_test=False)


def test_the_t2_kg_needs_the_test_flag(world):
    w = world
    with pytest.raises(ValueError, match="with_test=True"):
        load_kg(w.dir, "default", "t2", w.enc, w.meta, with_test=False)


def test_guard_rejects_another_item_order(world):
    w = world
    ent = pd.read_csv(w.kg_dir / "entities.csv", dtype={"key": str})
    ent.loc[[0, 1], "key"] = ent.loc[[1, 0], "key"].to_numpy()
    ent.to_csv(w.kg_dir / "entities.csv", index=False)
    with pytest.raises(ValueError, match="other item order"):
        kg_at(w, "t1")


def test_guard_rejects_another_number_of_items_and_triples(world):
    w = world
    for key, value, text in (("n_items", w.enc.n_items + 1, "items"), ("n_triples", {"t1": 99, "t2": 6}, "triples")):
        (w.kg_dir / "kg_meta.json").write_text(json.dumps({**w.info, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=text):
            kg_at(w, "t1")


def test_guard_rejects_the_wrong_variant_and_cutoff(world):
    w = world
    (w.kg_dir / "kg_meta.json").write_text(json.dumps({**w.info, "variant": "static"}), encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        kg_at(w, "t1")
    with pytest.raises(ValueError, match="cutoff"):
        kg_at(w, "t3")


def test_kg_triples_validate_their_ids():
    def make(rows):
        return KGTriples("default", "t1", np.array(rows, dtype=np.int64).reshape(-1, 3), 5, 8, ("a", "b"), True, None)

    make([[0, 0, 5], [1, 1, 7]])
    for rows, text in (
        ([[5, 0, 6]], "head"),  # head is not an item
        ([[0, 0, 4]], "tail"),  # tail is an item
        ([[0, 0, 8]], "tail"),  # tail outside the entity range
        ([[0, 2, 5]], "relation"),
        ([[0, 0, 5], [0, 0, 5]], "duplicate"),
    ):
        with pytest.raises(ValueError, match=text):
            make(rows)


ML100K = ROOT / "data" / "processed" / "ml-100k"


@pytest.mark.skipif(not (ML100K / "kg" / "default" / "kg_meta.json").exists(), reason="needs the ML-100K split and KG")
def test_real_ml100k_graph_matches_the_meta_counts():
    cfg = {"name": "ml-100k", "processed_dir": str(ML100K)}
    split, meta = load_saved(cfg)
    enc = encode(split)
    info = json.loads((ML100K / "kg" / "default" / "kg_meta.json").read_text(encoding="utf-8"))
    kg1, kg2 = (load_kg(ML100K, "default", c, enc, meta) for c in ("t1", "t2"))
    ckg1 = build_ckg(enc, kg1)
    ckg2 = build_ckg(enc, kg2, known_tails=used_tails(kg1))
    assert ckg1.n_edges == 2 * (meta["train_pos"] + info["n_triples"]["t1"])
    assert ckg1.n_isolated_entities() == info["n_entities"] - enc.n_items - info["n_tail_entities_used"]["t1"]
    assert ckg2.n_triples + ckg2.n_triples_dropped_unseen_tail == info["n_triples"]["t2"]
    assert ckg2.n_triples_dropped_unseen_tail >= info["n_entities_only_in_t2"]
    assert ckg1.time_consistent and ckg1.wikidata_snapshot == info["wikidata_snapshot"]
