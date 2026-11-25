"""Relation report and the pruned KG variant (decision 0016 item 8); pandas only, no torch."""

import json

import numpy as np
import pandas as pd
import pytest

from tkgat.kg.prune import prune_rule, read_kg, relation_table, write_pruned

N_ITEMS = 100
RELATIONS = ["wide", "dated_wide", "narrow", "tiny_wide"]


def make_variant(tmp_path):
    """4 relations: wide (60% of the items, 1,200 triples), dated_wide, narrow (3% of the items), tiny_wide."""
    rows = []

    def add(r, n_heads, n_triples, dated):
        heads = np.arange(n_heads)
        for k in range(n_triples):
            rows.append((int(heads[k % n_heads]), r, N_ITEMS + k, 1.4e9 if dated and k % 2 == 0 else np.nan))

    add(0, 60, 1200, False)
    add(1, 80, 1500, True)
    add(2, 3, 1100, True)  # many triples, but only 3% of the items
    add(3, 50, 40, False)  # many items, but only 40 triples
    dates = pd.DataFrame(rows, columns=["h", "r", "t", "ts"])
    dates = dates.sample(frac=1, random_state=1).reset_index(drop=True)
    d = tmp_path / "default"
    d.mkdir()
    dates[["h", "r", "t"]].to_csv(d / "kg_t1.tsv", sep="\t", index=False)
    dates.to_csv(d / "kg_t1_dates.tsv", sep="\t", index=False)
    t2 = dates.iloc[:-5]  # t2 differs from t1 only so that both cut-offs are exercised
    t2[["h", "r", "t"]].to_csv(d / "kg_t2.tsv", sep="\t", index=False)
    t2.to_csv(d / "kg_t2_dates.tsv", sep="\t", index=False)
    pd.DataFrame({"relation": range(4), "name": RELATIONS}).to_csv(d / "relations.csv", index=False)
    n_ent = N_ITEMS + int(dates["t"].max()) + 1
    pd.DataFrame(
        {"entity": range(n_ent), "kind": ["item"] * N_ITEMS + ["freebase"] * (n_ent - N_ITEMS), "key": 0, "label": ""}
    ).to_csv(d / "entities.csv", index=False)
    info = {
        "variant": "default",
        "n_items": N_ITEMS,
        "n_entities": n_ent,
        "n_relations": 4,
        "n_triples": {"t1": len(dates), "t2": len(t2)},
        "n_dated_triples": {"t1": int(dates["ts"].notna().sum()), "t2": int(t2["ts"].notna().sum())},
        "inputs": {"split_sha256": {"train": "x"}},
        "per_relation": [{"relation": n, "cutoff": "t1", "kept": 1} for n in RELATIONS],
        "date_kind_per_relation": [{"relation": n, "facts": 1} for n in RELATIONS],
        "n_dropped": {"t1": {"future_tail": 2}},
        "n_items_with_facts": {"t1": 90, "t2": 90},
    }
    (d / "kg_meta.json").write_text(json.dumps(info), encoding="utf-8")
    return d, dates


def test_table_counts_triples_items_dates_and_tails(tmp_path):
    d, dates = make_variant(tmp_path)
    t = relation_table(d, "t1").set_index("name")
    assert t.loc["wide", ["triples", "items", "tails", "dated"]].tolist() == [1200, 60, 1200, 0]
    assert t.loc["narrow", "item_share"] == pytest.approx(0.03)
    assert t.loc["dated_wide", "dated"] == 750 and t.loc["dated_wide", "dated_share"] == pytest.approx(0.5)
    assert t["triples"].sum() == len(dates)
    assert relation_table(d, "t2")["triples"].sum() == len(dates) - 5


def test_rule_drops_few_items_or_few_triples(tmp_path):
    d, _ = make_variant(tmp_path)
    keep = prune_rule(relation_table(d))
    assert dict(zip(RELATIONS, keep.tolist(), strict=True)) == {
        "wide": True,
        "dated_wide": True,
        "narrow": False,  # 3% of the items < 5%
        "tiny_wide": False,  # 40 triples < 1,000
    }
    assert prune_rule(relation_table(d), min_item_share=0.7, min_triples=0).tolist() == [False, True, False, False]


def test_pruned_variant_renumbers_relations_and_keeps_every_other_file_consistent(tmp_path):
    d, dates = make_variant(tmp_path)
    info = write_pruned(d, tmp_path / "pruned", ["narrow", "tiny_wide"])
    p = tmp_path / "pruned"
    rel = pd.read_csv(p / "relations.csv")
    assert rel["name"].tolist() == ["wide", "dated_wide"] and rel["relation"].tolist() == [0, 1]
    for cut, n in (("t1", len(dates)), ("t2", len(dates) - 5)):
        tri, dts = read_kg(p, cut)  # also checks that triples and dates are in the same order
        assert len(tri) == info["n_triples"][cut] < n and set(tri["r"]) == {0, 1}
        assert dts["ts"].notna().sum() == info["n_dated_triples"][cut]
    kept_old = dates[dates["r"].isin([0, 1])].reset_index(drop=True)
    tri, dts = read_kg(p, "t1")
    assert tri[["h", "t"]].equals(kept_old[["h", "t"]]) and tri["r"].equals(kept_old["r"])  # ids 0 and 1 are unchanged
    assert pd.read_csv(p / "entities.csv").equals(pd.read_csv(d / "entities.csv"))
    meta = json.loads((p / "kg_meta.json").read_text())
    assert meta["variant"] == "pruned" and meta["pruned_from"] == "default" and meta["n_relations"] == 2
    assert meta["dropped_relations"] == ["narrow", "tiny_wide"] and "n_dropped" not in meta
    assert [r["relation"] for r in meta["per_relation"]] == ["wide", "dated_wide"]
    assert meta["inputs"] == json.loads((d / "kg_meta.json").read_text())["inputs"]  # the split hashes are kept


def test_renumbering_follows_the_old_order_when_a_middle_relation_goes(tmp_path):
    d, dates = make_variant(tmp_path)
    write_pruned(d, tmp_path / "pruned", ["dated_wide"])
    tri, _ = read_kg(tmp_path / "pruned", "t1")
    assert pd.read_csv(tmp_path / "pruned" / "relations.csv")["name"].tolist() == ["wide", "narrow", "tiny_wide"]
    old = dates[dates["r"] == 2]
    assert len(tri[tri["r"] == 1]) == len(old)  # old relation 2 (narrow) is now 1
    assert set(tri["r"]) == {0, 1, 2}


def test_bad_requests_are_refused(tmp_path):
    d, _ = make_variant(tmp_path)
    with pytest.raises(ValueError, match="not relations"):
        write_pruned(d, tmp_path / "x", ["nope"])
    with pytest.raises(ValueError, match="every relation"):
        write_pruned(d, tmp_path / "y", RELATIONS)
    broken = pd.read_csv(d / "kg_t1_dates.tsv", sep="\t").iloc[::-1]
    broken.to_csv(d / "kg_t1_dates.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="same order"):
        relation_table(d)
