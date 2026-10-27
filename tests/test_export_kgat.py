"""The KGAT export is the official split and KG, re-labelled: parse the files back and compare with the source."""

import filecmp
import importlib.util
import json

import pandas as pd
import pytest

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.export.kgat import export_kgat

from .conftest import ROOT


def parse_items(path):
    """``{user: [item, ...]}`` of a ``user item item ...`` file."""
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {int(r[0]): [int(x) for x in r[1:]] for r in rows}


def parse_labels(path):
    """``{(user, item): label}`` of a ``user item:label ...`` file."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        u, *pairs = line.split()
        for pair in pairs:
            item, label = pair.split(":")
            out[(int(u), int(item))] = int(label)
    return out


def table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines[1:]]


@pytest.fixture
def exported(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    enc = encode(split)
    n = enc.n_items
    entities = pd.DataFrame(
        {
            "entity": range(n + 3),
            "kind": ["item"] * n + ["wikidata"] * 3,
            "key": [*[str(i) for i in enc.item_ids], "Q1", "Q2", "Q3"],
            "label": [""] * n + ["genre A", "director B", "award C"],
        }
    )
    relations = pd.DataFrame({"relation": [0, 1], "name": ["award_received", "genre"]})
    kg_t1 = pd.DataFrame({"h": [0, 1, 2], "r": [1, 1, 0], "t": [n, n + 1, n + 1]})
    kg_t2 = pd.concat([kg_t1, pd.DataFrame({"h": [3], "r": [0], "t": [n + 2]})], ignore_index=True)
    freebase = {int(enc.item_ids[0]): "m.abc", int(enc.item_ids[1]): "m.def"}
    out = {}
    for name, ts in (("plain", False), ("time", True)):
        folder = tmp_path / name
        meta = export_kgat(
            folder, split, enc, entities, relations, kg_t1, kg_t2, freebase, ts, {"run_name": "synthetic"}
        )
        out[name] = (folder, meta)
    return split, enc, kg_t1, kg_t2, freebase, out


def test_interactions_round_trip(exported):
    split, enc, *_, out = exported
    folder, meta = out["plain"]
    users = {int(raw): i for i, raw in enumerate(enc.user_ids)}
    items = {int(raw): i for i, raw in enumerate(enc.item_ids)}
    for name, df in (("train", split.train), ("val", split.val), ("test", split.test)):
        got = {(u, i) for u, its in parse_items(folder / f"{name}.txt").items() for i in its}
        want = {(users[int(u)], items[int(i)]) for u, i in zip(df["user"], df["item"], strict=True)}
        assert got == want and meta["interactions"][name] == len(df)
        ratings = parse_labels(folder / f"{name}_rating.txt")
        assert set(ratings) == want and all(r >= 4 for r in ratings.values())  # rating >= 4 only


def test_items_are_in_time_order_per_user(exported):
    split, enc, *_, out = exported
    folder, _ = out["time"]
    items = {int(raw): i for i, raw in enumerate(enc.item_ids)}
    users = {int(raw): i for i, raw in enumerate(enc.user_ids)}
    ts = {
        (users[int(u)], items[int(i)]): int(t)
        for u, i, t in zip(split.train["user"], split.train["item"], split.train["ts"], strict=True)
    }
    for u, its in parse_items(folder / "train.txt").items():
        stamps = [ts[(u, i)] for i in its]
        assert stamps == sorted(stamps)


def test_masks_hold_every_earlier_interaction(exported):
    split, enc, *_, out = exported
    folder, meta = out["plain"]
    users = {int(raw): i for i, raw in enumerate(enc.user_ids)}
    items = {int(raw): i for i, raw in enumerate(enc.item_ids)}
    for name, df in (("val", split.seen_before_val), ("test", split.seen_before_test)):
        known = df[df["user"].isin(users) & df["item"].isin(items)]
        want = {(users[int(u)], items[int(i)]) for u, i in zip(known["user"], known["item"], strict=True)}
        got = {(u, i) for u, its in parse_items(folder / f"{name}_mask.txt").items() for i in its}
        assert got == want
    # the test mask holds the val interactions too, the val mask does not hold val
    val_pairs = {(u, i) for u, its in parse_items(folder / "val.txt").items() for i in its}
    test_mask = {(u, i) for u, its in parse_items(folder / "test_mask.txt").items() for i in its}
    val_mask = {(u, i) for u, its in parse_items(folder / "val_mask.txt").items() for i in its}
    assert val_pairs <= test_mask and not (val_pairs & val_mask)


def test_ids_lists_and_kg(exported):
    split, enc, kg_t1, kg_t2, freebase, out = exported
    folder, meta = out["plain"]
    item_list = table(folder / "item_list.txt")
    assert [(int(r[0]), int(r[1])) for r in item_list] == [(int(raw), i) for i, raw in enumerate(enc.item_ids)]
    assert {int(r[0]): r[2] for r in item_list if r[2] != "NA"} == freebase  # unlinked items are "NA"
    assert [r[1] for r in table(folder / "entity_list.txt")] == [str(i) for i in range(meta["n_entities"])]
    assert [r[0] for r in table(folder / "entity_list.txt")][: enc.n_items] == [
        f"movielens.item.{i}" for i in enc.item_ids
    ]
    assert [int(r[1]) for r in table(folder / "relation_list.txt")] == [0, 1]
    for name, kg in (("kg_final.txt", kg_t1), ("kg_test.txt", kg_t2)):
        got = [tuple(int(x) for x in line.split()) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
        assert got == list(zip(kg["h"], kg["r"], kg["t"], strict=True))
    assert meta["kg_triples"] == {"kg_final.txt (t1)": 3, "kg_test.txt (t2)": 4}


def test_timestamp_variant_only_adds_timestamps(exported):
    split, enc, *_, out = exported
    plain, _ = out["plain"]
    time, meta = out["time"]
    extra = {p.name for p in time.iterdir()} - {p.name for p in plain.iterdir()}
    assert extra == {"train_timestamp.txt", "val_timestamp.txt", "test_timestamp.txt"}
    same = [
        "train.txt",
        "val.txt",
        "test.txt",
        "val_mask.txt",
        "test_mask.txt",
        "kg_final.txt",
        "kg_test.txt",
        "item_list.txt",
        "entity_list.txt",
        "train_rating.txt",
    ]
    assert all(filecmp.cmp(plain / f, time / f, shallow=False) for f in same)
    users = {int(raw): i for i, raw in enumerate(enc.user_ids)}
    items = {int(raw): i for i, raw in enumerate(enc.item_ids)}
    want = {
        (users[int(u)], items[int(i)]): int(t)
        for u, i, t in zip(split.test["user"], split.test["item"], split.test["ts"], strict=True)
    }
    assert parse_labels(time / "test_timestamp.txt") == want
    assert "timestamp" in (time / "interactions_chronological.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert "timestamp" not in (plain / "interactions_chronological.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert (
        meta["timestamp_variant"] is True
        and json.loads((time / "dataset_meta.json").read_text())["run_name"] == "synthetic"
    )


def test_rejects_entities_that_do_not_start_with_the_train_items(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    enc = encode(split)
    bad = pd.DataFrame({"entity": range(2), "kind": ["wikidata"] * 2, "key": ["Q1", "Q2"], "label": ["", ""]})
    empty = pd.DataFrame({"h": [], "r": [], "t": []}, dtype=int)
    with pytest.raises(ValueError, match="first entities"):
        export_kgat(
            tmp_path / "x", split, enc, bad, pd.DataFrame({"relation": [0], "name": ["a"]}), empty, empty, {}, False
        )


ML100K = ROOT / "data" / "processed" / "ml-100k"


@pytest.mark.skipif(not (ML100K / "kg" / "default" / "kg_meta.json").exists(), reason="needs the ML-100K split and KG")
@pytest.mark.parametrize("timestamps", [False, True])
def test_real_ml100k_export_matches_the_split_meta(tmp_path, timestamps):
    spec = importlib.util.spec_from_file_location("export_script", ROOT / "scripts" / "08_export_kgat.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    argv = ["--data", str(ROOT / "configs" / "data" / "ml100k.yaml"), "--out", str(tmp_path / "out")]
    script.main([*argv, *(["--timestamps"] if timestamps else [])])

    meta = json.loads((tmp_path / "out" / "dataset_meta.json").read_text(encoding="utf-8"))
    split_meta = json.loads((ML100K / "meta.json").read_text(encoding="utf-8"))
    assert meta["interactions"] == {
        "train": split_meta["train_pos"],
        "val": split_meta["val_pos"],
        "test": split_meta["test_pos"],
    }
    assert meta["n_users"] == split_meta["n_users"] and meta["n_items"] == split_meta["n_items"]
    assert meta["cutoffs"] == {"t1": split_meta["t1"], "t2": split_meta["t2"]}
    kg = json.loads((ML100K / "kg" / "default" / "kg_meta.json").read_text(encoding="utf-8"))
    assert meta["kg_triples"] == {"kg_final.txt (t1)": kg["n_triples"]["t1"], "kg_test.txt (t2)": kg["n_triples"]["t2"]}
    assert (tmp_path / "out" / "test_timestamp.txt").exists() == timestamps
    with pytest.raises(FileExistsError):  # a second export into the same folder needs --overwrite
        script.main(argv)
