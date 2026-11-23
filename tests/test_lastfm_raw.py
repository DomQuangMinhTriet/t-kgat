"""Raw rebuild: new pairs survive, ids round-trip, dates align, and leakage checks hold."""

import gzip
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml

from tkgat.data.encode import encode
from tkgat.data.lastfm_audit import audit_raw_split
from tkgat.data.lastfm_raw import cut_pairs, load_raw_pairs, utc_seconds, window_start
from tkgat.data.leakage import check_split
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.ckg import load_kg, load_kg_dates
from tkgat.kg.lastfm import parse_freebase_date
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS, read_freebase_source
from tkgat.pipeline.build import load_saved, save_split
from tkgat.pipeline.lastfm_raw import build_raw_dataset, check_snapshot_containment, read_item_links
from tkgat.registry import sha256_file

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROWS = (
    "m.track1\tmusic.recording.artist\tm.artist\n"
    "m.track2\tmusic.recording.artist\tm.future\n"
    "m.track1\tmusic.recording.canonical_version\tm.track2\n"
)


def raw_file(path, rows):
    path.write_text(
        "user_id:token\titem_id:token\ttimestamp:float\n"
        + "".join(f"{u}\t{i}\t{ts}\n" for u, i, ts in rows), encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("core", [1, 2])
@pytest.mark.parametrize("empty_links", [False, True])
def test_disk_engine_matches_all_export_hashes_and_processed_rules(inputs, tmp_path, core, empty_links):
    pytest.importorskip("duckdb")
    pairs, mapping, freebase = inputs
    links = mapping / "item_list.txt"
    if empty_links:
        links.write_text("org_id remap_id freebase_id\n", encoding="utf-8")
    expected = build_raw_dataset(pairs, tmp_path / "pandas", tmp_path / "e1", freebase,
                                 item_links=links, kcore=core)
    actual = build_raw_dataset(pairs, tmp_path / "disk", tmp_path / "e2", freebase,
                               item_links=links, kcore=core, engine="duckdb", memory_limit="256MB")
    assert actual["sha256"] == expected["sha256"]
    for key in ("interactions", "coverage", "cutoffs", "n_users", "n_items", "mask_rows"):
        assert actual[key] == expected[key]
    old_meta = json.loads((tmp_path / "pandas/meta.json").read_text())
    new_meta = json.loads((tmp_path / "disk/meta.json").read_text())
    assert new_meta["sha256"] == old_meta["sha256"]
    for part in ("train", "val", "test", "seen_before_val", "seen_before_test", "dropped_interactions"):
        assert (tmp_path / f"pandas/{part}.csv").read_bytes() == (tmp_path / f"disk/{part}.csv").read_bytes()
    split, saved = load_saved({"processed_dir": tmp_path / "disk"})
    assert check_split(split) == []
    audit, _ = audit_raw_split(load_raw_pairs(pairs)[0], split)
    for key, value in audit.items():
        assert actual["interaction_audit"][key] == value
    kg = load_kg(tmp_path / "disk", "default", "t1", encode(split), saved)
    assert len(load_kg_dates(tmp_path / "disk", kg)) == len(kg)


def test_disk_engine_rejects_nonchronological_cut_and_cleans_only_owned_scratch(inputs, tmp_path):
    pytest.importorskip("duckdb")
    pairs, mapping, freebase = inputs
    lines = pairs.read_text().splitlines(keepends=True)
    pairs.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8", newline="\n")
    manifest = Path(str(pairs) + ".meta.json")
    meta = json.loads(manifest.read_text())
    meta["sha256"] = sha256_file(pairs)
    manifest.write_text(json.dumps(meta), encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="chronological"):
        build_raw_dataset(pairs, tmp_path / "p", tmp_path / "e", freebase,
                          item_links=mapping / "item_list.txt", engine="duckdb", scratch_dir=scratch)
    assert [p.name for p in scratch.iterdir()] == ["keep.txt"]
    assert pairs.is_file() and manifest.is_file()


def test_disk_engine_cascading_core_and_empty_eval_match_pandas(inputs, tmp_path):
    pytest.importorskip("duckdb")
    _, mapping, freebase = inputs
    edges = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4)]
    rows = [(u, i, t) for t, (u, i) in enumerate(edges, 1)] + [(6, 6, 100), (1, 3, 200), (6, 2, 300)]
    raw = raw_file(tmp_path / "cascade.inter", rows)
    pairs = tmp_path / "cascade.tsv"
    cut_pairs(raw, pairs, progress=False)
    common = {"item_links": mapping / "item_list.txt", "kcore": 2}
    expected = build_raw_dataset(pairs, tmp_path / "p1", tmp_path / "e1", freebase, **common)
    actual = build_raw_dataset(pairs, tmp_path / "p2", tmp_path / "e2", freebase, engine="duckdb", **common)
    assert actual["interactions"] == {"train": 4, "val": 0, "test": 0}
    assert len(actual["interaction_audit"]["core_iterations"]) >= 3
    assert actual["sha256"] == expected["sha256"]


def test_disk_engine_reads_only_batches_and_overwrite_is_repeatable(inputs, tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pairs, mapping, freebase = inputs
    read_csv = pd.read_csv

    def bounded_read(*args, **kwargs):
        assert kwargs.get("chunksize", 0) > 0, "disk engine must never load all pairs into pandas"
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", bounded_read)
    options = {"item_links": mapping / "item_list.txt", "engine": "duckdb", "kcore": 1}
    first = build_raw_dataset(pairs, tmp_path / "p", tmp_path / "e", freebase, **options)
    second = build_raw_dataset(pairs, tmp_path / "p", tmp_path / "e", freebase, overwrite=True, **options)
    assert first["sha256"] == second["sha256"]
    assert pairs.is_file() and first["pairs_sha256"] == sha256_file(pairs)


def test_unsorted_events_are_reduced_without_reference_ids(tmp_path):
    rows = [(901, 99999, 40), (1, 2, 30), (901, 99999, 10), (1, 2, 20), (1, 3, 15)]
    raw = raw_file(tmp_path / "events.inter", rows)
    first, second = tmp_path / "a.tsv", tmp_path / "b.tsv"
    meta = cut_pairs(raw, first, chunksize=2, progress=False)
    raw_file(raw, list(reversed(rows)))
    other = cut_pairs(raw, second, chunksize=3, progress=False)
    assert meta["sha256"] == other["sha256"]
    pairs, _ = load_raw_pairs(first)
    assert list(pairs[["user", "item", "ts"]].itertuples(index=False, name=None)) == [
        (901, 99999, 10), (1, 3, 15), (1, 2, 20),
    ]
    assert meta["events_in_window"] == 5 and meta["pairs"] == 3
    assert meta["accounting"]["repeated_listens_collapsed"] == 2
    assert sum(meta["accounting"].values()) == meta["rows_scanned"]
    with pytest.raises(FileExistsError):
        cut_pairs(raw, first, progress=False)
    first.write_text(first.read_text() + "1\t4\t50\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_raw_pairs(first)


def test_window_is_cut_before_earliest_pair_and_user_threshold(tmp_path):
    raw = raw_file(tmp_path / "events.inter", [(1, 9, 1), (1, 9, 10), (1, 9, 15), (2, 8, 20), (3, 7, 30)])
    out = tmp_path / "pairs.tsv"
    meta = cut_pairs(raw, out, start=10, end=30, min_events_user=2, chunksize=1, progress=False)
    pairs, _ = load_raw_pairs(out)
    assert pairs[["user", "item", "ts"]].values.tolist() == [[1, 9, 10]]
    assert meta["events_before_start"] == meta["events_at_or_after_end"] == 1
    assert meta["events_dropped_user_threshold"] == 1
    assert sum(meta["accounting"].values()) == meta["rows_scanned"]
    assert window_start(utc_seconds("2024-03-31"), 1) == utc_seconds("2024-02-29")
    with pytest.raises(ValueError, match="explicit end"):
        cut_pairs(raw, tmp_path / "bad.tsv", window_months=3, progress=False)


def test_supplied_drive_tracks_id_header_preserves_raw_tracks_and_earliest_times(tmp_path):
    source = tmp_path / "drive.inter"
    source.write_text(
        "user_id:token\ttracks_id:token\ttimestamp:float\n"
        "31435741\t4\t1385212958\n"
        "31435741\t4\t1385210000\n"
        "31435741\t5\t1385212000\n", encoding="utf-8",
    )
    output = tmp_path / "pairs.tsv"
    meta = cut_pairs(source, output, progress=False, chunksize=1)
    pairs, _ = load_raw_pairs(output)
    assert pairs[["user", "item", "ts"]].values.tolist() == [
        [31435741, 4, 1385210000], [31435741, 5, 1385212000],
    ]
    assert meta["rows_scanned"] == 3 and meta["accounting"]["repeated_listens_collapsed"] == 1


@pytest.mark.parametrize("kind", ["gzip", "zip"])
def test_local_archives(kind, tmp_path):
    raw = raw_file(tmp_path / "events.inter", [(4, 5, 6)])
    if kind == "gzip":
        source = tmp_path / "events.gz"
        with gzip.open(source, "wb") as handle:
            handle.write(raw.read_bytes())
    else:
        archive = tmp_path / "events.zip"
        with ZipFile(archive, "w") as handle:
            handle.write(raw, "nested/events.inter")
        source = f"{archive}::nested/events.inter"
    assert cut_pairs(source, tmp_path / "pairs.tsv", progress=False)["pairs"] == 1


@pytest.fixture
def inputs(tmp_path):
    base = utc_seconds("2014-01-01")
    rows = [(u, 1000 + i, base + (j * 6 + i) * 86400) for j, u in enumerate((10, 20)) for i in range(1, 7)]
    rows += [(10, 1001, base + 30 * 86400)]
    raw = raw_file(tmp_path / "events.inter", list(reversed(rows)))
    pairs = tmp_path / "pairs.tsv"
    cut_pairs(raw, pairs, chunksize=4, progress=False)
    author = tmp_path / "author"
    author.mkdir()
    (author / "item_list.txt").write_text(
        "org_id remap_id freebase_id\n1001 0 m.track1\n1002 1 m.track2\n", encoding="utf-8",
    )
    (author / "entity_list.txt").write_text(
        "org_id remap_id\nm.track1 0\nm.track2 1\nm.artist 2\nm.future 3\n", encoding="utf-8",
    )
    (author / "relation_list.txt").write_text(
        "org_id remap_id\n" + "".join(
            f"http://rdf.freebase.com/ns/{name} {index}\n" for index, name in enumerate(PAPER_RELATIONS)
        ), encoding="utf-8",
    )
    (author / "kg_final.txt").write_text("0 2 2\n1 2 3\n0 5 1\n", encoding="utf-8")
    # No author train.txt, test.txt or user_list.txt exists: they must never be read.
    freebase = tmp_path / "freebase.tsv"
    freebase.write_text(
        'subject\tpredicate\tobject\n' + GRAPH_ROWS + 'm.artist\tpeople.person.date_of_birth\t"1980"\n'
        'm.artist\tpeople.person.date_of_birth\t"1990"\n'
        'm.future\tmusic.release.release_date\t"2020-03-10"\n', encoding="utf-8",
    )
    return pairs, author, freebase


def test_raw_rebuild_roundtrip_and_fact_timestamps(inputs, tmp_path):
    pairs, author, freebase = inputs
    processed, exported = tmp_path / "processed", tmp_path / "export"
    meta = build_raw_dataset(pairs, processed, exported, freebase, item_links=author / "item_list.txt", kcore=1)
    assert meta["n_users"] == 2 and meta["n_items"] == 6
    assert meta["coverage"]["items_without_links"] == 4
    assert meta["uses_published_interactions"] is False
    assert meta["interactions"] == {"train": 9, "val": 1, "test": 2}
    s, _ = load_saved({"processed_dir": processed})
    assert check_split(s) == []
    users = pd.read_csv(exported / "user_list.txt", sep=" ").set_index("remap_id").org_id.to_dict()
    items = pd.read_csv(exported / "item_list.txt", sep=" ", keep_default_na=False)
    item_ids = items.set_index("remap_id").org_id.to_dict()
    assert items.freebase_id.tolist() == ["m.track1", "m.track2", "NA", "NA", "NA", "NA"]
    for part in ("train", "val", "test"):
        expected = set(getattr(s, part)[["user", "item", "ts"]].itertuples(index=False, name=None))
        got = set()
        for line in (exported / f"{part}_timestamp.txt").read_text().splitlines():
            user, *events = line.split()
            for event in events:
                item, ts = map(int, event.split(":"))
                got.add((users[int(user)], item_ids[item], ts))
        assert got == expected
    entities = pd.read_csv(exported / "entity_list.txt", sep=" ")
    assert entities.org_id.is_unique and entities.org_id.iloc[:2].tolist() == ["m.track1", "m.track2"]
    dated = pd.read_csv(exported / "kg_final_timestamp.tsv", sep="\t")
    triple_rows = [list(map(int, row.split())) for row in (exported / "kg_final.txt").read_text().splitlines()]
    assert dated[["h", "r", "t"]].values.tolist() == triple_rows
    assert dated.ts.dropna().tolist() == [utc_seconds("1980-12-31")]
    assert dated.ts.isna().sum() == 1  # undated canonical-version link
    assert dated.iloc[1].t == 1  # a tail which is an item uses that item's entity id
    assert "m.future" not in set(entities.loc[dated.t, "org_id"])
    assert (exported / "fact_date_evidence.tsv").is_file()
    assert meta["sha256"]["train.txt"]
    with pytest.raises(FileExistsError):
        build_raw_dataset(pairs, processed, exported, freebase, item_links=author / "item_list.txt", kcore=1)


def test_unlinked_items_are_not_dropped_even_with_empty_kg(inputs, tmp_path):
    pairs, author, freebase = inputs
    links = tmp_path / "links.tsv"
    links.write_text("item\tfreebase_id\n9999\tm.other\n", encoding="utf-8")
    meta = build_raw_dataset(
        pairs, tmp_path / "processed", tmp_path / "export", freebase, item_links=links, kcore=1,
    )
    assert meta["n_items"] == 6
    assert meta["coverage"]["items_without_links"] == 6
    assert meta["kg_triples"]["kg_final.txt (t1)"] == 0


def test_freebase_facts_and_expanded_links_do_not_need_reference_tree(inputs, tmp_path):
    pairs, _, freebase = inputs
    links, graph = tmp_path / "links.tsv", tmp_path / "graph.tsv"
    links.write_text("item\tfreebase_id\n1006\t/m/new\n", encoding="utf-8")
    graph.write_text(
        "subject\tpredicate\tobject\nm.new\tmusic.recording.artist\tm.artist\n"
        'm.new\ttype.object.name\t"A literal"\n', encoding="utf-8",
    )
    meta = build_raw_dataset(
        pairs, tmp_path / "processed", tmp_path / "export", graph,
        item_links=links, kcore=1,
    )
    assert meta["coverage"]["items_with_facts"]["t1"] == 1
    assert meta["coverage"]["triples"]["t1"] == 1  # names are labels, not a selected graph relation


def test_reject_ambiguous_links(tmp_path):
    links = tmp_path / "links.tsv"
    links.write_text("item\tfreebase_id\n1\tm.one\n1\tm.two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        read_item_links(links)


def test_cli_pipeline(inputs, tmp_path):
    pairs, author, freebase = inputs
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/02b_build_lastfm_from_raw.py"),
         "--kg-mode", "one-hop", "--pairs", str(pairs),
         "--item-links", str(author / "item_list.txt"), "--freebase", str(freebase), "--kcore", "1",
         "--out", str(tmp_path / "processed"), "--export-dir", str(tmp_path / "export")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "export/dataset_meta.json").read_text())
    assert meta["n_items"] == 6


@pytest.mark.parametrize("bad_row", ["1\t2\t3\t4\n", "1\t2\n", "\n", "1\t2\t-3\n", "1\t2\t1.5\n"])
def test_invalid_raw_rows_fail_instead_of_disappearing(tmp_path, bad_row):
    raw = tmp_path / "bad.inter"
    raw.write_text("user_id:token\titem_id:token\ttimestamp:float\n" + bad_row, encoding="utf-8")
    output = tmp_path / "pairs.tsv"
    with pytest.raises((ValueError, TypeError)):
        cut_pairs(raw, output, chunksize=1, progress=False)
    assert not output.exists()


@pytest.mark.parametrize(
    ("literal", "expected"),
    [('"2013"', "2013-12-31"), ('"2013-02"', "2013-02-28"),
     ('"2012-02"', "2012-02-29"), ('"2013-05-17"', "2013-05-17")],
)
def test_conservative_date_precision(literal, expected):
    assert parse_freebase_date(literal).isoformat() == expected


def test_audit_rejects_a_past_only_but_incomplete_mask(inputs):
    pairs, _, _ = inputs
    frame, _ = load_raw_pairs(pairs)
    split = temporal_split(frame, SplitConfig(min_rating=1, kcore=1))
    for name in ("seen_before_val", "seen_before_test"):
        changed = replace(split, **{name: getattr(split, name).iloc[1:].copy()})
        assert check_split(changed) == []  # the generic check only tests absence of future rows
        with pytest.raises(ValueError, match="Incomplete"):
            audit_raw_split(frame, changed)


def test_5core_does_not_use_future_activity_and_every_exclusion_has_a_reason(inputs, tmp_path):
    _, author, freebase = inputs
    rows = [(u, 1000 + i, (u - 1) * 5 + i) for u in range(1, 6) for i in range(1, 6)]
    rows += [(99, 1001, 26), (99, 1002, 27), (99, 1003, 28), (99, 1004, 100), (99, 1005, 101)]
    rows += [(200, 1000 + i, 101 + i) for i in range(1, 5)]
    raw = raw_file(tmp_path / "fivecore.inter", rows)
    pairs = tmp_path / "fivecore.tsv"
    cut_pairs(raw, pairs, progress=False)
    out, export = tmp_path / "processed", tmp_path / "export"
    meta = build_raw_dataset(pairs, out, export, freebase, item_links=author / "item_list.txt")
    split, _ = load_saved({"processed_dir": out})
    assert len(split.train) == 25
    assert 99 not in set(split.train.user)  # five total items cannot rescue a user with one train item
    assert split.train.groupby("user").item.nunique().min() == 5
    assert split.train.groupby("item").user.nunique().min() == 5
    dropped = pd.read_csv(out / "dropped_interactions.csv")
    assert len(dropped) + sum(meta["interactions"].values()) == len(rows)
    assert set(dropped.reason) == {"train_kcore", "cold_user"}
    report = json.loads((out / "validation_report.json").read_text())
    assert report["unexplained_pairs"] == 0 and report["accounted_pairs"] == len(rows)


def test_boundaries_masks_and_cold_item_accounting():
    frame = pd.DataFrame(
        [(1, 1, 1, 1), (2, 1, 1, 2), (1, 2, 1, 3), (2, 2, 1, 4),
         (1, 3, 1, 5), (2, 3, 1, 6), (1, 4, 1, 7), (2, 4, 1, 8),
         (1, 5, 1, 9), (3, 1, 1, 10)], columns=["user", "item", "rating", "ts"],
    )
    split = temporal_split(frame, SplitConfig(q_val=0.5, q_test=0.8, min_rating=1, kcore=1))
    assert (split.t1, split.t2) == (5, 8)
    report, dropped = audit_raw_split(frame, split)
    assert split.seen_before_val.ts.tolist() == [1, 2, 3, 4]
    assert split.seen_before_test.ts.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert set(dropped.reason) == {"cold_user", "cold_item"}
    assert report["accounted_pairs"] == len(frame)
    # ts == t1 and ts == t2 belong to val/test periods, even if then removed as cold items.
    assert dropped.loc[dropped.ts.eq(5), "split"].item() == "val"
    assert dropped.loc[dropped.ts.eq(8), "split"].item() == "test"


def test_exported_masks_match_all_past_known_interactions(inputs, tmp_path):
    pairs, author, freebase = inputs
    out, export = tmp_path / "processed", tmp_path / "export"
    build_raw_dataset(pairs, out, export, freebase, item_links=author / "item_list.txt", kcore=1)
    split, _ = load_saved({"processed_dir": out})
    users = pd.read_csv(export / "user_list.txt", sep=" ").set_index("remap_id").org_id.to_dict()
    items = pd.read_csv(export / "item_list.txt", sep=" ").set_index("remap_id").org_id.to_dict()
    for phase, history in (("val", split.seen_before_val), ("test", split.seen_before_test)):
        actual = set()
        for line in (export / f"{phase}_mask.txt").read_text().splitlines():
            user, *row_items = map(int, line.split())
            actual.update((users[user], items[item]) for item in row_items)
        expected = history[history.user.isin(users.values()) & history.item.isin(items.values())]
        assert actual == set(expected[["user", "item"]].itertuples(index=False, name=None))


def test_repeated_builds_have_identical_data_hashes(inputs, tmp_path):
    pairs, author, freebase = inputs
    first = build_raw_dataset(
        pairs, tmp_path / "p1", tmp_path / "e1", freebase, item_links=author / "item_list.txt", kcore=1,
    )
    second = build_raw_dataset(
        pairs, tmp_path / "p2", tmp_path / "e2", freebase, item_links=author / "item_list.txt", kcore=1,
    )
    assert first["sha256"] == second["sha256"]
    meta1 = json.loads((tmp_path / "p1/meta.json").read_text())
    meta2 = json.loads((tmp_path / "p2/meta.json").read_text())
    assert meta1["sha256"] == meta2["sha256"]
    for filename in ("train.csv", "val.csv", "test.csv"):
        assert b"\r\n" not in (tmp_path / "p1" / filename).read_bytes()


def test_audit_does_not_label_arbitrary_training_loss_as_kcore():
    rows = []
    for offset in (0, 10):
        rows += [(u + offset, i + offset, 1, len(rows) + (u - 1) * 5 + i)
                 for u in range(1, 6) for i in range(1, 6)]
    rows += [(100 + i, 1000 + i, 1, 100 + i) for i in range(20)]
    frame = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    split = temporal_split(frame, SplitConfig(min_rating=1, kcore=5))
    assert len(split.train) == 50
    changed = replace(split, train=split.train[split.train.user.lt(10)].copy())
    assert check_split(changed) == []  # the remaining component still satisfies degree >= 5
    with pytest.raises(ValueError, match="complete train-only k-core"):
        audit_raw_split(frame, changed)


def test_shuffled_mapping_rows_and_duplicate_mid_do_not_merge_raw_interactions(inputs, tmp_path):
    pairs, author, freebase = inputs
    links = tmp_path / "aliases.tsv"
    links.write_text("item\tfreebase_id\n1002\tm.track1\n1001\tm.track1\n", encoding="utf-8")
    out, export = tmp_path / "processed", tmp_path / "export"
    result = build_raw_dataset(pairs, out, export, freebase, item_links=links, kcore=1)
    assert result["n_items"] == 6 and sum(result["interactions"].values()) == 12
    items = pd.read_csv(export / "item_list.txt", sep=" ", keep_default_na=False)
    assert items.iloc[:2].org_id.tolist() == [1001, 1002]
    assert items.iloc[:2].freebase_id.tolist() == ["m.track1", "m.track1"]
    entities = pd.read_csv(export / "entity_list.txt", sep=" ")
    assert entities.org_id.is_unique
    assert entities.org_id.iloc[:2].tolist() == ["m.track1", "lfm1b.track.1002"]


def test_unparseable_fact_date_keeps_fact_with_blank_timestamp(inputs, tmp_path):
    pairs, author, freebase = inputs
    freebase.write_text(
        'subject\tpredicate\tobject\n' + GRAPH_ROWS + 'm.artist\tpeople.person.date_of_birth\t"unknown"\n',
        encoding="utf-8",
    )
    export = tmp_path / "export"
    meta = build_raw_dataset(
        pairs, tmp_path / "processed", export, freebase, item_links=author / "item_list.txt", kcore=1,
    )
    dates = pd.read_csv(export / "kg_final_timestamp.tsv", sep="\t")
    assert len(dates) == 3 and dates.ts.isna().all()
    assert dates.date_source.eq("undated").all()
    assert meta["coverage"]["dated_triples"]["t1"] == 0


def test_new_processed_kg_loads_with_existing_training_loader(inputs, tmp_path):
    pairs, author, freebase = inputs
    out = tmp_path / "processed"
    build_raw_dataset(pairs, out, tmp_path / "export", freebase, item_links=author / "item_list.txt", kcore=1)
    split, meta = load_saved({"processed_dir": out})
    encoded = encode(split)
    for cutoff in ("t1", "t2"):
        kg = load_kg(out, "default", cutoff, encoded, meta)
        dates = load_kg_dates(out, kg)
        assert len(kg) == 2 and len(dates) == len(kg)
        assert kg.time_consistent and kg.wikidata_snapshot is None
        assert kg.n_items == 6
        assert pd.isna(dates).sum() == 1
    # Reuse the old loader's val-only boundary and split-hash guards unchanged.
    load_kg(out, "default", "t1", encoded, meta, with_test=False)
    with pytest.raises(ValueError, match="only for the test"):
        load_kg(out, "default", "t2", encoded, meta, with_test=False)
    with (out / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("10,1001,1,0\n")
    with pytest.raises(ValueError, match="changed after the split"):
        load_kg(out, "default", "t1", encoded, meta)


def test_existing_training_loader_handles_new_empty_kg(inputs, tmp_path):
    pairs, author, freebase = inputs
    links = tmp_path / "empty-links.tsv"
    links.write_text("item\tfreebase_id\n", encoding="utf-8")
    out = tmp_path / "processed"
    build_raw_dataset(pairs, out, tmp_path / "export", freebase, item_links=links, kcore=1)
    split, meta = load_saved({"processed_dir": out})
    kg = load_kg(out, "default", "t1", encode(split), meta)
    assert len(kg) == 0 and len(load_kg_dates(out, kg)) == 0


def test_snapshot_guard_rejects_disappearing_facts():
    earlier = pd.DataFrame({"h": [0], "relation": ["artist"], "tail": ["m.a"]})
    later = pd.DataFrame({"h": [0, 0], "relation": ["artist", "release"], "tail": ["m.a", "m.b"]})
    check_snapshot_containment(earlier, later)
    with pytest.raises(ValueError, match="not contained"):
        check_snapshot_containment(later, earlier)


def test_default_kg_matches_old_builder_on_shared_raw_items(inputs, tmp_path, monkeypatch):
    """Exercise the actual old script, not a reimplementation of its rules."""
    pairs, author, freebase = inputs
    with (author / "item_list.txt").open("a", encoding="utf-8") as handle:
        handle.write("1003 4 m.track3\n1004 5 m.track4\n1005 6 m.track5\n1006 7 m.track6\n")
    with (author / "entity_list.txt").open("a", encoding="utf-8") as handle:
        handle.write("m.track3 4\nm.track4 5\nm.track5 6\nm.track6 7\n")
    # One literal per predicate, as required by the legacy Freebase reader.
    # Jan 10 is on t1's UTC day but before t2's UTC day: excluded at t1, kept at t2.
    freebase.write_text(
        'subject\tpredicate\tobject\n' + GRAPH_ROWS + 'm.artist\tpeople.person.date_of_birth\t"1980"\n'
        'm.future\tmusic.release.release_date\t"2014-01-10"\n', encoding="utf-8",
    )
    new_out, old_out = tmp_path / "new", tmp_path / "old"
    build_raw_dataset(pairs, new_out, tmp_path / "export", freebase, item_links=author / "item_list.txt", kcore=1)
    frame, _ = load_raw_pairs(pairs)
    old_ids = {1001: 0, 1002: 1, 1003: 4, 1004: 5, 1005: 6, 1006: 7}
    frame["item"] = frame.item.map(old_ids)
    legacy = temporal_split(frame, SplitConfig(min_rating=1, kcore=1))
    cfg = {"name": "lastfm", "processed_dir": str(old_out), "freebase": str(freebase)}
    save_split(legacy, cfg, {})
    config = tmp_path / "legacy.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "06_build_kg_lastfm.py", "--data", str(config), "--author-data", str(author), "--variant", "default",
    ])
    runpy.run_path(str(ROOT / "scripts/06_build_kg_lastfm.py"), run_name="__main__")
    for cutoff in ("t1", "t2"):
        for suffix in ("", "_dates"):
            name = f"kg_{cutoff}{suffix}.tsv"
            new_frame = pd.read_csv(new_out / "kg/default" / name, sep="\t")
            old_frame = pd.read_csv(old_out / "kg/default" / name, sep="\t")
            pd.testing.assert_frame_equal(new_frame, old_frame, check_dtype=False)
    kg_info = json.loads((new_out / "kg/default/kg_meta.json").read_text())
    assert kg_info["n_triples"] == {"t1": 2, "t2": 3}
    assert kg_info["n_entities_only_in_t2"] == 1


def test_overwrite_cannot_replace_the_original_author_files(inputs, tmp_path):
    pairs, author, freebase = inputs
    before = {path.name: path.read_bytes() for path in author.iterdir()}
    with pytest.raises(ValueError, match="source files must be preserved"):
        build_raw_dataset(
            pairs, tmp_path / "processed", author, freebase,
            item_links=author / "item_list.txt", kcore=1, overwrite=True,
        )
    assert {path.name: path.read_bytes() for path in author.iterdir()} == before


def test_cut_cannot_overwrite_its_source_zip(tmp_path):
    raw = raw_file(tmp_path / "raw.inter", [(1, 2, 3)])
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as handle:
        handle.write(raw, "raw.inter")
    before = archive.read_bytes()
    with pytest.raises(ValueError, match="source and output must differ"):
        cut_pairs(f"{archive}::raw.inter", archive, overwrite=True, progress=False)
    assert archive.read_bytes() == before


@pytest.fixture
def freebase_with_facts(inputs):
    pairs, fallback, freebase = inputs
    freebase.write_text(
        'subject\tpredicate\tobject\n'
        'm.track1\tmusic.recording.artist\tm.drive_artist\n'
        'm.track1\tmusic.recording.releases\tm.drive_release\n'
        'm.track1\tmusic.recording.releases\tm.drive_release\n'
        'm.track1\ttype.object.type\tmusic.recording\n'
        'm.track1\ttype.object.name\t"Drive Track"@en\n'
        'm.track1\tmusic.recording.length\t"280.506"\n'
        'm.track1\thttp://rdf.freebase.com/key/authority.musicbrainz.release\t"mbid-123"\n'
        'm.drive_artist\tpeople.person.date_of_birth\t"1980"\n'
        'm.drive_release\tmusic.release.release_date\t"2010-05"\n'
        'm.future\tmusic.release.release_date\t"2020-03-10"\n', encoding="utf-8",
    )
    return pairs, fallback, freebase


def test_only_paper_relations_are_rebuilt_from_drive_freebase(freebase_with_facts, tmp_path):
    pairs, mapping_dir, freebase = freebase_with_facts
    out, export = tmp_path / "processed", tmp_path / "export"
    meta = build_raw_dataset(pairs, out, export, freebase, item_links=mapping_dir / "item_list.txt", kcore=1)
    assert meta["n_items"] == 6 and sum(meta["interactions"].values()) == 12
    assert meta["coverage"]["fact_sources"] == {"freebase": 2}
    assert meta["coverage"]["triples"] == {"t1": 2, "t2": 2}
    assert meta["uses_published_kg"] is False
    sources = pd.read_csv(export / "fact_sources.tsv", sep="\t")
    assert set(sources.fact_source) == {"freebase"}
    assert set(sources.source_path) == {str(freebase)}
    assert len(sources[sources.cutoff.eq("t1")]) == 2
    assert set(sources.raw_object) == {"m.drive_artist", "m.drive_release"}
    entities = pd.read_csv(export / "entity_metadata.tsv", sep="\t")
    assert entities.label.iloc[0] == "Drive Track"  # outside-schema name still supplies a label
    assert "m.drive_artist" in set(entities.org_id)
    assert {"m.artist", "music.recording"}.isdisjoint(entities.org_id)
    relations = pd.read_csv(export / "relation_list.txt", sep=" ")
    assert relations.org_id.tolist() == [f"http://rdf.freebase.com/ns/{name}" for name in PAPER_RELATIONS]
    assert relations.remap_id.tolist() == list(range(9))
    dates = pd.read_csv(export / "kg_final_timestamp.tsv", sep="\t")
    assert set(dates.ts) == {utc_seconds("1980-12-31"), utc_seconds("2010-05-31")}
    report = json.loads((out / "freebase_source_report.json").read_text())
    assert report["rows_scanned"] == 10 and report["duplicate_fact_rows"] == 1
    assert report["relation_not_selected"] == 4
    assert (report["outside_linked_train_heads"] + report["relation_not_selected"]
            + report["unique_facts"] + report["duplicate_fact_rows"]) == 10
    assert report["selected_relation_counts"]["music.recording.artist"] == 1
    assert report["selected_relation_counts"]["music.recording.canonical_version"] == 0
    assert "music.recording.canonical_version" in report["relations_without_linked_train_facts"]


def test_freebase_build_needs_only_mapping_and_not_old_graph_files(freebase_with_facts, tmp_path):
    pairs, mapping_dir, freebase = freebase_with_facts
    for name in ("kg_final.txt", "entity_list.txt", "relation_list.txt"):
        (mapping_dir / name).unlink()
    out, export = tmp_path / "p", tmp_path / "e"
    meta = build_raw_dataset(pairs, out, export, freebase, item_links=mapping_dir / "item_list.txt", kcore=1)
    assert meta["coverage"]["fact_sources"] == {"freebase": 2}
    assert meta["uses_published_kg"] is False
    split, saved = load_saved({"processed_dir": out})
    kg = load_kg(out, "default", "t1", encode(split), saved)
    assert len(kg) == 2 and len(load_kg_dates(out, kg)) == 2


def test_changes_to_old_graph_cannot_supply_missing_relations(freebase_with_facts, tmp_path):
    pairs, mapping_dir, freebase = freebase_with_facts
    first = build_raw_dataset(pairs, tmp_path / "p1", tmp_path / "e1", freebase,
                              item_links=mapping_dir / "item_list.txt", kcore=1)
    for name in ("kg_final.txt", "entity_list.txt", "relation_list.txt"):
        (mapping_dir / name).write_text("invalid old graph data", encoding="utf-8")
    second = build_raw_dataset(pairs, tmp_path / "p2", tmp_path / "e2", freebase,
                               item_links=mapping_dir / "item_list.txt", kcore=1)
    assert first["sha256"] == second["sha256"]
    assert second["coverage"]["fact_sources"] == {"freebase": 2}


def test_other_predicates_are_not_relabelled_as_paper_relations(tmp_path):
    freebase = tmp_path / "facts.tsv"
    freebase.write_text(
        'subject\tpredicate\tobject\n'
        'm.track\ttype.object.type\tmusic.recording\n'
        'm.track\tmusic.release.track\tm.other\n'
        'm.track\tmusic.release.release_date\t"2030-05"\n', encoding="utf-8",
    )
    result = read_freebase_source(freebase, {"m.track"})
    assert result.graph.empty
    assert result.report["relation_not_selected"] == 3
    assert len(result.report["relations_without_linked_train_facts"]) == 9
    assert result.evidence.loc["m.track", "tail_release_date"] == '\"2030-05\"'


def test_freebase_hashes_do_not_depend_on_source_row_order(freebase_with_facts, tmp_path):
    pairs, fallback, freebase = freebase_with_facts
    first = build_raw_dataset(
        pairs, tmp_path / "p1", tmp_path / "e1", freebase, item_links=fallback / "item_list.txt", kcore=1,
    )
    lines = freebase.read_text().splitlines(keepends=True)
    freebase.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8")
    second = build_raw_dataset(
        pairs, tmp_path / "p2", tmp_path / "e2", freebase, item_links=fallback / "item_list.txt", kcore=1,
    )
    assert first["sha256"] == second["sha256"]


def test_drive_defaults_never_require_an_external_paper_clone():
    from tkgat.pipeline.lastfm_raw import ITEM_LINKS_DRIVE

    assert ITEM_LINKS_DRIVE.startswith("gdrive://")
    entrypoint = (ROOT / "scripts/02b_build_lastfm_from_raw.py").read_text()
    assert "knowledge_graph_attention_network/Data" not in entrypoint
    for removed in ("--metadata-dir", "--tracks", "--artists", "--albums", "--events",
                    "--fallback-data", "--author-data", "--no-paper-fallback"):
        assert removed not in entrypoint


def test_cli_reads_the_supplied_archive_layout_without_external_metadata(freebase_with_facts, tmp_path):
    pairs, fallback, freebase = freebase_with_facts
    archive = tmp_path / "support.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(freebase, "freebase/freebase-subset.tsv")
        for filename in ("item_list.txt", "kg_final.txt", "entity_list.txt", "relation_list.txt"):
            zipped.write(fallback / filename, "kgat-output/last-fm-temporal/" + filename)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/02b_build_lastfm_from_raw.py"),
         "--kg-mode", "one-hop", "--pairs", str(pairs),
         "--freebase", f"{archive}::freebase/freebase-subset.tsv",
         "--item-links", f"{archive}::kgat-output/last-fm-temporal/item_list.txt", "--kcore", "1",
         "--out", str(tmp_path / "p"), "--export-dir", str(tmp_path / "e")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "e/dataset_meta.json").read_text())
    assert meta["coverage"]["fact_sources"] == {"freebase": 2}
    assert "support.zip::" in meta["item_links"]


