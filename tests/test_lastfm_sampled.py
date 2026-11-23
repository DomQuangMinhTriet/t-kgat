"""Train-only, paper-scale user sampling with an iterative 10-core and compact KG ids."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tkgat.data.encode import encode
from tkgat.data.lastfm_raw import cut_pairs, load_raw_pairs, utc_seconds, validate_pair_cache
from tkgat.data.leakage import check_split
from tkgat.data.split import kcore
from tkgat.kg.ckg import build_ckg, load_kg, load_kg_dates
from tkgat.kg.freebase_dump import extract_dump
from tkgat.kg.lastfm_freebase import read_freebase_source
from tkgat.kg.lastfm_twohop import build_twohop
from tkgat.pipeline.build import load_saved
from tkgat.pipeline.lastfm_raw import build_raw_dataset

from .conftest import ROOT
from .test_lastfm_twohop import sources as _sources

sources = _sources


@pytest.fixture
def sampled_inputs(tmp_path):
    pytest.importorskip("duckdb")
    rows = []
    for u in range(40):
        for j in range(20):
            ts = 1000 + u * 14 + j if j < 14 else (2000 if j < 17 else 3000) + u * 3 + j % 3
            rows.append((u, (u * 3 + j) % 20, ts))
    raw = tmp_path / "events.inter"
    raw.write_text("user\titem\tts\n" + "".join(f"{u}\t{i}\t{ts}\n" for u, i, ts in rows))
    pairs = tmp_path / "pairs.tsv"
    cut_pairs(raw, pairs, progress=False)
    links = tmp_path / "links.tsv"
    links.write_text("item\tfreebase_id\n")
    freebase = tmp_path / "fb.tsv"
    freebase.write_text("subject\tpredicate\tobject\nm.x\tmusic.recording.artist\tm.y\n")
    return pairs, links, freebase


def build_sample(inputs, root, *, seed=2019, target=360):
    pairs, links, freebase = inputs
    root.mkdir()
    meta = build_raw_dataset(
        pairs, root / "processed", root / "export", freebase,
        item_links=links, q_val=0.7, q_test=0.85, kcore=10, engine="auto",
        memory_limit="256MB", target_interactions=target, user_seed=seed,
    )
    split, _ = load_saved({"processed_dir": root / "processed"})
    return split, meta


def test_sampled_core_cutoffs_complete_masks_and_full_source_accounting(sampled_inputs, tmp_path):
    split, meta = build_sample(sampled_inputs, tmp_path / "built")
    frame, _ = load_raw_pairs(sampled_inputs[0])
    sampled_users = pd.read_csv(tmp_path / "built/processed/sampled_users.tsv", sep="\t").user
    assert len(sampled_users) < frame.user.nunique()
    assert (split.t1, split.t2) == tuple(int(np.quantile(frame.ts, q)) for q in (0.7, 0.85))
    assert check_split(split) == []
    history = frame[frame.user.isin(sampled_users)]
    expected = kcore(history[history.ts < split.t1], 10)
    pd.testing.assert_frame_equal(
        split.train.sort_values(["user", "item"]).reset_index(drop=True),
        expected.sort_values(["user", "item"]).reset_index(drop=True),
    )
    assert split.train.groupby("user").item.nunique().min() >= 10
    assert split.train.groupby("item").user.nunique().min() >= 10
    for name, cutoff in (("seen_before_val", split.t1), ("seen_before_test", split.t2)):
        want = history.loc[history.ts < cutoff, ["user", "item", "ts"]].sort_values(["ts", "user", "item"])
        pd.testing.assert_frame_equal(getattr(split, name).reset_index(drop=True), want.reset_index(drop=True))
    report = meta["interaction_audit"]
    assert report["unexplained_pairs"] == 0
    assert sum(report["kept"].values()) + report["dropped_pairs"] == len(frame)
    outside = sum(row["outside_user_sample"] for row in report["dropped_by_reason"].values())
    assert outside == (~frame.user.isin(sampled_users)).sum()
    assert report["user_sampling"]["actual_train_pairs"] == len(split.train)
    assert abs(len(split.train) - 252) < 30
    dropped = pd.read_csv(tmp_path / "built/processed/dropped_interactions.csv")
    assert "outside_user_sample" not in set(dropped.reason)


def test_user_selection_is_reproducible_and_seed_changes_membership(sampled_inputs, tmp_path):
    for name, seed in (("one", 2019), ("two", 2019), ("other", 2020)):
        build_sample(sampled_inputs, tmp_path / name, seed=seed)
    a = (tmp_path / "one/processed/sampled_users.tsv").read_bytes()
    assert a == (tmp_path / "two/processed/sampled_users.tsv").read_bytes()
    assert a != (tmp_path / "other/processed/sampled_users.tsv").read_bytes()
    for filename in ("train.csv", "val.csv", "test.csv", "seen_before_val.csv", "seen_before_test.csv"):
        assert (tmp_path / "one/processed" / filename).read_bytes() == (
            tmp_path / "two/processed" / filename
        ).read_bytes()


def test_future_id_changes_cannot_change_selected_users_or_train(sampled_inputs, tmp_path):
    pairs, links, freebase = sampled_inputs
    original, _ = load_raw_pairs(pairs)
    build_sample(sampled_inputs, tmp_path / "before")
    future = original.ts >= int(np.quantile(original.ts, 0.7))
    changed = original.copy()
    changed.loc[future, "user"] = 900000 + np.arange(future.sum())
    changed.loc[future, "item"] = 800000 + np.arange(future.sum())
    raw = tmp_path / "future.inter"
    changed[["user", "item", "ts"]].to_csv(raw, sep="\t", index=False)
    changed_pairs = tmp_path / "future-pairs.tsv"
    cut_pairs(raw, changed_pairs, progress=False)
    build_sample((changed_pairs, links, freebase), tmp_path / "after")
    for filename in ("sampled_users.tsv", "train.csv"):
        assert (tmp_path / "before/processed" / filename).read_bytes() == (
            tmp_path / "after/processed" / filename
        ).read_bytes()


def test_sampled_prefix_is_the_closest_nonempty_train_core(sampled_inputs, tmp_path):
    split, _ = build_sample(sampled_inputs, tmp_path / "built")
    frame, _ = load_raw_pairs(sampled_inputs[0])
    train = kcore(frame[frame.ts < split.t1], 10)
    users = sorted(train.user.unique(), key=lambda u: (hashlib.sha256(f"2019:{u}".encode()).digest(), u))
    distances = []
    for prefix in range(1, len(users) + 1):
        size = len(kcore(train[train.user.isin(users[:prefix])], 10))
        if size:
            distances.append(abs(size - int(360 * 0.7)))
    assert abs(len(split.train) - int(360 * 0.7)) == min(distances)


def test_sampled_twohop_pipeline_exports_aligned_compact_ids(sampled_inputs, sources, tmp_path):
    pairs, _, _ = sampled_inputs
    frame, _ = load_raw_pairs(pairs)
    frame["item"] += 1000
    frame["ts"] += utc_seconds("2014-01-01")
    raw, pairs = tmp_path / "dated.inter", tmp_path / "dated-pairs.tsv"
    frame[["user", "item", "ts"]].to_csv(raw, sep="\t", index=False)
    cut_pairs(raw, pairs, progress=False)
    links, dump = sources
    cache = tmp_path / "freebase.tsv"
    extract_dump(dump, links, cache)
    output, export = tmp_path / "processed", tmp_path / "export"
    result = build_raw_dataset(
        pairs, output, export, cache, item_links=links, kg_mode="two-hop", kcore=10,
        q_val=0.7, q_test=0.85, target_interactions=360, kg_entity_min=1, kg_relation_min=1,
        engine="auto", memory_limit="256MB",
    )
    split, metadata = load_saved({"processed_dir": output})
    encoded = encode(split)
    kg = load_kg(output, "default", "t1", encoded, metadata)
    timestamps = load_kg_dates(output, kg)
    graph = build_ckg(encoded, kg, kg_ts=timestamps)
    assert graph.n_triples > 0 and len(timestamps) == len(kg)
    entities = pd.read_csv(output / "kg/default/entities.csv")
    assert {"m.future", "m.release"}.isdisjoint(entities.key)
    used = set(range(encoded.n_items)) | set(kg.triples[:, 0]) | set(kg.triples[:, 2])
    assert used == set(entities.entity)
    final = pd.read_csv(export / "kg_final.txt", sep=" ", header=None)
    dated = pd.read_csv(export / "kg_final_timestamp.tsv", sep="\t")
    assert np.array_equal(final.values, dated[["h", "r", "t"]].values)
    assert result["user_sampling"]["actual_train_pairs"] == len(split.train)


def test_compacted_ids_remove_unused_entities_without_changing_facts_or_dates(sources, tmp_path):
    links, dump = sources
    cache = tmp_path / "cache.tsv"
    extract_dump(dump, links, cache)
    data = read_freebase_source(cache, {"m.track"}, all_heads=True)
    items = pd.DataFrame({"h": [0, 1], "item": [1001, 1002], "freebase_id": ["m.track", "m.track"]})
    options = {"entity_min": 1, "relation_min": 1}
    cutoffs = {"t1": utc_seconds("2014-01-01"), "t2": utc_seconds("2015-01-01")}
    full = build_twohop(items, data.graph, data.evidence, cutoffs, **options)
    compact = build_twohop(items, data.graph, data.evidence, cutoffs, compact_entities=True, **options)
    assert len(compact.entities) < len(full.entities)
    assert compact.entities.entity.tolist() == list(range(len(compact.entities)))
    used = set(range(len(items)))
    for cut in ("t1", "t2"):
        assert list(zip(compact.kept[cut]["head"], compact.kept[cut]["tail"], strict=True)) == list(
            zip(full.kept[cut]["head"], full.kept[cut]["tail"], strict=True)
        )
        assert compact.kept[cut].ts.tolist() == full.kept[cut].ts.tolist()
        used.update(compact.triples[cut].h)
        used.update(compact.triples[cut].t)
        assert compact.dates[cut][["h", "r", "t"]].equals(compact.triples[cut])
    assert used == set(compact.entities.entity)
    assert compact.dropped["t1"]["head"].notna().all()
    assert compact.dropped["t1"][["h", "t"]].isna().any().any()


def test_01b_reuse_verifies_cache_and_never_opens_source_again(sampled_inputs, monkeypatch):
    pairs, _, _ = sampled_inputs
    meta = json.loads(Path(str(pairs) + ".meta.json").read_text())
    assert validate_pair_cache(pairs, meta["source"])["sha256"] == meta["sha256"]
    main = runpy.run_path(str(ROOT / "scripts/01b_cut_lastfm.py"))["main"]

    def forbidden(*args, **kwargs):
        raise AssertionError("raw data must not be streamed again")

    monkeypatch.setitem(main.__globals__, "cut_pairs", forbidden)
    monkeypatch.setattr("sys.argv", ["01b", "--interactions", meta["source"], "--out", str(pairs), "--reuse-existing"])
    main()
    with pytest.raises(ValueError, match="other source/window"):
        validate_pair_cache(pairs, meta["source"], end=123)
    pairs.write_text(pairs.read_text() + "999\t999\t999\n")
    with pytest.raises(ValueError, match="hash"):
        validate_pair_cache(pairs, meta["source"])


def test_01c_reuse_does_not_scan_dump_and_rejects_mapping_changes(sources, tmp_path, monkeypatch):
    links, dump = sources
    cache = tmp_path / "cache.tsv"
    extract_dump(dump, links, cache)
    main = runpy.run_path(str(ROOT / "scripts/01c_extract_freebase.py"))["main"]

    def forbidden(*args, **kwargs):
        raise AssertionError("full dump must not be scanned again")

    monkeypatch.setitem(main.__globals__, "extract_dump", forbidden)
    args = ["--dump", str(dump), "--item-links", str(links), "--out", str(cache), "--reuse-existing"]
    main(args)
    links.write_text("item\tfreebase_id\n1001\tm.changed\n")
    with pytest.raises(SystemExit):
        main(args)
