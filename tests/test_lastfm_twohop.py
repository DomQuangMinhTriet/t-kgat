"""Raw Freebase provenance, bidirectional depth, cutoff leakage and loader compatibility."""

import gzip
import runpy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tkgat.data.encode import encode
from tkgat.data.lastfm_raw import cut_pairs, utc_seconds
from tkgat.kg.ckg import KGTriples, build_ckg, load_kg, load_kg_dates, used_tails
from tkgat.kg.freebase_dump import extract_dump, validate_extract
from tkgat.kg.lastfm_freebase import read_freebase_source
from tkgat.kg.lastfm_twohop import build_twohop
from tkgat.pipeline.build import load_saved
from tkgat.pipeline.lastfm_raw import build_raw_dataset

NS = "http://rdf.freebase.com/ns/"


def nt(head, relation, tail, *, literal=False):
    obj = tail if literal else f"<{NS}{tail}>"
    return f"<{NS}{head}>\t<{NS}{relation}>\t{obj}\t.\n"


@pytest.fixture
def sources(tmp_path):
    links, dump = tmp_path / "links.tsv", tmp_path / "dump.nt"
    links.write_text("item\tfreebase_id\n1001\tm.track\n1002\tm.track\n", encoding="utf-8")
    rows = [
        nt("m.artist", "people.person.date_of_birth", '"1980"', literal=True),
        nt("m.parent", "music.recording.artist", "m.artist"),  # second hop occurs before first in the file
        nt("m.artist", "music.recording.releases", "m.too_far"),  # third hop: must not leak into cache
        nt("m.parent", "music.recording.canonical_version", "m.track"),  # incoming first hop
        nt("m.track", "music.recording.releases", "m.release"),
        nt("m.release", "music.release.release_date", '"2020-05"', literal=True),
        nt("m.future", "music.recording.canonical_version", "m.track"),
        nt("m.future", "music.release.release_date", '"2020"', literal=True),
        nt("m.parent", "type.object.name", '"Parent name"@en', literal=True),
        nt("m.track", "type.object.type", "music.recording"),  # never relabel this as notable_types
        nt("m.parent", "music.recording.canonical_version", "m.track"),  # exact duplicate
    ]
    dump.write_text("".join(rows), encoding="utf-8", newline="\n")
    return links, dump


@pytest.mark.parametrize("compressed", [False, True])
def test_dump_extraction_keeps_incoming_two_hops_and_evidence_only(sources, tmp_path, compressed):
    links, dump = sources
    if compressed:
        zipped = tmp_path / "dump.nt.gz"
        zipped.write_bytes(gzip.compress(dump.read_bytes()))
        dump = zipped
    out = tmp_path / "freebase.tsv"
    meta = extract_dump(dump, links, out)
    assert meta["resource_facts"] == 4 and meta["source_is_paper_kg"] is False
    assert meta["source_dump_saved_locally"] is False
    assert len(meta["passes"]) == 3
    assert meta["passes"][0]["duplicates_collapsed"] == 1
    text = out.read_text()
    assert "m.parent\tmusic.recording.artist" in text
    assert "m.parent\tmusic.recording.canonical_version\t" in text
    assert "m.too_far" not in text and "type.object.type" not in text
    assert '"1980"' in text and '"2020-05"' in text and '"Parent name"@en' in text
    assert validate_extract(out)["sha256"] == meta["sha256"]


def test_dump_row_order_does_not_change_extracted_data_hash(sources, tmp_path):
    links, dump = sources
    one = extract_dump(dump, links, tmp_path / "one.tsv")
    dump.write_text("".join(reversed(dump.read_text().splitlines(keepends=True))), encoding="utf-8", newline="\n")
    two = extract_dump(dump, links, tmp_path / "two.tsv")
    assert one["sha256"] == two["sha256"]
    assert one["source_fingerprint"] != two["source_fingerprint"]


def test_changed_dump_between_passes_never_publishes_extraction(sources, tmp_path, monkeypatch):
    from tkgat.kg import freebase_dump

    links, dump = sources
    actual = freebase_dump.dump_text
    calls = []

    @contextmanager
    def changed(source, fingerprint):
        with actual(source, fingerprint) as handle:
            yield handle
        calls.append(1)
        if len(calls) == 2:
            fingerprint["sha256"] = "changed"

    monkeypatch.setattr(freebase_dump, "dump_text", changed)
    out = tmp_path / "out.tsv"
    with pytest.raises(ValueError, match="changed between"):
        extract_dump(dump, links, out)
    assert not out.exists() and not Path(str(out) + ".meta.json").exists()


def test_truncated_gzip_cannot_publish_partial_facts(sources, tmp_path):
    links, dump = sources
    bad = tmp_path / "truncated.gz"
    bad.write_bytes(gzip.compress(dump.read_bytes())[:-12])
    with pytest.raises((EOFError, OSError)):
        extract_dump(bad, links, tmp_path / "partial.tsv")
    assert not (tmp_path / "partial.tsv").exists()


def test_twohop_dates_check_future_heads_and_retain_undated_edges(sources, tmp_path):
    links, dump = sources
    cache = tmp_path / "cache.tsv"
    extract_dump(dump, links, cache)
    primary = read_freebase_source(cache, {"m.track"}, all_heads=True)
    items = pd.DataFrame({"h": [0, 1], "item": [1001, 1002], "freebase_id": ["m.track", "m.track"]})
    graph = build_twohop(items, primary.graph, primary.evidence,
                         {"t1": utc_seconds("2014-01-01"), "t2": utc_seconds("2022-01-01")},
                         entity_min=1, relation_min=1)
    assert len(graph.kept["t1"]) == 3  # incoming edge has two item aliases, plus second-hop artist edge
    assert graph.kept["t1"].ts.isna().sum() == 2
    assert "future_head" in set(graph.dropped["t1"].reason)
    assert "future_tail" in set(graph.dropped["t1"].reason)
    assert set(graph.triples["t1"].t) >= {0, 1}
    assert (graph.triples["t1"].h >= 2).all()  # non-item heads are preserved
    assert graph.report["frequency_reference"].startswith("t1 only")
    assert set(graph.kept["t1"]["head"]) == {"m.parent"}


def test_future_facts_cannot_boost_quality_thresholds():
    graph = pd.DataFrame({
        "head": ["m.a", "m.a"], "relation": ["music.recording.artist"] * 2,
        "tail": ["m.b", "m.future"], "tail_kind": ["freebase"] * 2,
        "tail_label": [""] * 2, "fact_source": ["freebase"] * 2,
        "source_path": ["original"] * 2, "raw_object": ["m.b", "m.future"],
    })
    evidence = pd.DataFrame({"tail_release_date": ["", '"2020"'], "tail_active_start": ["", ""],
                             "tail_date_of_birth": ["", ""], "tail_label": ["", ""]}, index=["m.b", "m.future"])
    result = build_twohop(pd.DataFrame({"h": [0], "item": [1], "freebase_id": ["m.a"]}), graph, evidence,
                          {"t1": utc_seconds("2014-01-01"), "t2": utc_seconds("2022-01-01")},
                          entity_min=1, relation_min=2)
    assert result.report["relation_counts_after_entity_filter_t1"]["music.recording.artist"] == 1
    assert result.triples["t1"].empty and result.triples["t2"].empty


def test_full_twohop_pipeline_and_training_loader(sources, tmp_path):
    links, dump = sources
    cache = tmp_path / "freebase.tsv"
    extract_dump(dump, links, cache)
    raw, pairs = tmp_path / "events.inter", tmp_path / "pairs.tsv"
    base = utc_seconds("2014-01-01")
    raw.write_text("user_id\ttracks_id\ttimestamp\n" + "".join(
        f"{u}\t{1000+i}\t{base+(j*6+i)*86400}\n" for j, u in enumerate((10, 20)) for i in range(1, 7)
    ), encoding="utf-8")
    cut_pairs(raw, pairs, progress=False)
    kwargs = {"item_links": links, "kcore": 1, "kg_mode": "two-hop", "kg_entity_min": 1, "kg_relation_min": 1}
    meta = build_raw_dataset(pairs, tmp_path / "p", tmp_path / "e", cache, **kwargs)
    assert meta["uses_published_kg"] is False and meta["graph_scope"] == "two-hop"
    assert meta["freebase_extraction"]["source"] == str(dump)
    assert meta["interactions"] == {"train": 9, "val": 1, "test": 2}
    split, saved = load_saved({"processed_dir": tmp_path / "p"})
    kg = load_kg(tmp_path / "p", "default", "t1", encode(split), saved)
    assert kg.graph_scope == "two-hop"
    assert len(kg) == len(load_kg_dates(tmp_path / "p", kg)) == 3
    assert (kg.triples[:, 0] >= kg.n_items).any()
    assert (kg.triples[:, 2] < kg.n_items).any()
    assert build_ckg(encode(split), kg).n_edges == 2 * (9 + 3)
    if pytest.importorskip("duckdb"):
        other = build_raw_dataset(pairs, tmp_path / "p2", tmp_path / "e2", cache, engine="duckdb", **kwargs)
        assert other["sha256"] == meta["sha256"]


def test_twohop_scope_preserves_legacy_guards_and_rejects_untrained_test_heads():
    rows = np.array([[1, 0, 0]], dtype="int64")
    with pytest.raises(ValueError, match="head outside"):
        KGTriples("default", "t1", rows, 1, 3, ("artist",), True, None)
    train = KGTriples("default", "t1", np.array([[0, 0, 1]], dtype="int64"), 1, 3,
                      ("artist",), True, None, graph_scope="two-hop")
    test = KGTriples("default", "t2", np.array([[0, 0, 1], [2, 0, 0]], dtype="int64"), 1, 3,
                     ("artist",), True, None, graph_scope="two-hop")
    enc = SimpleNamespace(n_items=1, n_users=1, train=np.array([[0, 0]], dtype="int64"), train_ts=np.array([10]))
    ckg = build_ckg(enc, test, known_tails=used_tails(train))
    assert ckg.n_edges == 4  # one interaction plus one KG edge, each with its inverse


def test_modified_extracted_cache_is_rejected_before_any_output(sources, tmp_path):
    links, dump = sources
    cache = tmp_path / "freebase.tsv"
    extract_dump(dump, links, cache)
    cache.write_text(cache.read_text() + 'm.extra\ttype.object.name\t"bad"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_extract(cache)


def test_default_mirror_fingerprint_is_enforced_before_publishing(sources, tmp_path, monkeypatch):
    from tkgat.kg import freebase_dump

    links, dump = sources
    monkeypatch.setattr(freebase_dump, "DEFAULT_DUMP", str(dump))
    with pytest.raises(ValueError, match="pinned size/SHA-1"):
        extract_dump(dump, links, tmp_path / "out.tsv")
    assert not (tmp_path / "out.tsv").exists()


def test_cli_defaults_to_verified_twohop_and_fails_before_reading_pairs(tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/02b_build_lastfm_from_raw.py"
    main = runpy.run_path(str(script))["main"]
    with pytest.raises(SystemExit) as error:
        main(["--pairs", str(tmp_path / "missing-pairs.tsv"), "--freebase", str(tmp_path / "missing-cache.tsv"),
              "--out", str(tmp_path / "processed"), "--export-dir", str(tmp_path / "export")])
    assert error.value.code == 2
    assert "01c_extract_freebase.py first" in capsys.readouterr().err
    assert not (tmp_path / "processed").exists() and not (tmp_path / "export").exists()
