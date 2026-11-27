"""Raw-event, source-provenance and leakage regressions for the data-only port."""

import gzip
import json
import runpy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from tkgat.data import drive
from tkgat.data import protocol_v2 as protocol
from tkgat.data.lastfm_check import check_clean_dataset
from tkgat.data.lastfm_raw import cut_pairs, load_raw_pairs, source_sha256, validate_pair_cache
from tkgat.kg.freebase_dump import extract_dump, validate_extract
from tkgat.kg.lastfm import DATE_COLUMNS
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS, read_freebase_source
from tkgat.kg.lastfm_twohop import build_twohop, reachable
from tkgat.pipeline.build import load_saved
from tkgat.pipeline.lastfm_raw import build_raw_dataset, read_item_links
from tkgat.registry import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def rdf(head, relation, tail, *, literal=False):
    obj = tail if literal else f"<http://rdf.freebase.com/ns/{tail}>"
    return f"<http://rdf.freebase.com/ns/{head}>\t<http://rdf.freebase.com/ns/{relation}>\t{obj}\t.\n"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # A dense enough core to survive k=10, with warm and cold held-out pairs.
    rng = np.random.default_rng(5)
    times = rng.permutation(750)
    rows = [(100 + u, 1000 + i, protocol.WINDOW_START + int(times[u * 30 + i]) * 1500)
            for u in range(25) for i in range(30)]
    rows += [(100, 1000, protocol.WINDOW_START - 1), (100, 1000, protocol.WINDOW_END),
             (999, 1000, protocol.WINDOW_END - 1), (100, 9999, protocol.WINDOW_END - 2)]
    raw = tmp_path / "lfm1b-tracks.inter"
    raw.write_text("user_id:token\ttracks_id:token\ttimestamp:float\n" + "".join(
        f"{u}\t{i}\t{ts}\n" for u, i, ts in reversed(rows)), encoding="utf-8")
    pairs = tmp_path / "pairs.tsv"
    cut_pairs(raw, pairs, start=protocol.WINDOW_START, end=protocol.WINDOW_END, chunksize=91, progress=False)
    links = tmp_path / "item_links.tsv"
    links.write_text("item_id:token\tentity_id:token\n1000\t/m/track0\n1001\t/m/track1\n", encoding="utf-8")
    # Only the expected source SIZE is reduced in fixtures; the production gate
    # still requires the full mapping. All other v2 rules are exercised unchanged.
    monkeypatch.setattr(protocol, "MAPPING_TRACKS", 2)
    rdf_path = tmp_path / "freebase.nt.gz"
    pair_data, _ = load_raw_pairs(pairs)
    t1 = int(np.quantile(pair_data.ts, 0.8))
    # Existing conservative KG rule excludes dates on the cutoff's UTC day.
    between = pd.Timestamp(t1, unit="s", tz="UTC").strftime('%Y-%m-%d')
    rows = [
        rdf("m.track0", "music.recording.artist", "m.artist"),
        rdf("m.track1", "music.recording.releases", "m.old_release"),
        rdf("m.neighbor", "music.recording.artist", "m.artist"),  # incoming second hop
        rdf("m.too_far", "music.recording.artist", "m.neighbor"),  # third hop, excluded
        rdf("m.track0", "music.recording.releases", "m.new_release"),
        rdf("m.track0", "music.recording.releases", "m.future_release"),
        rdf("m.artist", "people.person.date_of_birth", '"1970-02-03"', literal=True),
        rdf("m.new_release", "music.release.release_date", f'"{between}"', literal=True),
        rdf("m.future_release", "music.release.release_date", '"2016-02-03"', literal=True),
        rdf("m.unrelated", "music.recording.artist", "m.elsewhere"),
    ]
    with gzip.open(rdf_path, "wt", encoding="utf-8") as handle:
        handle.write("".join(rows))
    freebase = tmp_path / "freebase.tsv"
    extract_dump(rdf_path, links, freebase, snapshot_date=protocol.SNAPSHOT, train_pairs=pairs)
    return {"raw": raw, "pairs": pairs, "links": links, "freebase": freebase, "dump": rdf_path}


def build(inputs, root, engine="pandas", **kwargs):
    return build_raw_dataset(inputs["pairs"], root / "processed", root / "export", inputs["freebase"],
                             item_links=inputs["links"], engine=engine, protocol_v2=True, **kwargs)


def test_constants_match_decision():
    text = (ROOT / "docs/decisions/0021-protocol-v2.md").read_text(encoding="utf-8")
    assert "14 days, 2014-03-18 to 2014-04-01 (end exclusive)" in text
    assert "10-core, iterative, on train only" in text
    assert "0.8 and 0.9 quantiles" in text
    assert "--target-interactions 0" in text
    assert "snapshot of 2015-04-16" in text
    assert f"{protocol.MAPPING_TRACKS:,} tracks" in text
    assert (protocol.WINDOW_END - protocol.WINDOW_START) == 14 * 86400
    assert (protocol.KCORE, protocol.Q_VAL, protocol.Q_TEST) == (10, 0.8, 0.9)


def test_earliest_is_selected_inside_window_not_from_history(inputs):
    pairs, meta = load_raw_pairs(inputs["pairs"])
    assert meta["events_before_start"] == meta["events_at_or_after_end"] == 1
    assert meta["accounting"]["output_pairs"] == len(pairs)
    assert pairs.ts.is_monotonic_increasing
    row = pairs[(pairs.user == 100) & (pairs.item == 1000)].iloc[0]
    assert row.ts >= protocol.WINDOW_START
    with pytest.raises(ValueError, match="other source/window"):
        validate_pair_cache(inputs["pairs"], inputs["raw"])


@pytest.mark.parametrize("engine", ["pandas", "duckdb"])
def test_full_pipeline_and_independent_checker(inputs, tmp_path, engine):
    root = tmp_path / engine
    meta = build(inputs, root, engine)
    split, _ = load_saved({"processed_dir": str(root / "processed")})
    assert split.config.kcore == 10
    assert split.train.groupby("user").size().min() >= 10
    assert split.train.groupby("item").size().min() >= 10
    assert set(split.val.user) <= set(split.train.user)
    assert set(split.test.item) <= set(split.train.item)
    assert meta["protocol_v2"]["target_interactions"] == 0
    assert meta["coverage"]["items_without_links"] > 0
    assert meta["coverage"]["item_mid_link_rate"] == 2 / 30
    assert tuple(pd.read_csv(root / "processed/kg/default/relations.csv").name) == PAPER_RELATIONS
    assert sum(v["dropped"] for v in meta["coverage"]["evaluation_retention"].values()) == 2
    checked = check_clean_dataset(root / "processed", root / "export")
    assert checked["passed"]
    assert checked["coverage"]["item_mid_link_rate"] < 1  # low coverage is allowed
    assert checked["checked_rows"]["train"] == len(split.train)


def test_temporal_twohop_is_train_seeded_and_keeps_new_t2_edges(inputs, tmp_path):
    meta = build(inputs, tmp_path / "build")
    sources = pd.read_csv(tmp_path / "build/export/fact_sources.tsv", sep="\t")
    assert not sources.raw_object.str.contains("m.too_far|m.elsewhere|m.future_release", regex=True).any()
    assert sources.raw_object.str.contains("m.artist", regex=False).any()
    assert not sources[sources.cutoff == "t1"].raw_object.str.contains("m.new_release", regex=False).any()
    assert sources[sources.cutoff == "t2"].raw_object.str.contains("m.new_release", regex=False).any()
    assert meta["kg_construction"]["entity_min"] == meta["kg_construction"]["relation_min"] == 1
    assert meta["coverage"]["triples"]["t2"] > meta["coverage"]["triples"]["t1"]
    dates = pd.read_csv(tmp_path / "build/processed/kg/default/kg_t2_dates.tsv", sep="\t")
    assert dates.ts.isna().any()  # undated facts stay undated


def test_both_engines_write_identical_artifact_hashes(inputs, tmp_path):
    a = build(inputs, tmp_path / "a", "pandas")
    b = build(inputs, tmp_path / "b", "duckdb")
    assert a["sha256"] == b["sha256"]
    assert a["coverage"] == b["coverage"]


@pytest.mark.parametrize("changed", [
    {"kcore": 5}, {"q_val": 0.7}, {"target_interactions": 100}, {"kg_entity_min": 10}, {"kg_relation_min": 50},
])
def test_unapproved_rules_fail_before_outputs(inputs, tmp_path, changed):
    root = tmp_path / "refused"
    with pytest.raises(ValueError, match="0021 requires"):
        build(inputs, root, **changed)
    assert not (root / "processed").exists()


def test_history_pair_cache_is_refused(inputs, tmp_path):
    manifest = Path(str(inputs["pairs"]) + ".meta.json")
    meta = json.loads(manifest.read_text())
    meta["start_inclusive"] = None
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="rebuild 01b from raw"):
        build(inputs, tmp_path / "refused")


def test_old_mapping_and_wrong_dump_snapshot_are_refused(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "MAPPING_TRACKS", 1_254_923)
    with pytest.raises(ValueError, match="new mapping"):
        build(inputs, tmp_path / "refused")
    monkeypatch.setattr(protocol, "MAPPING_TRACKS", 2)
    manifest = Path(str(inputs["freebase"]) + ".meta.json")
    meta = json.loads(manifest.read_text())
    meta["source_snapshot"] = "2015-08-09"
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="snapshot 2015-04-16"):
        build(inputs, tmp_path / "refused")


def test_mapping_change_with_same_mid_set_is_detected(inputs, tmp_path):
    inputs["links"].write_text("item\tfreebase_id\n1001\tm.track0\n1000\tm.track1\n")
    with pytest.raises(ValueError, match="another mapping"):
        build(inputs, tmp_path / "refused")


@pytest.mark.parametrize("area,name", [("processed", "train.csv"), ("export", "item_list.txt")])
def test_checker_detects_tampered_artifacts(inputs, tmp_path, area, name):
    root = tmp_path / "build"
    build(inputs, root)
    with (root / area / name).open("a") as handle:
        handle.write("corruption\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check_clean_dataset(root / "processed", root / "export")


def test_checker_detects_changed_split_even_if_hash_is_updated(inputs, tmp_path):
    root = tmp_path / "build"
    build(inputs, root)
    train = root / "processed/train.csv"
    rows = pd.read_csv(train)
    rows.loc[0, "ts"] += 1
    rows.to_csv(train, index=False, lineterminator="\n")
    manifest = root / "processed/meta.json"
    meta = json.loads(manifest.read_text())
    meta["artifact_sha256"]["train.csv"] = sha256_file(train)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="differs from the raw-source"):
        check_clean_dataset(root / "processed", root / "export")


def test_extract_hashes_dump_and_mapping_and_rejects_corruption(inputs):
    manifest = validate_extract(inputs["freebase"])
    assert manifest["source_fingerprint"]["sha256"] == sha256_file(inputs["dump"])
    assert manifest["item_links_sha256"] == sha256_file(inputs["links"])
    with inputs["freebase"].open("a") as handle:
        handle.write("corruption\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_extract(inputs["freebase"])


def test_rec_bole_and_kgat_mapping_headers(tmp_path):
    path = tmp_path / "links.tsv"
    for header in ("item_id:token\tentity_id:token", "item\tfreebase_id", "org_id\tfreebase_id"):
        path.write_text(header + "\n23\t/m/abc\n")
        assert read_item_links(path) == {23: "m.abc"}


def test_01b_cli_defaults_to_the_fixed_window(inputs, tmp_path):
    script = runpy.run_path(str(ROOT / "scripts/01b_cut_lastfm.py"))
    target = tmp_path / "cli-pairs.tsv"
    script["main"](["--interactions", str(inputs["raw"]), "--out", str(target)])
    protocol.check_cut_manifest(json.loads(Path(str(target) + ".meta.json").read_text()))


def test_output_cannot_overwrite_input_or_other_outputs(inputs, tmp_path):
    with pytest.raises(ValueError, match="contains input source"):
        build_raw_dataset(inputs["pairs"], tmp_path, tmp_path.parent / "separate-export", inputs["freebase"],
                          item_links=inputs["links"])


def test_zero_kg_coverage_does_not_stop_the_build(inputs, tmp_path):
    with gzip.open(inputs["dump"], "wt", encoding="utf-8") as handle:
        handle.write(rdf("m.unrelated", "music.recording.artist", "m.someone"))
    extract_dump(inputs["dump"], inputs["links"], inputs["freebase"],
                 snapshot_date=protocol.SNAPSHOT, overwrite=True, train_pairs=inputs["pairs"])
    root = tmp_path / "empty-kg"
    meta = build(inputs, root)
    assert meta["coverage"]["item_fact_rate"] == {"t1": 0, "t2": 0}
    assert meta["interactions"]["train"] > 0
    assert meta["kg_construction"]["cutoff_graph_stats"]["t1"]["max_degree"] == 0
    assert check_clean_dataset(root / "processed", root / "export")["passed"]


def test_mapped_cold_items_do_not_seed_the_kg(inputs, tmp_path, monkeypatch):
    with inputs["links"].open("a") as handle:
        handle.write("9999\t/m/cold\n")
    with gzip.open(inputs["dump"], "at", encoding="utf-8") as handle:
        handle.write(rdf("m.cold", "music.recording.artist", "m.cold_artist"))
    extract_dump(inputs["dump"], inputs["links"], inputs["freebase"],
                 snapshot_date=protocol.SNAPSHOT, overwrite=True, train_pairs=inputs["pairs"])
    monkeypatch.setattr(protocol, "MAPPING_TRACKS", 3)
    root = tmp_path / "cold-kg"
    build(inputs, root)
    sources = pd.read_csv(root / "export/fact_sources.tsv", sep="\t")
    assert not sources.raw_object.str.contains("m.cold", regex=False).any()


def test_truncated_dump_cannot_publish_extraction(inputs, tmp_path):
    damaged = tmp_path / "damaged.gz"
    damaged.write_bytes(inputs["dump"].read_bytes()[:-7])
    target = tmp_path / "incomplete.tsv"
    with pytest.raises((EOFError, OSError)):
        extract_dump(damaged, inputs["links"], target, snapshot_date=protocol.SNAPSHOT)
    assert not target.exists()
    assert not Path(str(target) + ".meta.json").exists()


def test_cli_extract_reuse_build_and_check(inputs, tmp_path):
    extract_script = runpy.run_path(str(ROOT / "scripts/01c_extract_freebase.py"))
    extract_script["main"]([
        "--dump", str(inputs["dump"]), "--snapshot-date", protocol.SNAPSHOT,
        "--pairs", str(inputs["pairs"]),
        "--item-links", str(inputs["links"]), "--out", str(inputs["freebase"]), "--reuse-existing",
    ])
    root = tmp_path / "cli"
    build_script = runpy.run_path(str(ROOT / "scripts/02b_build_lastfm_from_raw.py"))
    build_script["main"]([
        "--pairs", str(inputs["pairs"]), "--item-links", str(inputs["links"]),
        "--freebase", str(inputs["freebase"]), "--out", str(root / "processed"),
        "--export-dir", str(root / "export"),
    ])
    check_script = runpy.run_path(str(ROOT / "scripts/16_check_clean_dataset.py"))
    report = root / "report.json"
    check_script["main"]([
        "--processed-dir", str(root / "processed"), "--export-dir", str(root / "export"), "--report", str(report),
    ])
    assert json.loads(report.read_text())["passed"]


def test_no_linked_train_items_does_not_stop_the_pipeline(inputs, tmp_path):
    inputs["links"].write_text("item\tfreebase_id\n9000\tm.outside0\n9001\tm.outside1\n")
    extract_dump(inputs["dump"], inputs["links"], inputs["freebase"],
                 snapshot_date=protocol.SNAPSHOT, overwrite=True, train_pairs=inputs["pairs"])
    root = tmp_path / "no-linked-items"
    meta = build(inputs, root)
    assert meta["coverage"]["item_mid_link_rate"] == 0
    assert meta["coverage"]["item_fact_rate"] == {"t1": 0, "t2": 0}
    assert check_clean_dataset(root / "processed", root / "export")["passed"]


def test_mapping_member_hash_is_the_same_locally_and_on_drive(inputs, tmp_path, monkeypatch):
    archive_path = tmp_path / "mapping.zip"
    member = "lastfm-links-recbole/item_links.tsv"
    with ZipFile(archive_path, "w") as archive:
        archive.write(inputs["links"], member)
    with ZipFile(archive_path) as archive:
        monkeypatch.setattr(drive, "drive_archive", lambda file_id: archive)
        assert source_sha256(protocol.ITEM_LINKS_DRIVE) == sha256_file(inputs["links"])
        assert source_sha256(f"{archive_path}::{member}") == sha256_file(inputs["links"])
        assert read_item_links(protocol.ITEM_LINKS_DRIVE) == read_item_links(inputs["links"])


def test_cli_uses_drive_mapping_by_default_without_local_download(inputs, tmp_path, monkeypatch):
    archive_path = tmp_path / "mapping.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(inputs["links"], "lastfm-links-recbole/item_links.tsv")
    # The source file is removed to prove that neither CLI requires a local TSV.
    inputs["links"].unlink()
    root = tmp_path / "drive-cli"
    with ZipFile(archive_path) as archive:
        def supplied_archive(file_id):
            assert file_id == "1SqoD1_3VgqjrsOo4TJfoR_AeKe_jHl6x"
            return archive

        monkeypatch.setattr(drive, "drive_archive", supplied_archive)
        extract_script = runpy.run_path(str(ROOT / "scripts/01c_extract_freebase.py"))
        extract_script["main"]([
            "--dump", str(inputs["dump"]), "--snapshot-date", protocol.SNAPSHOT,
            "--pairs", str(inputs["pairs"]), "--out", str(inputs["freebase"]), "--overwrite",
        ])
        build_script = runpy.run_path(str(ROOT / "scripts/02b_build_lastfm_from_raw.py"))
        build_script["main"]([
            "--pairs", str(inputs["pairs"]), "--freebase", str(inputs["freebase"]),
            "--out", str(root / "processed"), "--export-dir", str(root / "export"),
        ])
        meta = json.loads((root / "export/dataset_meta.json").read_text())
        assert meta["item_links"] == protocol.ITEM_LINKS_DRIVE
        assert len(meta["item_links_sha256"]) == 64
        assert check_clean_dataset(root / "processed", root / "export")["passed"]


def test_extraction_and_build_record_hops_degrees_and_timings(inputs, tmp_path, capsys):
    extraction = validate_extract(inputs["freebase"])
    assert [report["graph_stats"] for report in extraction["passes"][:2]] == [
        {"triples": 4, "entities": 6, "max_degree": 3},
        {"triples": 5, "entities": 7, "max_degree": 3},
    ]
    assert all(report["elapsed_seconds"] >= 0 for report in extraction["passes"])
    assert extraction["timing_seconds"]["total"] >= sum(
        report["elapsed_seconds"] for report in extraction["passes"]
    )
    root = tmp_path / "logged"
    meta = build(inputs, root)
    assert [stats["triples"] for stats in meta["kg_construction"]["hop_stats"]] == [4, 5]
    assert meta["timing_seconds"]["total_before_metadata_finalization"] >= meta["timing_seconds"]["kg_construction"]
    for cut, stats in meta["kg_construction"]["cutoff_graph_stats"].items():
        assert stats["triples"] == meta["coverage"]["triples"][cut]
    kg_meta = json.loads((root / "processed/kg/default/kg_meta.json").read_text())
    assert kg_meta["timing_seconds"] == meta["timing_seconds"]
    assert kg_meta["date_semantics"]["fact_date"] == "max_available_endpoint_dates"
    assert kg_meta["date_semantics"]["both_endpoints_undated"] == "keep_fact_with_null_timestamp"
    log = capsys.readouterr().out
    assert "Train-seeded hop 1" in log and "Train-seeded hop 2" in log
    assert "max_degree=" in log and "KG construction complete: elapsed=" in log


def test_hop_degree_includes_incoming_parallel_edges_and_self_loops():
    graph = pd.DataFrame({"head": ["a", "a", "b", "hub"], "tail": ["hub"] * 4})
    stats = []
    result = reachable(graph, {"a"}, hop_stats=stats)
    assert len(result) == 4
    assert stats == [
        {"hop": 1, "triples": 2, "entities": 2, "max_degree": 2},
        {"hop": 2, "triples": 4, "entities": 3, "max_degree": 5},
    ]


def test_documented_date_rule_matches_two_dated_one_dated_and_undated_edges(inputs):
    primary = read_freebase_source(inputs["freebase"], {"m.track0", "m.track1"}, all_heads=True)
    evidence = primary.evidence.copy()
    evidence.loc["m.track0", [*DATE_COLUMNS, "tail_label"]] = ["2000-03-04", "", "1980-05-06", ""]
    items = pd.DataFrame({"h": [0, 1], "item": [1000, 1001], "freebase_id": ["m.track0", "m.track1"]})
    bundle = build_twohop(items, primary.graph, evidence, {"t1": protocol.WINDOW_START, "t2": protocol.WINDOW_END})
    facts = bundle.kept["t1"]
    # Head chooses its earlier birth date (1980), then max(head, artist 1970) is 1980.
    expected = int(pd.Timestamp("1980-05-06", tz="UTC").timestamp())
    assert facts.loc[(facts["head"] == "m.track0") & (facts["tail"] == "m.artist"), "ts"].item() == expected
    # Incoming neighbor/artist has only the artist's date; old release has neither.
    expected = int(pd.Timestamp("1970-02-03", tz="UTC").timestamp())
    assert facts.loc[facts["head"] == "m.neighbor", "ts"].item() == expected
    assert facts.loc[facts["head"] == "m.track1", "ts"].isna().all()
