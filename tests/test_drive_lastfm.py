"""Drive ZIP I/O and the complete timestamp pipeline, without network access."""

import hashlib
import io
import runpy
import sys
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
import yaml

from tkgat.data import drive
from tkgat.data.encode import encode
from tkgat.kg.ckg import load_kg, load_kg_dates
from tkgat.pipeline.build import PARTS, build_split, load_saved, save_split

from .conftest import ROOT
from .test_export_kgat import parse_labels
from .test_kg_lastfm import interactions, write_author, write_freebase

pytest.importorskip("stream_unzip", reason="Drive streaming tests require the drive extra")


@pytest.fixture(autouse=True)
def clear_drive_cache(monkeypatch):
    drive.drive_archive.cache_clear()
    monkeypatch.delenv("TKGAT_DRIVE_ACCESS_TOKEN", raising=False)
    yield
    drive.drive_archive.cache_clear()


def mock_public(monkeypatch, content):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert isinstance(kwargs["output"], io.BytesIO)
        assert kwargs["use_cookies"] is False
        kwargs["output"].write(content)
        return kwargs["output"]

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    return calls


def make_zip(entries):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_reads_members_once_without_extracting_to_disk(monkeypatch, tmp_path):
    content = make_zip({"pairs/input.tsv": "user\titem\tts\n1\t2\t3\n", "other.txt": "hello"})
    calls = mock_public(monkeypatch, content)
    drive.verify_archive("test-id", hashlib.sha256(content).hexdigest())
    with drive.open_text_source("gdrive://test-id/pairs/input.tsv") as handle:
        assert pd.read_csv(handle, sep="\t").iloc[0].tolist() == [1, 2, 3]
    with drive.open_text_source("gdrive://test-id/other.txt") as handle:
        assert handle.read() == "hello"
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_private_file_uses_the_explicit_token(monkeypatch):
    content = make_zip({"x.txt": "private"})
    monkeypatch.setenv("TKGAT_DRIVE_ACCESS_TOKEN", "test-token")
    requests = []

    def open_response(request, timeout):
        requests.append(request)
        assert timeout == 60
        return io.BytesIO(content)

    monkeypatch.setattr(drive, "urlopen", open_response)
    with drive.open_text_source("gdrive://private-id/x.txt") as handle:
        assert handle.read() == "private"
    assert requests[0].get_header("Authorization") == "Bearer test-token"
    assert requests[0].full_url.startswith("https://www.googleapis.com/drive/v3/files/private-id?")


def test_corruption_and_missing_members_are_reported(monkeypatch):
    content = make_zip({"x.txt": "x"})
    mock_public(monkeypatch, content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        drive.verify_archive("test-id", "0" * 64)
    with pytest.raises(FileNotFoundError, match="member not found"):
        with drive.open_binary_source("gdrive://test-id/missing.tsv"):
            pass
    drive.drive_archive.cache_clear()
    mock_public(monkeypatch, b"<html>Sign in</html>")
    with pytest.raises(ValueError, match="invalid ZIP"):
        drive.drive_archive("test-id")


def test_public_access_failure_does_not_fall_back_to_a_local_download(monkeypatch):
    def denied(**kwargs):
        raise RuntimeError("access denied")

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=denied))
    with pytest.raises(RuntimeError, match="TKGAT_DRIVE_ACCESS_TOKEN"):
        drive.drive_archive("private-id")


def test_memory_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(drive, "MAX_ARCHIVE_BYTES", 3)
    buffer = drive._MemoryBuffer()
    buffer.write(b"abc")
    with pytest.raises(ValueError, match="in-memory limit"):
        buffer.write(b"d")


def test_large_zip_member_is_read_incrementally_without_memory_archive(monkeypatch, tmp_path):
    content = make_zip({"skip.txt": "skip", "lfm1b-tracks/events.inter": "user\titem\tts\n1\t2\t300\n"})
    consumed = []

    def source(file_id):
        assert file_id == "raw-id"
        for pos in range(0, len(content), 7):
            consumed.append(pos)
            yield content[pos:pos + 7]

    monkeypatch.setattr(drive, "drive_chunks", source)
    with drive.open_text_source("gdrive-stream://raw-id/events.inter") as handle:
        assert handle.readline() == "user\titem\tts\n"
        assert handle.read() == "1\t2\t300\n"
    assert consumed and drive.drive_archive.cache_info().currsize == 0
    assert list(tmp_path.iterdir()) == []


def test_streaming_supports_zip64_and_validates_member_crc(monkeypatch):
    from stream_unzip import CRC32IntegrityError

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("raw/events.inter", "w", force_zip64=True) as handle:
            handle.write(b"user\titem\tts\n1\t2\t400\n")
    content = buffer.getvalue()
    monkeypatch.setattr(drive, "drive_chunks", lambda file_id: iter([content]))
    with drive.open_text_source("gdrive-stream://raw-id/events.inter") as handle:
        assert handle.read().endswith("1\t2\t400\n")
    damaged = bytearray(content)
    damaged[14] ^= 1  # corrupt the expected CRC in the local ZIP header
    monkeypatch.setattr(drive, "drive_chunks", lambda file_id: iter([bytes(damaged)]))
    with pytest.raises(CRC32IntegrityError):
        with drive.open_text_source("gdrive-stream://raw-id/events.inter") as handle:
            handle.read()


def test_latest_config_cannot_silently_load_earliest_cache():
    with pytest.raises(ValueError, match="require raw events"):
        build_split({"name": "lastfm", "timestamp_policy": "latest"})


@pytest.mark.parametrize("value", ["https://drive.google.com/drive/folders/abc", "https://example.com/file/d/abc/view"])
def test_only_drive_file_urls_are_accepted(value):
    with pytest.raises(ValueError, match="Drive ZIP file"):
        drive.drive_file_id(value)


def test_drive_and_local_user_samples_have_identical_splits(monkeypatch, tmp_path, synthetic):
    raw = synthetic[["user", "item", "ts"]].to_csv(sep="\t", index=False)
    mock_public(monkeypatch, make_zip({"pairs/kgat_author_earliest.tsv": raw}))
    local = tmp_path / "raw"
    local.mkdir()
    (local / "kgat_author_earliest.tsv").write_text(raw, encoding="utf-8")
    cfg = {
        "name": "lastfm", "raw_dir": str(local), "user_fraction": 0.25, "user_seed": 2019,
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 2},
    }
    expected, report = build_split(cfg)
    actual, actual_report = build_split({**cfg, "raw_dir": "gdrive://test-id/pairs"})
    assert report == actual_report
    for part in PARTS:
        pd.testing.assert_frame_equal(getattr(expected, part), getattr(actual, part))


@pytest.fixture
def pipeline_inputs(monkeypatch, tmp_path):
    author = write_author(tmp_path / "author")
    freebase = write_freebase(tmp_path / "freebase.tsv")
    raw_dir = tmp_path / "local-raw"
    raw_dir.mkdir()
    pairs = interactions()[["user", "item", "ts"]].to_csv(sep="\t", index=False)
    (raw_dir / "kgat_author_earliest.tsv").write_text(pairs, encoding="utf-8")
    cfg = {
        "name": "lastfm", "run_name": "lastfm-test", "raw_dir": str(raw_dir),
        "processed_dir": str(tmp_path / "reference"), "clean": False,
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 2},
    }
    split, report = build_split(cfg)
    reference_dir = save_split(split, cfg, report)
    entries = {
        "pairs/kgat_author_earliest.tsv": pairs,
        "freebase/freebase-subset.tsv": freebase.read_bytes(),
        "reference/meta.json": (reference_dir / "meta.json").read_bytes(),
    }
    content = make_zip(entries)
    calls = mock_public(monkeypatch, content)
    cfg.update(
        raw_dir="gdrive://test-id/pairs", freebase="gdrive://test-id/freebase/freebase-subset.tsv",
        drive_zip_id="test-id", drive_zip_sha256=hashlib.sha256(content).hexdigest(),
        reference_meta="reference/meta.json", processed_dir=str(tmp_path / "processed"),
    )
    path = tmp_path / "drive.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, path, author, calls


def test_complete_pipeline_preserves_timestamps_kg_and_hashes(pipeline_inputs, tmp_path):
    cfg, path, author, calls = pipeline_inputs
    main = runpy.run_path(str(ROOT / "scripts" / "generate_lastfm_drive.py"))["main"]
    output = tmp_path / "export"
    args = ["--data", str(path), "--author-data", str(author), "--export-dir", str(output)]
    main(args)
    split, meta = load_saved(cfg)
    enc = encode(split)
    assert len(calls) == 1  # pairs, Freebase, hashes and reference share the RAM ZIP
    for name in ("train", "val", "test"):
        frame = getattr(split, name)
        expected = {
            (enc.user_ids.tolist().index(int(row.user)), enc.item_ids.tolist().index(int(row.item))): int(row.ts)
            for row in frame.itertuples(index=False)
        }
        assert parse_labels(output / f"{name}_timestamp.txt") == expected
    for variant in ("default", "drop_undated", "static"):
        for cutoff in ("t1", "t2"):
            kg = load_kg(cfg["processed_dir"], variant, cutoff, enc, meta)
            assert len(load_kg_dates(cfg["processed_dir"], kg)) == len(kg)
    assert "lastfm.item." in (output / "entity_list.txt").read_text()
    assert "freebase.music." in (output / "relation_list.txt").read_text()
    assert not list(tmp_path.rglob("*.zip"))
    with pytest.raises(FileExistsError, match="Output is not empty"):
        main(args)
    assert len(calls) == 1


def test_reference_mismatch_stops_before_kg_and_export(pipeline_inputs, monkeypatch, tmp_path):
    cfg, path, author, _ = pipeline_inputs
    cfg["split"]["q_val"] = 0.7
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main = runpy.run_path(str(ROOT / "scripts" / "generate_lastfm_drive.py"))["main"]
    with pytest.raises(RuntimeError, match="differ from the Drive reference"):
        main(["--data", str(path), "--author-data", str(author), "--export-dir", str(tmp_path / "export")])
    assert not (tmp_path / "processed" / "kg").exists()
    assert not (tmp_path / "export").exists()


def test_latest_drive_pipeline_uses_raw_maxima_and_never_the_earliest_cache(pipeline_inputs, monkeypatch, tmp_path):
    cfg, path, author, _ = pipeline_inputs
    events = interactions()
    users = sorted(events.user.unique())
    (author / "user_list.txt").write_text(
        "org_id remap_id\n" + "".join(f"raw-{user} {user}\n" for user in users), encoding="utf-8"
    )
    by_user = events.groupby("user")["item"].apply(list)
    (author / "train.txt").write_text(
        "".join(f"{user} {' '.join(map(str, items))}\n" for user, items in by_user.items()), encoding="utf-8"
    )
    (author / "test.txt").write_text("", encoding="utf-8")
    raw_items = pd.read_csv(author / "item_list.txt", sep=" ").set_index("remap_id")["org_id"].to_dict()
    raw = "user_id\ttracks_id\ttimestamp\n" + "".join(
        f"raw-{row.user}\t{raw_items[row.item]}\t{ts}\n"
        for row in events.itertuples(index=False) for ts in (int(row.ts) + 9, int(row.ts) - 100, int(row.ts))
    )
    raw_zip = make_zip({"lfm1b-tracks/lfm1b-tracks.inter": raw})
    monkeypatch.setattr(drive, "drive_chunks", lambda file_id: iter([raw_zip[:31], raw_zip[31:]]))

    def no_cache(*args, **kwargs):
        raise AssertionError("latest mode must not load an earliest-only TSV")

    monkeypatch.setattr("tkgat.pipeline.build.load", no_cache)
    cfg.pop("reference_meta")
    cfg.update(timestamp_policy="latest", raw_interactions="gdrive-stream://raw-id/lfm1b-tracks.inter")
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    output = tmp_path / "latest-export"
    main = runpy.run_path(str(ROOT / "scripts" / "generate_lastfm_drive.py"))["main"]
    main(["--data", str(path), "--author-data", str(author), "--export-dir", str(output)])
    got, meta = load_saved(cfg)
    expected, _ = build_split(cfg, interactions=events.assign(ts=events.ts + 9))
    for part in PARTS:
        pd.testing.assert_frame_equal(
            getattr(got, part), getattr(expected, part), check_dtype=not getattr(expected, part).empty
        )
    assert meta["clean"]["lastfm_timestamps"]["timestamp_policy"] == "latest"
    assert meta["clean"]["lastfm_timestamps"]["lfm_candidate_rows"] == 3 * len(events)
    assert "timestamp" in (output / "interactions_chronological.tsv").read_text().splitlines()[0]
    assert not list(tmp_path.rglob("kgat_author_latest.tsv"))  # no raw-derived disk cache in Drive mode


def test_preloaded_latest_events_keep_maxima_before_splitting(synthetic):
    cfg = {
        "name": "lastfm", "timestamp_policy": "latest", "clean": False,
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 2},
    }
    base = synthetic.assign(rating=1)
    repeated = pd.concat([base.assign(ts=base.ts + 100), base.assign(ts=base.ts - 100), base], ignore_index=True)
    actual, _ = build_split(cfg, interactions=repeated)
    expected, _ = build_split(cfg, interactions=base.assign(ts=base.ts + 100))
    for part in PARTS:
        pd.testing.assert_frame_equal(getattr(actual, part), getattr(expected, part))


def test_shipped_latest_config_has_separate_outputs_and_raw_drive_source():
    cfg = yaml.safe_load((ROOT / "configs/data/lastfm_u25_latest_drive.yaml").read_text(encoding="utf-8"))
    assert cfg["timestamp_policy"] == "latest" and cfg["user_fraction"] == 0.25 and cfg["user_seed"] == 2019
    assert cfg["raw_interactions"].startswith("gdrive-stream://")
    assert cfg["run_name"] == "lastfm-u25-latest" and cfg["processed_dir"].endswith("lastfm-u25-latest")
    assert "reference_meta" not in cfg
