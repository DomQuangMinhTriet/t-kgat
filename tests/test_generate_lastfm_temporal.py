"""Pure functions of scripts/generate_lastfm_temporal_kgat.py on tiny synthetic data (CPU, no real data)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from tkgat.data.leakage import check_split

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gen_lastfm_temporal", ROOT / "scripts" / "generate_lastfm_temporal_kgat.py"
)
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script  # @dataclass looks the module up
_spec.loader.exec_module(script)

MAX32 = (1 << 32) - 1


@pytest.mark.parametrize("user,item", [(0, 0), (1, 2), (MAX32, 0), (0, MAX32), (MAX32, MAX32), (123456, 789)])
def test_pair_key_round_trip(user, item):
    assert script.decode_pair(script.pair_key(user, item)) == (user, item)


def test_pair_key_is_injective_near_the_32_bit_edge():
    keys = {script.pair_key(u, i) for u in (0, 1, MAX32 - 1, MAX32) for i in (0, 1, MAX32 - 1, MAX32)}
    assert len(keys) == 16


@pytest.mark.parametrize("user,item", [(-1, 0), (0, -1), (MAX32 + 1, 0), (0, MAX32 + 1)])
def test_pair_key_rejects_ids_outside_32_bits(user, item):
    with pytest.raises(ValueError):
        script.pair_key(user, item)


def _write_cache(path, rows):
    pd.DataFrame(rows, columns=["user", "item", "ts"]).to_csv(path, sep="\t", index=False)


def test_load_cache_accepts_the_expected_universe(tmp_path):
    path = tmp_path / "cache.tsv"
    _write_cache(path, [(0, 1, 10), (0, 2, 20), (1, 1, 30)])
    expected = {script.pair_key(0, 1), script.pair_key(0, 2), script.pair_key(1, 1)}
    frame = script.load_cache(path, expected)
    assert len(frame) == 3
    assert (frame["rating"] == 1).all()


def test_load_cache_with_a_missing_pair_fails(tmp_path):
    path = tmp_path / "cache.tsv"
    _write_cache(path, [(0, 1, 10)])
    with pytest.raises(ValueError, match="pair universe differs"):
        script.load_cache(path, {script.pair_key(0, 1), script.pair_key(0, 2)})


def test_load_cache_with_an_extra_pair_fails(tmp_path):
    path = tmp_path / "cache.tsv"
    _write_cache(path, [(0, 1, 10), (0, 2, 20)])
    with pytest.raises(ValueError, match="pair universe differs"):
        script.load_cache(path, {script.pair_key(0, 1)})


def test_load_cache_rejects_duplicate_pairs_and_bad_columns(tmp_path):
    dup = tmp_path / "dup.tsv"
    _write_cache(dup, [(0, 1, 10), (0, 1, 20)])
    with pytest.raises(ValueError, match="duplicate"):
        script.load_cache(dup, {script.pair_key(0, 1)})
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"user": [0], "item": [1], "time": [5]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="columns"):
        script.load_cache(bad, {script.pair_key(0, 1)})


def test_read_kg_reads_triples_and_rejects_bad_input(tmp_path):
    ok = tmp_path / "kg.txt"
    ok.write_text("0 1 5\n2 0 6\n", encoding="utf-8")
    assert script.read_kg(ok) == [(0, 1, 5), (2, 0, 6)]
    dup = tmp_path / "dup.txt"
    dup.write_text("0 1 5\n0 1 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        script.read_kg(dup)
    short = tmp_path / "short.txt"
    short.write_text("0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected h r t"):
        script.read_kg(short)


def test_grouped_maps_ids_and_keeps_row_order():
    frame = pd.DataFrame({"user": [10, 10, 20], "item": [7, 8, 7], "ts": [3, 4, 5]})
    result = script.grouped(frame, {10: 0, 20: 1}, {7: 0, 8: 1})
    assert result == {0: [(0, 3), (1, 4)], 1: [(0, 5)]}


def _kg_dir(tmp_path):
    author = tmp_path / "author"
    author.mkdir()
    # items 0 and 1 are train items; triple (5, 2, 6) touches neither and must be dropped by the seeding
    (author / "kg_final.txt").write_text("0 1 5\n1 1 6\n5 2 6\n7 0 0\n", encoding="utf-8")
    return author


def test_kg_snapshots_static_gives_three_identical_snapshots(tmp_path):
    author = _kg_dir(tmp_path)
    kg_t1, kg_t2, times, stats = script.kg_snapshots(author, {0, 1}, 100, 200, "static", None)
    assert kg_t1 == kg_t2 == [(0, 1, 5), (1, 1, 6), (7, 0, 0)]
    assert kg_t1 is not kg_t2
    assert times == {}
    assert stats["policy"] == "static"
    assert stats["time_consistent"] is False
    assert stats["source_triples"] == 4
    assert stats["train_item_seeded_triples"] == 3


def test_kg_snapshots_strict_without_fact_times_fails(tmp_path):
    with pytest.raises(ValueError, match="--kg-fact-times"):
        script.kg_snapshots(_kg_dir(tmp_path), {0, 1}, 100, 200, "strict", None)


def test_kg_snapshots_strict_drops_undated_and_future_facts(tmp_path):
    author = _kg_dir(tmp_path)
    facts = tmp_path / "facts.tsv"
    # (0,1,5): before t1; (1,1,6): between t1 and t2; (7,0,0): undated (absent); (5,2,6): dated but not seeded
    facts.write_text("head relation tail timestamp\n0 1 5 50\n1 1 6 150\n5 2 6 10\n", encoding="utf-8")
    kg_t1, kg_t2, times, stats = script.kg_snapshots(author, {0, 1}, 100, 200, "strict", facts)
    assert kg_t1 == [(0, 1, 5)]
    assert kg_t2 == [(0, 1, 5), (1, 1, 6)]
    assert set(kg_t1) <= set(kg_t2)
    assert times[(1, 1, 6)] == 150
    assert stats["policy"] == "strict_drop_undated"
    assert stats["dated_seeded_triples"] == 2
    assert stats["undated_seeded_triples_dropped"] == 1
    assert stats["triples_t1"] == 1 and stats["triples_t2"] == 2


def test_kg_snapshots_strict_drops_a_fact_dated_exactly_at_the_cutoff(tmp_path):
    author = _kg_dir(tmp_path)
    facts = tmp_path / "facts.tsv"
    facts.write_text("0 1 5 100\n1 1 6 200\n", encoding="utf-8")
    kg_t1, kg_t2, _, _ = script.kg_snapshots(author, {0, 1}, 100, 200, "strict", facts)
    assert kg_t1 == [] and kg_t2 == [(0, 1, 5)]


def test_kg_snapshots_strict_rejects_facts_absent_from_the_author_kg(tmp_path):
    author = _kg_dir(tmp_path)
    facts = tmp_path / "facts.tsv"
    facts.write_text("9 9 9 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from author KG"):
        script.kg_snapshots(author, {0, 1}, 100, 200, "strict", facts)


def test_make_split_passes_check_split_and_sorts_by_time():
    rows = [(u, i, 1, 1000 + 10 * k) for k, (u, i) in enumerate((u, i) for u in range(6) for i in range(10))]
    frame = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    split = script.make_split(frame, 0.8, 0.9, 1)
    assert check_split(split) == []
    assert split.t1 < split.t2
    assert len(split.train) and len(split.val) + len(split.test) > 0
    for part in (split.train, split.val, split.test):
        assert part["ts"].is_monotonic_increasing
    assert split.train["ts"].max() < split.t1 <= split.val["ts"].min() if len(split.val) else True
