"""The Last-FM KG builder on a synthetic author KG and Freebase subset (CPU, no real data needed).

Checks the date parser of ``tkgat.kg.lastfm`` literal form by literal form, the time-consistency rule
(no kept fact has a tail existence date on or after the cut-off) and that the three variants differ only
where decisions 0004 and 0006 say they do. The temporal rule itself lives in ``tkgat.kg.temporal`` and is
tested there; here it is exercised through the Last-FM sources.
"""

import datetime as dt
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tkgat.data.encode import encode
from tkgat.data.leakage import check_split
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.ckg import build_ckg, load_kg, load_kg_dates
from tkgat.kg.lastfm import (
    DATE_COLUMNS,
    HEAD_NOT_ITEM,
    annotate_lastfm,
    check_kg,
    item_entities,
    lastfm_facts,
    literal_value,
    parse_freebase_date,
    read_author_kg,
    read_freebase_subset,
    seeded_triples,
)
from tkgat.kg.temporal import cutoff_facts, triple_dates
from tkgat.pipeline.build import load_saved, save_split

from .conftest import ROOT

T1 = int(dt.datetime(2014, 4, 28, 12, tzinfo=dt.timezone.utc).timestamp())
T2 = int(dt.datetime(2014, 6, 6, 12, tzinfo=dt.timezone.utc).timestamp())
GYEAR = '"{}"^^<http://www.w3.org/2001/XMLSchema#gYear>'
GYEARMONTH = '"{}"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
DATE = '"{}"^^<http://www.w3.org/2001/XMLSchema#date>'


# --- the date parser ------------------------------------------------------------------------------------


def test_literal_value_strips_the_type_and_the_language_tag():
    assert literal_value(DATE.format("2007-09-10")) == "2007-09-10"
    assert literal_value('"In My Arms"@en') == "In My Arms"
    assert literal_value("m.0314xwk") == "m.0314xwk"  # a plain entity object


def test_a_year_is_the_last_day_of_that_year():
    assert parse_freebase_date(GYEAR.format("1999")) == dt.date(1999, 12, 31)


def test_a_year_and_month_is_the_last_day_of_that_month():
    assert parse_freebase_date(GYEARMONTH.format("1999-03")) == dt.date(1999, 3, 31)
    assert parse_freebase_date(GYEARMONTH.format("1996-02")) == dt.date(1996, 2, 29)  # leap year
    assert parse_freebase_date(GYEARMONTH.format("1997-02")) == dt.date(1997, 2, 28)


def test_a_full_date_is_that_day():
    assert parse_freebase_date(DATE.format("2007-09-10")) == dt.date(2007, 9, 10)


def test_a_full_date_ending_in_january_the_first_is_year_only():
    # decision 0006 A: Freebase writes a year-only date as -01-01, so it is read as 31 December
    assert parse_freebase_date(DATE.format("2001-01-01")) == dt.date(2001, 12, 31)
    assert parse_freebase_date(DATE.format("2001-01-02")) == dt.date(2001, 1, 2)


@pytest.mark.parametrize(
    "literal",
    [
        DATE.format("-0044-03-15"),  # a BC year (decision 0006 C)
        GYEAR.format("-0044"),
        DATE.format("1999-13-45"),  # not a calendar date
        '"not a date"@en',
        "m.0314xwk",  # an entity, not a literal
        '""',
        "",
    ],
)
def test_unusable_literals_have_no_date(literal):
    assert parse_freebase_date(literal) is None


# --- the sources ----------------------------------------------------------------------------------------

# The ids of the interaction file are the author's *remap* ids (so are the ids of kg_final.txt), and only
# some of them survive the split, so an item index h is never the author id: 1, 2, 4, 5 -> h = 0, 1, 2, 3.
AUTHOR_ITEMS = [  # org_id, remap_id (= entity id), freebase_id
    ("7000", 0, "m.cold0"),  # only in the test period, so never a train item
    ("7001", 1, "m.item0"),
    ("7002", 2, "m.item1"),
    ("7003", 3, "m.cold3"),  # in no interaction at all
    ("7004", 4, "m.item2"),
    ("7005", 5, "m.item3"),
]
AUTHOR_TAILS = [("m.old", 6), ("m.future", 7), ("m.nodate", 8), ("m.month", 9)]
AUTHOR_RELATIONS = [
    "http://rdf.freebase.com/ns/music.recording.artist",
    "http://rdf.freebase.com/ns/common.topic.notable_types",
    "http://rdf.freebase.com/ns/music.recording.canonical_version",
]
AUTHOR_TRIPLES = [
    (1, 0, 6),  # h = 0 -> an artist born long ago: kept at both cut-offs
    (1, 1, 8),  # h = 0 -> an undated tail: kept by default, dropped by drop_undated
    (2, 0, 7),  # h = 1 -> a release of 2014-05-10: dropped at t1, kept at t2
    (2, 1, 9),  # h = 1 -> a tail dated 2014-04 (last day 30 April): dropped at t1, kept at t2
    (4, 2, 1),  # h = 2 -> item 1: the tail becomes its own attribute entity
    (0, 2, 2),  # head 0 is not a train item: item-seeded (its tail is), but not 1-hop
]
FREEBASE_ROWS = [
    ("m.old", "people.person.date_of_birth", DATE.format("1946-10-18")),
    ("m.old", "type.object.name", '"An Old Artist"@en'),
    ("m.future", "music.release.release_date", DATE.format("2014-05-10")),
    ("m.month", "music.release.release_date", GYEARMONTH.format("2014-04")),
    ("m.nodate", "type.object.name", '"No Date"@en'),
    ("m.item0", "music.release.release_date", GYEAR.format("2001")),
    ("m.item1", "music.artist.active_start", GYEAR.format("1990")),
    ("m.unused", "music.release.release_date", DATE.format("2020-01-02")),
]
TRAIN_ITEMS = [1, 2, 4, 5]  # author remap ids; their item indices h are 0, 1, 2, 3


def write_author(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    lines = lambda rows: "\n".join(rows) + "\n"  # noqa: E731
    (d / "item_list.txt").write_text(
        lines(["org_id remap_id freebase_id", *(f"{o} {r} {f}" for o, r, f in AUTHOR_ITEMS)]), encoding="utf-8"
    )
    entities = [(f, r) for _, r, f in AUTHOR_ITEMS] + AUTHOR_TAILS
    (d / "entity_list.txt").write_text(lines(["org_id remap_id", *(f"{o} {r}" for o, r in entities)]), encoding="utf-8")
    (d / "relation_list.txt").write_text(
        lines(["org_id remap_id", *(f"{o} {i}" for i, o in enumerate(AUTHOR_RELATIONS))]), encoding="utf-8"
    )
    (d / "kg_final.txt").write_text(lines(f"{h} {r} {t}" for h, r, t in AUTHOR_TRIPLES), encoding="utf-8")
    return d


def write_freebase(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "subject\tpredicate\tobject\n" + "".join(f"{s}\t{p}\t{o}\n" for s, p, o in FREEBASE_ROWS),
        encoding="utf-8",
    )
    return path


def interactions() -> pd.DataFrame:
    """21 pairs placed so that the 0.8 / 0.9 quantiles of the split are exactly T1 and T2.

    16 pairs (4 users x the four train items) lie before T1 and become the train part, 2 lie at T1
    (validation) and 3 at T2 (test). With 21 values the quantiles fall on the sorted positions 16 and 18,
    which hold T1 and T2, so the cut-offs of the split are the constants the fact dates are chosen against.
    """
    rows = [(u, item, 1, T1 - (20 + u * 4 + n) * 86_400) for u in range(4) for n, item in enumerate(TRAIN_ITEMS)]
    rows += [(u, 1, 1, T1) for u in (4, 5)]
    rows += [(u, 1, 1, T2) for u in (6, 7)] + [(8, 0, 1, T2)]  # item 0 appears only in the test period
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


@pytest.fixture
def sources(tmp_path):
    author = write_author(tmp_path / "author")
    freebase = write_freebase(tmp_path / "freebase-subset.tsv")
    split = temporal_split(interactions(), SplitConfig(min_rating=1, kcore=2))
    assert (split.t1, split.t2) == (T1, T2)  # the fact dates below are chosen against these cut-offs
    assert check_split(split) == []  # the synthetic split follows the protocol too
    enc = encode(split)
    items = item_entities(enc.item_ids, author / "item_list.txt")
    triples, entity_ids, relations = read_author_kg(author)
    tails = read_freebase_subset(freebase, set(entity_ids.loc[triples["t"].unique()]))
    facts, dropped = lastfm_facts(items, triples, entity_ids, relations, tails)
    return dict(
        author=author,
        freebase=freebase,
        split=split,
        enc=enc,
        items=items,
        triples=triples,
        relations=relations,
        facts=facts,
        format_dropped=dropped,
        tails=tails,
    )


def test_the_split_has_the_expected_train_items(sources):
    assert list(sources["enc"].item_ids) == TRAIN_ITEMS  # item 0 is a test-period item, so never a train item
    assert list(sources["items"]["freebase_id"]) == ["m.item0", "m.item1", "m.item2", "m.item3"]
    assert list(sources["items"]["h"]) == [0, 1, 2, 3]  # the item index is not the author id


def test_freebase_is_read_only_for_the_entities_asked_for(sources):
    tails = sources["tails"]
    assert "m.unused" not in tails.index  # not an entity of the author KG
    assert tails.loc["m.old", "tail_label"] == "An Old Artist"
    assert tails.loc["m.old", "tail_date_of_birth"] == DATE.format("1946-10-18")
    assert tails.loc["m.nodate", list(DATE_COLUMNS)].tolist() == ["", "", ""]


def test_two_values_for_one_date_predicate_are_refused(tmp_path):
    path = tmp_path / "dup.tsv"
    path.write_text(
        "subject\tpredicate\tobject\n"
        f"m.a\tmusic.release.release_date\t{GYEAR.format('1999')}\n"
        f"m.a\tmusic.release.release_date\t{GYEAR.format('2000')}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two values"):
        read_freebase_subset(path, {"m.a"})


# --- the 1-hop shape ------------------------------------------------------------------------------------


def test_the_item_seeded_filter_keeps_triples_with_an_item_at_either_end(sources):
    seeded = seeded_triples(sources["items"], sources["triples"])
    assert len(seeded) == len(AUTHOR_TRIPLES)  # every synthetic triple touches a train item


def test_a_triple_whose_head_is_not_a_train_item_is_dropped_with_its_reason(sources):
    dropped = sources["format_dropped"]
    assert len(dropped) == 1 and (dropped["reason"] == HEAD_NOT_ITEM).all()
    assert dropped["author_head"].tolist() == [0] and dropped["author_tail"].tolist() == [2]
    assert len(sources["facts"]) == len(AUTHOR_TRIPLES) - 1


def test_a_tail_that_is_also_a_train_item_stays_an_attribute_entity(sources):
    # the author KG has one id space for items and entities, the repository's format has two
    item_to_item = sources["facts"].query("relation == 'music.recording.canonical_version'")
    assert item_to_item["h"].tolist() == [2] and item_to_item["tail"].tolist() == ["m.item0"]


def test_every_head_is_a_train_item(sources):
    h = sources["facts"]["h"].to_numpy()
    assert h.min() >= 0 and h.max() < sources["enc"].n_items


# --- the time-consistency rule --------------------------------------------------------------------------


def kept_tails(sources, ts: int, variant: str = "default") -> set[str]:
    kept, _ = cutoff_facts(sources["facts"], ts, variant)
    return set(kept["tail"])


def test_a_tail_dated_after_the_cutoff_is_dropped_and_comes_back_later(sources):
    assert "m.future" not in kept_tails(sources, T1)  # released 2014-05-10, after t1
    assert "m.future" in kept_tails(sources, T2)


def test_a_month_only_tail_date_is_the_last_day_of_that_month(sources):
    assert "m.month" not in kept_tails(sources, T1)  # 2014-04 -> 30 April, after t1 (28 April)
    assert "m.month" in kept_tails(sources, T2)


def test_an_old_tail_is_kept_at_both_cutoffs(sources):
    assert "m.old" in kept_tails(sources, T1) and "m.old" in kept_tails(sources, T2)


def test_the_head_items_own_release_never_drops_its_facts(sources):
    # m.item0 is released in 2001 and m.item1's artist starts in 1990, but a head's own date is not a filter
    # (decision 0006 B); the fact of item 2 -> m.item0 is dated by the *tail*, which is 2001
    assert kept_tails(sources, T1) >= {"m.item0"}


@pytest.mark.parametrize("ts", [T1, T2])
@pytest.mark.parametrize("variant", ["default", "drop_undated", "static"])
def test_check_kg_passes_on_what_the_builder_keeps(sources, ts, variant):
    kept, _ = cutoff_facts(sources["facts"], ts, variant)
    assert check_kg(kept, ts, variant, sources["enc"].n_items) == []


@pytest.mark.parametrize("variant", ["default", "drop_undated"])
def test_check_kg_reports_a_future_tail_that_slipped_through(sources, variant):
    kept, _ = cutoff_facts(sources["facts"], T2, variant)  # holds m.future, dated after t1
    problems = check_kg(kept, T1, variant, sources["enc"].n_items)
    assert any("exists only on/after" in p for p in problems)


def test_check_kg_is_independent_of_the_annotated_columns(sources):
    """A wrong ``tail_date`` cannot hide a leak: the check re-parses the raw literals."""
    kept, _ = cutoff_facts(sources["facts"], T2, "default")
    lying = kept.assign(tail_date=None, statement_date=None)
    assert any("exists only on/after" in p for p in check_kg(lying, T1, "default", sources["enc"].n_items))


def test_check_kg_reports_a_head_outside_the_item_range(sources):
    kept, _ = cutoff_facts(sources["facts"], T1, "default")
    assert "head entity outside the train item range" in check_kg(kept, T1, "default", 1)


def test_check_kg_reports_duplicate_triples(sources):
    kept, _ = cutoff_facts(sources["facts"], T1, "default")
    doubled = pd.concat([kept, kept.iloc[:1]], ignore_index=True)
    assert "duplicate triples" in check_kg(doubled, T1, "default", sources["enc"].n_items)


def test_the_kg_has_no_statement_dates(sources):
    assert sources["facts"]["statement_date"].isna().all()


def test_annotate_takes_the_earliest_of_the_three_tail_dates():
    row = dict.fromkeys(DATE_COLUMNS, "")
    row.update(
        tail_release_date=DATE.format("2005-06-07"),
        tail_active_start=GYEAR.format("1990"),
        tail_date_of_birth=DATE.format("1972-02-29"),
    )
    out = annotate_lastfm(pd.DataFrame([row]))
    assert out.loc[0, "tail_date"] == dt.date(1972, 2, 29)


# --- the variants ---------------------------------------------------------------------------------------


def test_the_variants_differ_only_where_expected(sources):
    facts = sources["facts"]
    key = ["h", "relation", "tail"]
    kept = {v: cutoff_facts(facts, T1, v)[0].set_index(key).index for v in ("default", "drop_undated", "static")}
    # static = every fact of the train items, no time filter; default = static minus the future tails
    assert set(kept["static"]) - set(kept["default"]) == {
        (1, "music.recording.artist", "m.future"),
        (1, "common.topic.notable_types", "m.month"),
    }
    # drop_undated = default minus the facts with no usable date at all
    assert set(kept["default"]) - set(kept["drop_undated"]) == {(0, "common.topic.notable_types", "m.nodate")}
    assert set(kept["drop_undated"]) <= set(kept["default"]) <= set(kept["static"])


def test_static_is_the_same_graph_at_both_cutoffs(sources):
    key = ["h", "relation", "tail"]
    at = {c: cutoff_facts(sources["facts"], c, "static")[0].set_index(key).index for c in (T1, T2)}
    assert set(at[T1]) == set(at[T2])


def test_the_kg_at_t1_is_contained_in_the_kg_at_t2(sources):
    key = ["h", "relation", "tail"]
    for variant in ("default", "drop_undated", "static"):
        t1 = cutoff_facts(sources["facts"], T1, variant)[0].set_index(key).index
        t2 = cutoff_facts(sources["facts"], T2, variant)[0].set_index(key).index
        assert set(t1) <= set(t2), variant


def test_stage2_dates_follow_the_triples_and_are_nan_when_undated(sources):
    from tkgat.kg.temporal import assign_ids

    kept = {c: cutoff_facts(sources["facts"], ts, "default")[0] for c, ts in (("t1", T1), ("t2", T2))}
    universe, _ = cutoff_facts(sources["facts"], T1, "static")
    ent, rel, tri = assign_ids(sources["enc"].n_items, sources["items"], kept, sources["relations"], universe=universe)
    dates = triple_dates(kept["t1"], ent, rel)
    assert dates[["h", "r", "t"]].equals(tri["t1"][["h", "r", "t"]])
    undated = ent.loc[ent["key"] == "m.nodate", "entity"].item()
    assert np.isnan(dates.loc[dates["t"] == undated, "ts"]).all()
    assert dates["ts"].notna().sum() == len(dates) - 1


# --- the script -----------------------------------------------------------------------------------------


def load_build_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return runpy.run_path(str(ROOT / "scripts" / "06_build_kg_lastfm.py"), run_name="not_main")
    finally:
        sys.path.pop(0)


@pytest.fixture
def built(sources, tmp_path, monkeypatch):
    cfg = {
        "name": "lastfm",
        "run_name": "lastfm-synthetic",
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "clean": False,
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 2},
    }
    save_split(sources["split"], cfg, {})
    data_cfg = tmp_path / "lastfm-synthetic.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    script = load_build_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "06_build_kg_lastfm.py",
            "--data",
            str(data_cfg),
            "--author-data",
            str(sources["author"]),
            "--freebase",
            str(sources["freebase"]),
        ],
    )
    script["main"]()
    return cfg


@pytest.mark.parametrize("variant", ["default", "drop_undated", "static"])
def test_the_output_loads_through_load_kg_and_load_kg_dates(built, variant):
    cfg = built
    split, meta = load_saved(cfg)
    enc = encode(split)
    processed = Path(cfg["processed_dir"])
    kg1 = load_kg(processed, variant, "t1", enc, meta)
    kg2 = load_kg(processed, variant, "t2", enc, meta)
    ts1 = load_kg_dates(processed, kg1)
    assert len(ts1) == len(kg1) and kg1.n_relations == len(AUTHOR_RELATIONS)
    assert kg1.time_consistent == (variant != "static")
    assert kg1.wikidata_snapshot is None  # Last-FM uses Freebase, not Wikidata
    ckg = build_ckg(enc, kg1, kg_ts=ts1)  # the guards of KGTriples and build_ckg accept the files
    assert ckg.n_triples == len(kg1) and ckg.n_interactions == len(enc.train)
    assert len(kg2) >= len(kg1)


def test_the_meta_records_the_sources_and_the_dropped_triples(built):
    info = json.loads((Path(built["processed_dir"]) / "kg" / "default" / "kg_meta.json").read_text(encoding="utf-8"))
    assert info["variant"] == "default" and info["wikidata_snapshot"] is None
    assert "Freebase dump 2015-08-09" in info["kg_snapshot"]
    assert info["n_item_seeded_triples"] == len(AUTHOR_TRIPLES)
    assert info["n_head_seeded_triples"] == len(AUTHOR_TRIPLES) - 1
    assert info["n_dropped_head_not_train_item"] == 1
    assert info["n_triples"]["t1"] == len(AUTHOR_TRIPLES) - 3  # minus the non-item head and the two future tails
    assert info["inputs"]["split_sha256"]
    assert any("1-hop" in limitation for limitation in info["known_limitations"])


def test_dropped_csv_lists_every_reason(built):
    dropped = pd.read_csv(Path(built["processed_dir"]) / "kg" / "default" / "dropped.csv")
    assert set(dropped["reason"]) == {"future_tail", HEAD_NOT_ITEM}
    assert set(dropped["cutoff"]) == {"t1", "t2"}


# --- the shipped model configs on a Last-FM run ----------------------------------------------------------
#
# Nothing in the KGAT / T-KGAT configs names a dataset: scripts/run.py reads the selected point from
# results/runs/<run_name>/<tag>_summary.json, and `run_name` comes from the data config. These tests check
# that the shipped configs therefore resolve for `run_name: lastfm` with no new model config.


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


KGAT_POINT = {"lr": 0.001, "l2": 0.00001, "l2_kg": 0.00001, "mess_dropout": 0.1}
MODELS = ROOT / "configs" / "model"


@pytest.fixture
def lastfm_runs(tmp_path):
    """``results/runs/lastfm/`` as it looks after the KGAT grid of the shipped kgat.yaml was selected."""
    out = tmp_path / "runs" / "lastfm"
    out.mkdir(parents=True)
    assert KGAT_POINT in yaml.safe_load((MODELS / "kgat.yaml").read_text(encoding="utf-8"))["grid"]
    (out / "kgat_summary.json").write_text(json.dumps({"selected": KGAT_POINT}), encoding="utf-8")
    return out


@pytest.mark.parametrize("tag", ["tkgat", "tkgat_kg"])
def test_run_py_resolves_from_kgat_for_a_lastfm_run(lastfm_runs, tag):
    run = load_run_script()
    mcfg = yaml.safe_load((MODELS / f"{tag}.yaml").read_text(encoding="utf-8"))
    points, base = run.tkgat_points(mcfg, MODELS / f"{tag}.yaml", lastfm_runs)
    assert base == KGAT_POINT
    assert [p["gamma"] for p in points] == [1, 3, 10]
    for point in points:  # (lr, l2, l2_kg, mess_dropout) come from the KGAT run, never from the file
        assert {k: point[k] for k in KGAT_POINT} == KGAT_POINT
        assert point.get("time_on") == ("kg" if tag == "tkgat_kg" else None)


@pytest.mark.parametrize("tag, source", [("tkgat_perm", "tkgat"), ("tkgat_kg_perm", "tkgat_kg")])
def test_run_py_resolves_from_tkgat_for_a_lastfm_run(lastfm_runs, tag, source):
    run = load_run_script()
    selected = {**KGAT_POINT, "gamma": 3, **({"time_on": "kg"} if source == "tkgat_kg" else {})}
    (lastfm_runs / f"{source}_summary.json").write_text(json.dumps({"selected": selected}), encoding="utf-8")
    mcfg = yaml.safe_load((MODELS / f"{tag}.yaml").read_text(encoding="utf-8"))
    points, base = run.tkgat_points(mcfg, MODELS / f"{tag}.yaml", lastfm_runs)
    assert points == [selected] and base == selected  # a control re-runs exactly the selected point


@pytest.mark.parametrize("tag", ["tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm"])
def test_a_lastfm_run_needs_its_own_kgat_run_first(lastfm_runs, tag):
    run = load_run_script()
    (lastfm_runs / "kgat_summary.json").unlink()
    mcfg = yaml.safe_load((MODELS / f"{tag}.yaml").read_text(encoding="utf-8"))
    with pytest.raises(FileNotFoundError, match="run .* first"):
        run.tkgat_points(mcfg, MODELS / f"{tag}.yaml", lastfm_runs)


def test_the_kg_variant_of_the_shipped_configs_exists_for_lastfm(built):
    for tag in ("kgat", "tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm"):
        variant = yaml.safe_load((MODELS / f"{tag}.yaml").read_text(encoding="utf-8"))["kg"]["variant"]
        assert (Path(built["processed_dir"]) / "kg" / variant).is_dir(), tag
