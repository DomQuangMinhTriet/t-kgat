import csv
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.temporal import (
    annotate,
    assign_ids,
    build_kg,
    check_kg,
    cutoff_day,
    end_of_period,
    item_entities,
)

from .conftest import ROOT

COLS = [
    "freebase_id",
    "relation",
    "tail",
    "tail_label",
    "point_in_time",
    "point_in_time_precision",
    "start_time",
    "tail_publication",
    "tail_inception",
    "tail_birth",
    "film_publication",
]
T1 = int(dt.datetime(1998, 3, 7, 12, tzinfo=dt.timezone.utc).timestamp())
T2 = int(dt.datetime(1998, 3, 31, 12, tzinfo=dt.timezone.utc).timestamp())


def fact(fb="m.a", relation="genre", tail="Q1", **dates) -> dict:
    row = dict.fromkeys(COLS, "")
    row.update(freebase_id=fb, relation=relation, tail=tail, tail_label=tail, **dates)
    return row


def facts(*rows: dict) -> pd.DataFrame:
    return annotate(pd.DataFrame(list(rows), columns=COLS))


ITEMS = pd.DataFrame({"h": [0, 1, 2], "item": [10, 20, 30], "freebase_id": ["m.a", "m.b", None]})


def kept_tails(f: pd.DataFrame, ts: int, variant: str = "default") -> set[str]:
    kept, _ = build_kg(f, ITEMS, ts, variant)
    return set(kept["tail"])


def test_end_of_period_by_precision():
    assert end_of_period("1998-03-23", 11) == dt.date(1998, 3, 23)
    assert end_of_period("1996-02-01", 10) == dt.date(1996, 2, 29)  # leap year
    assert end_of_period("1997-02-01", 10) == dt.date(1997, 2, 28)
    assert end_of_period("1998-01-01", 9) == dt.date(1998, 12, 31)
    assert end_of_period("1990-01-01", 8) == dt.date(1999, 12, 31)
    assert end_of_period("1998-01-01", 3) is None  # precision outside 8-11 = no date (0006 C)
    assert end_of_period("", 11) is None
    assert end_of_period("-0427-01-0") is None  # BC year, unparseable


def test_dates_without_precision():
    assert end_of_period("1998-01-01") == dt.date(1998, 12, 31)  # read as year-only (0006 A)
    assert end_of_period("1998-02-01") == dt.date(1998, 2, 1)
    assert end_of_period("1998-05-14") == dt.date(1998, 5, 14)


def test_cutoff_day_is_utc():
    assert cutoff_day(T1) == dt.date(1998, 3, 7)
    assert cutoff_day(889237198) == dt.date(1998, 3, 7)  # ML-100K t1


def test_titanic_award_between_t1_and_t2():
    f = facts(
        fact(
            relation="award_received",
            tail="Q_oscar",
            point_in_time="1998-03-23",
            point_in_time_precision="11",
            tail_inception="1929-01-01",
        )
    )
    assert kept_tails(f, T1) == set()
    assert kept_tails(f, T2) == {"Q_oscar"}


def test_statement_date_used_and_target_date_still_checked():
    f = facts(
        # statement before t1, award founded long ago -> kept
        fact(
            tail="Q_old",
            relation="award_received",
            point_in_time="1997-03-24",
            point_in_time_precision="11",
            tail_inception="1929-01-01",
        ),
        # statement before t1 but the target exists only later -> dropped (0006 rule)
        fact(
            tail="Q_new",
            relation="award_received",
            point_in_time="1997-03-24",
            point_in_time_precision="11",
            tail_inception="1998-03-10",
        ),
        # start_time is used when point_in_time is missing
        fact(tail="Q_start", relation="award_received", start_time="1998-03-20"),
    )
    assert kept_tails(f, T1) == {"Q_old"}
    assert kept_tails(f, T2) == {"Q_old", "Q_new", "Q_start"}  # Q_new exists by t2


def test_sequel_released_after_cutoff_is_dropped():
    f = facts(
        fact(relation="follows", tail="Q_prequel", tail_publication="1995-06-30"),
        fact(relation="followed_by", tail="Q_sequel", tail_publication="1999-11-24"),
    )
    assert kept_tails(f, T2) == {"Q_prequel"}


def test_year_only_and_boundary_day():
    f = facts(
        fact(relation="based_on", tail="Q_1998", tail_publication="1998-01-01"),  # -> 1998-12-31
        fact(relation="based_on", tail="Q_onday", tail_publication="1998-03-07"),
        fact(relation="based_on", tail="Q_before", tail_publication="1998-03-06"),
    )
    assert kept_tails(f, T1) == {"Q_before"}  # strict: a date on the cut-off day is dropped


def test_film_release_never_drops_its_facts():
    f = facts(fact(tail="Q_drama", film_publication="1998-01-01"))  # 31 Dec 1998 > t1, still kept (0006 B)
    assert kept_tails(f, T1) == {"Q_drama"}


def test_unusable_precision_follows_undated_rule():
    f = facts(fact(relation="award_received", tail="Q_x", point_in_time="2001-01-01", point_in_time_precision="3"))
    assert kept_tails(f, T1, "default") == {"Q_x"}
    assert kept_tails(f, T1, "drop_undated") == set()


def test_variants_and_monotonicity():
    f = facts(
        fact(tail="Q_genre"),
        fact(relation="director", tail="Q_dir", tail_birth="1950-05-05"),
        fact(relation="award_received", tail="Q_aw", point_in_time="1997-03-24", point_in_time_precision="11"),
        fact(fb="m.b", relation="award_received", tail="Q_t", point_in_time="1998-03-23", point_in_time_precision="11"),
    )
    assert kept_tails(f, T1, "default") == {"Q_genre", "Q_dir", "Q_aw"}
    assert kept_tails(f, T1, "drop_undated") == {"Q_dir", "Q_aw"}  # a birth date dates the fact (0006 D)
    for v in ("default", "drop_undated"):
        assert kept_tails(f, T1, v) <= kept_tails(f, T2, v)
    assert kept_tails(f, T2, "drop_undated") <= kept_tails(f, T2, "default")


def test_undated_means_no_usable_date_at_all():
    f = facts(
        fact(relation="award_received", tail="Q_tail_only", tail_inception="1929-01-01"),
        fact(
            relation="award_received",
            tail="Q_bad_both",
            point_in_time="1990-01-01",
            point_in_time_precision="3",
            tail_birth="-0427-01-0",
        ),
        fact(tail="Q_genre"),
    )
    assert kept_tails(f, T1, "drop_undated") == {"Q_tail_only"}
    assert kept_tails(f, T1, "default") == {"Q_tail_only", "Q_bad_both", "Q_genre"}


def test_static_keeps_everything_and_is_not_date_checked():
    f = facts(
        fact(relation="award_received", tail="Q_aw", point_in_time="1998-03-23", point_in_time_precision="11"),
        fact(relation="followed_by", tail="Q_seq", tail_publication="1999-11-24"),
        fact(tail="Q_genre"),
        fact(fb="m.zzz", tail="Q_other_film"),
    )  # still restricted to train items
    k1, d1 = build_kg(f, ITEMS, T1, "static")
    k2, _ = build_kg(f, ITEMS, T2, "static")
    assert set(k1["tail"]) == set(k2["tail"]) == {"Q_aw", "Q_seq", "Q_genre"} and d1.empty
    assert check_kg(k1, T1, "static", n_items=3) == []
    assert check_kg(k1, T1, "default", n_items=3) != []  # the same KG fails as a time-consistent one


def test_variants_share_entity_ids():
    f = facts(
        fact(tail="Q_genre"),
        fact(relation="award_received", tail="Q_aw", point_in_time="1998-03-23", point_in_time_precision="11"),
        fact(relation="followed_by", tail="Q_seq", tail_publication="1999-11-24"),
    )
    universe, _ = build_kg(f, ITEMS, T1, "static")
    rels = ["award_received", "followed_by", "genre"]
    ents = {}
    for v in ("default", "drop_undated", "static"):
        kgs = {c: build_kg(f, ITEMS, ts, v)[0] for c, ts in (("t1", T1), ("t2", T2))}
        ents[v], _, _ = assign_ids(3, ITEMS, kgs, rels, universe=universe)
    assert ents["default"].equals(ents["drop_undated"]) and ents["default"].equals(ents["static"])
    with pytest.raises(ValueError, match="not in the universe"):
        assign_ids(3, ITEMS, {"t1": universe}, rels, universe=universe.iloc[:1])


def test_dropped_reasons_and_unlinked_items():
    f = facts(
        fact(relation="award_received", tail="Q_aw", point_in_time="1998-03-23", point_in_time_precision="11"),
        fact(relation="followed_by", tail="Q_seq", tail_publication="1999-11-24"),
        fact(tail="Q_genre"),
        fact(fb="m.zzz", tail="Q_other_film"),
    )  # film not among the train items
    kept, dropped = build_kg(f, ITEMS, T1, "drop_undated")
    assert kept.empty
    assert dict(zip(dropped["tail"], dropped["reason"], strict=True)) == {
        "Q_aw": "future_statement",
        "Q_seq": "future_tail",
        "Q_genre": "undated",
    }


def test_check_kg_catches_injected_future_facts():
    f = facts(fact(tail="Q_genre"), fact(relation="director", tail="Q_dir", tail_birth="1950-05-05"))
    kept, _ = build_kg(f, ITEMS, T1, "default")
    assert check_kg(kept, T1, "default", n_items=3) == []
    leak = facts(
        fact(relation="award_received", tail="Q_aw", point_in_time="1998-03-23", point_in_time_precision="11"),
        fact(relation="followed_by", tail="Q_seq", tail_publication="1999-11-24"),
    )
    leak = ITEMS.iloc[[0]].merge(leak, on="freebase_id")  # bypass build_kg on purpose
    bad = pd.concat([kept, leak], ignore_index=True)
    problems = check_kg(bad, T1, "default", n_items=3)
    assert any("statement date" in p for p in problems) and any("target exists" in p for p in problems)
    assert any("undated" in p for p in check_kg(kept, T1, "drop_undated", n_items=3))
    assert any("item range" in p for p in check_kg(kept.assign(h=5), T1, "default", n_items=3))


def test_item_ids_come_from_encode(synthetic):
    s = temporal_split(synthetic, SplitConfig(kcore=2))
    enc = encode(s)
    raw = sorted(set(synthetic["item"]))
    links = pd.DataFrame({"item": raw[::-1], "freebase_id": [f"m.{r}" for r in raw[::-1]]})  # unordered on purpose
    items = item_entities(enc.item_ids, links, "item")
    f = facts(*[fact(fb=f"m.{r}", tail=f"Q{r % 3}") for r in raw])
    kept, _ = build_kg(f, items, s.t1, "default")
    ent, rel, tri = assign_ids(enc.n_items, items, {"t1": kept}, sorted(set(kept["relation"])))
    fb_of = dict(zip(links["item"], links["freebase_id"], strict=True))
    for h, fb in zip(kept["h"], kept["freebase_id"], strict=True):
        assert fb_of[int(enc.item_ids[h])] == fb
    assert (ent["entity"].to_numpy() == np.arange(len(ent))).all()
    assert (ent.loc[: enc.n_items - 1, "key"].astype(int).to_numpy() == enc.item_ids).all()
    assert tri["t1"]["t"].min() >= enc.n_items and tri["t1"]["h"].max() < enc.n_items


def test_entity_ids_shared_across_cutoffs():
    f = facts(
        fact(tail="Q_genre"),
        fact(relation="award_received", tail="Q_aw", point_in_time="1998-03-23", point_in_time_precision="11"),
    )
    k1, _ = build_kg(f, ITEMS, T1, "default")
    k2, _ = build_kg(f, ITEMS, T2, "default")
    ent, _, tri = assign_ids(3, ITEMS, {"t1": k1, "t2": k2}, ["award_received", "genre"])
    assert set(map(tuple, tri["t1"].to_numpy())) <= set(map(tuple, tri["t2"].to_numpy()))
    assert list(ent["key"]) == ["10", "20", "30", "Q_aw", "Q_genre"]


ML100K = ROOT / "data" / "processed" / "ml-100k"
FACTS = ROOT / "data" / "raw" / "wikidata" / "facts.tsv"


@pytest.mark.skipif(not (ML100K / "meta.json").exists() or not FACTS.exists(), reason="ML-100K KG data not built")
def test_real_ml100k_kg_has_no_future_facts():
    from tkgat.pipeline.build import load_config, load_saved

    s, meta = load_saved(load_config(ROOT / "configs" / "data" / "ml100k.yaml"))
    enc = encode(s)
    links = pd.read_csv(ROOT / "data" / "processed" / "links" / "ml100k_links.csv", dtype={"freebase_id": str})
    items = item_entities(enc.item_ids, links, "item_100k")
    raw = pd.read_csv(FACTS, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    f = annotate(raw[raw["freebase_id"].isin(set(items["freebase_id"].dropna()))])
    fb_of = dict(zip(links["item_100k"], links["freebase_id"], strict=True))
    for variant in ("default", "drop_undated", "static"):
        k1, _ = build_kg(f, items, meta["t1"], variant)
        k2, _ = build_kg(f, items, meta["t2"], variant)
        assert check_kg(k1, meta["t1"], variant, enc.n_items) == []
        assert check_kg(k2, meta["t2"], variant, enc.n_items) == []
        assert all(fb_of[int(enc.item_ids[h])] == fb for h, fb in zip(k1["h"], k1["freebase_id"], strict=True))
    titanic = (
        (f["freebase_id"] == "m.0dr_4") & (f["relation"] == "award_received") & (f["point_in_time"] == "1998-03-23")
    )
    if titanic.any():
        k1, _ = build_kg(f, items, meta["t1"], "default")
        k2, _ = build_kg(f, items, meta["t2"], "default")
        # compare (relation, tail): the same award is also a nomination dated 1998-02-10, before t1
        oscars = set(zip(f.loc[titanic, "relation"], f.loc[titanic, "tail"], strict=True))

        def pairs(k: pd.DataFrame) -> set[tuple[str, str]]:
            t = k[k["freebase_id"] == "m.0dr_4"]
            return set(zip(t["relation"], t["tail"], strict=True))

        assert not oscars & pairs(k1)
        assert oscars <= pairs(k2)
