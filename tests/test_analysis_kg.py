import pandas as pd
import pytest

from tkgat.analysis import kg

META = {
    "time_consistent": True,
    "n_triples": {"t1": 10, "t2": 12},
    "date_kind_per_relation": [
        {"relation": "genre", "facts": 5, "statement_dated": 0.0, "tail_date_only": 0.0, "no_date": 1.0},
        {"relation": "award_received", "facts": 8, "statement_dated": 0.75, "tail_date_only": 0.25, "no_date": 0.0},
    ],
    "per_relation": [
        {"relation": "genre", "cutoff": "t1", "kept": 5, "future_statement": 0, "future_tail": 0, "undated": 0},
        {"relation": "genre", "cutoff": "t2", "kept": 5, "future_statement": 0, "future_tail": 0, "undated": 0},
        {
            "relation": "award_received",
            "cutoff": "t1",
            "kept": 5,
            "future_statement": 3,
            "future_tail": 0,
            "undated": 0,
        },
        {
            "relation": "award_received",
            "cutoff": "t2",
            "kept": 7,
            "future_statement": 1,
            "future_tail": 0,
            "undated": 0,
        },
    ],
}

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


def fact(relation, tail, **dates):
    row = dict.fromkeys(COLS, "")
    row.update(freebase_id="m.x", relation=relation, tail=tail, tail_label=tail, **dates)
    return row


def test_relation_date_kinds_sorted_by_facts():
    t = kg.relation_date_kinds(META)
    assert list(t.index) == ["award_received", "genre"] and t.loc["genre", "no_date"] == 1.0


def test_relation_date_kinds_splits_wikidata_and_movielens():
    meta = {
        "date_kind_per_relation": [
            *META["date_kind_per_relation"],
            {"relation": "movielens.genre", "facts": 9, "statement_dated": 1.0, "tail_date_only": 0.0, "no_date": 0.0},
        ]
    }
    assert list(kg.relation_date_kinds(meta, "wikidata").index) == ["award_received", "genre"]
    assert list(kg.relation_date_kinds(meta, "movielens").index) == ["movielens.genre"]
    assert kg.relation_date_kinds(meta)["facts"].sum() == 22
    with pytest.raises(ValueError):
        kg.relation_date_kinds(meta, "freebase")


def test_dropped_by_relation_wide():
    t = kg.dropped_by_relation(META)
    assert t.loc["award_received", "future_statement_t1"] == 3
    assert t.loc["award_received", "kept_t2"] == 7 and "undated_t2" in t.columns


def test_variant_triples():
    t = kg.variant_triples({"default": META, "static": {**META, "time_consistent": False}})
    assert t.loc["default", "t2"] == 12 and t.loc["static", "time_consistent"] is False


def test_date_precision():
    f = pd.DataFrame(
        [
            fact("award_received", "Q1", point_in_time="1998-03-23", point_in_time_precision="11"),
            fact("award_received", "Q2", point_in_time="1998-01-01", point_in_time_precision="9"),
            fact("award_received", "Q3", point_in_time="1000-01-01", point_in_time_precision="3"),
            fact("director", "Q4", tail_birth="1950-01-01", film_publication="1997-11-01"),
            fact("director", "Q5", tail_birth="1950-05-05", film_publication="1997-01-01"),
        ],
        columns=COLS,
    )
    by_prec, jan1 = kg.date_precision(f)
    assert by_prec.to_dict() == {"day": 1, "year": 1, "other (3)": 1}
    assert jan1.loc["tail_birth", "filled"] == 2 and jan1.loc["tail_birth", "share_jan1"] == pytest.approx(0.5)
    assert jan1.loc["start_time", "filled"] == 0


def test_item_fact_counts_maps_heads_to_raw_ids():
    ent = pd.DataFrame({"entity": [0, 1, 2], "kind": ["item", "item", "wikidata"], "key": ["10", "20", "Q1"]})
    tri = pd.DataFrame({"h": [0, 0, 1], "r": [0, 1, 0], "t": [2, 2, 2]})
    assert kg.item_fact_counts(tri, ent).to_dict() == {10: 2, 20: 1}


def test_popular_concentration():
    # item 1 is the most popular (3 positives) of 4 items; top 25% = {1}
    train = pd.DataFrame({"item": [1, 1, 1, 2, 2, 3, 4]})
    fact_counts = pd.Series({1: 2, 2: 3, 3: 3, 4: 2})
    dropped = pd.DataFrame(
        {
            "item": ["1", "1", "2", "1"],
            "relation": ["award_received", "nominated_for", "award_received", "followed_by"],
            "reason": ["future_statement", "future_statement", "future_statement", "future_statement"],
            "cutoff": "t1",
        }
    )
    c = kg.popular_concentration(fact_counts, dropped, train, frac=0.25)
    assert c["top_items"] == 1 and c["share_of_all_facts"] == pytest.approx(0.2)
    assert c["future_award_facts"] == 3  # the followed_by fact is not an award/nomination
    assert c["share_of_future_award_facts"] == pytest.approx(2 / 3) and c["films_losing_award_facts"] == 2
    with pytest.raises(ValueError):
        kg.popular_concentration(fact_counts, pd.concat([dropped, dropped.assign(cutoff="t2")]), train)


def test_film_fact_status():
    film = pd.DataFrame(
        [
            fact("nominated_for", "Q_pic", point_in_time="1998-02-10", point_in_time_precision="11"),
            fact("award_received", "Q_pic", point_in_time="1998-03-23", point_in_time_precision="11"),
            fact("followed_by", "Q_seq", tail_publication="1999-11-24"),
            fact("genre", "Q_drama"),
        ],
        columns=COLS,
    )
    dropped = pd.DataFrame(
        {
            "freebase_id": "m.x",
            "relation": ["award_received", "followed_by", "followed_by"],
            "tail": ["Q_pic", "Q_seq", "Q_seq"],
            "reason": ["future_statement", "future_tail", "future_tail"],
            "cutoff": ["t1", "t1", "t2"],
        }
    )
    s = kg.film_fact_status(film, dropped).set_index(["relation", "tail"])
    assert s.loc[("nominated_for", "Q_pic"), "status"] == "kept_t1"
    assert s.loc[("award_received", "Q_pic"), "status"] == "added_t2"
    assert s.loc[("award_received", "Q_pic"), "reason_t1"] == "future_statement"
    assert s.loc[("followed_by", "Q_seq"), "status"] == "dropped_both"
    assert s.loc[("genre", "Q_drama"), "status"] == "kept_t1" and s.loc[("genre", "Q_drama"), "reason_t1"] == ""
    assert str(s.loc[("award_received", "Q_pic"), "statement_date"]) == "1998-03-23"


def test_items_with_filtered_facts_and_positive_share():
    ent = pd.DataFrame(
        {"entity": [0, 1, 2, 3], "kind": ["item", "item", "item", "wikidata"], "key": ["10", "20", "30", "Q1"]}
    )
    static = pd.DataFrame({"h": [0, 0, 1, 2], "r": [0, 1, 0, 0], "t": [3, 3, 3, 3]})
    filtered = pd.DataFrame({"h": [0, 1], "r": [0, 0], "t": [3, 3]})  # item 10 loses one fact, item 30 all
    assert kg.items_with_filtered_facts(static, filtered, ent) == {10, 30}
    assert kg.items_with_filtered_facts(static, static, ent) == set()
    val = pd.DataFrame({"user": [1, 1, 2, 3], "item": [10, 20, 30, 20]})
    assert kg.positive_share(val, {10, 30}) == pytest.approx(0.5)
    assert kg.positive_share(val.iloc[:0], {10}) == 0.0
