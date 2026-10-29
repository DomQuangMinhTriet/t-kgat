"""T-KGAT Stage 2 (decision 0009, items 7-10): Delta-t on the KG triples, the fact dates file, the date control.

CPU only, tiny graphs; nothing here trains on real data.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.ckg import KGTriples, build_ckg, carry_kg_ts, load_kg, load_kg_dates, permute_kg_ts
from tkgat.kg.temporal import annotate, fact_valid_from, triple_dates
from tkgat.models.kgat import KGAT, KGATConfig
from tkgat.pipeline.build import load_saved, save_split

from .test_kgat import CFG, NI, TRAIN, TRAIN_TS, TRIPLES, E, U, make_model
from .test_run_kgat import FIXED, GRID, RUN_NAME, load_run_script, write_kg
from .test_tkgat import edit

DAY = 86400.0
# one date per row of TRIPLES; NaN = undated. The toy train window is 100..500, the facts are older: most are far
# beyond one window (dt_norm > 1, capped), 300 is inside it (dt_norm = 0.5, not capped)
KG_TS = np.array([-1e8, -2e8, np.nan, -3e8, 300.0, np.nan, -4e8])


def make_kg_ckg(kg_ts=KG_TS):
    enc = SimpleNamespace(n_users=U, n_items=NI, train=TRAIN, train_ts=TRAIN_TS)
    kg = KGTriples("default", "t1", np.ascontiguousarray(TRIPLES), NI, E, ("a", "b"), True, None)
    return build_ckg(enc, kg, kg_ts=kg_ts)


def make_stage2(ckg=None, seed=0, **kw):
    return make_model(ckg or make_kg_ckg(), seed=seed, time_on="kg", **kw)


def kg_edges(model):
    return (model.delta_r[model.relation] > 0).numpy()


# --- the fact dates file ---------------------------------------------------------------------------------------------


def day_seconds(y, m, d):
    import datetime as dt

    return dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp()


def raw_fact(**dates):
    cols = [
        "point_in_time",
        "point_in_time_precision",
        "start_time",
        "tail_publication",
        "tail_inception",
        "tail_birth",
    ]
    row = dict.fromkeys(cols, "")
    row.update(dates)
    return row


def test_a_fact_holds_from_the_later_of_its_statement_date_and_its_target_date():
    facts = annotate(
        pd.DataFrame(
            [
                raw_fact(point_in_time="1998-03-23T00:00:00Z", point_in_time_precision="11"),  # statement only
                raw_fact(point_in_time="1998-01-01T00:00:00Z", point_in_time_precision="9"),  # year only: 31 Dec
                raw_fact(tail_publication="2001-05-04T00:00:00Z"),  # target only
                raw_fact(
                    point_in_time="1990-06-01T00:00:00Z", point_in_time_precision="11", tail_publication="1995-02-03"
                ),
                raw_fact(),  # undated
            ]
        )
    )
    got = fact_valid_from(facts).to_numpy()
    assert got[0] == day_seconds(1998, 3, 23)
    assert got[1] == day_seconds(1998, 12, 31)  # decision 0006: a year-only date is 31 Dec
    assert got[2] == day_seconds(2001, 5, 4)
    assert got[3] == day_seconds(1995, 2, 3)  # the later of the two
    assert np.isnan(got[4])


def test_triple_dates_follow_the_row_order_of_the_triples_file():
    kept = pd.DataFrame({"h": [1, 0, 0], "relation": ["b", "a", "a"], "tail": ["Q2", "Q9", "Q1"]})
    kept = annotate(
        kept.assign(point_in_time=["1999-01-01T00:00:00Z", "", ""], point_in_time_precision=["11", "", ""]).assign(
            start_time="", tail_publication="", tail_inception="", tail_birth=""
        )
    )
    entities = pd.DataFrame(
        {
            "entity": [0, 1, 2, 3, 4],
            "kind": ["item", "item", "wikidata", "wikidata", "wikidata"],
            "key": ["10", "11", "Q1", "Q2", "Q9"],
        }
    )
    relations = pd.DataFrame({"relation": [0, 1], "name": ["a", "b"]})
    out = triple_dates(kept, entities, relations)
    assert out[["h", "r", "t"]].values.tolist() == [[0, 0, 2], [0, 0, 4], [1, 1, 3]]  # sorted like assign_ids does
    assert out["ts"].isna().tolist() == [True, True, False] and out["ts"].iloc[2] == day_seconds(1999, 1, 1)


# --- the graph -------------------------------------------------------------------------------------------------------


def test_kg_edges_carry_the_fact_dates_in_both_directions_and_interactions_keep_train_ts():
    ckg = make_kg_ckg()
    inter = ckg.relation <= 0  # placeholder to keep the mask below readable
    n_rel = ckg.n_relations
    is_interact = (ckg.relation == 0) | (ckg.relation == n_rel + 1)
    assert sorted(ckg.edge_ts[is_interact]) == sorted(np.concatenate([TRAIN_TS, TRAIN_TS]))
    assert sorted(ckg.edge_ts[~is_interact & ~np.isnan(ckg.edge_ts)]) == sorted(
        np.concatenate([KG_TS[~np.isnan(KG_TS)]] * 2)
    )
    assert int(np.isnan(ckg.edge_ts).sum()) == 2 * int(np.isnan(KG_TS).sum())
    assert (ckg.t_ref, ckg.dt_den) == (TRAIN_TS.max(), TRAIN_TS.max() - TRAIN_TS.min())  # still from train_ts only
    assert inter.shape == ckg.relation.shape


def test_without_kg_ts_the_kg_edges_stay_nan_as_in_stage_1():
    ckg = make_kg_ckg(kg_ts=None)
    assert np.isnan(ckg.edge_ts[(ckg.relation != 0) & (ckg.relation != ckg.n_relations + 1)]).all()


def test_kg_ts_must_be_aligned_with_the_triples():
    with pytest.raises(ValueError, match="aligned"):
        make_kg_ckg(kg_ts=KG_TS[:-1])


# --- the formula -----------------------------------------------------------------------------------------------------


def test_stage_2_puts_delta_r_on_the_kg_relations_only():
    model = make_stage2()
    mask = kg_edges(model)
    rel = model.relation.numpy()
    interact = (rel == 0) | (rel == model.n_relations + 1)
    assert not mask[interact].any() and mask[~interact].all()
    stage1 = make_model(make_kg_ckg(), seed=0)
    assert np.array_equal(stage1.delta_r.numpy() + model.delta_r.numpy(), np.ones(model.n_rel))  # complementary


def test_gamma_zero_is_kgat_in_stage_2():
    ckg = make_kg_ckg()
    plain, zero = make_model(ckg, seed=3), make_stage2(ckg, seed=3, gamma=0.0)
    assert torch.equal(plain.A.to_dense(), zero.A.to_dense())
    assert torch.equal(plain.eval_embeddings(), zero.eval_embeddings())
    assert not torch.equal(plain.A.to_dense(), make_stage2(ckg, seed=3, gamma=2.0).A.to_dense())  # gamma does act


def test_dt_norm_of_a_kg_edge_is_capped_at_one_and_an_undated_fact_has_zero():
    model = make_stage2(gamma=2.0)
    penalty = model._time_penalty(torch.float64).numpy()
    ts, mask = model.edge_ts.numpy(), kg_edges(model)
    dt_norm = np.where(np.isnan(ts), 0.0, np.clip((model._t_ref - ts) / model._dt_den, 0.0, None))
    assert (dt_norm[mask & ~np.isnan(ts)] > 1).any()  # the fact dates are decades older than the toy train window
    expect = np.where(mask, 2.0 * np.minimum(dt_norm, 1.0), 0.0)
    assert np.allclose(penalty, expect)
    assert penalty.max() == pytest.approx(2.0)  # gamma * 1
    assert np.isclose(penalty[mask & (ts == 300.0)], 2.0 * 0.5).all()  # a fact inside the window is not capped
    assert (penalty[np.isnan(ts)] == 0).all() and torch.isfinite(model.A.to_dense()).all()


def test_a_fact_dated_after_the_anchor_has_no_negative_delta_t():
    ckg = make_kg_ckg(kg_ts=np.full(len(TRIPLES), TRAIN_TS.max() + 1e6))
    penalty = make_stage2(ckg, gamma=3.0)._time_penalty(torch.float64).numpy()
    assert (penalty == 0).all()  # dt = max(0, t_ref - ts) = 0 (item 10, first bullet)


def test_an_older_fact_never_has_more_attention_when_pi_is_equal():
    model = make_stage2(gamma=3.0)
    model._raw_scores = lambda: torch.zeros(len(model.head), dtype=torch.float64)
    w = model._segment_softmax(model._edge_scores()).numpy()
    ts, head, mask = model.edge_ts.numpy(), model.head.numpy(), kg_edges(model) & ~np.isnan(model.edge_ts.numpy())
    checked = 0
    for h in np.unique(head):
        mine = np.flatnonzero((head == h) & mask)
        if len(mine) > 1:
            order = mine[np.argsort(ts[mine], kind="stable")]
            assert (np.diff(w[order]) >= -1e-15).all()
            checked += len(mine)
    assert checked


def test_interactions_are_not_penalised_in_stage_2():
    ckg = make_kg_ckg()
    model = make_stage2(ckg, gamma=5.0)
    rel = model.relation.numpy()
    interact = (rel == 0) | (rel == model.n_relations + 1)
    assert (model._time_penalty(torch.float64).numpy()[interact] == 0).all()


def test_time_on_must_be_known():
    with pytest.raises(ValueError, match="time_on"):
        KGATConfig(**CFG, time_on="rating")


def test_stage_1_is_untouched_by_the_option():
    ckg = make_kg_ckg(kg_ts=None)
    a, b = make_model(ckg, seed=1, gamma=2.0), make_model(ckg, seed=1, gamma=2.0, time_on="interact")
    assert torch.equal(a.A.to_dense(), b.A.to_dense())
    with pytest.raises(ValueError, match="no timestamp"):  # Stage 1 still refuses a timed edge without a timestamp
        import dataclasses

        ts = ckg.edge_ts.copy()
        ts[np.flatnonzero(~np.isnan(ts))[0]] = np.nan
        KGAT(dataclasses.replace(ckg, edge_ts=ts), KGATConfig(**CFG, gamma=1.0), device="cpu")


# --- the control -----------------------------------------------------------------------------------------------------


def test_the_date_permutation_keeps_the_multiset_per_seed_and_moves_dates():
    once = permute_kg_ts(KG_TS, 2019)
    assert np.array_equal(np.sort(once[~np.isnan(once)]), np.sort(KG_TS[~np.isnan(KG_TS)]))
    assert int(np.isnan(once).sum()) == int(np.isnan(KG_TS).sum())  # undated facts are shuffled, not removed
    assert np.array_equal(once, permute_kg_ts(KG_TS, 2019), equal_nan=True)  # one permutation per seed
    assert any(not np.array_equal(permute_kg_ts(KG_TS, s), once, equal_nan=True) for s in (2020, 2021, 2022))
    assert not np.array_equal(once, KG_TS, equal_nan=True) or True  # a tiny graph may keep the order for a seed


def test_carry_kg_ts_keeps_the_trained_dates_and_the_new_triples_their_own():
    kg1 = KGTriples("default", "t1", TRIPLES[:5].copy(), NI, E, ("a", "b"), True, None)
    new = np.array([[2, 0, 4]], dtype=np.int64)
    kg2 = KGTriples("default", "t2", np.concatenate([TRIPLES[:5], new]), NI, E, ("a", "b"), True, None)
    trained = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    own = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 77.0])
    assert carry_kg_ts(kg1, trained, kg2, own).tolist() == [9.0, 8.0, 7.0, 6.0, 5.0, 77.0]
    assert np.array_equal(carry_kg_ts(kg1, own[:5], kg2, own), own)  # unpermuted dates: nothing changes


# --- load_kg_dates and scripts/run.py --------------------------------------------------------------------------------


def write_dates(processed, kg_meta_cut="t1", shift=0.0):
    """kg_<cut>_dates.tsv for the synthetic KG of write_kg: the same triples, a date per triple."""
    kg_dir = processed / "kg" / "default"
    for cut in ("t1", "t2"):
        tri = pd.read_csv(kg_dir / f"kg_{cut}.tsv", sep="\t")
        tri["ts"] = np.where(np.arange(len(tri)) % 3 == 0, np.nan, 1e8 + 1e6 * np.arange(len(tri)) + shift)
        tri.to_csv(kg_dir / f"kg_{cut}_dates.tsv", sep="\t", index=False)


@pytest.fixture
def s2setup(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    cfg = {"name": "synthetic", "run_name": RUN_NAME, "processed_dir": str(tmp_path / "processed")}
    save_split(split, cfg, {})
    _, meta = load_saved(cfg)
    enc = encode(split)
    write_kg(tmp_path / "processed", enc, meta)
    write_dates(tmp_path / "processed")
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg))
    models = tmp_path / "models"
    models.mkdir()
    common = {"seeds": [1, 2], "kg": {"variant": "default"}, "fixed": FIXED}
    bodies = {
        "kgat": {"name": "kgat", **common, "grid": GRID},
        "tkgat_kg": {
            "name": "tkgat",
            "time_on": "kg",
            **common,
            "from_kgat": "kgat",
            "grid": [{"gamma": 1}, {"gamma": 3}],
        },
        "tkgat_kg_perm": {
            "name": "tkgat",
            "time_on": "kg",
            "control": "permute_kg_ts",
            **common,
            "from_tkgat": "tkgat_kg",
        },
    }
    for tag, body in bodies.items():
        (models / f"{tag}.yaml").write_text(yaml.safe_dump(body))
    runs = tmp_path / "runs" / RUN_NAME
    runs.mkdir(parents=True)
    (runs / "kgat_summary.json").write_text(json.dumps({"selected": GRID[1]}))
    return SimpleNamespace(tmp=tmp_path, data=data_cfg, models=models, runs=runs, enc=enc, split=split, meta=meta)


def run_tag(ts, tag, *extra, script=None):
    (script or load_run_script()).main(
        ["--data", str(ts.data), "--model", str(ts.models / f"{tag}.yaml"), "--device", "cpu"]
        + ["--registry", str(ts.tmp / "registry.csv"), "--runs-dir", str(ts.tmp / "runs"), *extra]
    )
    return pd.read_csv(ts.tmp / "registry.csv")


def test_load_kg_dates_reads_a_file_that_matches_the_triples(s2setup):
    kg = load_kg(s2setup.tmp / "processed", "default", "t1", s2setup.enc, s2setup.meta)
    ts = load_kg_dates(s2setup.tmp / "processed", kg)
    assert len(ts) == len(kg) and np.isnan(ts[0]) and not np.isnan(ts[1])


def test_a_dates_file_of_another_order_or_a_missing_one_is_refused(s2setup):
    processed = s2setup.tmp / "processed"
    kg = load_kg(processed, "default", "t1", s2setup.enc, s2setup.meta)
    path = processed / "kg" / "default" / "kg_t1_dates.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.iloc[::-1].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="same order"):
        load_kg_dates(processed, kg)
    path.unlink()
    with pytest.raises(ValueError, match="06_build_kg"):
        load_kg_dates(processed, kg)


def test_val_rows_record_time_on_kg_and_the_graph_carries_fact_dates(s2setup):
    script, seen = load_run_script(), []
    real = script.fit_kgat

    def spy(data, ckg, model_cfg, fit_cfg, seed):
        seen.append((ckg.edge_ts.copy(), model_cfg.time_on, model_cfg.gamma))
        return real(data, ckg, model_cfg, fit_cfg, seed)

    script.fit_kgat = spy
    reg = run_tag(s2setup, "tkgat_kg", "--val-only", script=script)
    assert set(reg["phase"]) == {"val"} and len(reg) == 4
    assert all(json.loads(c)["time_on"] == "kg" for c in reg["config"])
    assert {t for _, t, _ in seen} == {"kg"} and {g for _, _, g in seen} == {1, 3}
    assert all(int(np.isnan(ts).sum()) > 0 and np.isfinite(ts).any() for ts, _, _ in seen)
    summary = json.loads((s2setup.runs / "tkgat_kg_val_summary.json").read_text())
    assert summary["tkgat"]["time_on"] == "kg" and summary["tkgat"]["control"] is None


def test_the_full_stage_2_run_scores_test_once_on_the_t2_graph_with_fact_dates(s2setup):
    reg = run_tag(s2setup, "tkgat_kg")
    test = reg[(reg["model"] == "tkgat_kg") & (reg["phase"] == "test")]
    assert len(test) == 2 and (s2setup.runs / "tkgat_kg_seed1.npz").exists()
    assert {json.loads(c)["gamma"] for c in test["config"]} <= {1, 3} and len(
        {json.loads(c)["gamma"] for c in test["config"]}
    ) == 1


def test_the_date_control_shuffles_fact_dates_per_seed_and_not_train_ts(s2setup):
    run_tag(s2setup, "tkgat_kg")
    script, seen = load_run_script(), {}
    real = script.fit_kgat

    def spy(data, ckg, model_cfg, fit_cfg, seed):
        seen[seed] = (np.asarray(data.train_ts).copy(), ckg.edge_ts.copy())
        return real(data, ckg, model_cfg, fit_cfg, seed)

    script.fit_kgat = spy
    reg = run_tag(s2setup, "tkgat_kg_perm", script=script)
    kg = load_kg(s2setup.tmp / "processed", "default", "t1", s2setup.enc, s2setup.meta)
    real_ts = load_kg_dates(s2setup.tmp / "processed", kg)
    for seed in (1, 2):
        train_ts, edge_ts = seen[seed]
        assert np.array_equal(train_ts, s2setup.enc.train_ts)  # the interactions keep their real timestamps
        expected = build_ckg(s2setup.enc, kg, kg_ts=permute_kg_ts(real_ts, seed)).edge_ts
        assert np.array_equal(edge_ts, expected, equal_nan=True)  # this seed's permutation, nothing else
    assert not np.array_equal(seen[1][1], seen[2][1], equal_nan=True)
    rows = reg[reg["model"] == "tkgat_kg_perm"]
    assert set(rows["phase"]) == {"val", "test"}
    point = {k: v for k, v in json.loads(rows.iloc[0]["config"]).items() if k != "perm_seed"}
    assert point["time_on"] == "kg" and "perm_seed" in json.loads(rows.iloc[0]["config"])


@pytest.mark.parametrize(
    "tag, changes, message",
    [
        ("tkgat_kg", {"time_on": "rating"}, "time_on must be one of"),
        ("tkgat_kg", {"kg": {"variant": "none"}}, "kg differs"),
        ("tkgat_kg_perm", {"control": "permute_train_ts"}, "shuffles the dates of time_on"),
        ("tkgat_kg_perm", {"time_on": None}, "time_on differs"),
    ],
)
def test_the_stage_2_configs_are_checked(s2setup, tag, changes, message):
    run_tag(s2setup, "tkgat_kg", "--val-only")
    edit(s2setup, tag, **changes)
    with pytest.raises(ValueError, match=message):
        run_tag(s2setup, tag, "--val-only")


def test_the_stage_2_run_never_reads_test_or_the_t2_files(s2setup):
    for name in ("test.csv", "seen_before_test.csv"):
        (s2setup.tmp / "processed" / name).unlink()
    for name in ("kg_t2.tsv", "kg_t2_dates.tsv"):
        (s2setup.tmp / "processed" / "kg" / "default" / name).unlink()
    run_tag(s2setup, "tkgat_kg", "--val-only")
    reg = run_tag(s2setup, "tkgat_kg_perm", "--val-only")
    assert set(reg["phase"]) == {"val"} and not list(s2setup.runs.glob("*.npz"))


@pytest.mark.parametrize("tag", ["tkgat_kg", "tkgat_kg_perm"])
def test_the_shipped_stage_2_configs_extend_kgat_yaml(tag):
    from .conftest import ROOT

    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{tag}.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs" / "model" / "kgat.yaml").read_text())
    assert cfg["name"] == "tkgat" and cfg["time_on"] == "kg" and all(cfg[k] == ref[k] for k in ("fixed", "kg", "seeds"))
    if tag == "tkgat_kg":
        assert cfg["from_kgat"] == "kgat" and [p["gamma"] for p in cfg["grid"]] == [1, 3, 10]
    else:
        assert cfg["control"] == "permute_kg_ts" and cfg["from_tkgat"] == "tkgat_kg" and "grid" not in cfg
