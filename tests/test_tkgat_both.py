"""T-KGAT combined (decision 0016, item 7): Delta-t on the user-item edges and on the KG facts at once.

CPU only, tiny graphs; nothing here trains on real data.
"""

import json

import numpy as np
import pytest
import torch
import yaml

from tkgat.models.kgat import SECONDS_PER_DAY, SECONDS_PER_YEAR, KGATConfig

from .test_kg_lastfm import KGAT_POINT, MODELS, load_run_script
from .test_kgat import CFG, TRAIN_TS, make_model
from .test_tkgat_stage2 import KG_TS, make_kg_ckg

# the toy train timestamps are 100..500 s: pick time constants that make dt / tau visible instead of ~0
TAU = dict(tau_ui_days=400.0 / SECONDS_PER_DAY, tau_kg_years=1e8 / SECONDS_PER_YEAR)  # tau_ui = 400 s, tau_kg = 1e8 s


def make_both(ckg=None, seed=0, **kw):
    return make_model(ckg or make_kg_ckg(), seed=seed, time_on="both", **{**TAU, **kw})


def kinds(model):
    rel = model.relation.numpy()
    ui = (rel == 0) | (rel == model.n_relations + 1)
    return ui, ~ui


def test_delta_ui_is_interact_and_delta_kg_is_the_kg_relations_and_their_inverses():
    model = make_both(gamma=1.0)
    ui, kg = kinds(model)
    assert (model.delta_ui[model.relation].numpy() > 0).tolist() == ui.tolist()
    assert (model.delta_kg[model.relation].numpy() > 0).tolist() == kg.tolist()
    assert np.array_equal(model.delta_r.numpy(), np.maximum(model.delta_ui.numpy(), model.delta_kg.numpy()))


def test_penalty_is_the_sum_of_two_absolute_terms_without_cap_or_window():
    model = make_both(gamma=2.0)
    penalty = model._time_penalty(torch.float64).numpy()
    ts, (ui, kg) = model.edge_ts.numpy(), kinds(model)
    dt = np.clip(model._t_ref - ts, 0.0, None)  # seconds
    expect_ui = np.where(ui, 2.0 * dt / 400.0, 0.0)
    expect_kg = np.where(kg & ~np.isnan(ts), 2.0 * np.nan_to_num(dt) / 1e8, 0.0)
    assert np.allclose(penalty, expect_ui + expect_kg)
    assert penalty[kg & ~np.isnan(ts)].max() > 2.0 * 1.0 * 0 and np.isfinite(penalty).all()
    # no cap: a fact 4e8 s older than the anchor is penalised 4x as much as one 1e8 s older (Stage 2 would clip both)
    old = penalty[kg & (ts == -4e8)]
    older_ref = penalty[kg & (ts == -1e8)]
    assert np.allclose(old / older_ref, (4e8 + TRAIN_TS.max()) / (1e8 + TRAIN_TS.max()))
    # no window: the user-item term is dt / tau, not dt / (t_ref - t_min)
    assert np.isclose(penalty[ui].max(), 2.0 * (TRAIN_TS.max() - TRAIN_TS.min()) / 400.0)


def test_an_undated_fact_adds_nothing_and_attention_stays_finite():
    model = make_both(gamma=3.0)
    ts, kg = model.edge_ts.numpy(), kinds(model)[1]
    penalty = model._time_penalty(torch.float64).numpy()
    assert (penalty[kg & np.isnan(ts)] == 0).all() and np.isnan(ts[kg]).any()
    assert torch.isfinite(model.A.to_dense()).all()


def test_a_fact_dated_after_the_anchor_has_no_negative_delta_t():
    ckg = make_kg_ckg(kg_ts=np.full(len(KG_TS), TRAIN_TS.max() + 1e6))
    penalty = make_both(ckg, gamma=3.0)._time_penalty(torch.float64).numpy()
    ui, kg = kinds(make_both(ckg, gamma=3.0))
    assert (penalty[kg] == 0).all() and (penalty[ui] >= 0).all()


def test_gamma_zero_is_plain_kgat():
    ckg = make_kg_ckg()
    plain, zero = make_model(ckg, seed=3), make_both(ckg, seed=3, gamma=0.0)
    assert torch.equal(plain.A.to_dense(), zero.A.to_dense())
    assert torch.equal(plain.eval_embeddings(), zero.eval_embeddings())
    assert not torch.equal(plain.A.to_dense(), make_both(ckg, seed=3, gamma=2.0).A.to_dense())


def test_each_term_alone_equals_stage_1_and_stage_2_shaped_penalties():
    ckg = make_kg_ckg()
    ui, kg = kinds(make_both(ckg, gamma=1.0))
    only_ui = make_both(ckg, gamma=2.0, gamma_kg=0.0)._time_penalty(torch.float64).numpy()
    only_kg = make_both(ckg, gamma=2.0, gamma_ui=0.0)._time_penalty(torch.float64).numpy()
    both = make_both(ckg, gamma=2.0)._time_penalty(torch.float64).numpy()
    assert (only_ui[kg] == 0).all() and (only_kg[ui] == 0).all()
    assert np.allclose(only_ui + only_kg, both)
    assert not torch.equal(
        make_both(ckg, seed=2, gamma=2.0, gamma_kg=0.0).A.to_dense(), make_both(ckg, seed=2, gamma=2.0).A.to_dense()
    )


def test_both_gammas_zero_by_override_is_plain_kgat_even_with_gamma_set():
    ckg = make_kg_ckg()
    plain = make_model(ckg, seed=4)
    off = make_both(ckg, seed=4, gamma=5.0, gamma_ui=0.0, gamma_kg=0.0)
    assert torch.equal(plain.A.to_dense(), off.A.to_dense())


def test_an_older_fact_never_has_more_attention_when_pi_is_equal():
    model = make_both(gamma=3.0, gamma_ui=0.0)
    model._raw_scores = lambda: torch.zeros(len(model.head), dtype=torch.float64)
    w = model._segment_softmax(model._edge_scores()).numpy()
    ts, head = model.edge_ts.numpy(), model.head.numpy()
    mask = kinds(model)[1] & ~np.isnan(ts)
    checked = 0
    for h in np.unique(head):
        mine = np.flatnonzero((head == h) & mask)
        if len(mine) > 1:
            order = mine[np.argsort(ts[mine], kind="stable")]
            assert (np.diff(w[order]) >= -1e-15).all()
            checked += len(mine)
    assert checked


def test_kg_timed_relations_restricts_delta_kg_to_those_relations_and_their_inverses():
    ckg = make_kg_ckg()
    model = make_both(ckg, gamma=1.0, kg_timed_relations=[1])  # KG relation index 1 = relation "b"
    rel = model.relation.numpy()
    r = model.n_relations
    timed = model.delta_kg[model.relation].numpy() > 0
    assert set(np.unique(rel[timed]).tolist()) == {2, 2 + r + 1}  # relation_id(1) = 2 and its inverse
    penalty = model._time_penalty(torch.float64).numpy()
    assert (penalty[~timed & ~kinds(model)[0]] == 0).all()
    with pytest.raises(ValueError, match="not a KG relation"):
        make_both(ckg, gamma=1.0, kg_timed_relations=[2])


def test_an_interact_edge_without_a_timestamp_is_refused_but_an_undated_fact_is_not():
    import dataclasses

    ckg = make_kg_ckg()
    ts = ckg.edge_ts.copy()
    first_ui = np.flatnonzero(np.isin(ckg.relation, [0, ckg.n_relations + 1]))[0]
    ts[first_ui] = np.nan
    with pytest.raises(ValueError, match="Interact edge has no timestamp"):
        make_both(dataclasses.replace(ckg, edge_ts=ts), gamma=1.0)
    make_both(ckg, gamma=1.0)  # NaN facts only: fine


def test_config_validation():
    with pytest.raises(ValueError, match="belong to time_on"):
        KGATConfig(**CFG, time_on="kg", gamma_kg=0.0)
    with pytest.raises(ValueError, match="belong to time_on"):
        KGATConfig(**CFG, time_on="interact", kg_timed_relations=[0])
    with pytest.raises(ValueError, match="gamma_ui"):
        KGATConfig(**CFG, time_on="both", gamma_ui=-1.0)
    with pytest.raises(ValueError, match="tau"):
        KGATConfig(**CFG, time_on="both", tau_kg_years=0.0)
    assert KGATConfig(**CFG, time_on="both", kg_timed_relations=[3, 1, 1]).kg_timed_relations == (1, 3)


def test_stage_1_and_stage_2_are_untouched_by_the_new_options():
    ckg = make_kg_ckg()
    for time_on in ("interact", "kg"):
        a = make_model(ckg, seed=1, gamma=2.0, time_on=time_on)
        b = make_model(ckg, seed=1, gamma=2.0, time_on=time_on, tau_ui_days=7.0, tau_kg_years=1.0)  # tau unused
        assert torch.equal(a.A.to_dense(), b.A.to_dense())


# --- run.py: grid, ablations, control, the shipped configs ------------------------------------------------------------


def shipped(tag):
    return yaml.safe_load((MODELS / f"{tag}.yaml").read_text(encoding="utf-8"))


def test_the_both_grid_has_twelve_points_and_extends_the_kgat_point(tmp_path):
    out = tmp_path / "runs"
    out.mkdir()
    (out / "kgat_summary.json").write_text(json.dumps({"selected": KGAT_POINT}), encoding="utf-8")
    run = load_run_script()
    points, base = run.tkgat_points(shipped("tkgat_both"), MODELS / "tkgat_both.yaml", out)
    assert base == KGAT_POINT and len(points) == 12 and len({json.dumps(p, sort_keys=True) for p in points}) == 12
    assert {p["gamma"] for p in points} == {1, 3, 10}
    assert {p["tau_ui_days"] for p in points} == {90, 365} and {p["tau_kg_years"] for p in points} == {5, 20}
    for p in points:
        assert p["time_on"] == "both" and {k: p[k] for k in KGAT_POINT} == KGAT_POINT


def test_a_both_grid_point_may_hold_only_gamma_and_the_two_taus(tmp_path):
    out = tmp_path / "runs"
    out.mkdir()
    (out / "kgat_summary.json").write_text(json.dumps({"selected": KGAT_POINT}), encoding="utf-8")
    run = load_run_script()
    mcfg = shipped("tkgat_both")
    mcfg["grid"] = [{"gamma": 1}]
    with pytest.raises(ValueError, match="tau_kg_years"):
        run.tkgat_points(mcfg, MODELS / "tkgat_both.yaml", out)


def test_delta_kg_relations_go_to_the_points_as_names_and_resolve_to_indices(tmp_path):
    out = tmp_path / "runs"
    out.mkdir()
    (out / "kgat_summary.json").write_text(json.dumps({"selected": KGAT_POINT}), encoding="utf-8")
    run = load_run_script()
    mcfg = {**shipped("tkgat_both"), "delta_kg_relations": ["b"]}
    points, _ = run.tkgat_points(mcfg, MODELS / "tkgat_both.yaml", out)
    assert all(p["kg_timed_relations"] == ["b"] for p in points)
    assert run.resolve_kg_relations(["b"], ("a", "b")) == (1,) and run.resolve_kg_relations(None, ("a", "b")) is None
    with pytest.raises(ValueError, match="unknown relation"):
        run.resolve_kg_relations(["zzz"], ("a", "b"))


@pytest.mark.parametrize(
    "tag, extra",
    [("tkgat_both_nokg", {"gamma_kg": 0.0}), ("tkgat_both_noui", {"gamma_ui": 0.0}), ("tkgat_both_perm", {})],
)
def test_ablations_and_the_control_re_run_the_selected_point(tmp_path, tag, extra):
    out = tmp_path / "runs"
    out.mkdir()
    selected = {**KGAT_POINT, "gamma": 3, "tau_ui_days": 90, "tau_kg_years": 20, "time_on": "both"}
    (out / "tkgat_both_summary.json").write_text(json.dumps({"selected": selected}), encoding="utf-8")
    run = load_run_script()
    points, base = run.tkgat_points(shipped(tag), MODELS / f"{tag}.yaml", out)
    assert base == selected and points == [{**selected, **extra}]


def test_an_ablation_needs_the_combined_model_and_cannot_also_be_a_control(tmp_path):
    out = tmp_path / "runs"
    out.mkdir()
    (out / "tkgat_kg_summary.json").write_text(
        json.dumps({"selected": {**KGAT_POINT, "gamma": 3, "time_on": "kg"}}), encoding="utf-8"
    )
    run = load_run_script()
    mcfg = {**shipped("tkgat_kg_perm"), "control": None, "ablate": "kg"}
    with pytest.raises(ValueError, match="time_on = both"):
        run.tkgat_points(mcfg, MODELS / "tkgat_kg_perm.yaml", out)
    both = {**shipped("tkgat_both_perm"), "ablate": "kg"}
    with pytest.raises(ValueError, match="control or an ablation"):
        run.tkgat_points(both, MODELS / "tkgat_both_perm.yaml", out)


def test_the_shipped_combined_configs_share_fixed_kg_and_seeds_with_kgat():
    kgat = shipped("kgat")
    for tag in ("tkgat_both", "tkgat_both_perm", "tkgat_both_nokg", "tkgat_both_noui"):
        cfg = shipped(tag)
        assert cfg["time_on"] == "both" and cfg["name"] == "tkgat", tag
        for key in ("fixed", "kg", "seeds"):
            assert cfg[key] == kgat[key], (tag, key)
