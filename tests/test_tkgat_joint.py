"""Joint time attention: independent train windows, one softmax, and frozen windows at test."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tkgat.models.kgat import KGAT, KGATConfig

from .test_kgat import CFG, TRAIN_TS, TRIPLES, make_model
from .test_run_kgat import load_run_script
from .test_tkgat import edit
from .test_tkgat_stage2 import KG_TS, make_kg_ckg, run_tag
from .test_tkgat_stage2 import s2setup as _s2setup

s2setup = _s2setup


def interaction_mask(ckg):
    return (ckg.relation == 0) | (ckg.relation == ckg.n_relations + 1)


def expected_penalty(ckg, train, gamma):
    """Independent NumPy reference, fitting each type's bounds on the training graph only."""
    result = np.zeros(ckg.n_edges)
    for is_interaction in (True, False):
        fit = train.edge_ts[interaction_mask(train) == is_interaction]
        fit = fit[np.isfinite(fit)]
        use = (interaction_mask(ckg) == is_interaction) & np.isfinite(ckg.edge_ts)
        if len(fit) and fit.max() > fit.min():
            result[use] = gamma * np.clip((fit.max() - ckg.edge_ts[use]) / (fit.max() - fit.min()), 0, 1)
    return result


def test_both_types_use_their_own_window_in_one_attention_softmax():
    ckg = make_kg_ckg()
    model = make_model(ckg, gamma=2.0, time_on="both")
    expected = expected_penalty(ckg, ckg, 2.0)
    assert np.allclose(model._time_penalty(torch.float64).numpy(), expected)
    assert (model.delta_r == 1).all()
    mask = interaction_mask(ckg)
    assert (expected[mask] > 0).any() and (expected[~mask] > 0).any()
    # The newest fact (ts=300) has age zero in the fact window, despite being older than the newest interaction.
    assert (expected[(~mask) & (ckg.edge_ts == 300)] == 0).all()
    raw = model._raw_scores().detach().numpy()
    dense = np.zeros((ckg.n_nodes, ckg.n_nodes))
    for head in np.unique(ckg.head):
        edges = ckg.head == head
        logits = (raw - expected)[edges]
        weights = np.exp(logits - logits.max())
        np.add.at(dense[head], ckg.tail[edges], weights / weights.sum())
    assert np.allclose(model.A.to_dense().numpy(), dense)
    assert np.isfinite(dense).all() and (expected[np.isnan(ckg.edge_ts)] == 0).all()
    # Forward and inverse edges use the identical timestamp and penalty.
    penalties = dict(zip(zip(ckg.head, ckg.relation, ckg.tail, strict=True), expected, strict=True))
    for (h, r, t), value in penalties.items():
        if r <= ckg.n_relations:
            assert value == penalties[(t, r + ckg.n_relations + 1, h)]


@pytest.mark.parametrize("fact_dates", [None, np.full(len(TRIPLES), np.nan), np.full(len(TRIPLES), 100.0), KG_TS])
@pytest.mark.parametrize("flat_interactions", [False, True])
def test_missing_dates_and_zero_width_disable_only_their_own_type(fact_dates, flat_interactions):
    ckg = make_kg_ckg(fact_dates)
    if flat_interactions:
        ts = ckg.edge_ts.copy()
        ts[interaction_mask(ckg)] = TRAIN_TS.max()
        ckg = dataclasses.replace(ckg, edge_ts=ts, dt_den=0.0)
    model = make_model(ckg, gamma=3.0, time_on="both")
    assert np.allclose(model._time_penalty(torch.float64).numpy(), expected_penalty(ckg, ckg, 3.0))
    assert torch.isfinite(model.A.to_dense()).all()


def test_t2_dates_cannot_refit_either_train_window():
    train = make_kg_ckg()
    model = make_model(train, gamma=3.0, time_on="both")
    dates = train.edge_ts.copy()
    facts = np.flatnonzero((~interaction_mask(train)) & np.isfinite(dates))
    dates[facts[:2]] = [-1e10, 1e10]  # dates outside both ends of the frozen t1 window
    test = dataclasses.replace(train, cutoff="t2", edge_ts=dates, t_ref=1e10, dt_den=1e10)
    model.set_graph(test)
    model.update_attention()
    assert (model._t_ref, model._dt_den) == (train.t_ref, train.dt_den)
    assert np.allclose(model._time_penalty(torch.float64).numpy(), expected_penalty(test, train, 3.0))
    with pytest.raises(ValueError, match="t1 training graph first"):
        KGAT(test, KGATConfig(**CFG, gamma=3.0, time_on="both"), device="cpu")


def test_gamma_zero_stays_exactly_kgat_and_missing_interaction_dates_are_rejected():
    ckg = make_kg_ckg()
    plain = make_model(ckg, seed=4)
    joint = make_model(ckg, seed=4, gamma=0.0, time_on="both")
    assert torch.equal(plain.A.to_dense(), joint.A.to_dense())
    assert torch.equal(plain.eval_embeddings(), joint.eval_embeddings())
    dates = ckg.edge_ts.copy()
    dates[np.flatnonzero(interaction_mask(ckg))[0]] = np.nan
    with pytest.raises(ValueError, match="no timestamp"):
        make_model(dataclasses.replace(ckg, edge_ts=dates), gamma=1.0, time_on="both")


def test_runner_loads_fact_dates_for_joint_training_and_test_and_keeps_control_working(s2setup):
    edit(s2setup, "tkgat_kg", time_on="both")
    edit(s2setup, "tkgat_kg_perm", time_on="both", control="permute_train_ts")
    script, seen, test_windows = load_run_script(), [], []
    real_fit = script.fit_kgat

    def spy_fit(data, ckg, model_cfg, fit_cfg, seed):
        assert model_cfg.time_on == "both"
        facts = ckg.edge_ts[~interaction_mask(ckg)]
        assert np.isfinite(facts).any()
        seen.append((seed, data.train_ts.copy(), facts.copy()))
        return real_fit(data, ckg, model_cfg, fit_cfg, seed)

    class SpyKGAT(KGAT):
        def set_graph(self, ckg):
            super().set_graph(ckg)
            test_windows.append((ckg.cutoff, self._t_ref, self._dt_den, self._fact_window))

    script.fit_kgat, script.KGAT = spy_fit, SpyKGAT
    reg = run_tag(s2setup, "tkgat_kg", script=script)
    assert len(reg[reg.phase == "test"]) == 2
    assert all(json.loads(c)["time_on"] == "both" for c in reg.config)
    assert len({entry[1:] for entry in test_windows}) == 1
    assert {entry[0] for entry in test_windows} == {"t1", "t2"}
    summary = json.loads((s2setup.runs / "tkgat_kg_summary.json").read_text())
    assert summary["tkgat"]["time_on"] == "both"
    before = seen.copy()
    seen.clear()
    run_tag(s2setup, "tkgat_kg_perm", script=script)
    for seed, train_ts, facts in seen:
        _, original_ts, original_facts = next(row for row in before if row[0] == seed)
        assert np.array_equal(np.sort(train_ts), np.sort(original_ts))
        assert np.array_equal(facts, original_facts, equal_nan=True)


def test_primary_configs_enable_joint_attention():
    root = Path(__file__).resolve().parents[1]
    for tag in ("tkgat_joint", "tkgat_both"):
        cfg = yaml.safe_load((root / "configs" / "model" / f"{tag}.yaml").read_text())
        assert cfg["time_on"] == "both"
    for tag in ("tkgat", "tkgat_perm"):
        cfg = yaml.safe_load((root / "configs" / "model" / f"{tag}.yaml").read_text())
        assert cfg.get("time_on", "interact") == "interact"  # historical Stage 1 notebooks keep their semantics


def test_joint_validation_does_not_open_test_or_t2_files(s2setup):
    edit(s2setup, "tkgat_kg", time_on="both")
    for name in ("test.csv", "seen_before_test.csv"):
        (s2setup.tmp / "processed" / name).unlink()
    for name in ("kg_t2.tsv", "kg_t2_dates.tsv"):
        (s2setup.tmp / "processed" / "kg" / "default" / name).unlink()
    reg = run_tag(s2setup, "tkgat_kg", "--val-only")
    assert set(reg.phase) == {"val"} and not list(s2setup.runs.glob("*.npz"))


def test_joint_control_rejects_a_selected_point_from_the_old_mode(s2setup):
    edit(s2setup, "tkgat_kg", time_on="both")
    edit(s2setup, "tkgat_kg_perm", time_on="both")
    (s2setup.runs / "tkgat_kg_summary.json").write_text(json.dumps({"selected": {"gamma": 1, "time_on": "kg"}}))
    cfg_path = s2setup.models / "tkgat_kg_perm.yaml"
    with pytest.raises(ValueError, match="older time mode"):
        load_run_script().tkgat_points(yaml.safe_load(cfg_path.read_text()), cfg_path, s2setup.runs)
