"""Merged protocol: window-based joint attention and restartable train+validation refit."""

import dataclasses
import json

import numpy as np
import torch

from tkgat.models.kgat import KGAT, KGATConfig

from .test_kgat import CFG
from .test_run_kgat import GRID, load_run_script
from .test_tkgat import edit
from .test_tkgat_stage2 import make_kg_ckg, run_tag
from .test_tkgat_stage2 import s2setup as _s2setup

s2setup = _s2setup


def test_refit_fits_fresh_windows_at_t2_then_freezes_them():
    base = make_kg_ckg()
    ui = (base.relation == 0) | (base.relation == base.n_relations + 1)
    dates = base.edge_ts.copy()
    dates[ui & (dates == 500)] = 1000
    dates[(~ui) & (dates == 300)] = 700
    graph = dataclasses.replace(base, cutoff="t2", edge_ts=dates, t_ref=1000.0, dt_den=900.0)
    model = KGAT(graph, KGATConfig(**CFG, gamma=2.0, time_on="both"), device="cpu", refit=True)
    assert (model._t_ref, model._dt_den) == (1000.0, 900.0)
    assert model._fact_window == (700.0, 400000700.0)
    penalty = model._time_penalty(torch.float64).numpy()
    assert np.isclose(penalty[ui & (dates == 100)], 2.0).all()
    assert (penalty[(~ui) & (dates == 700)] == 0).all()
    script = load_run_script()
    before, after = script.joint_time_windows(base), script.joint_time_windows(graph)
    assert before["interaction"]["min_ts"] == after["interaction"]["min_ts"] == 100
    assert after["interaction"]["max_ts"] == model._t_ref == 1000
    assert after["interaction"]["width_seconds"] == model._dt_den == 900
    assert after["fact"]["max_ts"] == model._fact_window[0] == 700
    assert after["fact"]["width_seconds"] == model._fact_window[1] == 400000700
    drift = script.compare_time_windows(before, after)
    assert drift["interaction"] == {
        "min_shift_seconds": 0.0,
        "max_shift_seconds": 500.0,
        "width_change_seconds": 500.0,
        "width_ratio": 2.25,
        "width_change_pct": 125.0,
    }
    model.set_graph(dataclasses.replace(graph, edge_ts=dates + 1000, t_ref=2000.0, dt_den=2000.0))
    assert (model._t_ref, model._dt_den, model._fact_window) == (1000.0, 900.0, (700.0, 400000700.0))


def test_standalone_refit_is_bounded_resumable_and_uses_validation_epochs_only(s2setup, monkeypatch):
    edit(s2setup, "tkgat_kg", standalone=True, from_kgat=None, time_on="both", base_point=GRID[1])
    (s2setup.models / "kgat.yaml").unlink()
    (s2setup.runs / "kgat_summary.json").unlink()
    script, calls, used_windows = load_run_script(), [], {}
    real = script.fit_kgat
    train_size = len(s2setup.enc.train)
    merged_size = train_size + len(s2setup.split.val)

    def spy(data, ckg, model_cfg, fit_cfg, seed, fixed_epochs=None):
        fit = real(data, ckg, model_cfg, fit_cfg, seed, fixed_epochs=fixed_epochs)
        calls.append((seed, len(data.train), ckg.cutoff, fixed_epochs, fit.best_epoch))
        phase = "refit" if fixed_epochs is not None else "selection"
        used_windows[phase] = {
            "interaction": (fit.model._t_ref - fit.model._dt_den, fit.model._t_ref, fit.model._dt_den),
            "fact": (fit.model._fact_window[0] - fit.model._fact_window[1], *fit.model._fact_window),
        }
        if fixed_epochs is not None:
            facts = ckg.edge_ts[(ckg.relation != 0) & (ckg.relation != ckg.n_relations + 1)]
            facts = facts[np.isfinite(facts)]
            assert fit.model._fact_window == (facts.max(), np.ptp(facts))
            assert fit.best_val == {} and all("recall@20" not in row for row in fit.history)
        return fit

    monkeypatch.setattr(script, "fit_kgat", spy)
    for step in range(6):
        reg = run_tag(s2setup, "tkgat_kg", "--refit", "--resume", "--max-new-runs", "1", script=script)
        assert len(calls) == step + 1
        assert len(reg[reg.phase == "test"]) == max(0, step - 3)
        if step < 5:
            assert not (s2setup.runs / "tkgat_kg_refit_summary.json").exists()
        if step == 3:
            # The fit budget pauses before refit/test, but the window comparison is already reviewable.
            comparison = json.loads((s2setup.runs / "tkgat_kg_window_summary.json").read_text())
            assert set(comparison["time_windows"]) >= {"selection", "refit", "refit_vs_selection"}
            assert (reg.phase == "test").sum() == 0
    assert all(size == train_size and cut == "t1" for _, size, cut, epoch, _ in calls if epoch is None)
    assert all(size == merged_size and cut == "t2" for _, size, cut, epoch, _ in calls if epoch is not None)
    summary = json.loads((s2setup.runs / "tkgat_kg_refit_summary.json").read_text())
    windows = summary["tkgat"]["time_windows"]
    assert windows["policy"] == "fit_each_training_phase"
    assert comparison["time_windows"] == windows
    for phase, kinds in used_windows.items():
        for kind, expected_bounds in kinds.items():
            row = windows[phase][kind]
            assert (row["min_ts"], row["max_ts"], row["width_seconds"]) == expected_bounds
            assert row["width_days"] == row["width_seconds"] / 86400.0
    expected = int(np.ceil(np.median(summary["refit"]["best_epochs"])))
    assert all(epoch == expected for _, _, _, epoch, _ in calls if epoch is not None)
    for value in reg.loc[reg.phase == "test", "config"]:
        config = json.loads(value)
        assert config["refit"] is True and config["epochs"] == expected and config["time_on"] == "both"
        assert not any(key.startswith("tau_") for key in config)
    assert len(list((s2setup.runs / "states").glob("*_refit_seed*.pt"))) == 2
    assert len(list(s2setup.runs.glob("*_refit_seed*.npz"))) == 2
    before = len(reg)
    reg = run_tag(s2setup, "tkgat_kg", "--refit", "--resume", script=script)
    assert len(calls) == 6 and len(reg) == before


def test_window_summary_keeps_empty_and_single_date_facts_distinct():
    graph = make_kg_ckg()
    script = load_run_script()
    ui = (graph.relation == 0) | (graph.relation == graph.n_relations + 1)
    dates = graph.edge_ts.copy()
    dates[~ui] = np.nan
    empty = script.joint_time_windows(dataclasses.replace(graph, edge_ts=dates))
    assert empty["fact"] == {"min_ts": None, "max_ts": None, "width_seconds": 0.0, "width_days": 0.0}
    dates[~ui] = 0.0
    single = script.joint_time_windows(dataclasses.replace(graph, edge_ts=dates))
    assert single["fact"] == {"min_ts": 0.0, "max_ts": 0.0, "width_seconds": 0.0, "width_days": 0.0}
    drift = script.compare_time_windows(empty, single)["fact"]
    assert drift["min_shift_seconds"] is None and drift["max_shift_seconds"] is None
    assert drift["width_ratio"] is None and drift["width_change_pct"] is None
