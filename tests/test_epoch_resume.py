"""Epoch continuation restores both optimizers, dropout/sampler RNG and validation selection."""

import json
from dataclasses import replace

import pandas as pd
import pytest
import torch

from tkgat.train.kgat import fit_kgat

from .test_kaggle_tkgat_joint import load_wrapper
from .test_refit_train import sealed
from .test_run_kgat import FIXED, GRID, load_run_script
from .test_tkgat import edit
from .test_tkgat_stage2 import s2setup as _s2setup
from .test_train_kgat import FIT, MODEL, make_ckg, same_state, stable
from .test_train_kgat import world as _world

s2setup = _s2setup
world = _world


@pytest.mark.parametrize("schedule", ["none", "cosine"])
def test_epoch_resume_matches_uninterrupted_training_including_optimizers_and_rng(world, tmp_path, schedule):
    _, data = world
    graph = make_ckg(data)
    cfg = replace(FIT, max_epochs=4, eval_every=2, lr_schedule=schedule)
    whole = fit_kgat(data, graph, MODEL, cfg, seed=5)
    checkpoint = tmp_path / "fit.epoch.pt"
    for step in range(4):
        resumed = fit_kgat(
            data, graph, MODEL, cfg, seed=5, checkpoint_path=checkpoint, checkpoint_key="same-split", max_new_epochs=1
        )
        assert len(resumed.history) == step + 1
        assert resumed.complete == (step == 3)
        if step == 0:
            assert resumed.best_val == {} and resumed.best_epoch == 0
    assert stable(resumed.history) == stable(whole.history)
    assert same_state(resumed.best_state, whole.best_state)
    assert resumed.best_epoch == whole.best_epoch and resumed.best_val == whole.best_val
    assert checkpoint.with_suffix(".csv").exists()
    assert not checkpoint.with_name(checkpoint.name + ".tmp").exists()


def test_fixed_epoch_refit_resumes_without_reading_validation_or_other_forbidden_parts(world, tmp_path):
    _, data = world
    graph = make_ckg(data)
    cfg = replace(FIT, max_epochs=4)
    whole = fit_kgat(sealed(data), graph, MODEL, cfg, seed=3, fixed_epochs=3)
    for step in range(3):
        result = fit_kgat(
            sealed(data),
            graph,
            MODEL,
            cfg,
            seed=3,
            fixed_epochs=3,
            checkpoint_path=tmp_path / "refit.pt",
            checkpoint_key="refit-data",
            max_new_epochs=1,
        )
        assert result.complete == (step == 2)
    assert result.best_val == {} and result.best_epoch == 3
    assert stable(result.history) == stable(whole.history)
    assert same_state(result.best_state, whole.best_state)


def test_mismatched_data_seed_or_hyperparameters_cannot_resume(world, tmp_path):
    _, data = world
    graph = make_ckg(data)
    checkpoint = tmp_path / "fit.pt"
    fit_kgat(data, graph, MODEL, FIT, seed=0, checkpoint_path=checkpoint, checkpoint_key="a", max_new_epochs=1)
    for cfg, seed, key in ((FIT, 1, "a"), (FIT, 0, "other-data"), (replace(FIT, lr=0.02), 0, "a")):
        with pytest.raises(ValueError, match="another configuration"):
            fit_kgat(data, graph, MODEL, cfg, seed=seed, checkpoint_path=checkpoint, checkpoint_key=key)


def test_benchmark_reads_only_training_batches_and_saves_no_epoch_checkpoint(world):
    _, data = world
    result = fit_kgat(sealed(data), make_ckg(data), MODEL, FIT, seed=0, benchmark_batches=2)
    assert result.complete and len(result.history) == 1 and result.best_val == {}
    row = result.history[0]
    assert row["cf_batches"] == 2 and row["kg_batches"] == 2 and "recall@20" not in row
    assert torch.isfinite(result.model.A.to_dense()).all()


def test_pausing_requires_a_checkpoint_and_logs_progress(world, tmp_path, capsys):
    _, data = world
    graph = make_ckg(data)
    with pytest.raises(ValueError, match="requires a checkpoint_path"):
        fit_kgat(data, graph, MODEL, FIT, seed=0, max_new_epochs=1)
    fit_kgat(
        data,
        graph,
        MODEL,
        FIT,
        seed=0,
        checkpoint_path=tmp_path / "fit.pt",
        checkpoint_key="split",
        max_new_epochs=1,
    )
    log = capsys.readouterr().out
    assert "CF 1/" in log and "KG 1/" in log and "Saved epoch 1" in log


def setup_standalone(ts):
    edit(ts, "tkgat_kg", standalone=True, from_kgat=None, time_on="both", base_point=GRID[1])


def test_runner_pauses_mid_fit_without_selecting_or_scoring_and_finishes_after_resume(s2setup):
    setup_standalone(s2setup)
    script = load_run_script()
    args = [
        "--data",
        str(s2setup.data),
        "--model",
        str(s2setup.models / "tkgat_kg.yaml"),
        "--device",
        "cpu",
        "--registry",
        str(s2setup.tmp / "registry.csv"),
        "--runs-dir",
        str(s2setup.tmp / "runs"),
        "--resume",
        "--epochs-per-fit",
        "1",
        "--val-only",
    ]
    script.main(args)
    assert not (s2setup.tmp / "registry.csv").exists()
    window_summary = json.loads((s2setup.runs / "tkgat_kg_window_summary.json").read_text())
    assert "selection" in window_summary["time_windows"] and "refit" not in window_summary["time_windows"]
    assert not (s2setup.runs / "tkgat_kg_val_summary.json").exists()
    assert list((s2setup.runs / "states").glob("*.epoch.pt"))
    for _ in range(20):
        script.main(args)
        if (s2setup.runs / "tkgat_kg_val_summary.json").exists():
            break
    assert (s2setup.runs / "tkgat_kg_val_summary.json").exists()
    assert not list(s2setup.runs.glob("*_seed*.npz"))
    assert not list((s2setup.runs / "states").glob("*.epoch.pt"))


def test_runner_benchmark_does_not_open_heldout_files_or_write_scores(s2setup):
    setup_standalone(s2setup)
    for name in ("test.csv", "seen_before_test.csv"):
        (s2setup.tmp / "processed" / name).unlink()
    for name in ("kg_t2.tsv", "kg_t2_dates.tsv"):
        (s2setup.tmp / "processed/kg/default" / name).unlink()
    load_run_script().main(
        [
            "--data",
            str(s2setup.data),
            "--model",
            str(s2setup.models / "tkgat_kg.yaml"),
            "--device",
            "cpu",
            "--registry",
            str(s2setup.tmp / "registry.csv"),
            "--runs-dir",
            str(s2setup.tmp / "runs"),
            "--val-only",
            "--benchmark-batches",
            "2",
        ]
    )
    report = json.loads((s2setup.runs / "tkgat_kg_benchmark.json").read_text())
    assert report["benchmark_only"] and not report["test_read"] and not report["validation_scored"]
    assert report["estimated_epoch_seconds_excluding_validation"] > 0
    assert not (s2setup.tmp / "registry.csv").exists()
    assert not list(s2setup.runs.rglob("*.pt"))


def test_runner_resumes_selection_and_refit_epochs_without_duplicate_or_premature_test_rows(s2setup):
    setup_standalone(s2setup)
    # Earliest selected epoch is 2, ensuring that refit also has an incomplete epoch boundary.
    edit(s2setup, "tkgat_kg", fixed={**FIXED, "eval_every": 2, "patience": 10})
    script = load_run_script()
    registry = s2setup.tmp / "registry.csv"
    args = [
        "--data",
        str(s2setup.data),
        "--model",
        str(s2setup.models / "tkgat_kg.yaml"),
        "--device",
        "cpu",
        "--registry",
        str(registry),
        "--runs-dir",
        str(s2setup.tmp / "runs"),
        "--resume",
        "--refit",
        "--epochs-per-fit",
        "1",
        "--max-new-runs",
        "1",
    ]
    saw_partial_refit = False
    summary = s2setup.runs / "tkgat_kg_refit_summary.json"
    for _ in range(40):
        script.main(args)
        saw_partial_refit |= bool(list((s2setup.runs / "states").glob("*_refit_seed*.epoch.pt")))
        if registry.exists():
            rows = pd.read_csv(registry)
            if (rows.phase == "test").any():
                assert (rows.phase == "val").sum() == 4
        if summary.exists():
            break
    assert summary.exists() and saw_partial_refit
    assert (rows.phase == "val").sum() == 4 and (rows.phase == "test").sum() == 2
    assert not list((s2setup.runs / "states").glob("*.epoch.pt"))
    before = registry.read_bytes()
    script.main(args)
    assert registry.read_bytes() == before


def test_two_completed_fits_per_session_restore_into_new_output_without_retraining(s2setup, monkeypatch):
    setup_standalone(s2setup)
    script, calls = load_run_script(), []
    real = script.fit_kgat

    def spy(*args, **kwargs):
        fit = real(*args, **kwargs)
        calls.append((args[4], args[2].gamma, kwargs.get("fixed_epochs")))
        assert fit.complete
        return fit

    monkeypatch.setattr(script, "fit_kgat", spy)
    identity = {"source": "unchanged", "split": "unchanged"}
    previous = None
    for session in range(3):
        output = s2setup.tmp / f"session-{session}"
        output.mkdir()
        if previous is not None:
            load_wrapper().restore_results(previous, output, identity)
        (output / "run-identity.json").write_text(json.dumps(identity))
        script.main(
            [
                "--data",
                str(s2setup.data),
                "--model",
                str(s2setup.models / "tkgat_kg.yaml"),
                "--device",
                "cpu",
                "--registry",
                str(output / "registry.csv"),
                "--runs-dir",
                str(output / "runs"),
                "--refit",
                "--resume",
                "--max-new-runs",
                "2",
                "--epochs-per-fit",
                str(FIXED["max_epochs"]),
            ]
        )
        assert len(calls) == 2 * (session + 1)
        rows = pd.read_csv(output / "registry.csv")
        assert (rows.phase == "val").sum() == min(4, 2 * (session + 1))
        assert (rows.phase == "test").sum() == (2 if session == 2 else 0)
        previous = output
    summary = output / "runs" / s2setup.runs.name / "tkgat_kg_refit_summary.json"
    assert summary.exists()
    assert len(set(calls)) == 6
