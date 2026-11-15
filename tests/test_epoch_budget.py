"""scripts/13_epoch_budget.py: the epoch evidence for decision 0016 item 6 (from history files only, no test score)."""

import importlib.util
import json

import pandas as pd

from .conftest import ROOT

SCRIPT = ROOT / "scripts" / "13_epoch_budget.py"
DOC = ROOT / "docs" / "epoch-budget.md"


def load_script():
    spec = importlib.util.spec_from_file_location("epoch_budget", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_history(path, *, recalls, max_epochs=300, lr=0.001, model="kgat", seed=2019, grid=0, nested=True):
    """A history file: one row per epoch, a validation score every 5 epochs (``recalls``: epochs 5, 10, ...)."""
    last = 5 * len(recalls)
    fit = {"lr": lr, "max_epochs": max_epochs, "patience": 10, "eval_every": 5}
    config = json.dumps({"fit": fit, "model": {}} if nested else fit)
    best_epoch = 5 * (max(range(len(recalls)), key=recalls.__getitem__) + 1)
    rows = []
    for epoch in range(1, last + 1):
        score = recalls[epoch // 5 - 1] if epoch % 5 == 0 else None
        rows.append(
            {
                "epoch": epoch,
                "recall@20": score,
                "seconds": 2.0,
                "dataset": "toy",
                "model": model,
                "grid_index": grid,
                "seed": seed,
                "config": config,
                "best_epoch": best_epoch,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_read_run_early_stop_and_checkpoint_shares(tmp_path):
    module = load_script()
    # the score peaks at epoch 30 (0.5); by epoch 25 the best so far is 0.4, i.e. 80% of the peak
    write_history(tmp_path / "a_history.csv", recalls=[0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.45, 0.44, 0.43])
    run = module.read_run(tmp_path / "a_history.csv")
    assert run["best_epoch"] == run["best_epoch_from_curve"] == 30
    assert run["last_epoch"] == 45 and not run["censored"]
    assert run["share_at_25"] == 0.4 / 0.5 and run["share_at_50"] == 1.0
    assert run["seconds"] == 2.0 * 45 and run["lr"] == 0.001 and run["max_epochs"] == 300


def test_run_that_reaches_max_epochs_is_censored_and_flat_config_is_read(tmp_path):
    module = load_script()
    write_history(tmp_path / "b_history.csv", recalls=[0.1, 0.2, 0.3, 0.4], max_epochs=20, nested=False, model="bprmf")
    run = module.read_run(tmp_path / "b_history.csv")
    assert run["censored"] and run["best_epoch"] == 20 and run["max_epochs"] == 20


def test_file_without_validation_score_is_skipped(tmp_path):
    module = load_script()
    pd.DataFrame({"epoch": [1, 2], "recall@20": [None, None], "dataset": "toy"}).to_csv(
        tmp_path / "c_history.csv", index=False
    )
    assert module.read_run(tmp_path / "c_history.csv") is None


def test_collect_and_summarise_group_by_dataset_model_and_lr(tmp_path):
    module = load_script()
    folder = tmp_path / "toy"
    folder.mkdir()
    for seed, peak in ((2019, 6), (2020, 8)):
        write_history(
            folder / f"kgat_0_seed{seed}_history.csv", recalls=[0.1 * i for i in range(1, peak)] + [0.5] * 2, seed=seed
        )
    write_history(
        folder / "kgat_2_seed2019_history.csv", recalls=[0.1, 0.2, 0.3, 0.4], lr=0.0001, max_epochs=20, grid=2
    )
    (tmp_path / "other").mkdir()
    write_history(tmp_path / "other" / "kgat_0_seed2019_history.csv", recalls=[0.1, 0.2])
    runs = module.collect(tmp_path, datasets=["toy"])
    assert (
        len(runs) == 3
    )  # the "other" folder is not asked for (the dataset column says "toy" in both, the folder decides)
    summary = module.summarise(runs).set_index("lr")
    assert summary.loc[0.001, "runs"] == 2 and summary.loc[0.001, "cut_at_max"] == 0
    assert summary.loc[0.0001, "runs"] == 1 and summary.loc[0.0001, "cut_at_max"] == 1
    text = module.markdown(runs, module.summarise(runs))
    assert "## toy" in text and "| kgat | 0.0001 | 1 |" in text


def test_committed_document_equals_a_fresh_run():
    module = load_script()
    runs = module.collect(ROOT / "results" / "runs")
    assert DOC.read_text(encoding="utf-8") == module.markdown(runs, module.summarise(runs))
