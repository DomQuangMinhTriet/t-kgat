"""scripts/09_significance.py on a tiny fake results folder: counts, Holm, curves, and the guards."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tkgat.analysis.significance import compare_dataset, selected_point, warning
from tkgat.eval.stats import paired_wilcoxon
from tkgat.registry import FIELDS

from .conftest import ROOT

GRID = [{"dim": 8, "lr": 0.01, "l2": 0.0001}, {"dim": 16, "lr": 0.01, "l2": 0.0001}]
FIXED = {"batch_size": 64, "max_epochs": 10, "eval_every": 2, "patience": 2, "select_metric": "recall@20"}
N_USERS = 30


def load_script():
    spec = importlib.util.spec_from_file_location("significance_script", ROOT / "scripts" / "09_significance.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def write_npz(path, users, rng, base=None):
    """Per-user scores; ``base`` (if given) is copied for the first five users so their difference is exactly 0."""
    scores = {m: rng.random(len(users)) for m in ("recall@10", "ndcg@10", "recall@20", "ndcg@20")}
    if base is not None:
        for m in ("recall@20", "ndcg@20"):
            scores[m][:5] = base[m][:5]
    np.savez(path, users=users, **scores)
    return scores


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(1)
    users = np.arange(N_USERS)
    runs = tmp_path / "runs" / "fake"
    runs.mkdir(parents=True)
    seeds = [11, 12]
    bpr = {s: write_npz(runs / f"bprmf_seed{s}.npz", users, rng) for s in seeds}
    mean = {m: np.mean([bpr[s][m] for s in seeds], axis=0) for m in ("recall@20", "ndcg@20")}
    for base in ("mostpop", "mostpop_recent"):
        write_npz(runs / f"{base}_seed0.npz", users, rng, base=mean)
    (runs / "bprmf_summary.json").write_text(json.dumps({"selected": GRID[1]}), encoding="utf-8")
    for seed, n_epochs in zip(seeds, (6, 4), strict=True):  # the second seed stops earlier
        config = json.dumps({**FIXED, **GRID[1], "device": "cpu"})
        rows = [
            {
                "epoch": e,
                "loss": 0.7 - 0.05 * e + 0.01 * seed,
                "recall@20": 0.1 + 0.01 * e if e % 2 == 0 else np.nan,
                "best_epoch": n_epochs - 2,
                "config": config,
            }
            for e in range(1, n_epochs + 1)
        ]
        pd.DataFrame(rows).to_csv(runs / f"bprmf_1_seed{seed}_history.csv", index=False)
    reg_rows = []
    for model, model_seeds in (("bprmf", seeds), ("mostpop", [0]), ("mostpop_recent", [0])):
        for s in model_seeds:
            reg_rows.append(
                {f: "" for f in FIELDS}
                | {"run_at": "2026-10-28T00:00:00", "dataset": "fake", "split_sha256": "abc", "model": model}
                | {"config": "{}", "seed": s, "phase": "test", "n_users": N_USERS}
            )
    reg_path = tmp_path / "registry.csv"
    pd.DataFrame(reg_rows).to_csv(reg_path, index=False)
    model_path = tmp_path / "bprmf.yaml"
    model_path.write_text(yaml.safe_dump({"name": "bprmf", "seeds": seeds, "grid": GRID, "fixed": FIXED}))
    return tmp_path, reg_path, model_path, mean, runs


def test_wilcoxon_reports_the_ranked_differences_not_the_pairs():
    a = np.array([0.3, 0.2, 0.5, 0.1, 0.4, 0.6])
    b = np.array([0.3, 0.2, 0.1, 0.0, 0.1, 0.2])  # two users are tied
    res = paired_wilcoxon(a, b)
    assert res["n_pairs"] == 6 and res["n_nonzero"] == 4


def test_script_writes_counts_holm_curves_and_table(results):
    tmp, reg_path, model_path, mean, runs = results
    out = tmp / "out"
    load_script().main(
        [
            "--registry", str(reg_path), "--runs-dir", str(tmp / "runs"), "--model", str(model_path),
            "--out-csv", str(out / "sig.csv"), "--out-curves", str(out / "curves.csv"),
            "--figures-dir", str(out / "fig"), "--out-table", str(out / "table.md"),
        ]
    )  # fmt: skip
    sig = pd.read_csv(out / "sig.csv")
    assert len(sig) == 4 and set(sig["model_b"]) == {"mostpop", "mostpop_recent"}  # 2 baselines x 2 metrics
    assert (sig["n_pairs"] == N_USERS).all()
    for _, row in sig.iterrows():
        base = np.load(runs / f"{row['model_b']}_seed0.npz")[row["metric"]]
        diff = mean[row["metric"]] - base
        assert row["n_nonzero"] == np.count_nonzero(diff) <= N_USERS - 5  # the five tied users are not ranked
        assert row["p_raw"] <= row["p_holm"] <= 1  # Holm never lowers a p-value
        assert "only 30 evaluation users" in row["warning"]
    for _, g in sig.groupby("metric"):  # the smaller raw p is multiplied by the number of comparisons (2)
        small = g.loc[g["p_raw"].idxmin()]
        assert small["p_holm"] == pytest.approx(min(1.0, 2 * small["p_raw"]))

    curves = pd.read_csv(out / "curves.csv")
    assert curves["n_seeds"].tolist() == [2, 2, 2, 2, 1, 1]  # the second seed stopped after epoch 4
    assert curves["loss_std"].iloc[:4].notna().all() and curves["loss_std"].iloc[4:].isna().all()  # ddof=1, NaN for 1
    assert (out / "fig" / "fig5_bprmf_curve_fake.png").exists()
    table = (out / "table.md").read_text(encoding="utf-8")
    assert "| Batch size | 64 |" in table and "| Dim | 16 |" in table and "2 / 2" in table


def test_selected_point_must_be_in_the_grid():
    assert selected_point({"selected": GRID[1]}, GRID) == 1
    with pytest.raises(ValueError, match="not in the grid"):
        selected_point({"selected": {"dim": 99, "lr": 0.01, "l2": 0.0001}}, GRID)


def test_history_trained_with_another_config_is_rejected(results):
    tmp, reg_path, model_path, *_ = results
    cfg = yaml.safe_load(model_path.read_text())
    cfg["fixed"]["batch_size"] = 128
    model_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="batch_size"):
        load_script().main(
            ["--registry", str(reg_path), "--runs-dir", str(tmp / "runs"), "--model", str(model_path)]
            + ["--out-csv", str(tmp / "a.csv"), "--out-curves", str(tmp / "b.csv")]
            + ["--figures-dir", str(tmp / "f"), "--out-table", str(tmp / "t.md")]
        )


def test_rows_of_two_splits_cannot_be_paired(results):
    tmp, reg_path, *_ = results
    reg = pd.read_csv(reg_path)
    reg.loc[reg["model"] == "mostpop", "split_sha256"] = "other"
    with pytest.raises(ValueError, match="different splits"):
        compare_dataset("fake", reg, tmp / "runs")


def test_warning_text():
    assert warning(1000, 900) == ""
    assert "only 51 evaluation users" in warning(51, 40) and "non-zero" in warning(1000, 10)
