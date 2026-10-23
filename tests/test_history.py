import importlib.util
import json

import pandas as pd
import pytest
import yaml

from tkgat import registry
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline.build import save_split

from .conftest import ROOT


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_writes_one_history_per_grid_point_and_seed(synthetic, tmp_path):
    dcfg = {
        "name": "synthetic",
        "processed_dir": str(tmp_path / "processed"),
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 4, "kcore": 2},
    }
    save_split(temporal_split(synthetic, SplitConfig(**dcfg["split"])), dcfg, {})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(dcfg), encoding="utf-8")
    fixed = {"batch_size": 256, "max_epochs": 6, "eval_every": 2, "patience": 5}
    mcfg = {"name": "bprmf", "seeds": [0, 1], "grid": [{"dim": 8, "lr": 0.01}, {"dim": 4, "lr": 0.01}], "fixed": fixed}
    (tmp_path / "bprmf.yaml").write_text(yaml.safe_dump(mcfg), encoding="utf-8")

    load_run_script().main(
        [
            "--data",
            str(tmp_path / "data.yaml"),
            "--model",
            str(tmp_path / "bprmf.yaml"),
            "--registry",
            str(tmp_path / "registry.csv"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
        ]
    )

    out = tmp_path / "runs" / "synthetic"
    files = sorted(p.name for p in out.glob("*_history.csv"))
    assert files == [f"bprmf_{g}_seed{s}_history.csv" for g in (0, 1) for s in (0, 1)]
    reg = pd.read_csv(tmp_path / "registry.csv")
    val_rows = reg[reg["phase"] == "val"]
    for g, point in enumerate(mcfg["grid"]):
        for seed in mcfg["seeds"]:
            h = pd.read_csv(out / f"bprmf_{g}_seed{seed}_history.csv")
            assert list(h["epoch"]) == list(range(1, fixed["max_epochs"] + 1))
            assert h["loss"].notna().all()
            evaluated = h["epoch"] % fixed["eval_every"] == 0
            assert h.loc[evaluated, "recall@20"].notna().all() and h.loc[~evaluated, "recall@20"].isna().all()
            assert not any("test" in c for c in h.columns)
            assert (h["grid_index"] == g).all() and (h["seed"] == seed).all()
            cfg = json.loads(h["config"].iloc[0])
            assert {k: cfg[k] for k in point} == point and {k: cfg[k] for k in fixed} == fixed
            assert cfg["device"] == "cpu"
            # the history agrees with the registry row of the same run
            row = val_rows[(val_rows["config"] == json.dumps(point, sort_keys=True)) & (val_rows["seed"] == seed)]
            assert len(row) == 1
            best = int(h["best_epoch"].iloc[0])
            assert best == int(row["best_epoch"].iloc[0])
            assert h.loc[h["epoch"] == best, "recall@20"].iloc[0] == pytest.approx(row["recall@20"].iloc[0])


def test_save_history_refuses_test_scores(tmp_path):
    with pytest.raises(ValueError, match="test"):
        registry.save_history(tmp_path / "h.csv", [{"epoch": 1, "loss": 0.5, "test_recall@20": 0.1}], {})
    with pytest.raises(ValueError, match="test"):
        registry.save_history(tmp_path / "h.csv", [{"epoch": 1, "loss": 0.5}], {"phase_test": 1})
