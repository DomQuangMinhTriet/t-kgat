"""scripts/10_sensitivity.py on a tiny fake results folder: val rows only, order, weak-evidence note."""

import importlib.util
import json

import pandas as pd
import pytest
import yaml

from tkgat import registry
from tkgat.analysis.sensitivity import close_pairs, order_of
from tkgat.registry import FIELDS

from .conftest import ROOT

POINT = {"dim": 4, "lr": 0.01, "l2": 0.0001}
# val Recall@20 of (mostpop, mostpop_recent, bprmf seeds) per run
SCORES = {
    "base": (0.10, 0.30, [0.20, 0.22]),  # MostPop recent > BPR-MF > MostPop
    "same": (0.05, 0.25, [0.12, 0.14]),  # same order, other absolute scores
    "flip": (0.20, 0.10, [0.40, 0.44]),  # BPR-MF > MostPop > MostPop recent
}
VAL_USERS = {"base": 66, "same": 99, "flip": 34}


def load_script():
    spec = importlib.util.spec_from_file_location("sens_script", ROOT / "scripts" / "10_sensitivity.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def reg_row(run, model, seed, phase, recall, config):
    return (
        {f: "" for f in FIELDS}
        | {"run_at": "2026-10-28T00:00:00", "dataset": run, "model": model, "seed": seed, "phase": phase}
        | {"config": json.dumps(config), "recall@20": recall, "n_users": 1}
    )


@pytest.fixture
def setup(tmp_path):
    rows, data_cfgs = [], []
    for run, (pop, recent, bpr) in SCORES.items():
        processed = tmp_path / "processed" / run
        processed.mkdir(parents=True)
        meta = {
            "config": {"q_val": 0.8, "q_test": 0.9, "min_rating": 4, "kcore": 5},
            "t1": 889237198,
            "t2": 891382233,
            "val_users": VAL_USERS[run],
            "val_pos": 700,
        }
        (processed / "meta.json").write_text(json.dumps(meta))
        cfg = tmp_path / f"{run}.yaml"
        cfg.write_text(yaml.safe_dump({"name": "x", "run_name": run, "processed_dir": str(processed)}))
        data_cfgs.append(str(cfg))
        sha = registry.sha256_file(processed / "meta.json")[:12]
        runs = tmp_path / "runs" / run
        runs.mkdir(parents=True)
        for model, selected in (
            ("mostpop", {"window_days": None}),
            ("mostpop_recent", {"window_days": 7}),
            ("bprmf", POINT),
        ):
            # --val-only runs write <model>_val_summary.json; the base run has the normal summary
            name = f"{model}_summary.json" if run == "base" else f"{model}_val_summary.json"
            (runs / name).write_text(json.dumps({"selected": selected}))
        for model, selected, values in (
            ("mostpop", {"window_days": None}, [pop]),
            ("mostpop_recent", {"window_days": 7}, [recent]),
            ("bprmf", POINT, bpr),
        ):
            for seed, v in enumerate(values):
                rows.append(reg_row(run, model, seed, "val", v, selected) | {"split_sha256": sha})
        # a test row and a val row of a grid point that was not selected must not change the table
        rows.append(reg_row(run, "bprmf", 0, "test", 0.999, POINT) | {"split_sha256": sha})
        rows.append(reg_row(run, "bprmf", 0, "val", 0.999, {**POINT, "dim": 99}) | {"split_sha256": sha})
    reg_path = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(reg_path, index=False)
    return tmp_path, reg_path, data_cfgs


def test_table_order_and_notes(setup):
    tmp, reg_path, data_cfgs = setup
    load_script().main(
        ["--data", *data_cfgs, "--registry", str(reg_path), "--runs-dir", str(tmp / "runs")]
        + ["--out-csv", str(tmp / "s.csv"), "--out-md", str(tmp / "s.md")]
    )
    table = pd.read_csv(tmp / "s.csv")
    assert len(table) == 9 and set(table["model"]) == {"mostpop", "mostpop_recent", "bprmf"}
    assert not any(c.startswith("test") or c == "phase" for c in table.columns)  # q_test is a cut-off, not a score
    bpr = table[(table["run_name"] == "base") & (table["model"] == "bprmf")].iloc[0]
    assert bpr["val_recall20_mean"] == pytest.approx(0.21) and bpr["n_seeds"] == 2  # the 0.999 rows are ignored
    assert bpr["val_recall20_std"] == pytest.approx(pd.Series([0.20, 0.22]).std(ddof=1))  # sample std
    assert table[(table["run_name"] == "base") & (table["model"] == "mostpop")]["val_recall20_std"].isna().all()
    assert order_of(table[table["run_name"] == "base"]) == ["mostpop_recent", "bprmf", "mostpop"]

    text = (tmp / "s.md").read_text(encoding="utf-8")
    assert "- `same`: thứ tự giữ nguyên" in text and "- `flip`: thứ tự đổi" in text
    flip_note = next(line for line in text.splitlines() if line.startswith("- `flip`"))
    assert "34 user" in flip_note and "bằng chứng yếu" in flip_note  # few val users: weak evidence either way
    assert "bằng chứng yếu" not in next(line for line in text.splitlines() if line.startswith("- `same`"))
    assert "không so sánh được giữa các cấu hình" in text
    assert "| base | 5 | 0,8 / 0,9 |" in text


def test_close_pairs_uses_the_bprmf_std():
    t = pd.DataFrame(
        {
            "model": ["mostpop", "mostpop_recent", "bprmf"],
            "val_recall20_mean": [0.10, 0.30, 0.295],
            "val_recall20_std": [float("nan"), float("nan"), 0.02],
        }
    )
    assert close_pairs(t) == [("mostpop_recent", "bprmf")]  # 0.005 < 0.02; the BPR-MF vs MostPop gap is large


def test_a_summary_for_another_split_is_rejected(setup):
    tmp, reg_path, data_cfgs = setup
    reg = pd.read_csv(reg_path)
    reg["split_sha256"] = "000000000000"  # rows of another split: nothing matches
    reg.to_csv(reg_path, index=False)
    with pytest.raises(ValueError, match="no val rows"):
        load_script().main(
            ["--data", *data_cfgs, "--registry", str(reg_path), "--runs-dir", str(tmp / "runs")]
            + ["--out-csv", str(tmp / "s.csv"), "--out-md", str(tmp / "s.md")]
        )
