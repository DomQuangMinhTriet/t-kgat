"""scripts/run.py with KGAT: val-only without test files, the t2 graph at test, one state per run, resume."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.eval.metrics import evaluate
from tkgat.kg.ckg import build_ckg, load_kg, used_tails
from tkgat.models.kgat import KGAT
from tkgat.pipeline.build import load_saved, save_split
from tkgat.train.kgat import split_params

from .conftest import ROOT

RUN_NAME = "synthetic-kgat"
FIXED = {
    "dim": 4,
    "kge_dim": 4,
    "layers": [4, 2],
    "batch_size": 64,
    "batch_size_kg": 128,
    "max_epochs": 4,
    "eval_every": 1,
    "patience": 2,
    "select_metric": "recall@20",
}
GRID = [
    {"lr": 0.02, "l2": 1e-5, "l2_kg": 1e-5, "mess_dropout": 0.1},
    {"lr": 0.01, "l2": 1e-4, "l2_kg": 1e-5, "mess_dropout": 0.1},
]
N_ATTR = 5  # attribute entities; the one with the largest id appears only in the t2 KG


def write_kg(processed, enc, meta):
    """A KG in the layout of scripts/06_build_kg.py: t2 adds one fact on a tail that t1 never uses."""
    n = enc.n_items
    items = np.arange(n)
    t1 = np.unique(
        np.concatenate(
            [
                np.stack([items, np.zeros(n, dtype=int), n + items % 4], axis=1),
                np.stack([items, np.ones(n, dtype=int), n + (items * 3 + 1) % 4], axis=1),
            ]
        ),
        axis=0,
    )
    t2 = np.concatenate([t1, [[0, 0, n + 4]]])  # the new tail n + 4 is unseen at t1
    kg_dir = processed / "kg" / "default"
    kg_dir.mkdir(parents=True)
    for name, tri in (("t1", t1), ("t2", t2)):
        pd.DataFrame(tri, columns=["h", "r", "t"]).to_csv(kg_dir / f"kg_{name}.tsv", sep="\t", index=False)
    keys = [*[str(i) for i in enc.item_ids], *[f"Q{i}" for i in range(N_ATTR)]]
    pd.DataFrame(
        {"entity": range(n + N_ATTR), "kind": ["item"] * n + ["wikidata"] * N_ATTR, "key": keys, "label": ""}
    ).to_csv(kg_dir / "entities.csv", index=False)
    pd.DataFrame({"relation": [0, 1], "name": ["a", "b"]}).to_csv(kg_dir / "relations.csv", index=False)
    info = {
        "variant": "default",
        "time_consistent": True,
        "wikidata_snapshot": "2026-10-17",
        "inputs": {"split_sha256": meta["sha256"]},
        "n_items": n,
        "n_entities": n + N_ATTR,
        "n_relations": 2,
        "n_triples": {"t1": len(t1), "t2": len(t2)},
    }
    (kg_dir / "kg_meta.json").write_text(json.dumps(info), encoding="utf-8")
    return len(t1), len(t2)


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def setup(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    cfg = {"name": "synthetic", "run_name": RUN_NAME, "processed_dir": str(tmp_path / "processed")}
    save_split(split, cfg, {})
    _, meta = load_saved(cfg)
    sizes = write_kg(tmp_path / "processed", encode(split), meta)
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg))
    models = {}
    for tag, variant in (("kgat", "default"), ("kgat_nokg", "none")):
        models[tag] = tmp_path / f"{tag}.yaml"
        models[tag].write_text(
            yaml.safe_dump({"name": "kgat", "seeds": [1, 2], "kg": {"variant": variant}, "grid": GRID, "fixed": FIXED})
        )
    return tmp_path, data_cfg, models, sizes


def run(setup, tag, *extra, registry="registry.csv", runs="runs"):
    tmp, data_cfg, models, _ = setup
    load_run_script().main(
        ["--data", str(data_cfg), "--model", str(models[tag]), "--device", "cpu"]
        + ["--registry", str(tmp / registry), "--runs-dir", str(tmp / runs), *extra]
    )
    return tmp / registry, tmp / runs / RUN_NAME


def test_val_only_needs_no_test_file_and_writes_no_test_output(setup):
    tmp, *_ = setup
    (tmp / "processed" / "test.csv").unlink()
    (tmp / "processed" / "seen_before_test.csv").unlink()
    registry, out = run(setup, "kgat", "--val-only")
    reg = pd.read_csv(registry)
    assert set(reg["phase"]) == {"val"} and set(reg["model"]) == {"kgat"} and len(reg) == 4  # 2 points x 2 seeds
    assert not list(out.glob("*.npz")) and not (out / "kgat_summary.json").exists()
    summary = json.loads((out / "kgat_val_summary.json").read_text())
    assert summary["val_only"] is True and "t2" not in summary["kg"] and "test_mean_std" not in summary
    assert summary["kg"]["t1"]["kg_variant"] == "default" and summary["kg"]["t1"]["time_consistent"] is True
    for g in range(2):
        for seed in (1, 2):
            assert (out / f"kgat_{g}_seed{seed}_history.csv").exists()
            assert (out / "states" / f"kgat_{g}_seed{seed}.pt").exists()


def test_val_only_never_reads_the_t2_graph(setup):
    tmp, *_ = setup
    (tmp / "processed" / "kg" / "default" / "kg_t2.tsv").unlink()
    run(setup, "kgat", "--val-only")  # would fail if the t2 file were read


def test_each_state_is_a_cpu_state_dict_of_the_best_epoch(setup):
    _, out = run(setup, "kgat", "--val-only")
    for path in (out / "states").glob("*.pt"):
        ck = torch.load(path, weights_only=True)
        assert all(t.device.type == "cpu" for t in ck["state"].values())
        assert ck["best_epoch"] >= 1 and "recall@20" in ck["best_val"]
        assert "test" not in " ".join(ck)  # nothing about test in a checkpoint


def test_test_is_scored_on_the_t2_graph_with_frozen_weights(setup):
    tmp, _, _, (n_t1, n_t2) = setup
    registry, out = run(setup, "kgat")
    reg = pd.read_csv(registry)
    summary = json.loads((out / "kgat_summary.json").read_text())
    selected = summary["selected"]
    test = reg[reg["phase"] == "test"]
    assert len(test) == 2 and set(test["config"]) == {json.dumps(selected, sort_keys=True)}  # only the selected point
    t1, t2 = summary["kg"]["t1"], summary["kg"]["t2"]
    assert (t1["n_triples"], t1["n_triples_dropped_unseen_tail"]) == (n_t1, 0)
    assert (t2["n_triples"], t2["n_triples_dropped_unseen_tail"]) == (n_t2 - 1, 1)  # Q2: the unseen-tail fact is gone
    assert t1["n_interactions"] == t2["n_interactions"] and t2["cutoff"] == "t2"

    # rebuild the score by hand: same state, t2 graph, attention recomputed
    cfg = yaml.safe_load((tmp / "data.yaml").read_text())
    split, meta = load_saved(cfg)
    data = encode(split)
    processed = tmp / "processed"
    kg1, kg2 = (load_kg(processed, "default", c, data, meta) for c in ("t1", "t2"))
    g = GRID.index(selected)
    model_cfg = split_params({**selected, **FIXED}, "cpu")[0]
    for seed in (1, 2):
        ck = torch.load(out / "states" / f"kgat_{g}_seed{seed}.pt", weights_only=True)
        net = KGAT(build_ckg(data, kg1), model_cfg, device="cpu")
        net.load_state_dict(ck["state"])
        net.set_graph(build_ckg(data, kg2, known_tails=used_tails(kg1)))
        net.update_attention()
        res = evaluate(net.scorer(), data.test, data.mask_test)
        saved = np.load(out / f"kgat_seed{seed}.npz")
        assert np.array_equal(saved["users"], res["users"])
        assert np.allclose(saved["recall@20"], res["per_user"]["recall@20"])
        row = test[test["seed"] == seed].iloc[0]
        assert row["recall@20"] == pytest.approx(res["recall@20"]) and row["n_users"] == res["n_users"]


def test_the_val_rows_do_not_depend_on_the_test_files(setup):
    tmp, *_ = setup
    full_reg, _ = run(setup, "kgat", registry="full.csv", runs="runs_full")
    for name in ("test.csv", "seen_before_test.csv"):
        (tmp / "processed" / name).unlink()
    only_reg, _ = run(setup, "kgat", "--val-only", registry="only.csv", runs="runs_only")
    full = pd.read_csv(full_reg)
    full = full[full["phase"] == "val"].drop(columns=["run_at", "seconds", "git_commit"])
    only = pd.read_csv(only_reg).drop(columns=["run_at", "seconds", "git_commit"])
    pd.testing.assert_frame_equal(full.reset_index(drop=True), only.reset_index(drop=True), check_dtype=False)


def test_nokg_has_its_own_model_tag_and_a_graph_without_triples(setup):
    registry, out = run(setup, "kgat_nokg")
    reg = pd.read_csv(registry)
    assert set(reg["model"]) == {"kgat_nokg"}
    summary = json.loads((out / "kgat_nokg_summary.json").read_text())
    assert summary["kg"]["t1"]["kg_variant"] == "none" and summary["kg"]["t1"]["n_triples"] == 0
    assert summary["kg"]["t2"]["n_edges"] == summary["kg"]["t1"]["n_edges"]  # no KG: nothing changes at test


def test_resume_trains_nothing_and_logs_nothing_twice(setup, monkeypatch):
    registry, out = run(setup, "kgat")
    before = pd.read_csv(registry)
    summary = (out / "kgat_summary.json").read_text()

    def no_training(*args, **kwargs):
        raise AssertionError("a finished run must not be trained again")

    script = load_run_script()
    monkeypatch.setattr(script, "fit_kgat", no_training)
    tmp, data_cfg, models, _ = setup
    script.main(
        ["--data", str(data_cfg), "--model", str(models["kgat"]), "--device", "cpu", "--resume"]
        + ["--registry", str(registry), "--runs-dir", str(tmp / "runs")]
    )
    after = pd.read_csv(registry)
    pd.testing.assert_frame_equal(before, after)  # no val row and no test row was added
    resumed = json.loads((out / "kgat_summary.json").read_text())
    assert resumed["test_mean_std"] == json.loads(summary)["test_mean_std"]


def test_resume_finishes_a_run_that_stopped_before_test(setup):
    registry, out = run(setup, "kgat", "--val-only")  # states and val rows exist, no test row
    assert set(pd.read_csv(registry)["phase"]) == {"val"}
    tmp, data_cfg, models, _ = setup
    load_run_script().main(
        ["--data", str(data_cfg), "--model", str(models["kgat"]), "--device", "cpu", "--resume"]
        + ["--registry", str(registry), "--runs-dir", str(tmp / "runs")]
    )
    reg = pd.read_csv(registry)
    assert (reg["phase"] == "val").sum() == 4 and (reg["phase"] == "test").sum() == 2


def test_resume_refuses_a_checkpoint_of_another_configuration(setup):
    run(setup, "kgat", "--val-only")
    tmp, data_cfg, models, _ = setup
    body = yaml.safe_load(models["kgat"].read_text())
    body["fixed"]["layers"] = [4, 4]
    models["kgat"].write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="another configuration"):
        load_run_script().main(
            ["--data", str(data_cfg), "--model", str(models["kgat"]), "--device", "cpu", "--resume"]
            + ["--registry", str(tmp / "registry.csv"), "--runs-dir", str(tmp / "runs")]
        )


def test_resume_is_only_for_kgat(setup):
    tmp, data_cfg, *_ = setup
    mostpop = tmp / "mostpop.yaml"
    mostpop.write_text(yaml.safe_dump({"name": "mostpop", "grid": [{"window_days": None}]}))
    with pytest.raises(SystemExit):
        load_run_script().main(["--data", str(data_cfg), "--model", str(mostpop), "--resume"])


def test_an_unknown_kg_variant_is_refused(setup):
    tmp, data_cfg, models, _ = setup
    body = yaml.safe_load(models["kgat"].read_text())
    body["kg"] = {"variant": "future"}
    models["kgat"].write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="kg.variant"):
        run(setup, "kgat", "--val-only")


@pytest.mark.parametrize("tag", ["kgat", "kgat_nokg", "kgat_drop_undated", "kgat_static"])
def test_the_shipped_configs_parse_and_agree_on_grid_and_fixed(tag):
    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{tag}.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs" / "model" / "kgat.yaml").read_text())
    assert cfg["name"] == "kgat" and cfg["grid"] == ref["grid"] and cfg["fixed"] == ref["fixed"]
    assert len(cfg["grid"]) == 4 and cfg["seeds"] == [2019, 2020, 2021, 2022, 2023]
    assert cfg["kg"]["variant"] == {"kgat": "default", "kgat_nokg": "none"}.get(tag, tag.removeprefix("kgat_"))
    for point in cfg["grid"]:
        split_params({**point, **cfg["fixed"]}, "cpu")  # every key is a known setting
