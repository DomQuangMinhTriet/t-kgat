"""The official configs of decision 0016 (configs/model/v02) carry the values fixed by the small-set gate."""

from pathlib import Path

import yaml

V02 = Path(__file__).resolve().parents[1] / "configs" / "model" / "v02"
TRAINED = ["bprmf", "kgat", "kgat_nokg", "tkgat_both", "tkgat_both_perm", "tkgat_both_nokg", "tkgat_both_noui"]


def load(name: str) -> dict:
    return yaml.safe_load((V02 / f"{name}.yaml").read_text(encoding="utf-8"))


def test_every_trained_model_has_the_gate_budget():
    for name in TRAINED:
        fixed = load(name)["fixed"]
        assert fixed["max_epochs"] == 150, name
        assert fixed["patience"] == 10, name
        assert fixed["eval_every"] == 5, name
        assert fixed["select_metric"] == "recall@20", name
        assert fixed.get("lr_schedule", "none") == "none", name


def test_no_lr_1e4_grid_point_is_left():
    for name in ["bprmf", "kgat", "kgat_nokg"]:
        assert all(point["lr"] == 1e-3 for point in load(name)["grid"]), name
    assert len(load("kgat")["grid"]) == 2 and len(load("kgat_nokg")["grid"]) == 2
    assert len(load("tkgat_both")["grid"]) == 3


def test_kgat_family_shares_everything_but_the_kg():
    kgat, nokg, both = load("kgat"), load("kgat_nokg"), load("tkgat_both")
    assert kgat["fixed"] == nokg["fixed"] == both["fixed"]
    assert kgat["grid"] == nokg["grid"] and kgat["seeds"] == nokg["seeds"] == both["seeds"]
    assert kgat["kg"] != nokg["kg"]


def test_derived_configs_resolve_inside_the_directory():
    assert load("tkgat_both")["from_kgat"] == "kgat"
    for name in ["tkgat_both_nokg", "tkgat_both_noui", "tkgat_both_perm"]:
        assert load(name)["from_tkgat"] == "tkgat_both", name
        assert (V02 / "tkgat_both.yaml").exists()
    assert (V02 / "kgat.yaml").exists()
