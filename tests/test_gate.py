"""Rules of the small-set gate (decision 0016 items 5, 6, 8, 9) and the script that writes docs/small-set-gate.md."""

import importlib.util
import json

import pandas as pd
import pytest

from tkgat.analysis import gate

from .conftest import ROOT


def rows(model, schedule, lr, means, point_tag="a"):
    """Three seeds per grid point with the given mean validation Recall@20."""
    out = []
    for i, mean in enumerate(means):
        for seed, delta in zip((1, 2, 3), (-0.001, 0.0, 0.001), strict=True):
            point = json.dumps({"lr": lr, "l2": 1e-5, "tag": f"{point_tag}{i}"}, sort_keys=True)
            out.append(
                {
                    "model": model,
                    "schedule": schedule,
                    "point": point,
                    "lr": lr,
                    "seed": seed,
                    gate.METRIC: mean + delta,
                }
            )
    return out


def table(*parts):
    frames = [p if isinstance(p, pd.DataFrame) else pd.DataFrame(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def test_best_points_marks_the_best_mean_per_model_and_schedule():
    pts = gate.best_points(table(rows("kgat", "none", 1e-3, [0.04, 0.05]), rows("kgat", "none", 1e-4, [0.01])))
    best = pts[pts["best"]]
    assert len(best) == 1 and best["mean"].iloc[0] == pytest.approx(0.05) and best["seeds"].iloc[0] == 3


def test_schedule_verdict_needs_cosine_not_lower_for_every_model():
    ok = table(
        rows("bprmf", "none", 1e-3, [0.040]),
        rows("bprmf", "cosine", 1e-3, [0.040]),  # equal counts as not lower
        rows("kgat", "none", 1e-3, [0.050]),
        rows("kgat", "cosine", 1e-3, [0.052]),
    )
    assert gate.schedule_verdict(gate.best_points(ok))["verdict"] == "cosine"
    worse = table(
        ok[ok["model"] == "bprmf"], rows("kgat", "none", 1e-3, [0.050]), rows("kgat", "cosine", 1e-3, [0.049])
    )
    assert gate.schedule_verdict(gate.best_points(worse))["verdict"] == "none"
    partial = table(
        rows("bprmf", "none", 1e-3, [0.04]), rows("bprmf", "cosine", 1e-3, [0.04]), rows("kgat", "none", 1e-3, [0.05])
    )
    assert gate.schedule_verdict(gate.best_points(partial))["verdict"] == "pending"


def test_lr_verdict_drops_1e4_only_if_it_is_lower_everywhere_it_exists():
    low = table(
        rows("kgat", "none", 1e-3, [0.05]),
        rows("kgat", "none", 1e-4, [0.03]),
        rows("bprmf", "cosine", 1e-3, [0.04]),
        rows("bprmf", "cosine", 1e-4, [0.01]),
    )
    assert gate.lr_verdict(gate.best_points(low))["verdict"] == "drop"
    wins_once = table(
        low[low["model"] == "kgat"], rows("bprmf", "cosine", 1e-3, [0.04]), rows("bprmf", "cosine", 1e-4, [0.05])
    )
    assert gate.lr_verdict(gate.best_points(wins_once))["verdict"] == "keep"
    assert gate.lr_verdict(gate.best_points(table(rows("kgat", "none", 1e-3, [0.05]))))["verdict"] == "pending"


def test_pruned_needs_a_mean_not_lower_and_fewer_triples():
    pts = gate.best_points(table(rows("kgat", "none", 1e-3, [0.050]), rows("kgat_pruned", "none", 1e-3, [0.050])))
    assert gate.pruned_verdict(pts, {"default": 100, "pruned": 90})["verdict"] == "pruned"
    assert gate.pruned_verdict(pts, {"default": 100, "pruned": 100})["verdict"] == "default"  # no fewer triples
    lower = gate.best_points(table(rows("kgat", "none", 1e-3, [0.050]), rows("kgat_pruned", "none", 1e-3, [0.049])))
    assert gate.pruned_verdict(lower, {"default": 100, "pruned": 90})["verdict"] == "default"


def test_gate_verdict_tells_the_advisor_when_kgat_is_not_above_nokg():
    above = gate.best_points(table(rows("kgat", "none", 1e-3, [0.05]), rows("kgat_nokg", "none", 1e-3, [0.04])))
    assert gate.gate_verdict(above)["verdict"] == "pass"
    equal = gate.best_points(table(rows("kgat", "none", 1e-3, [0.05]), rows("kgat_nokg", "none", 1e-3, [0.05])))
    assert gate.gate_verdict(equal)["verdict"] == "tell the advisor"


def runs(model, shares, lr=1e-3):
    return [{"model": model, "lr": lr, "share_at_100": a, "share_at_150": b, "share_at_200": c} for a, b, c in shares]


def test_epoch_budget_is_the_smallest_value_where_every_model_reaches_98_percent():
    data = pd.DataFrame(runs("kgat", [(0.97, 0.99, 1.0)] * 3) + runs("kgat_nokg", [(0.99, 0.995, 1.0)] * 3))
    got = gate.epoch_budget(data, ("kgat", "kgat_nokg"))
    assert got["verdict"] == 150 and len(got["rows"]) == 2
    slow = pd.DataFrame(runs("kgat", [(0.5, 0.9, 0.97)] * 3))
    assert gate.epoch_budget(slow, ("kgat",))["verdict"] == "more than 200"
    only_1e4 = pd.DataFrame(runs("kgat", [(1, 1, 1)], lr=1e-4))
    assert gate.epoch_budget(only_1e4, ("kgat",))["verdict"] == "pending"  # lr 1e-4 runs do not count


def load_script():
    spec = importlib.util.spec_from_file_location("gate_script", ROOT / "scripts" / "15_small_set_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def registry_rows(run):
    out = []
    for tag, mean in (
        ("bprmf_w3_none", 0.04),
        ("bprmf_w3_cosine", 0.041),
        ("kgat_w3_none", 0.05),
        ("kgat_nokg_w3_none", 0.04),
    ):
        for seed in (1, 2, 3):
            cfg = {"lr": 0.001, "l2": 1e-5} | (
                {"lr_schedule": "cosine", "lr_final_ratio": 0.01} if "cosine" in tag else {}
            )
            out.append(
                {
                    "dataset": run,
                    "model": tag,
                    "config": json.dumps(cfg, sort_keys=True),
                    "seed": seed,
                    "phase": "val",
                    "best_epoch": 100,
                    "recall@20": mean,
                    "ndcg@20": mean / 2,
                    "seconds": 5.0,
                }
            )
    out.append({**out[0], "phase": "test", "recall@20": 0.99})  # a test row must never be used
    out.append({**out[0], "dataset": "other-set", "recall@20": 0.99})
    return pd.DataFrame(out)


def test_script_reads_validation_rows_of_the_small_set_only():
    script = load_script()
    res = script.val_results(registry_rows("lastfm-w3"), "lastfm-w3")
    assert len(res) == 12 and set(res["model"]) == {"bprmf", "kgat", "kgat_nokg"} and res[gate.METRIC].max() < 0.99
    assert set(res.loc[res["model"] == "bprmf", "schedule"]) == {"none", "cosine"}
    assert all("lr_schedule" not in p for p in res["point"])  # the schedule is a column, not part of the point
    assert script.split_tag("kgat_pruned_w3_cosine") == ("kgat_pruned", "cosine") and script.split_tag("kgat") is None


def test_markdown_states_the_verdicts_and_never_calls_the_numbers_results(tmp_path):
    script = load_script()
    res = script.val_results(registry_rows("lastfm-w3"), "lastfm-w3")
    meta = {
        "clean": {"time_window": {"window_months": 3, "ratings_kept": 1000}},
        "n_users": 10,
        "n_items": 20,
        "train_pos": 300,
        "val_pos": 40,
    }
    text = script.markdown("lastfm-w3", res, pd.DataFrame(), meta, {"default": 100, "pruned": 90})
    assert "**Nothing here is a result**" in text and "### schedule: **pending**" in text
    assert "### KGAT vs KGAT without KG: **pass**" in text and "### epoch budget: **pending**" in text
    empty = script.markdown("lastfm-w3", pd.DataFrame(), pd.DataFrame(), meta, {"default": 100})
    assert "No validation row" in empty
