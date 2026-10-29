"""scripts/10_rq3_table.py on a fake registry in a temporary folder (no training, no real results)."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from tkgat import registry
from tkgat.analysis import rq3
from tkgat.eval.stats import holm, paired_wilcoxon
from tkgat.registry import FIELDS

from .conftest import ROOT

DS, SHA, N_USERS = "fake", "abc123def456", 40
SEEDS = [2019, 2020, 2021, 2022, 2023]
POINT = {"lr": 0.001, "l2": 0.0001}
TKGAT_POINT = {**POINT, "gamma": 3, "t_ref": 1000, "dt_den": 500}


def load_script():
    spec = importlib.util.spec_from_file_location("rq3_script", ROOT / "scripts" / "10_rq3_table.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


class World:
    """A registry and the per-user files that ``scripts/run.py`` would have written for it."""

    def __init__(self, tmp):
        self.tmp, self.rng, self.clock = tmp, np.random.default_rng(7), 0
        self.rows, self.arrays = [], {}
        self.runs = tmp / "runs"

    def add(self, tag, seeds, config=None, commit="abc1234", phase="test", dataset=DS, sha=SHA, npz=True, shift=0.0):
        self.clock += 1  # every call is a later run than the one before
        config = config if config is not None else POINT
        for seed in seeds:
            cfg = {**config, "perm_seed": seed} if tag == "tkgat_perm" else config
            metrics = {
                m: np.clip(self.rng.random(N_USERS) * 0.4 + shift, 0, 1)
                for m in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
            }
            self.rows.append(
                {f: "" for f in FIELDS}
                | {"run_at": f"2026-10-29T10:{self.clock:02d}:00Z", "git_commit": commit, "dataset": dataset}
                | {"split_sha256": sha, "model": tag, "config": json.dumps(cfg, sort_keys=True), "seed": seed}
                | {"phase": phase, "n_users": N_USERS, "device": "cpu"}
                | {m: float(np.nanmean(v)) for m, v in metrics.items()}
            )
            if npz and phase == "test":
                (self.runs / dataset).mkdir(parents=True, exist_ok=True)
                np.savez(self.runs / dataset / f"{tag}_seed{seed}.npz", users=np.arange(N_USERS), **metrics)
                self.arrays[(tag, seed)] = metrics
        return self

    def full(self):
        self.add("mostpop", [0], {"window_days": None}).add("mostpop_recent", [0], {"window_days": 7})
        self.add("bprmf", SEEDS, {"dim": 32, **POINT})
        for tag, config in (("kgat", POINT), ("kgat_nokg", POINT), ("tkgat", TKGAT_POINT), ("tkgat_perm", TKGAT_POINT)):
            self.add(tag, SEEDS, config)
        return self

    def build(self, datasets=None):
        path = self.tmp / "registry.csv"
        pd.DataFrame(self.rows, columns=FIELDS).to_csv(path, index=False)
        return rq3.build(path, self.runs, datasets or [DS])

    def history(self, tag, seed, values, point=POINT, g=0, commit="abc1234", sha=SHA, dataset=DS):
        """A KGAT-style history file: val Recall@20 every 5 epochs (``values``), the settings in ``config``."""
        epochs = range(1, 5 * len(values) + 1)
        recall = {5 * (i + 1): v for i, v in enumerate(values)}
        config = {
            "model": {"l2": point["l2"], "gamma": point.get("gamma", 0.0), "dim": 4},
            "fit": {"lr": point["lr"], "patience": 2},
        }
        frame = pd.DataFrame(
            {"epoch": list(epochs), "loss_cf": 0.5, "recall@20": [recall.get(e, np.nan) for e in epochs]}
        )
        for name, value in {
            "dataset": dataset,
            "split_sha256": sha,
            "model": tag,
            "grid_index": g,
            "seed": seed,
        }.items():
            frame[name] = value
        frame["config"], frame["git_commit"] = json.dumps(config, sort_keys=True), commit
        (self.runs / dataset).mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.runs / dataset / f"{tag}_{g}_seed{seed}_history.csv", index=False)
        return np.array(values)

    def mean_of(self, tag, metric, seeds=SEEDS):
        return np.mean([self.arrays[(tag, s)][metric] for s in seeds], axis=0)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def info(result):
    return result["datasets"][DS]


def test_the_table_holds_the_registry_means_and_sample_stds(world):
    world.full()
    frame = rq3.models_frame(world.build())
    reg = pd.DataFrame(world.rows)
    assert len(frame) == 7 and set(frame["status"]) == {"ok"}
    for tag in ("bprmf", "kgat", "kgat_nokg", "tkgat", "tkgat_perm"):
        rows = reg[reg["model"] == tag]
        mine = frame[frame["model"] == tag].iloc[0]
        for metric in rq3.METRICS:
            assert mine[f"{metric}_mean"] == pytest.approx(rows[metric].mean())
            assert mine[f"{metric}_std"] == pytest.approx(rows[metric].std(ddof=1))  # sample std, not ddof=0
        assert mine["n_seeds"] == 5 and mine["n_users"] == N_USERS
    single = frame[frame["model"] == "mostpop"].iloc[0]
    assert single["n_seeds"] == 1 and np.isnan(single["recall@20_std"])  # one seed has no sample std
    assert "perm_seed" not in frame[frame["model"] == "tkgat_perm"].iloc[0]["point"]
    assert "gamma = 3" in frame[frame["model"] == "tkgat"].iloc[0]["point"]


def test_pairs_are_paired_wilcoxon_per_user_with_a_holm_family_of_three(world):
    world.full()
    pairs = info(world.build())["pairs"]
    assert len(pairs) == 6 and all(p["status"] == "ok" for p in pairs)
    for metric in rq3.METRICS:
        block = [p for p in pairs if p["metric"] == metric]
        expected = []
        for p in block:
            a, b = world.mean_of(p["model_a"], metric), world.mean_of(p["model_b"], metric)
            res = paired_wilcoxon(a, b)  # the function of tkgat.eval, on the per-user mean over the seeds
            assert (p["n_pairs"], p["n_nonzero"]) == (res["n_pairs"], res["n_nonzero"])
            assert p["p_raw"] == pytest.approx(res["p_value"]) and p["mean_diff"] == pytest.approx(res["mean_diff"])
            expected.append(res["p_value"])
        assert [p["p_holm"] for p in block] == pytest.approx(holm(expected)["p_adjusted"])
        assert [(p["model_a"], p["model_b"]) for p in block] == [(a, b) for a, b, _ in rq3.PAIRS]


def test_a_model_that_has_not_run_is_printed_not_an_error(world):
    world.add("bprmf", SEEDS, {"dim": 32, **POINT}).add("kgat", SEEDS)
    result = world.build()
    text = rq3.markdown(result)
    frame = rq3.models_frame(result).set_index("model")
    assert frame.loc["tkgat", "status"] == rq3.NOT_RUN and frame.loc["kgat", "status"] == "ok"
    assert text.count(rq3.NOT_RUN) >= 5
    statuses = {p["pair"]: p["status"] for p in info(result)["pairs"]}
    assert statuses["T-KGAT so với KGAT"].startswith(rq3.NOT_RUN) and "T-KGAT" in statuses["T-KGAT so với KGAT"]
    assert statuses["KGAT so với KGAT không KG"].startswith(rq3.NOT_RUN)


def test_holm_does_not_depend_on_which_comparisons_have_run(world):
    world.add("kgat", SEEDS).add("tkgat", SEEDS, TKGAT_POINT)  # only T-KGAT against KGAT can be computed
    only = [p for p in info(world.build())["pairs"] if p["status"] == "ok" and p["metric"] == "recall@20"]
    assert len(only) == 1
    assert only[0]["p_holm"] == pytest.approx(min(1.0, 3 * only[0]["p_raw"]))  # the family has three, run or not
    world.add("kgat_nokg", SEEDS)
    full = [p for p in info(world.build())["pairs"] if p["status"] == "ok" and p["metric"] == "recall@20"]
    assert {p["pair"] for p in full} == {"T-KGAT so với KGAT", "KGAT so với KGAT không KG"}


def test_no_row_at_all_is_not_an_error(world):
    result = world.build()  # an empty registry
    assert info(result)["chosen"] == {} and info(result)["pairs"][0]["status"].startswith(rq3.NOT_RUN)
    assert rq3.NOT_RUN in rq3.markdown(result) and (rq3.models_frame(result)["status"] == rq3.NOT_RUN).all()


def test_only_test_rows_count(world):
    world.add("bprmf", SEEDS, {"dim": 32, **POINT})
    world.add("bprmf", SEEDS, {"dim": 32, **POINT}, phase="val", shift=0.5)  # val rows of the same model
    mine = rq3.models_frame(world.build()).set_index("model").loc["bprmf"]
    test_rows = [r for r in world.rows if r["phase"] == "test"]
    assert mine["recall@20_mean"] == pytest.approx(np.mean([r["recall@20"] for r in test_rows]))


def test_dirty_rows_are_ignored_and_do_not_hide_an_older_clean_run(world):
    world.add("tkgat", SEEDS, TKGAT_POINT)  # clean, older
    clean = [r["recall@20"] for r in world.rows]
    world.add("tkgat", SEEDS, TKGAT_POINT, commit="def5678-dirty", shift=0.5)  # newer, dirty: must not count
    result = world.build()
    mine = rq3.models_frame(result).set_index("model").loc["tkgat"]
    assert mine["recall@20_mean"] == pytest.approx(np.mean(clean))
    assert info(result)["dirty_ignored"] == 5 and "5 dòng test mang nhãn `-dirty` bị bỏ qua" in rq3.markdown(result)


def test_a_model_with_only_dirty_rows_has_not_run(world):
    world.add("kgat", SEEDS, commit="abc1234-dirty")
    frame = rq3.models_frame(world.build()).set_index("model")
    assert frame.loc["kgat", "status"] == rq3.NOT_RUN


def test_the_newest_point_and_the_newest_split_win(world):
    world.add("bprmf", SEEDS, {"dim": 32, **POINT}, sha="oldsplit00000")  # bprmf was only run on an older split
    world.add("kgat", SEEDS, {"lr": 0.01, "l2": 0.01}, sha="oldsplit00000")
    world.add("kgat", SEEDS, {"lr": 0.0001, "l2": 0.001})  # an older run on the current split
    world.add("kgat", SEEDS, POINT)  # the newest run selected this point
    result = world.build()
    assert info(result)["split"] == SHA
    mine = rq3.models_frame(result).set_index("model")
    assert "lr = 0.001" in mine.loc["kgat", "point"] and mine.loc["kgat", "n_seeds"] == 5
    assert mine.loc["bprmf", "status"] == rq3.NOT_RUN  # rows of another split are never mixed in
    newest = [r for r in world.rows if r["model"] == "kgat" and json.loads(r["config"]) == POINT]
    assert mine.loc["kgat", "recall@20_mean"] == pytest.approx(np.mean([r["recall@20"] for r in newest]))


def test_the_control_is_one_point_although_perm_seed_differs_per_row(world):
    world.add("tkgat_perm", SEEDS, TKGAT_POINT)
    assert len({r["config"] for r in world.rows}) == 5  # perm_seed makes every row's config different
    mine = rq3.models_frame(world.build()).set_index("model").loc["tkgat_perm"]
    assert mine["n_seeds"] == 5 and "perm_seed" not in mine["point"]


def test_fewer_than_five_seeds_is_flagged_and_the_pair_uses_the_common_seeds(world):
    world.add("kgat", SEEDS[:3]).add("tkgat", SEEDS, TKGAT_POINT)
    result = world.build()
    assert "chưa đủ 5 seed" in rq3.markdown(result)
    pair = next(p for p in info(result)["pairs"] if p["pair"] == "T-KGAT so với KGAT" and p["metric"] == "recall@20")
    assert pair["status"] == "ok" and pair["n_seeds"] == 3
    expected = paired_wilcoxon(
        world.mean_of("tkgat", "recall@20", SEEDS[:3]), world.mean_of("kgat", "recall@20", SEEDS[:3])
    )
    assert pair["p_raw"] == pytest.approx(expected["p_value"])


def test_a_per_user_file_of_another_run_is_refused(world):
    world.add("kgat", SEEDS).add("tkgat", SEEDS, TKGAT_POINT)
    other = np.load(world.runs / DS / "kgat_seed2019.npz")
    changed = {k: other[k] for k in other.files}
    changed["recall@20"] = np.clip(changed["recall@20"] + 0.05, 0, 1)  # not the numbers of the registry row
    np.savez(world.runs / DS / "kgat_seed2019.npz", **changed)
    result = world.build()
    recall = next(p for p in info(result)["pairs"] if p["pair"] == "T-KGAT so với KGAT" and p["metric"] == "recall@20")
    assert recall["status"].startswith("không so sánh được") and "registry" in recall["status"]
    ndcg = next(p for p in info(result)["pairs"] if p["pair"] == "T-KGAT so với KGAT" and p["metric"] == "ndcg@20")
    assert ndcg["status"] == "ok"  # the other metric of that file still matches
    assert rq3.models_frame(result).set_index("model").loc["kgat", "status"] == "ok"  # the table itself is untouched


def test_a_missing_per_user_file_leaves_the_table_and_skips_the_pair(world):
    world.add("kgat", SEEDS, npz=False).add("tkgat", SEEDS, TKGAT_POINT)
    result = world.build()
    pair = next(p for p in info(result)["pairs"] if p["pair"] == "T-KGAT so với KGAT")
    assert "no per-user file" in pair["status"]
    assert rq3.models_frame(result).set_index("model").loc["kgat", "status"] == "ok"


@pytest.mark.parametrize("tag_present, expect", [(False, "Chưa có tag `v0.2-kgat`"), (True, "có trong repo")])
def test_the_output_says_the_kgat_numbers_are_preliminary(world, monkeypatch, tag_present, expect):
    monkeypatch.setattr(rq3, "freeze_tag_exists", lambda tag=rq3.FREEZE_TAG: tag_present)
    world.full()
    text = rq3.markdown(world.build())
    assert "kết quả sơ bộ, chưa đóng băng" in text and "v0.2-kgat" in text and expect in text
    for label in ("KGAT", "KGAT không KG", "T-KGAT", "T-KGAT (đối chứng hoán vị)"):
        assert f"| {label} [sơ bộ] |" in text  # each of the four rows carries the label
    for label in ("MostPop", "MostPop recent", "BPR-MF"):
        assert f"| {label} |" in text  # the baselines do not


def test_freeze_tag_is_false_outside_a_repository(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(rq3.subprocess, "run", no_git)
    assert rq3.freeze_tag_exists() is False


def test_the_script_writes_the_three_files_and_prints_the_table(world, capsys):
    world.full()
    pd.DataFrame(world.rows, columns=FIELDS).to_csv(world.tmp / "registry.csv", index=False)
    out = world.tmp / "out"
    load_script().main(
        ["--registry", str(world.tmp / "registry.csv"), "--runs-dir", str(world.runs), "--datasets", DS]
        + ["--out-md", str(out / "t.md"), "--out-models", str(out / "m.csv"), "--out-pairs", str(out / "p.csv")]
    )
    assert "Bảng RQ3" in capsys.readouterr().out
    assert "kết quả sơ bộ, chưa đóng băng" in (out / "t.md").read_text(encoding="utf-8")
    assert len(pd.read_csv(out / "m.csv")) == 7 and len(pd.read_csv(out / "p.csv")) == 6


def test_the_script_runs_on_a_registry_with_no_matching_rows(world):
    path = world.tmp / "registry.csv"
    pd.DataFrame([], columns=FIELDS).to_csv(path, index=False)  # a header only
    out = world.tmp / "out"
    load_script().main(
        ["--registry", str(path), "--runs-dir", str(world.runs), "--datasets", "nothing-here"]
        + ["--out-md", str(out / "t.md"), "--out-models", str(out / "m.csv"), "--out-pairs", str(out / "p.csv")]
    )
    assert rq3.NOT_RUN in (out / "t.md").read_text(encoding="utf-8")


def test_newest_is_what_load_keeps(world):
    world.add("bprmf", SEEDS, POINT).add("bprmf", SEEDS, POINT, shift=0.2)  # a re-run: the second wins
    path = world.tmp / "registry.csv"
    pd.DataFrame(world.rows, columns=FIELDS).to_csv(path, index=False)
    pd.testing.assert_frame_equal(registry.load(path), registry.newest(registry.load(path, latest=False)))
    assert len(registry.load(path)) == 5


# --- subset note, header note, missing baselines ---------------------------------------------------------------

SUBSET = "ml-1m-u10 là 10% user của ML-1M, không phải ML-1M đầy đủ"


def test_a_subset_of_ml_1m_is_never_presented_as_the_full_dataset(world):
    world.add("kgat", SEEDS, dataset="ml-1m-u10").add("kgat", SEEDS, dataset=DS)
    text = rq3.markdown(world.build(["ml-1m-u10", DS]))
    assert text.count(SUBSET) == 2  # once in the note at the top and once under the heading of the dataset
    assert text.index(SUBSET) < text.index("## ml-1m-u10")
    assert SUBSET not in rq3.markdown(world.build([DS]))  # a dataset that is not a subset gets no such note


@pytest.mark.parametrize("tag_present", [False, True])
def test_the_header_says_it_ran_before_the_advisor_approved_the_protocol(world, monkeypatch, tag_present):
    monkeypatch.setattr(rq3, "freeze_tag_exists", lambda tag=rq3.FREEZE_TAG: tag_present)
    text = rq3.markdown(world.full().build())
    assert "chạy trước khi cô duyệt giao thức" in text.split("\n## ")[0]  # in the note before the first dataset


def test_a_dataset_without_any_clean_baseline_row_says_so(world):
    world.add("kgat", SEEDS, dataset="ml-1m-u10")  # what ml-1m-u10 looks like today: KGAT, no baseline
    text = rq3.markdown(world.build(["ml-1m-u10"]))
    assert "**Baseline của ml-1m-u10 chưa chạy:**" in text and "MostPop, MostPop recent, BPR-MF" in text
    assert SUBSET in text  # both notes appear for that dataset


def test_a_dataset_with_some_baselines_lists_the_missing_ones(world):
    world.add("bprmf", SEEDS, {"dim": 32, **POINT}).add("kgat", SEEDS)
    text = rq3.markdown(world.build())
    assert "Baseline chưa chạy ở fake: MostPop, MostPop recent." in text and "**Baseline của fake" not in text


def test_no_baseline_note_when_all_three_baselines_have_run(world):
    text = rq3.markdown(world.full().build())
    assert "Baseline của" not in text and "Baseline chưa chạy ở" not in text


def test_dirty_baseline_rows_do_not_count_as_a_baseline_that_ran(world):
    for tag in rq3.BASELINES:
        world.add(tag, [0], {"window_days": None}, commit="abc1234-dirty")
    world.add("kgat", SEEDS)
    assert "**Baseline của fake chưa chạy:**" in rq3.markdown(world.build())


# --- validation learning curves ---------------------------------------------------------------------------------------


def curve_world(world, tags=("kgat", "tkgat", "tkgat_perm"), n_points=6):
    expected = {}
    for tag in tags:
        point = POINT if tag == "kgat" else TKGAT_POINT
        world.add(tag, SEEDS, point)
        expected[tag] = np.array([world.history(tag, seed, world.rng.random(n_points), point) for seed in SEEDS])
    return expected


def test_the_curve_is_the_mean_over_seeds_per_epoch_with_the_sample_std(world):
    expected = curve_world(world)
    curves = info(world.build())["curves"]
    assert set(curves) == {"kgat", "tkgat", "tkgat_perm"}
    for tag, values in expected.items():  # values: seeds x evaluations
        t = curves[tag]
        assert t["epoch"].tolist() == [5, 10, 15, 20, 25, 30] and (t["n_seeds"] == 5).all()
        assert t["mean"].to_numpy() == pytest.approx(values.mean(axis=0))
        assert t["std"].to_numpy() == pytest.approx(values.std(axis=0, ddof=1))  # sample std over seeds


def test_a_model_without_history_is_left_out_not_an_error(world):
    curve_world(world, tags=("kgat",))
    world.add("tkgat", SEEDS, TKGAT_POINT).add("tkgat_perm", SEEDS, TKGAT_POINT)  # run, but no history files
    result = world.build()
    assert set(info(result)["curves"]) == {"kgat"}
    assert info(result)["no_history"] == ["T-KGAT", "T-KGAT (đối chứng hoán vị)"]


def test_a_model_that_has_not_run_has_no_curve(world):
    world.history("kgat", 2019, [0.1, 0.2])  # a history file, but no clean test row: the point is unknown
    assert info(world.build())["curves"] == {}


def test_histories_of_another_point_split_or_dirty_commit_are_ignored(world):
    world.add("kgat", SEEDS)
    keep = [world.history("kgat", seed, world.rng.random(4)) for seed in SEEDS[:3]]
    world.history("kgat", 2022, [0.9] * 4, {"lr": 0.5, "l2": 0.0001}, g=1)  # another grid point
    world.history("kgat", 2023, [0.9] * 4, g=2, sha="oldsplit00000")  # another split
    world.history("kgat", 2023, [0.9] * 4, g=3, commit="abc1234-dirty")  # a dirty run
    t = info(world.build())["curves"]["kgat"]
    assert (t["n_seeds"] == 3).all() and t["mean"].to_numpy() == pytest.approx(np.mean(keep, axis=0))


def test_kgat_never_reads_the_history_of_kgat_nokg(world):
    world.add("kgat", SEEDS)
    keep = [world.history("kgat", seed, world.rng.random(4)) for seed in SEEDS[:2]]
    for seed in SEEDS:
        world.history("kgat_nokg", seed, [0.99] * 4)  # same settings, other model: its name starts with "kgat_"
    t = info(world.build())["curves"]["kgat"]
    assert (t["n_seeds"] == 2).all() and t["mean"].to_numpy() == pytest.approx(np.mean(keep, axis=0))


def test_a_seed_that_stops_early_leaves_fewer_seeds_at_the_late_epochs(world):
    world.add("kgat", SEEDS[:2])
    world.history("kgat", 2019, [0.1, 0.2, 0.3, 0.4])
    world.history("kgat", 2020, [0.3, 0.4])
    t = info(world.build())["curves"]["kgat"]
    assert t["n_seeds"].tolist() == [2, 2, 1, 1] and np.isnan(t["std"].iloc[-1]) and t["std"].iloc[0] > 0


def test_draw_curves_writes_a_png_or_nothing(world, tmp_path):
    path = tmp_path / "figs" / "c.png"
    assert rq3.draw_curves(DS, {}, path) is False and not path.exists()  # nothing to draw: no file
    curve_world(world, tags=("kgat", "tkgat"))
    assert rq3.draw_curves(DS, info(world.build())["curves"], path) is True
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def run_script(world, out):
    pd.DataFrame(world.rows, columns=FIELDS).to_csv(world.tmp / "registry.csv", index=False)
    load_script().main(
        ["--registry", str(world.tmp / "registry.csv"), "--runs-dir", str(world.runs), "--datasets", DS]
        + ["--out-md", str(out / "t.md"), "--out-models", str(out / "m.csv"), "--out-pairs", str(out / "p.csv")]
        + ["--figures-dir", str(out / "figures")]
    )
    return (out / "t.md").read_text(encoding="utf-8")


def test_the_script_draws_the_figure_and_the_table_links_to_it(world):
    curve_world(world, tags=("kgat", "tkgat"))
    world.add("tkgat_perm", SEEDS, TKGAT_POINT)  # run but no history
    out = world.tmp / "out"
    text = run_script(world, out)
    assert (out / f"figures/rq3_curves_{DS}.png").exists()
    assert f"](figures/rq3_curves_{DS}.png)" in text  # the path is relative to the markdown file
    assert "Chưa có history của: T-KGAT (đối chứng hoán vị)" in text


def test_the_script_draws_no_figure_without_any_history(world):
    world.full()
    out = world.tmp / "out"
    text = run_script(world, out)
    assert not (out / "figures").exists() and "![" not in text
    assert "Chưa có file history" in text and "chưa vẽ learning curve" in text


def test_kg_models_run_before_0011_go_to_the_appendix_not_the_table(world):
    world.full()
    world.rows = [  # the first KG-model runs happened before decision 0011, on the Wikidata-only KG
        r | {"run_at": "2026-10-28T10:00:00Z"} if r["model"] in rq3.KG_MODELS else r for r in world.rows
    ]
    old_kgat = np.mean([r["recall@20"] for r in world.rows if r["model"] == "kgat"])
    world.add("kgat", SEEDS)  # the re-run of 0011, same key, later
    result = world.build()
    assert set(result["superseded"][DS]) == set(rq3.KG_MODELS)
    assert result["superseded"][DS]["kgat"].mean_std("recall@20")[0] == pytest.approx(old_kgat)
    assert result["datasets"][DS]["chosen"]["kgat"].mean_std("recall@20")[0] != pytest.approx(old_kgat)
    text = rq3.markdown(result)
    assert "Phụ lục: KG chỉ có Wikidata" in text and "quyết định 0011" in text


def test_no_appendix_without_rows_before_0011(world):
    assert "Phụ lục" not in rq3.markdown(world.full().build())
