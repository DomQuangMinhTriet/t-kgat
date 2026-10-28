"""The registry records which commit produced each number."""

import subprocess

import pandas as pd

from tkgat import registry


def test_registry_commit_ignores_untracked_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "a.py"], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True)
    registry.append(tmp_path / "results" / "registry.csv", {"dataset": "d", "seed": 1})
    assert not registry.git_commit().endswith("-dirty")  # the untracked registry does not count
    (tmp_path / "a.py").write_text("x = 2\n")
    assert registry.git_commit().endswith("-dirty")


def _git(*args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True)


def test_registry_commit_ignores_tracked_results(tmp_path, monkeypatch):
    """registry.csv and run outputs are committed; changing them must not mark the commit dirty."""
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    registry.append(tmp_path / "results" / "registry.csv", {"dataset": "d", "seed": 1})
    (tmp_path / "results" / "runs").mkdir()
    (tmp_path / "results" / "runs" / "h.csv").write_text("epoch\n1\n")
    (tmp_path / "src").mkdir()
    _git("add", ".")
    _git("commit", "-qm", "init")
    assert not registry.git_commit().endswith("-dirty")

    registry.append(tmp_path / "results" / "registry.csv", {"dataset": "d", "seed": 2})
    (tmp_path / "results" / "runs" / "h.csv").write_text("epoch\n1\n2\n")
    assert not registry.git_commit().endswith("-dirty")
    monkeypatch.chdir(tmp_path / "src")  # the exclusion is relative to the repo root, not the cwd
    assert not registry.git_commit().endswith("-dirty")

    (tmp_path / "a.py").write_text("x = 2\n")
    assert registry.git_commit().endswith("-dirty")


def _row(run_at, seed, phase="val", config='{"dim": 64}', recall=0.1, commit="abc"):
    return {
        "run_at": run_at,
        "git_commit": commit,
        "dataset": "ml-1m",
        "split_sha256": "s",
        "model": "bprmf",
        "config": config,
        "seed": seed,
        "phase": phase,
        "best_epoch": 5,
        "recall@10": recall,
        "recall@20": recall,
        "ndcg@10": recall,
        "ndcg@20": recall,
        "n_users": 10,
        "seconds": 1.0,
        "device": "cpu",
    }


def test_load_keeps_newest_run_per_key(tmp_path):
    path = tmp_path / "registry.csv"
    rows = [
        _row("2026-10-23T10:00:00Z", 2019, recall=0.10),  # first run
        _row("2026-10-23T10:00:00Z", 2020, recall=0.20),  # only run of this seed
        _row("2026-10-23T12:00:00Z", 2019, recall=0.11),  # re-run of seed 2019: wins
        _row("2026-10-23T09:00:00Z", 2019, recall=0.09),  # older run placed last (union merge): loses
        _row("2026-10-23T12:00:00Z", 2019, phase="test", recall=0.30),  # same seed, other phase: kept
        _row("2026-10-23T12:00:00Z", 2019, config='{"dim": 32}', recall=0.40),  # other config: kept
    ]
    pd.DataFrame(rows, columns=registry.FIELDS).to_csv(path, index=False)

    everything = registry.load(path, latest=False)
    assert len(everything) == len(rows)
    latest = registry.load(path)
    assert len(latest) == 4
    assert not latest.duplicated(registry.RUN_KEY).any()
    val64 = latest[(latest["phase"] == "val") & (latest["config"] == '{"dim": 64}')].set_index("seed")
    assert val64.loc[2019, "recall@20"] == 0.11 and val64.loc[2020, "recall@20"] == 0.20


def test_load_breaks_run_at_ties_by_line_order(tmp_path):
    path = tmp_path / "registry.csv"
    rows = [_row("2026-10-23T10:00:00Z", 2019, recall=0.1), _row("2026-10-23T10:00:00Z", 2019, recall=0.2)]
    pd.DataFrame(rows, columns=registry.FIELDS).to_csv(path, index=False)
    assert registry.load(path)["recall@20"].tolist() == [0.2]


def test_load_keeps_runs_of_different_splits(tmp_path):
    """Two splits with the same dataset name are different runs, not re-runs of each other."""
    path = tmp_path / "registry.csv"
    a = _row("2026-10-23T10:00:00Z", 2019, recall=0.18) | {"split_sha256": "aaa5core"}
    b = _row("2026-10-24T10:00:00Z", 2019, recall=0.21) | {"split_sha256": "bbb10core"}
    pd.DataFrame([a, b], columns=registry.FIELDS).to_csv(path, index=False)
    latest = registry.load(path)
    assert sorted(latest["split_sha256"]) == ["aaa5core", "bbb10core"]


def test_load_keeps_a_hash_that_looks_like_a_number_as_text(tmp_path):
    """A hash such as 347285991e48 is a valid float literal: read as one it would be 3.47e+56 and never match."""
    path = tmp_path / "registry.csv"
    registry.append(path, _row("2026-10-23T10:00:00Z", 2019, recall=0.18) | {"split_sha256": "347285991e48"})
    assert registry.load(path)["split_sha256"].tolist() == ["347285991e48"]
