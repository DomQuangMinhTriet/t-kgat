"""The registry records which commit produced each number."""

import subprocess

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
