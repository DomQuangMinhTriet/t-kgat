"""scripts/11_numbers_latest_lastfm.py and the committed files of ``lastfm-u25-latest`` (Phon's latest-listen set).

The unit checks always run. The check that the committed numbers file equals a fresh run needs the processed data and
the merged registry of the five result zips (see ``scripts/12_merge_latest_zips.py``), which are not in git, so it is
skipped on a machine without them (set ``LATEST_REGISTRY`` to that registry to run it).
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

from tkgat.analysis.report import Numbers, parse_numbers

from .conftest import ROOT

SCRIPT = ROOT / "scripts" / "11_numbers_latest_lastfm.py"
NUMBERS = ROOT / "docs" / "data-report-lastfm-latest-numbers.md"
MODELS = ROOT / "results" / "rq3_models_lastfm-latest.csv"
PAIRS = ROOT / "results" / "rq3_pairs_lastfm-latest.csv"
RUN = "lastfm-u25-latest"
DAY = 86400


def load_script():
    spec = importlib.util.spec_from_file_location("numbers_latest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_edge_numbers_of_a_tiny_train_window():
    # 100 edges: ten users, one edge a day for ten days, from Monday 2014-04-07 00:00 UTC
    start = int(pd.Timestamp("2014-04-07", tz="UTC").timestamp())
    train = pd.DataFrame(
        {
            "user": [u for u in range(10) for _ in range(10)],
            "ts": [start + d * DAY for _ in range(10) for d in range(10)],
        }
    )
    train = train.assign(item=range(len(train)), rating=1)
    n = Numbers()
    load_script().add_edges_latest(n, RUN, train, int(train["ts"].max()))
    got = {key: value for _, key, value, _ in n.rows}
    assert got[f"{RUN}.edge.window_days"] == "9"
    assert got[f"{RUN}.edge.first_day"] == "07/04/2014"
    # last edge day is t_ref: the edges younger than 7 days are those of the 7 last days (age 0..6): 70 of 100
    assert got[f"{RUN}.edge.last_7d_share"] == "70,0%"
    assert got[f"{RUN}.edge.dt_norm_median"] == "0,50"
    assert got[f"{RUN}.spread.user_under_1h"] == "0,00%"


def test_the_numbers_file_holds_no_score_and_no_missing_value():
    rows = parse_numbers(NUMBERS.read_text(encoding="utf-8"))
    assert rows, "the numbers file should hold rows"
    assert all(key.startswith(("conv.", RUN)) for key in rows)
    assert not [key for key in rows if "recall" in key or "ndcg" in key], "a test score reached the numbers file"
    assert not [key for key, (value, _) in rows.items() if value == "chưa có"]
    # the registry rows are those of the zips of Phon, not of results/registry.csv of this repository
    assert not [key for key, (_, source) in rows.items() if source.startswith("results/registry.csv")]


def test_committed_tables_are_the_five_seeds_of_the_latest_dataset():
    models, pairs = pd.read_csv(MODELS), pd.read_csv(PAIRS)
    assert set(models["dataset"]) == {RUN} and set(pairs["dataset"]) == {RUN}
    neural = models[
        models["model"].isin(["bprmf", "kgat", "kgat_nokg", "tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm"])
    ]
    assert (neural["n_seeds"] == 5).all() and (neural["status"] == "ok").all()
    assert ((pairs["p_holm"] >= 0) & (pairs["p_holm"] <= 1)).all()
    assert (pairs["p_holm"] >= pairs["p_raw"] - 1e-12).all()


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / RUN / "meta.json").exists() or "LATEST_REGISTRY" not in os.environ,
    reason="needs data/processed/lastfm-u25-latest and LATEST_REGISTRY (merged registry of the result zips)",
)
def test_committed_numbers_file_matches_a_fresh_run(tmp_path: Path):
    script = load_script()
    text = script.build(ROOT / "data" / "processed" / RUN, Path(os.environ["LATEST_REGISTRY"])).render(
        title="Số liệu của bản Last-FM `lastfm-u25-latest` (của Phôn)",
        script="scripts/11_numbers_latest_lastfm.py",
        report="notebooks/00_present_lastfm.ipynb",
        test="tests/test_latest_numbers.py",
    )
    assert text == NUMBERS.read_text(encoding="utf-8")
