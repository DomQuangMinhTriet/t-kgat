"""The Last-FM data report only contains numbers that come from files written by the pipeline.

The same two layers as ``tests/test_data_report.py`` (whose tests cover the checker itself):
- ``docs/data-report-lastfm.md`` against the committed ``docs/data-report-lastfm-numbers.md`` (needs no data);
- the committed numbers file against a fresh run of ``scripts/07_report_numbers_lastfm.py`` (skipped without the data).
"""

import importlib.util
import re

import pytest

from tkgat.analysis import report
from tkgat.analysis.report import Numbers, parse_numbers, unknown_numbers

from .conftest import ROOT

REPORT = ROOT / "docs" / "data-report-lastfm.md"
NUMBERS = ROOT / "docs" / "data-report-lastfm-numbers.md"
REQUIRED = [
    ROOT / "data" / "processed" / "lastfm" / "meta.json",
    ROOT / "data" / "processed" / "lastfm" / "kg" / "static" / "kg_meta.json",
    ROOT / "data" / "processed" / "lastfm-u25" / "meta.json",
    ROOT / "data" / "processed" / "lastfm-u25" / "kg" / "static" / "kg_meta.json",
    ROOT / "data" / "processed" / "lastfm-u10" / "meta.json",
    ROOT / "data" / "processed" / "lastfm-u10" / "kg" / "static" / "kg_meta.json",
]


def test_render_can_name_another_report():
    text = Numbers().render(title="T", script="s.py", report="r.md", test="t.py")
    assert text.startswith("# T\n") and "`s.py`" in text and "`r.md`" in text and "`t.py`" in text
    assert "07_report_numbers.py" not in text  # the MovieLens defaults are not leaking in


def test_render_defaults_are_still_the_movielens_report():
    text = Numbers().render()
    assert "`scripts/07_report_numbers.py`" in text and "`docs/data-report.md`" in text


@pytest.mark.skipif(not (REPORT.exists() and NUMBERS.exists()), reason="report or numbers file not written yet")
def test_every_number_in_the_report_is_in_the_numbers_file():
    bad = unknown_numbers(REPORT.read_text(encoding="utf-8"), NUMBERS.read_text(encoding="utf-8"))
    assert not bad, "numbers not in docs/data-report-lastfm-numbers.md:\n" + "\n".join(
        f"{t!r} in: {line}" for t, line in bad
    )


@pytest.mark.skipif(not NUMBERS.exists(), reason="numbers file not written yet")
def test_the_numbers_file_holds_no_score_and_reads_only_validation_rows_of_the_registry():
    """The numbers file may quote the registry for training hours and best epochs, never for a score, and only
    from validation rows: a test score must not reach the data report (rule 1 of CLAUDE.md)."""
    rows = parse_numbers(NUMBERS.read_text(encoding="utf-8"))
    scores = [key for key in rows if re.search(r"recall|ndcg|precision|score", key, re.I)]
    assert not scores, f"score-like keys in the numbers file: {scores}"
    from_results = {
        key: source for key, (_, source) in rows.items() if "results/" in source or "registry" in source.lower()
    }
    not_val = [key for key, source in from_results.items() if "phase=val" not in source]
    assert not not_val, f"numbers read from results/ without saying phase=val: {not_val}"


def test_the_script_drops_test_rows_before_computing_anything():
    """``add_registry`` is only ever given validation rows; check the call site filters on the phase."""
    source = (ROOT / "scripts" / "07_report_numbers_lastfm.py").read_text(encoding="utf-8")
    assert "phase == 'val'" in source and source.index("phase == 'val'") < source.index("add_registry(n, run")


def load_script():
    spec = importlib.util.spec_from_file_location("lastfm_numbers", ROOT / "scripts" / "07_report_numbers_lastfm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(
    not NUMBERS.exists() or not all(p.exists() for p in REQUIRED), reason="needs the data and a committed numbers file"
)
def test_committed_numbers_file_matches_a_fresh_run(tmp_path):
    out = tmp_path / "numbers.md"
    load_script().main(["--out", str(out)])
    fresh = out.read_text(encoding="utf-8")
    msg = "docs/data-report-lastfm-numbers.md is out of date: run `python scripts/07_report_numbers_lastfm.py`"
    assert fresh == NUMBERS.read_text(encoding="utf-8"), msg
    assert report.parse_numbers(fresh)  # the file parses back into keys
