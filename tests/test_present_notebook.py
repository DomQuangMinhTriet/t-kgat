"""notebooks/00_present_2026-10-31.ipynb: structure, no hand-typed numbers, no training, and "run all" without error.

The structure checks always run. Running the notebook needs ``data/processed`` (and the notebook tooling), so it is
skipped on a machine without them (CI).
"""

import ast
import json
import re
import time

import pytest

from tkgat.analysis.report import unknown_numbers

from .conftest import ROOT

NOTEBOOK = ROOT / "notebooks" / "00_present_2026-10-31.ipynb"
NUMBERS = ROOT / "docs" / "data-report-numbers.md"
PLAN = ROOT / "docs" / "plan.md"

# numbers that may stay in markdown: the date of the meeting, the day of the 17/10 feedback, "100K" of a dataset name
ALLOWED_STATIC = {"31/10/2026", "17", "100"}
RUN_SECONDS = 120


def cells(kind):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == kind]


def source(cell):
    return "".join(cell["source"])


def sections():
    """``{number: (title, presenter)}`` from the markdown cells: ``## N. title`` then ``*Người trình bày: who*``."""
    found = {}
    for cell in cells("markdown"):
        m = re.match(r"## (\d+)\. (.+)\n\*Người trình bày: (.+)\*", source(cell))
        if m:
            found[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return found


def plan_outline():
    """``{number: (title, presenter)}`` of the 10-part outline in docs/plan.md section 12."""
    text = PLAN.read_text(encoding="utf-8")
    rows = re.findall(r"^(\d+)\. (.+) — (Phôn|Triết|Cả hai)\s*$", text, re.M)
    return {int(n): (title.strip(), who) for n, title, who in rows}


def test_sections_follow_the_outline_of_the_plan():
    outline, found = plan_outline(), sections()
    assert sorted(outline) == list(range(1, 11)), "docs/plan.md should hold the 10-part outline"
    assert sorted(found) == list(range(1, 12))  # the ten parts and the preliminary KGAT part
    for n, (title, who) in outline.items():
        nb_title, nb_who = found[n]
        assert nb_who.startswith(who), f"section {n}: presenter {nb_who!r}, the plan says {who!r}"
        a, b = (re.sub(r"\s*\([^)]*\)", "", t) for t in (title, nb_title))  # parentheses may be dropped
        assert a.startswith(b) or b.startswith(a), f"section {n}: {nb_title!r} vs {title!r}"
    assert [n for n, (_, who) in found.items() if who.startswith("Triết")] == [2, 4, 6, 7, 11]
    assert found[11][0] == "KGAT và T-KGAT (sơ bộ)"


def test_preliminary_part_is_empty_and_labelled():
    last = next(c for c in cells("markdown") if source(c).startswith("## 11."))
    text = source(last).lower()
    assert "kết quả sơ bộ, chưa đóng băng, chạy trước khi cô duyệt quy trình" in text
    assert "ml-1m-u10" in text and "10% user của ml-1m" in text
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    after = [c for c in nb["cells"][nb["cells"].index(last) :]]
    assert len(after) == 2 and after[1]["cell_type"] == "code"  # the heading cell, then one cell that only counts rows
    assert "to_markdown" not in source(after[1]) and "display(" not in source(after[1])  # no results table


def test_markdown_has_no_numbers_that_are_not_in_the_numbers_file():
    text = "\n\n".join(source(c) for c in cells("markdown"))
    bad = [
        (t, line) for t, line in unknown_numbers(text, NUMBERS.read_text(encoding="utf-8")) if t not in ALLOWED_STATIC
    ]
    assert not bad, f"hand-typed numbers in markdown: {bad}"


def test_code_cells_are_clean_python_without_outputs_and_without_training():
    for i, cell in enumerate(cells("code")):
        code = source(cell)
        ast.parse(code)
        assert cell["outputs"] == [] and cell["execution_count"] is None, f"code cell {i} was saved with output"
    joined = "\n".join(source(c) for c in cells("code"))
    # the notebook only reads results; fit_bpr appears only in inspect.signature(fit_bpr)
    for forbidden in (
        r"\bfit_bpr\(",
        r"\bsubprocess\b",
        r"scripts/run\.py",
        r"\.backward\(",
        r"\boptim\.",
        r"\bregistry\.append\b",
    ):
        assert not re.search(forbidden, joined), f"the notebook must not train or write results: {forbidden}"


def test_every_needed_numbers_key_exists():
    """Every ``V("key")`` of the notebook is a key of docs/data-report-numbers.md: stale keys fail here, not live."""
    from tkgat.analysis.report import parse_numbers

    keys = parse_numbers(NUMBERS.read_text(encoding="utf-8"))
    joined = "\n".join(source(c) for c in cells("code"))
    used = set(re.findall(r'V\(\s*"([^"{]+)"\s*\)', joined)) | set(re.findall(r"V\(\s*'([^'{]+)'\s*\)", joined))
    assert used, "the notebook should read its numbers with V(...)"
    missing = sorted(k for k in used if k not in keys)
    assert not missing, f"keys not in docs/data-report-numbers.md: {missing}"


@pytest.mark.skipif(
    not all((ROOT / "data" / "processed" / d / "meta.json").exists() for d in ("ml-100k", "ml-1m")),
    reason="needs data/processed for ML-100K and ML-1M",
)
def test_restart_and_run_all_without_error_in_under_two_minutes():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    start = time.time()
    nbclient.NotebookClient(
        nb, timeout=RUN_SECONDS, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    ).execute()
    assert time.time() - start < RUN_SECONDS
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
