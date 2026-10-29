"""notebooks/00_present_2026-10-31.ipynb: structure, no hand-typed numbers, no training, and "run all" without error.

The structure checks always run. Running the notebook needs ``data/processed`` (and the notebook tooling), so it is
skipped on a machine without them (CI).
"""

import ast
import json
import re
import time

import pandas as pd
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


def section_cells(n):
    """The cells of section ``n``: from its heading to the heading of the next section."""
    all_cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    is_heading = lambda c: c["cell_type"] == "markdown" and re.match(r"## \d+\.", source(c))  # noqa: E731
    first = next(i for i, c in enumerate(all_cells) if is_heading(c) and source(c).startswith(f"## {n}."))
    last = next((i for i in range(first + 1, len(all_cells)) if is_heading(all_cells[i])), len(all_cells))
    return all_cells[first:last]


def code_of(cells_, marker):
    """The source of the one code cell of ``cells_`` that contains ``marker``."""
    found = [source(c) for c in cells_ if c["cell_type"] == "code" and marker in source(c)]
    assert len(found) == 1, f"{len(found)} code cells contain {marker!r}"
    return found[0]


def run_cell(code, **namespace):
    """Run a code cell of the notebook with the names its setup cell defines replaced by stand-ins."""
    said, shown = [], []
    scope = {"pd": pd, "say": said.append, "display": shown.append, **namespace}
    exec(compile(code, "<notebook cell>", "exec"), scope)  # noqa: S102 - our own notebook, in a test
    return said, shown


def test_the_preliminary_part_is_labelled_and_only_reads_the_rq3_files():
    head = source(section_cells(11)[0])
    text = head.lower()
    assert "sơ bộ, chưa đóng băng (tag `v0.2-kgat`)" in text and "chạy trước khi cô duyệt quy trình" in text
    plain = text.replace("`", "").replace("\n> ", " ")  # no backticks, no line break inside the quoted note
    assert "ml-1m-u10 là 10% user của ml-1m, không phải ml-1m đầy đủ" in plain
    assert "*Người trình bày: Triết*" in head
    code = "\n".join(source(c) for c in section_cells(11) if c["cell_type"] == "code")
    for name in ("results/rq3_models.csv", "results/rq3_pairs.csv", "rq3_curves_"):
        assert name in code  # it reads what scripts/10_rq3_table.py wrote ...
    assert "chưa chạy" in code or "NOT_RUN" in code  # ... and says "chưa chạy" while a file is missing
    assert "registry.append" not in code and "fit_" not in code


def test_section_10_lists_the_decisions_of_the_repo_and_points_to_the_current_plan():
    cs = section_cells(10)
    markdown = "\n".join(source(c) for c in cs if c["cell_type"] == "markdown")
    code = "\n".join(source(c) for c in cs if c["cell_type"] == "code")
    assert "https://claude.ai/artifact/5KUTXrnhN5ZA8gR2XJNjXs" in markdown and "bản gốc" in markdown
    assert "Quyết định và trạng thái" in markdown and "docs/plan.md" in markdown
    assert "docs/decisions" in code and "- Status:" in code
    assert "plan.md" not in code  # the phases of the 17/10 plan are no longer read


def test_the_decision_table_is_the_titles_and_status_lines_of_the_files():
    cs = section_cells(10)
    _, shown = run_cell(code_of(cs, "docs/decisions"), ROOT=ROOT)
    table = shown[0].data  # the Styler of the displayed frame
    files = sorted((ROOT / "docs" / "decisions").glob("0*.md"))
    assert len(table) == len(files) >= 9 and table["quyết định"].tolist() == [f.stem for f in files]
    for (_, row), path in zip(table.iterrows(), files, strict=True):
        lines = path.read_text(encoding="utf-8").splitlines()
        status = next(x for x in lines if x.startswith("- Status:")).removeprefix("- Status:").strip()
        assert row["tiêu đề"] == lines[0].lstrip("# ").strip() and row["Status (nguyên văn)"] == status


# --- section 11 with and without the files of scripts/10_rq3_table.py ---------------------------------------

MODELS_COLUMNS = ["dataset", "model", "label", "split_sha256", "status", "point", "n_seeds", "n_users"]
MODELS_COLUMNS += ["recall@20_mean", "recall@20_std", "ndcg@20_mean", "ndcg@20_std"]
PAIRS_COLUMNS = ["dataset", "metric", "pair", "model_a", "model_b", "status", "n_seeds", "n_pairs", "n_nonzero"]
PAIRS_COLUMNS += ["mean_a", "mean_b", "mean_diff", "p_raw", "warning", "p_holm"]


def rq3_cell():
    return code_of(section_cells(11), "RQ3_MODELS")


def registry_stand_in():
    return pd.DataFrame({"model": ["bprmf", "mostpop"]})  # the cell counts the KGAT-like rows of the registry


def test_the_rq3_cell_says_not_run_while_the_files_do_not_exist(tmp_path):
    said, shown = run_cell(
        rq3_cell(), ROOT=tmp_path, REG=registry_stand_in(), Image=None
    )  # an empty folder: no results/
    text = "\n".join(said)
    assert text.count("chưa chạy") == 3 and "Số dòng KGAT/T-KGAT trong registry: **0**" in text
    assert "rq3_models.csv" in text and "rq3_pairs.csv" in text and "rq3_curves_" in text
    assert shown == []  # no table, no figure, and no exception


def test_the_rq3_cell_survives_empty_files(tmp_path):
    (tmp_path / "results").mkdir()
    for name in ("rq3_models.csv", "rq3_pairs.csv"):
        (tmp_path / "results" / name).write_text("\n", encoding="utf-8")  # what the script writes for no dataset at all
    said, shown = run_cell(rq3_cell(), ROOT=tmp_path, REG=registry_stand_in(), Image=None)
    assert "\n".join(said).count("chưa chạy") == 3 and shown == []


def test_the_rq3_cell_shows_the_tables_and_the_figure_when_the_files_exist(tmp_path):
    ipython_display = pytest.importorskip("IPython.display")
    models = pd.DataFrame(
        [
            ["ml-1m-u10", "mostpop", "MostPop", "abc", "chưa chạy"] + [None] * 7,
            ["ml-1m-u10", "kgat", "KGAT", "abc", "ok", "lr = 0.001", 5, 41, 0.12345, 0.01, 0.2, 0.02],
            ["ml-1m-u10", "tkgat", "T-KGAT", "abc", "ok", "lr = 0.001, gamma = 3", 1, 41, 0.2, None, 0.3, None],
        ],
        columns=MODELS_COLUMNS,
    )
    pairs = pd.DataFrame(
        [
            ["ml-1m-u10", "recall@20", "T-KGAT so với KGAT", "tkgat", "kgat", "ok", 1, 41, 30, 0.0766, 0.0, 0.0766]
            + [0.0123, "only 41 evaluation users (< 100)", 0.0292],
            ["ml-1m-u10", "recall@20", "KGAT so với KGAT không KG", "kgat", "kgat_nokg", "chưa chạy: KGAT không KG"]
            + [None] * 9,
        ],
        columns=PAIRS_COLUMNS,
    )
    (tmp_path / "results").mkdir()
    models.to_csv(tmp_path / "results" / "rq3_models.csv", index=False)
    pairs.to_csv(tmp_path / "results" / "rq3_pairs.csv", index=False)
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    fig.savefig(figures / "rq3_curves_ml-1m-u10.png")
    plt.close(fig)

    said, shown = run_cell(rq3_cell(), ROOT=tmp_path, REG=registry_stand_in(), Image=ipython_display.Image)
    text = "\n".join(said)
    assert "chưa chạy:" not in text and "bộ con của ML-1M, không phải ML-1M đầy đủ" in text  # the subset is labelled
    table, pair_table, image = shown[0].data, shown[1].data, shown[2]
    assert isinstance(image, ipython_display.Image) and len(shown) == 3
    by_model = table.set_index("mô hình")
    assert by_model.loc["KGAT [sơ bộ]", "Recall@20 (test)"] == "0,1235 ± 0,0100"  # mean ± sample std, Vietnamese format
    assert by_model.loc["T-KGAT [sơ bộ]", "Recall@20 (test)"] == "0,2000"  # one seed: no std
    assert by_model.loc["MostPop", "Recall@20 (test)"] == "chưa chạy" and by_model.loc["MostPop", "số seed"] == "—"
    ok, missing = pair_table.iloc[0], pair_table.iloc[1]
    assert (ok["n_pairs"], ok["n_nonzero"], ok["p Holm"]) == (41, 30, "0,029") and "41 evaluation users" in ok[
        "ghi chú"
    ]
    assert missing["ghi chú"] == "chưa chạy: KGAT không KG" and missing["p Holm"] == "—"


def test_the_rq3_cell_works_with_only_some_of_the_files(tmp_path):
    (tmp_path / "docs" / "figures").mkdir(parents=True)
    (tmp_path / "docs" / "figures" / "rq3_curves_ml-100k.png").write_bytes(b"not a real png")  # only a figure
    ipython_display = pytest.importorskip("IPython.display")
    said, shown = run_cell(rq3_cell(), ROOT=tmp_path, REG=registry_stand_in(), Image=ipython_display.Image)
    assert (
        "\n".join(said).count("chưa chạy") == 2 and len(shown) == 1
    )  # the two tables are missing, the figure is shown


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
