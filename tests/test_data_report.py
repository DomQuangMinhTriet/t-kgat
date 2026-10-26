"""The data report only contains numbers that come from files written by the pipeline.

Three layers:
- the checker itself, on synthetic text (always runs);
- ``docs/data-report.md`` against the committed ``docs/data-report-numbers.md`` (needs no data, runs in CI once the
  report exists);
- the committed numbers file against a fresh run of ``scripts/07_report_numbers.py`` (skipped without the data).
"""

import importlib.util

import pytest

from tkgat.analysis import report
from tkgat.analysis.report import Numbers, unknown_numbers

from .conftest import ROOT

REPORT = ROOT / "docs" / "data-report.md"
NUMBERS = ROOT / "docs" / "data-report-numbers.md"


def numbers_file() -> str:
    n = Numbers()
    n.add("Mục A", "ml1m.ratings", report.vn_int(1000209), "data/raw/ml-1m/ratings.dat")
    n.add("Mục A", "ml1m.density", report.vn_pct(0.0447), "tính từ ratings")
    n.add("Mục A", "ml1m.t1", report.vn_date("2000-12-02"), "meta.json")
    n.add("Mục A", "bprmf.test", report.vn_pm(0.119387, 0.003081), "registry")
    n.add("Mục A", "ml100k.baseline", report.MISSING, "registry")
    return n.render()


def test_formatting_uses_vietnamese_separators():
    assert report.vn_int(1000209) == "1.000.209" and report.vn_int(932) == "932"
    assert report.vn_dec(0.1336, 4) == "0,1336" and report.vn_pct(0.7243) == "72,4%"
    assert report.vn_date("2000-12-02") == "02/12/2000"
    assert report.vn_pm(0.1194, None) == "0,1194"  # one seed: no sample std


def test_render_and_parse_round_trip():
    parsed = report.parse_numbers(numbers_file())
    assert parsed["ml1m.ratings"] == ("1.000.209", "data/raw/ml-1m/ratings.dat")
    assert parsed["bprmf.test"][0] == "0,1194 ± 0,0031" and parsed["ml100k.baseline"][0] == report.MISSING
    with pytest.raises(ValueError, match="duplicate"):
        n = Numbers()
        n.add("S", "k", "1", "s")
        n.add("S", "k", "2", "s")
    with pytest.raises(ValueError, match="break the table"):
        Numbers().add("S", "k", "a | b", "s")


def test_sourced_numbers_pass():
    text = (
        "# Báo cáo\n\n## 1. Bộ dữ liệu\n\n"
        "ML-1M có 1.000.209 rating, mật độ 4,5%. Mốc t1 là 02/12/2000 (năm 2000).\n"
        "Theo quyết định 0004 và 0006, xem Hình 2, mục 3.1 và `ddof=1`; BPR-MF đạt 0,1194 ± 0,0031.\n"
        "1. Có 3 biến thể và 5-core.\n"
    )
    assert unknown_numbers(text, numbers_file()) == []


def test_unsourced_numbers_are_reported():
    text = "ML-1M có 1.000.210 rating, mật độ 4,4%, mốc 03/12/2000, BPR-MF đạt 0,1195 và 11 phim.\n"
    found = [t for t, _ in unknown_numbers(text, numbers_file())]
    assert found == ["1.000.210", "4,4%", "03/12/2000", "0,1195", "11"]


def test_percent_sign_and_decimals_must_match_exactly():
    found = [t for t, _ in unknown_numbers("mật độ 4,5 và 4,50% và 1000209 rating", numbers_file())]
    assert found == ["4,5", "4,50%", "1000209"]  # a bare 4,5 is not 4,5%; no thousands dots is another token


def test_names_references_and_code_are_not_numbers():
    text = (
        "## 4. Chia theo mốc\n\nXem Hình 3a, Bảng 2, §12, #10, PR #16, RQ3, `scripts/07_report_numbers.py`,\n"
        "[link 2026](docs/figures/fig1.png) và Recall@20, NDCG@10, t1, t2, 10-core, Việc 20.\n"
    )
    assert unknown_numbers(text, numbers_file()) == []


def test_years_of_known_dates_are_accepted_but_other_years_are_not():
    assert unknown_numbers("Dữ liệu năm 2000.", numbers_file()) == []
    assert [t for t, _ in unknown_numbers("Dữ liệu năm 2003.", numbers_file())] == ["2003"]


def test_missing_values_contribute_no_numbers():
    assert report.accepted_tokens(numbers_file()).isdisjoint({"chưa", "có"})


@pytest.mark.skipif(not (REPORT.exists() and NUMBERS.exists()), reason="report or numbers file not written yet")
def test_every_number_in_the_report_is_in_the_numbers_file():
    bad = unknown_numbers(REPORT.read_text(encoding="utf-8"), NUMBERS.read_text(encoding="utf-8"))
    assert not bad, "numbers not in docs/data-report-numbers.md:\n" + "\n".join(f"{t!r} in: {line}" for t, line in bad)


def load_script():
    spec = importlib.util.spec_from_file_location("report_numbers", ROOT / "scripts" / "07_report_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQUIRED = [
    ROOT / "data" / "raw" / "ml-100k",
    ROOT / "data" / "raw" / "ml-1m",
    ROOT / "data" / "raw" / "wikidata" / "facts.tsv",
    ROOT / "data" / "processed" / "ml-100k" / "meta.json",
    ROOT / "data" / "processed" / "ml-1m" / "meta.json",
    ROOT / "data" / "processed" / "ml-1m" / "kg" / "static" / "kg_meta.json",
    ROOT / "data" / "processed" / "links" / "linking_report.json",
]


@pytest.mark.skipif(
    not NUMBERS.exists() or not all(p.exists() for p in REQUIRED), reason="needs the data and a committed numbers file"
)
def test_committed_numbers_file_matches_a_fresh_run(tmp_path):
    out = tmp_path / "numbers.md"
    load_script().main(["--out", str(out), "--figures-dir", "none"])
    fresh = out.read_text(encoding="utf-8")
    committed = NUMBERS.read_text(encoding="utf-8")
    msg = "docs/data-report-numbers.md is out of date: run `python scripts/07_report_numbers.py` and commit it"
    assert fresh == committed, msg
    assert report.parse_numbers(fresh)  # the file parses back into keys
