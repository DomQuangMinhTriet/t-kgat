"""Numbers of the data report: Vietnamese formatting, the numbers table, and the "every number is sourced" check.

``scripts/07_report_numbers.py`` fills a :class:`Numbers` table from files written by the pipeline and renders
``docs/data-report-numbers.md``. ``docs/data-report.md`` may only contain numbers that appear in that file:
:func:`unknown_numbers` finds the ones that do not (used by ``tests/test_data_report.py``).
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

MISSING = "chưa có"
CONVENTIONS = "Quy ước cố định"

# --- Vietnamese formatting: dot for thousands, comma for decimals ---------------------------------------------


def vn_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def vn_dec(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def vn_pct(share: float, digits: int = 1) -> str:
    return vn_dec(100 * share, digits) + "%"


def vn_date(d: dt.date | str) -> str:
    return dt.date.fromisoformat(str(d)[:10]).strftime("%d/%m/%Y")


def vn_pm(mean: float, std: float | None, digits: int = 4) -> str:
    """``mean ± sample std``; a single seed has no sample std."""
    return f"{vn_dec(mean, digits)} ± {vn_dec(std, digits)}" if std is not None else vn_dec(mean, digits)


# --- the numbers table ------------------------------------------------------------------------------------------


@dataclass
class Numbers:
    """Ordered (section, key, value, source) rows; keys are unique."""

    rows: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add(self, section: str, key: str, value: str, source: str) -> None:
        if any(k == key for _, k, _, _ in self.rows):
            raise ValueError(f"duplicate key {key!r}")
        if "|" in value or "|" in source or "\n" in value + source:
            raise ValueError(f"value or source of {key!r} would break the table")
        self.rows.append((section, key, str(value), source))

    def value(self, key: str) -> str:
        return next(v for _, k, v, _ in self.rows if k == key)

    def render(self) -> str:
        out = [
            "# Số liệu của báo cáo dữ liệu",
            "",
            "Sinh tự động bởi `scripts/07_report_numbers.py` từ các file do pipeline ghi ra. **Không sửa tay.**",
            "Mọi con số trong `docs/data-report.md` phải có trong file này (kiểm tra bởi `tests/test_data_report.py`).",
            f'Giá trị "{MISSING}" nghĩa là nguồn chưa tồn tại; script điền số khi nguồn xuất hiện.',
            "Định dạng: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân; phần trăm có ký hiệu %.",
        ]
        sections: dict[str, list[tuple[str, str, str]]] = {}  # in order of first appearance
        for section, key, value, source in self.rows:
            sections.setdefault(section, []).append((key, value, source))
        for section, rows in sections.items():
            out += ["", f"## {section}", "", "| Khóa | Giá trị | Nguồn |", "|---|---|---|"]
            out += [f"| `{key}` | {value} | {source} |" for key, value, source in rows]
        return "\n".join(out) + "\n"


_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


def parse_numbers(text: str) -> dict[str, tuple[str, str]]:
    """``{key: (value, source)}`` from a rendered numbers file."""
    return {m.group(1): (m.group(2), m.group(3)) for line in text.splitlines() if (m := _ROW.match(line))}


# --- manual check of the entity links (item 15) -------------------------------------------------------------------

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a proportion ``k / n`` (here: the share of wrong links in a random sample)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # the bounds are exactly 0 for k = 0 and exactly 1 for k = n; floating point leaves ~1e-17 otherwise
    return (0.0 if k == 0 else max(0.0, centre - half)), (1.0 if k == n else min(1.0, centre + half))


def sample_counts(path: str | Path) -> dict[str, int]:
    """Rows, rows checked by hand and wrong links of ``results/linking_manual_check_sample.csv``.

    The columns are found by prefix (``needs_human_check``, ``correct``), because their headers carry a note.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    human = next((c for c in rows[0] if c.startswith("needs_human_check")), None)
    correct = next((c for c in rows[0] if c.startswith("correct")), None)
    if human is None or correct is None:
        raise ValueError(f"{path} needs columns starting with 'needs_human_check' and 'correct'")
    return {
        "n": len(rows),
        "n_human": sum(r[human].strip() == "1" for r in rows),
        "n_wrong": sum(r[correct].strip() == "0" for r in rows),
    }


# --- finding numbers in prose ---------------------------------------------------------------------------------

_DATE = r"\d{2}/\d{2}/\d{4}"
_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?"
TOKEN = re.compile(rf"({_DATE})|(?:{_NUM})%?")

# Text whose digits are names or references, not data: product names, metric names, cut-off names, figure and
# section references, decision ids, issue and PR numbers.
_IGNORE = re.compile(
    r"ML-100K|ML-1M|MovieLens-100K|MovieLens-1M|T-KGAT|KGAT|Recall@\d+|NDCG@\d+|\bt[12]\b|\bRQ\d\b|\b\d+-core\b"
    r"|ddof=\d|\b[Hh]ình \d+[a-z]?|\b[Bb]ảng \d+|\b[Mm]ục \d+(?:\.\d+)*[a-z]?|§\s*\d+(?:\.\d+)*"
    r"|\b[Qq]uyết định \d{4}(?:\s*(?:/|và|,)\s*\d{4})*|\b[Vv]iệc \d+|#\d+|\bSHA-?\d+\b|\bP\d{3,4}\b"
)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)|https?://\S+")
_HEADING_NUM = re.compile(r"^(#+\s*)\d+(?:\.\d+)*\.?\s*")
_LIST_NUM = re.compile(r"^(\s*)\d+\.\s+")


def _strip(text: str) -> list[str]:
    text = _CODE.sub(" ", _LINK.sub(" ", text))
    lines = []
    for line in text.splitlines():
        if set(line.strip()) <= set("|-: "):  # table rules
            continue
        line = _HEADING_NUM.sub(r"\1", line) if line.lstrip().startswith("#") else _LIST_NUM.sub(r"\1", line)
        lines.append(_IGNORE.sub(" ", line))
    return lines


def tokens(text: str) -> list[tuple[str, str]]:
    """``(token, line)`` for every number in ``text`` after dropping names and references."""
    return [(m.group(0), line.strip()) for line in _strip(text) for m in TOKEN.finditer(line)]


def _is_small_count(token: str) -> bool:
    """Bare integers 0-10 are counting words ("3 biến thể", "hai ...") and are not checked."""
    return token.isdigit() and int(token) <= 10


def accepted_tokens(numbers_text: str) -> set[str]:
    """Every number token in the value column of the numbers file, plus the years of its dates."""
    accepted: set[str] = set()
    for value, _ in parse_numbers(numbers_text).values():
        for tok, _ in tokens(value):
            accepted.add(tok)
            if re.fullmatch(_DATE, tok):
                accepted.add(tok[-4:])
    return accepted


def unknown_numbers(report_text: str, numbers_text: str) -> list[tuple[str, str]]:
    """Numbers of ``report_text`` that are not in ``numbers_text`` as ``(token, line)``; empty = all sourced."""
    accepted = accepted_tokens(numbers_text)
    return [(t, line) for t, line in tokens(report_text) if t not in accepted and not _is_small_count(t)]
