"""Sensitivity of the baseline ORDER to the split (``scripts/10_sensitivity.py``), from validation only.

Every number comes from ``results/registry.csv`` (phase ``val`` rows of the grid point selected on validation) and
from the split's ``meta.json``. Test files and test rows are never read. Validation sets differ between splits,
so their absolute scores are not comparable: the table compares the order MostPop / MostPop recent / BPR-MF
inside each split, and the splits are never ranked by their val scores.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.analysis.report import vn_dec, vn_int

MODELS = ("mostpop", "mostpop_recent", "bprmf")
NAMES = {"mostpop": "MostPop", "mostpop_recent": "MostPop recent", "bprmf": "BPR-MF"}
WEAK_VAL_USERS = 50  # below this many val users both "same order" and "different order" are weak evidence


def selected_point(run_dir: Path, model: str) -> dict:
    """Grid point selected on validation: ``<model>_val_summary.json`` (``--val-only``) or ``<model>_summary.json``."""
    for name in (f"{model}_val_summary.json", f"{model}_summary.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["selected"]
    raise FileNotFoundError(f"no summary of {model} in {run_dir}")


def val_scores(reg: pd.DataFrame, run_name: str, model: str, point: dict, split_sha: str) -> pd.DataFrame:
    rows = reg[(reg["dataset"] == run_name) & (reg["model"] == model) & (reg["phase"] == "val")]
    rows = rows[rows["config"].map(lambda c: json.loads(c) == point)]
    # the split hash of the registry rows is the first 12 characters of the sha256 of meta.json
    rows = rows[rows["split_sha256"] == split_sha[:12]]
    if rows.empty:
        raise ValueError(f"{run_name}: no val rows of {model} at {point} for this split")
    return rows


def split_info(meta: dict) -> dict:
    day = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    cfg = meta["config"]
    return {
        "kcore": cfg["kcore"],
        "q_val": cfg["q_val"],
        "q_test": cfg["q_test"],
        "t1": day(meta["t1"]),
        "t2": day(meta["t2"]),
        "val_users": meta["val_users"],
        "val_pos": meta["val_pos"],
    }


def build_table(configs: list[tuple[str, Path, str]], reg: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """One row per (run_name, model): ``configs`` holds (run_name, processed_dir, meta.json sha256)."""
    rows = []
    for run_name, processed, sha in configs:
        meta = json.loads((Path(processed) / "meta.json").read_text(encoding="utf-8"))
        info = split_info(meta)
        for model in MODELS:
            point = selected_point(Path(runs_dir) / run_name, model)
            v = val_scores(reg, run_name, model, point, sha)["recall@20"]
            rows.append(
                {
                    "run_name": run_name,
                    **info,
                    "model": model,
                    "selected": json.dumps(point, sort_keys=True),
                    "n_seeds": len(v),
                    "val_recall20_mean": float(v.mean()),
                    "val_recall20_std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def order_of(group: pd.DataFrame) -> list[str]:
    """Models from the highest to the lowest mean val Recall@20."""
    return group.sort_values("val_recall20_mean", ascending=False)["model"].tolist()


def close_pairs(group: pd.DataFrame) -> list[tuple[str, str]]:
    """Adjacent models in the order whose gap is below the std over seeds of BPR-MF (the only stochastic one)."""
    order = order_of(group)
    g = group.set_index("model")
    std = g.loc["bprmf", "val_recall20_std"]
    out = []
    for a, b in zip(order, order[1:], strict=False):
        gap = g.loc[a, "val_recall20_mean"] - g.loc[b, "val_recall20_mean"]
        if "bprmf" in (a, b) and not np.isnan(std) and gap < std:
            out.append((a, b))
    return out


def vn_q(q: float) -> str:
    """A cut-off quantile with the fewest decimals: 0,7 / 0,85 / 0,925."""
    return vn_dec(q, 3).rstrip("0")


def close_text(group: pd.DataFrame) -> str:
    close = close_pairs(group)
    if not close:
        return ""
    pairs = ", ".join(f"{NAMES[a]} và {NAMES[b]}" for a, b in close)
    return f" Cặp có chênh lệch nhỏ hơn std của BPR-MF qua seed: {pairs}."


def fmt_order(order: list[str]) -> str:
    return " > ".join(NAMES[m] for m in order)


def markdown(table: pd.DataFrame, base: str) -> str:
    runs = list(dict.fromkeys(table["run_name"]))
    orders = {r: order_of(table[table["run_name"] == r]) for r in runs}
    lines = [
        "# Độ nhạy của thứ tự baseline theo cách chia (chỉ val; tự sinh bởi `scripts/10_sensitivity.py`)",
        "",
        "Nguồn: `results/registry.csv` (dòng val của điểm lưới được chọn trên val) và `meta.json` của từng split. "
        "Không đọc dòng test hay file test nào. Recall@20 trên val, trung bình ± std mẫu (`ddof=1`) qua seed; "
        "MostPop và MostPop recent là tất định nên chỉ có một giá trị.",
        "",
        "| Cấu hình (run_name) | k-core | q_val / q_test | t1 / t2 (UTC) | user val | positive val "
        "| MostPop | MostPop recent | BPR-MF | Thứ tự (cao đến thấp) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        g = table[table["run_name"] == r]
        first = g.iloc[0]
        cells = []
        for m in MODELS:
            row = g[g["model"] == m].iloc[0]
            std = row["val_recall20_std"]
            cells.append(vn_dec(row["val_recall20_mean"], 4) + ("" if np.isnan(std) else f" ± {vn_dec(std, 4)}"))
        lines.append(
            f"| {r} | {first['kcore']} | {vn_q(first['q_val'])} / {vn_q(first['q_test'])} "
            f"| {first['t1']} / {first['t2']} | {vn_int(first['val_users'])} | {vn_int(first['val_pos'])} "
            f"| {' | '.join(cells)} | {fmt_order(orders[r])} |"
        )
    lines += [
        "",
        "**Cách đọc.** Mỗi split có tập val khác nhau (khác user, khác positive), nên điểm val tuyệt đối "
        "**không so sánh được giữa các cấu hình** và không được dùng để xếp hạng hay chọn cấu hình. "
        "Chỉ so thứ tự ba mô hình trong từng dòng.",
        "",
        "**Nhận xét (tự sinh).**",
        "",
    ]
    for r in runs:
        g = table[table["run_name"] == r]
        users = int(g["val_users"].iloc[0])
        if r == base:
            lines.append(f"- `{r}` là cấu hình gốc, thứ tự: {fmt_order(orders[r])}.{close_text(g)}")
            continue
        same = orders[r] == orders[base]
        text = f"- `{r}`: thứ tự {'giữ nguyên' if same else 'đổi'} so với `{base}` ({fmt_order(orders[r])})."
        text += close_text(g)
        if users < WEAK_VAL_USERS:
            text += (
                f' Val chỉ có {vn_int(users)} user (< {WEAK_VAL_USERS}), nên "giữ nguyên" hay "đổi" '
                "đều là bằng chứng yếu."
            )
        lines.append(text)
    base_users = int(table[table["run_name"] == base]["val_users"].iloc[0])
    lines += [
        f"- Cấu hình gốc cũng chỉ có {vn_int(base_users)} user val: chọn điểm lưới trên val còn nhiễu "
        "(xem `docs/study/` và CLAUDE.md).",
        "",
    ]
    return "\n".join(lines)
