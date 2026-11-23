"""How many epochs do the runs already in the repository need? (decision 0016 item 6; advisor 2026-11-15)

    python scripts/13_epoch_budget.py
    python scripts/13_epoch_budget.py --datasets lastfm-u25 --out-md /tmp/epoch-budget.md

Reads only the per-epoch history files ``results/runs/<run_name>/*_history.csv`` (validation curves written during
training; no test score is in them) and writes ``docs/epoch-budget.md`` and ``results/epoch_budget.csv``. For every run
(one file) it records the best epoch, the last epoch, whether the run stopped on its own (early stopping) or was cut at
``max_epochs`` (a *censored* run: its best epoch is a lower bound, not where validation would have peaked), the
seconds spent, and how much of its best validation Recall@20 it had reached by epoch 25, 50, 100, 150 and 200. The
tables group runs by dataset, model and learning rate, because a run at lr 1e-4 and a run at lr 1e-3 need different
numbers of epochs. Nothing is chosen here: the output is the evidence for ``max_epochs`` and ``patience`` of 0016.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = (25, 50, 100, 150, 200)
METRIC = "recall@20"
SMALL_SETS = ("lastfm-w3", "lastfm-w2")  # chosen-by-validation sets of decision 0016 item 9; see 15_small_set_gate.py


def _config(raw: str) -> dict:
    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def read_run(path: Path) -> dict | None:
    """One history file -> one record, or None for a file without a validation score."""
    df = pd.read_csv(path)
    if df.empty or METRIC not in df.columns:
        return None
    val = df.dropna(subset=[METRIC])
    if val.empty:
        return None
    first = df.iloc[0]
    cfg = _config(first.get("config"))
    fit = cfg.get("fit", cfg)  # KGAT-type configs nest the trainer settings under "fit", BPR-MF keeps them at the top
    best_by_value = int(val.loc[val[METRIC].idxmax(), "epoch"])
    last = int(df["epoch"].max())
    max_epochs = int(fit.get("max_epochs", 0)) or None
    best = float(val[METRIC].max())
    record = {
        "dataset": first["dataset"],
        "model": first["model"],
        "grid_index": int(first["grid_index"]),
        "seed": int(first["seed"]),
        "lr": fit.get("lr"),
        "max_epochs": max_epochs,
        "patience": fit.get("patience"),
        "eval_every": fit.get("eval_every"),
        "best_epoch": int(first["best_epoch"]),
        "best_epoch_from_curve": best_by_value,
        "last_epoch": last,
        "censored": bool(max_epochs is not None and last >= max_epochs),
        "seconds": float(df["seconds"].sum()) if "seconds" in df.columns else np.nan,
        f"best_{METRIC}": best,
        "file": path.name,
    }
    for epoch in CHECKPOINTS:
        upto = val[val["epoch"] <= epoch]
        record[f"share_at_{epoch}"] = float(upto[METRIC].max() / best) if len(upto) and best > 0 else np.nan
    return record


def collect(runs_dir: Path, datasets: list[str] | None = None) -> pd.DataFrame:
    records = []
    for path in sorted(Path(runs_dir).glob("*/*_history.csv")):
        if (datasets and path.parent.name not in datasets) or (not datasets and path.parent.name in SMALL_SETS):
            continue
        record = read_run(path)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, lr): how many runs, where they peak, how many were cut, how fast they get there."""
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, model, lr), g in runs.groupby(["dataset", "model", "lr"], dropna=False, sort=True):
        free = g[~g["censored"]]
        row = {
            "dataset": dataset,
            "model": model,
            "lr": lr,
            "runs": len(g),
            "best_epoch_median": float(g["best_epoch"].median()),
            "best_epoch_p90": float(g["best_epoch"].quantile(0.9)),
            "best_epoch_max": int(g["best_epoch"].max()),
            "stopped_early": int((~g["censored"]).sum()),
            "cut_at_max": int(g["censored"].sum()),
            "hours": float(g["seconds"].sum() / 3600),
        }
        row["best_epoch_median_early_only"] = float(free["best_epoch"].median()) if len(free) else np.nan
        for epoch in CHECKPOINTS:
            row[f"share_at_{epoch}"] = float(g[f"share_at_{epoch}"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _n(x, digits: int = 0) -> str:
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


def markdown(runs: pd.DataFrame, summary: pd.DataFrame) -> str:
    out = [
        "# Ngân sách epoch từ các run đã có (quyết định 0016, mục 6)",
        "",
        "Nguồn: `results/runs/*/*_history.csv` (đường cong validation, không có điểm test), sinh bởi `scripts/13_epoch_budget.py`.",  # noqa: E501
        "Không có giá trị nào được chọn ở đây; file này là bằng chứng để đặt `max_epochs` và `patience` trong 0016.",
        "",
        "Cách đọc: `best_epoch` là epoch có Recall@20 validation cao nhất của một run (đánh giá mỗi `eval_every` epoch). Một run "  # noqa: E501
        "**bị cắt** (`cut_at_max`) chạm `max_epochs` mà chưa dừng sớm; best epoch của nó chỉ là cận dưới. `share@E` là "
        "trung vị, qua các run của nhóm, của tỷ lệ (Recall@20 validation tốt nhất trong E epoch đầu) / (tốt nhất cả run): "  # noqa: E501
        "0,95 nghĩa là chạy tới E epoch đã đạt 95% mức tốt nhất.",
        "",
    ]
    if runs.empty:
        return "\n".join(out + ["Không tìm thấy file history nào."]) + "\n"
    for dataset, g in summary.groupby("dataset", sort=True):
        out += [f"## {dataset}", ""]
        out += [
            "| model | lr | run | best_epoch trung vị | p90 | max | dừng sớm | bị cắt | share@25 | @50 | @100 | @150 | @200 | giờ |",  # noqa: E501
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in g.iterrows():
            lr_text = "–" if pd.isna(r["lr"]) else f"{r['lr']:g}"
            out.append(
                f"| {r['model']} | {lr_text} | {r['runs']} | "
                f"{_n(r['best_epoch_median'])} | {_n(r['best_epoch_p90'])} | {r['best_epoch_max']} | {r['stopped_early']} | "  # noqa: E501
                f"{r['cut_at_max']} | {_n(r['share_at_25'], 3)} | {_n(r['share_at_50'], 3)} | {_n(r['share_at_100'], 3)} | "  # noqa: E501
                f"{_n(r['share_at_150'], 3)} | {_n(r['share_at_200'], 3)} | {_n(r['hours'], 1)} |"
            )
        out.append("")
    mism = runs[runs["best_epoch"] != runs["best_epoch_from_curve"]]
    out += [
        "## Kiểm tra nội bộ",
        "",
        f"- Số run đọc được: {len(runs)}; số run bị cắt ở `max_epochs`: {int(runs['censored'].sum())}.",
        f"- Run có cột `best_epoch` khác epoch cực đại của chính đường cong: {len(mism)}"
        + (
            "" if mism.empty else " (" + ", ".join(sorted(mism["file"].head(5))) + ("…" if len(mism) > 5 else "") + ")."
        ),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="run_name of the data configs; default: all found but the small sets",
    )
    ap.add_argument("--out-md", default="docs/epoch-budget.md")
    ap.add_argument("--out-csv", default="results/epoch_budget.csv")
    args = ap.parse_args(argv)

    runs = collect(Path(args.runs_dir), args.datasets)
    summary = summarise(runs)
    Path(args.out_md).write_text(markdown(runs, summary), encoding="utf-8", newline="\n")
    if not runs.empty:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        runs.drop(columns=["file"]).to_csv(args.out_csv, index=False)
    print(f"{len(runs)} runs -> {args.out_md}")


if __name__ == "__main__":
    main()
