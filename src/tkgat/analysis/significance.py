"""Significance tests, learning curves and the training table of the BPR-MF baseline (``scripts/09_significance.py``).

Everything is read from files the pipeline wrote, never typed by hand:

  results/registry.csv (through ``registry.load()``)   which seeds exist, split hash
  results/runs/<run_name>/<model>_summary.json         the grid point selected on validation
  results/runs/<run_name>/<model>_seed<seed>.npz       per-user test metrics of the SELECTED grid point only
  results/runs/<run_name>/bprmf_<grid>_seed<seed>_history.csv   per-epoch loss and validation metrics

Test scores of grid points that were not selected do not exist on disk, so they cannot be compared here.
The test files are used only to compare the already-selected models once (decision 0001, rule 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.eval.stats import holm, load_per_user, paired_wilcoxon

BASELINES = ("mostpop", "mostpop_recent")
METRICS = ("recall@20", "ndcg@20")
MIN_NONZERO = 30  # below this many ranked differences a Wilcoxon p-value says little
MIN_PAIRS = 100  # below this many evaluation users the whole comparison is noisy


def selected_point(summary: dict, grid: list[dict]) -> int:
    """Index in ``grid`` of the grid point stored in ``summary["selected"]``."""
    want = summary["selected"]
    for i, point in enumerate(grid):
        if all(point[k] == want[k] for k in want) and set(point) == set(want):
            return i
    raise ValueError(f"selected grid point {want} is not in the grid of the model config")


def warning(n_pairs: int, n_nonzero: int) -> str:
    notes = []
    if n_pairs < MIN_PAIRS:
        notes.append(f"only {n_pairs} evaluation users (< {MIN_PAIRS})")
    if n_nonzero < MIN_NONZERO:
        notes.append(f"only {n_nonzero} non-zero differences (< {MIN_NONZERO})")
    return "; ".join(notes)


def compare_dataset(dataset: str, reg: pd.DataFrame, runs_dir: Path) -> list[dict]:
    """BPR-MF against each baseline, and MostPop against MostPop recent, on the per-user test metrics.

    BPR-MF has one score per user and seed, the baselines are deterministic. The test is run on the
    per-user mean over the BPR-MF seeds (one pair per user); the p-values of the single seeds are kept as
    ``p_raw_seed_min`` / ``p_raw_seed_max`` because averaging seeds hides seed noise (NaN for the pair of
    baselines, which has no seeds). Holm is applied separately for each metric over the three comparisons of this
    dataset.
    """
    test = reg[(reg["dataset"] == dataset) & (reg["phase"] == "test")]
    seeds = sorted(int(s) for s in test.loc[test["model"] == "bprmf", "seed"])
    if not seeds:
        raise ValueError(f"{dataset}: no BPR-MF test rows in the registry")
    shas = set(test.loc[test["model"].isin(("bprmf", *BASELINES)), "split_sha256"])
    if len(shas) != 1:
        raise ValueError(f"{dataset}: test rows come from {len(shas)} different splits, cannot be paired")
    run = Path(runs_dir) / dataset
    rows: list[dict] = []
    for metric in METRICS:
        block = []
        for base in BASELINES:
            base_file = run / f"{base}_seed0.npz"
            per_seed, base_scores, bpr_scores = [], None, []
            for seed in seeds:
                a, b = load_per_user(run / f"bprmf_seed{seed}.npz", base_file, metric)
                bpr_scores.append(a)
                base_scores = b
                per_seed.append(paired_wilcoxon(a, b)["p_value"])
            mean_bpr = np.mean(bpr_scores, axis=0)
            res = paired_wilcoxon(mean_bpr, base_scores)
            block.append(
                {
                    "dataset": dataset,
                    "split_sha256": next(iter(shas)),
                    "metric": metric,
                    "model_a": "bprmf",
                    "model_b": base,
                    "n_seeds_a": len(seeds),
                    "n_pairs": res["n_pairs"],
                    "n_nonzero": res["n_nonzero"],
                    "mean_a": float(mean_bpr.mean()),
                    "mean_b": float(base_scores.mean()),
                    "mean_diff": res["mean_diff"],
                    "p_raw": res["p_value"],
                    "p_raw_seed_min": float(min(per_seed)),
                    "p_raw_seed_max": float(max(per_seed)),
                }
            )
        # the two deterministic baselines against each other: one pair per user, no seeds
        a, b = load_per_user(run / "mostpop_seed0.npz", run / "mostpop_recent_seed0.npz", metric)
        res = paired_wilcoxon(a, b)
        block.append(
            {
                "dataset": dataset,
                "split_sha256": next(iter(shas)),
                "metric": metric,
                "model_a": "mostpop",
                "model_b": "mostpop_recent",
                "n_seeds_a": 1,
                "n_pairs": res["n_pairs"],
                "n_nonzero": res["n_nonzero"],
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_diff": res["mean_diff"],
                "p_raw": res["p_value"],
                "p_raw_seed_min": float("nan"),
                "p_raw_seed_max": float("nan"),
            }
        )
        corrected = holm([r["p_raw"] for r in block])
        for r, p_adj, rej in zip(block, corrected["p_adjusted"], corrected["reject"], strict=True):
            r["p_holm"], r["reject_holm_05"] = p_adj, rej
            r["warning"] = warning(r["n_pairs"], r["n_nonzero"])
        rows.extend(block)
    return rows


# --- learning curves ---------------------------------------------------------------------------------------------


@dataclass
class Curve:
    dataset: str
    grid_index: int
    point: dict
    table: pd.DataFrame  # epoch, n_seeds, loss_mean, loss_std, val_recall20_n, val_recall20_mean, val_recall20_std
    best_epochs: dict[int, int]  # seed -> best epoch
    last_epochs: dict[int, int]  # seed -> last trained epoch


def load_curve(dataset: str, runs_dir: Path, model_cfg: dict, seeds: list[int]) -> Curve:
    run = Path(runs_dir) / dataset
    summary = json.loads((run / "bprmf_summary.json").read_text(encoding="utf-8"))
    idx = selected_point(summary, model_cfg["grid"])
    frames, best, last = [], {}, {}
    for seed in seeds:
        h = pd.read_csv(run / f"bprmf_{idx}_seed{seed}_history.csv")
        cfg = json.loads(h["config"].iloc[0])
        for k, v in {**model_cfg["fixed"], **model_cfg["grid"][idx]}.items():
            if cfg.get(k) != v:
                raise ValueError(f"{dataset} seed {seed}: history was trained with {k}={cfg.get(k)}, config says {v}")
        frames.append(h[["epoch", "loss", "recall@20"]].assign(seed=seed))
        best[seed], last[seed] = int(h["best_epoch"].iloc[0]), int(h["epoch"].max())
    long = pd.concat(frames)
    g = long.groupby("epoch")
    table = pd.DataFrame(
        {
            "n_seeds": g["loss"].count(),
            "loss_mean": g["loss"].mean(),
            "loss_std": g["loss"].std(ddof=1),  # sample std over seeds, NaN where one seed is left
            "val_recall20_n": g["recall@20"].count(),
            "val_recall20_mean": g["recall@20"].mean(),
            "val_recall20_std": g["recall@20"].std(ddof=1),
        }
    ).reset_index()
    return Curve(dataset, idx, model_cfg["grid"][idx], table, best, last)


def draw_curve(curve: Curve, path: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = curve.table
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ink, accent, muted = "#1f2933", "#2b6cb0", "#9aa5b1"
    panels = (
        (axes[0], t, "loss_mean", "loss_std", "n_seeds", "loss huấn luyện BPR", "loss"),
        (
            axes[1],
            t[t["val_recall20_n"] > 0],
            "val_recall20_mean",
            "val_recall20_std",
            "val_recall20_n",
            "Recall@20 trên val",
            "Recall@20",
        ),
    )
    n_all = len(curve.best_epochs)
    for ax, d, mean, std, n, title, ylab in panels:
        # Seeds stop at different epochs (early stopping). After the first stop the mean is over fewer seeds, so
        # it is drawn dashed and without a band: it would otherwise read as a trend (survivors only).
        full, rest = d[d[n] == n_all], d[d[n] < n_all]
        ax.plot(full["epoch"], full[mean], color=accent, lw=1.8, label=f"trung bình {n_all} seed")
        ax.fill_between(
            full["epoch"], full[mean] - full[std], full[mean] + full[std], color=accent, alpha=0.2, lw=0, label="± std"
        )
        if len(rest):
            tail = pd.concat([full.tail(1), rest])
            ax.plot(tail["epoch"], tail[mean], color=muted, lw=1.4, ls="--", label=f"ít hơn {n_all} seed còn chạy")
        for ep in curve.best_epochs.values():
            ax.axvline(ep, color=muted, lw=0.8, ls=":")
        ax.set_title(f"{label}: {title}", fontsize=9, color=ink)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#e4e7eb", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([], [], color=muted, ls=":", lw=0.8, label="epoch tốt nhất của từng seed")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- training table ----------------------------------------------------------------------------------------------


def training_table_markdown(curves: list[Curve], model_cfg: dict, labels: dict[str, str]) -> str:
    fixed = model_cfg["fixed"]
    head = "| Tham số | " + " | ".join(labels.get(c.dataset, c.dataset) for c in curves) + " |"
    rule = "|---|" + "---|" * len(curves)

    def row(name: str, cells: list[str]) -> str:
        return f"| {name} | " + " | ".join(cells) + " |"

    def span(values: dict[int, int]) -> str:
        v = list(values.values())
        return f"{min(v)} đến {max(v)}"

    lines = [
        "# Tham số huấn luyện BPR-MF (tự sinh bởi `scripts/09_significance.py`)",
        "",
        "Đọc từ `configs/model/bprmf.yaml` và các file history trong `results/runs/`; không sửa tay.",
        "",
        head,
        rule,
        row("Batch size", [str(fixed["batch_size"])] * len(curves)),
        row("Số epoch tối đa", [str(fixed["max_epochs"])] * len(curves)),
        row("Đánh giá val mỗi (epoch)", [str(fixed["eval_every"])] * len(curves)),
        row("Patience (số lần đánh giá)", [str(fixed["patience"])] * len(curves)),
        row("Chỉ số chọn", [str(fixed["select_metric"])] * len(curves)),
        row("Learning rate", [str(c.point["lr"]) for c in curves]),
        row("L2", [str(c.point["l2"]) for c in curves]),
        row("Dim", [str(c.point["dim"]) for c in curves]),
        row("Điểm lưới được chọn trên val", [f"{c.grid_index + 1} / {len(model_cfg['grid'])}" for c in curves]),
        row("Seed", [", ".join(str(s) for s in c.best_epochs) for c in curves]),
        row("Epoch đã chạy (nhỏ nhất đến lớn nhất qua seed)", [span(c.last_epochs) for c in curves]),
        row("Epoch tốt nhất (nhỏ nhất đến lớn nhất qua seed)", [span(c.best_epochs) for c in curves]),
        "",
    ]
    return "\n".join(lines)


def seeds_of(reg: pd.DataFrame, dataset: str) -> list[int]:
    test = reg[(reg["dataset"] == dataset) & (reg["phase"] == "test") & (reg["model"] == "bprmf")]
    return sorted(int(s) for s in test["seed"])


def datasets_with_all_models(reg: pd.DataFrame) -> list[str]:
    test = reg[reg["phase"] == "test"]
    needed = {"bprmf", *BASELINES}
    return sorted(d for d, g in test.groupby("dataset") if needed <= set(g["model"]))
