"""The RQ3 table (``scripts/10_rq3_table.py``): baselines, KGAT, KGAT without KG, T-KGAT and its permutation control.

Every number comes from ``results/registry.csv`` (rows of ``phase == test`` whose commit is not ``-dirty``) and, for the
paired tests, from the per-user test files that ``scripts/run.py`` wrote
(``results/runs/<run_name>/<tag>_seed<seed>.npz``); nothing is typed by hand. The paired tests use ``tkgat.eval.stats``.

Which rows count, for one dataset:

* ``-dirty`` rows are dropped first, then the newest row of each registry key is kept (``registry.newest``), so a dirty
  re-run never hides a clean older run.
* Only the split of the newest row is used (a rebuilt split has another ``split_sha256``); rows of an older split are
  not mixed.
* For each model, the grid point of its newest row, with all the seeds of that point. ``perm_seed`` is not part of the
  point (it is the run seed, so it differs from row to row of the control).

The Holm family is the three planned comparisons of a dataset and metric. A comparison that cannot be computed counts as
p = 1, so the adjusted p-values of the others do not change with what has been run so far.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat import registry
from tkgat.analysis.report import vn_dec, vn_p, vn_pm
from tkgat.analysis.significance import warning
from tkgat.eval.stats import holm, load_per_user, paired_wilcoxon

MODELS = (
    ("mostpop", "MostPop"),
    ("mostpop_recent", "MostPop recent"),
    ("bprmf", "BPR-MF"),
    ("kgat", "KGAT"),
    ("kgat_nokg", "KGAT không KG"),
    ("tkgat", "T-KGAT"),
    ("tkgat_perm", "T-KGAT (đối chứng hoán vị)"),
    ("tkgat_kg", "T-KGAT giai đoạn 2 (Δt trên KG)"),
    ("tkgat_kg_perm", "T-KGAT giai đoạn 2 (đối chứng hoán vị ngày fact)"),
)
NAME = dict(MODELS)
PRELIMINARY = ("kgat", "kgat_nokg", "tkgat", "tkgat_perm", "tkgat_kg", "tkgat_kg_perm")
PAIRS = (
    ("tkgat", "kgat", "T-KGAT so với KGAT"),
    ("tkgat", "tkgat_perm", "T-KGAT thật so với hoán vị"),
    ("kgat", "kgat_nokg", "KGAT so với KGAT không KG"),
)
# Stage 2 (decision 0009 items 7-10) is its own Holm family; it appears only for a dataset where its models have run
OPTIONAL_MODELS = ("tkgat_kg", "tkgat_kg_perm")  # Stage 2: a row only where the model has run
PAIRS_STAGE2 = (
    ("tkgat_kg", "kgat", "T-KGAT giai đoạn 2 so với KGAT"),
    ("tkgat_kg", "tkgat_kg_perm", "T-KGAT giai đoạn 2 thật so với hoán vị"),
)
METRICS = ("recall@20", "ndcg@20")
METRIC_NAME = {"recall@20": "Recall@20", "ndcg@20": "NDCG@20"}
DEFAULT_DATASETS = ("ml-100k", "ml-1m-u10")  # where decision 0009 runs the official KGAT and T-KGAT grids
FREEZE_TAG = "v0.2-kgat"
NOT_RUN = "chưa chạy"
EXTRA_KEYS = ("t_ref", "dt_den", "perm_seed")  # recorded by T-KGAT rows, not part of the grid point shown
BASELINES = ("mostpop", "mostpop_recent", "bprmf")
CURVE_MODELS = ("kgat", "tkgat", "tkgat_perm")  # the models of the validation learning-curve figure
SUBSET_NOTE = {  # random user subsets of ML-1M (decision 0008): never to be read as the full dataset
    "ml-1m-u10": "ml-1m-u10 là 10% user của ML-1M, không phải ML-1M đầy đủ",
    "ml-1m-u05": "ml-1m-u05 là 5% user của ML-1M, không phải ML-1M đầy đủ",
}
CURVE_COLOR = {"kgat": "#2b6cb0", "tkgat": "#eb6834", "tkgat_perm": "#1baf7a"}
KG_MODELS = ("kgat", "tkgat", "tkgat_perm")  # the models that read the KG, re-run by decision 0011
KG_0011_SINCE = "2026-10-29T08:28:49Z"  # commit of decision 0011 (UTC): earlier KG-model rows used the Wikidata-only KG


def freeze_tag_exists(tag: str = FREEZE_TAG) -> bool:
    """Is the git tag that freezes the KGAT and T-KGAT numbers there? Outside a checkout it is not."""
    try:
        out = subprocess.run(["git", "tag", "--list", tag], capture_output=True, text=True, check=True)
    except Exception:  # noqa: BLE001 - no git, not a repository
        return False
    return tag in out.stdout.split()


def official_test_rows(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(rows, dirty)``: the newest clean test row of each key, and the test rows left out for being -dirty."""
    test = registry.load(path, latest=False)
    test = test[test["phase"] == "test"]
    is_dirty = test["git_commit"].fillna("").astype(str).str.endswith("-dirty")
    return registry.newest(test[~is_dirty]), test[is_dirty]


def superseded_rows(path: str | Path) -> pd.DataFrame:
    """Clean test rows of the KG models run before decision 0011, i.e. on the Wikidata-only KG (newest per key).

    The re-run of 0011 has the same registry keys, so ``registry.load`` hides these rows; 0011 (point 4) asks for them
    in an appendix.
    """
    test = registry.load(path, latest=False)
    test = test[(test["phase"] == "test") & test["model"].isin(KG_MODELS)]
    clean = ~test["git_commit"].fillna("").astype(str).str.endswith("-dirty")
    return registry.newest(test[clean & (test["run_at"].astype(str) < KG_0011_SINCE)])


def point_key(config: str) -> str:
    """The grid point of a registry ``config`` as a canonical string (``perm_seed`` is the run seed, not a setting)."""
    cfg = json.loads(config)
    return json.dumps({k: v for k, v in cfg.items() if k != "perm_seed"}, sort_keys=True)


def shown_point(key: str) -> str:
    cfg = {k: v for k, v in json.loads(key).items() if k not in EXTRA_KEYS}
    return ", ".join(f"{k} = {v}" for k, v in cfg.items()) or "—"


@dataclass
class ModelRows:
    tag: str
    rows: pd.DataFrame  # one row per seed, sorted by seed
    point: str

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.rows["seed"]]

    def values(self, metric: str) -> np.ndarray:
        return self.rows[metric].to_numpy(dtype=float)

    def mean_std(self, metric: str) -> tuple[float, float | None]:
        v = self.values(metric)
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)


def choose(rows: pd.DataFrame, dataset: str) -> tuple[dict[str, ModelRows], str | None]:
    """The rows that count for each model of ``MODELS`` in ``dataset`` and the split they belong to."""
    here = rows[(rows["dataset"] == dataset) & rows["model"].isin(NAME)].copy()
    if here.empty:
        return {}, None
    here = here.reset_index(drop=True).reset_index(names="_line").sort_values(["run_at", "_line"], kind="stable")
    split = here.iloc[-1]["split_sha256"]  # the split of the newest test row of any of these models
    chosen = {}
    for tag in NAME:
        mine = here[(here["split_sha256"] == split) & (here["model"] == tag)]
        if mine.empty:
            continue
        key = point_key(mine.iloc[-1]["config"])  # the point of the newest run of this model
        keep = mine[mine["config"].map(point_key) == key]
        chosen[tag] = ModelRows(tag, keep.sort_values("seed").drop(columns="_line"), key)
    return chosen, split


class PerUserMismatch(Exception):
    """A per-user file does not hold the numbers of the registry row it should belong to."""


def per_user_means(
    a: ModelRows, b: ModelRows, metric: str, runs_dir: Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-user mean over the common seeds of ``a`` and of ``b`` (users matched by ``load_per_user``); their count."""
    common = sorted(set(a.seeds) & set(b.seeds))
    if not common:
        raise PerUserMismatch("no seed in common")
    run = Path(runs_dir) / dataset
    xs, ys = [], []
    for seed in common:
        fa, fb = run / f"{a.tag}_seed{seed}.npz", run / f"{b.tag}_seed{seed}.npz"
        for f in (fa, fb):
            if not f.exists():
                raise PerUserMismatch(f"no per-user file {f.name}")
        x, y = load_per_user(fa, fb, metric)  # raises ValueError unless both hold the same users in the same order
        for model, values in ((a, x), (b, y)):
            logged = float(model.rows.loc[model.rows["seed"] == seed, metric].iloc[0])
            if not np.isclose(np.nanmean(values), logged, rtol=0, atol=1e-9):
                raise PerUserMismatch(f"{model.tag}_seed{seed}.npz does not match the registry row (another run)")
        xs.append(x)
        ys.append(y)
    return np.mean(xs, axis=0), np.mean(ys, axis=0), len(common)


def compare(chosen: dict[str, ModelRows], runs_dir: Path, dataset: str) -> list[dict]:
    """Paired Wilcoxon of the planned pairs, per metric, Holm-corrected within a family.

    The family of Stage 1 has three pairs. The two pairs of Stage 2 form a second family, shown only if a Stage 2 model
    has a clean test row, so the table and the adjusted p-values of Stage 1 do not depend on whether Stage 2 has run.
    """
    families = [PAIRS]
    if any(t in chosen for t in ("tkgat_kg", "tkgat_kg_perm")):
        families.append(PAIRS_STAGE2)
    out = []
    for metric in METRICS:
        for family in families:
            out.extend(compare_family(chosen, runs_dir, dataset, metric, family))
    return out


def compare_family(chosen: dict[str, ModelRows], runs_dir: Path, dataset: str, metric: str, family) -> list[dict]:
    """One metric, one Holm family: the rows of its pairs with ``p_holm`` set where the pair could be computed."""
    block = []
    for a, b, label in family:
        row = {"dataset": dataset, "metric": metric, "pair": label, "model_a": a, "model_b": b}
        missing = [NAME[t] for t in (a, b) if t not in chosen]
        if missing:
            row["status"] = f"{NOT_RUN}: {' và '.join(missing)}"
        else:
            try:
                x, y, n_seeds = per_user_means(chosen[a], chosen[b], metric, runs_dir, dataset)
            except (PerUserMismatch, ValueError, KeyError) as err:
                row["status"] = f"không so sánh được: {err}"
            else:
                res = paired_wilcoxon(x, y)
                row.update(
                    status="ok",
                    n_seeds=n_seeds,
                    n_pairs=res["n_pairs"],
                    n_nonzero=res["n_nonzero"],
                    mean_a=float(np.nanmean(x)),
                    mean_b=float(np.nanmean(y)),
                    mean_diff=res["mean_diff"],
                    p_raw=res["p_value"],
                    warning=warning(res["n_pairs"], res["n_nonzero"]),
                )
        block.append(row)
    padded = [r["p_raw"] if r["status"] == "ok" else 1.0 for r in block]  # a missing comparison is never significant
    for r, p_adj in zip(block, holm(padded)["p_adjusted"], strict=True):
        if r["status"] == "ok":
            r["p_holm"] = p_adj
    return block


# --- validation learning curves -------------------------------------------------------------------------------------


def selected_histories(run_dir: Path, tag: str, point: str, split: str) -> list[pd.DataFrame]:
    """Validation Recall@20 per epoch of the runs of ``tag`` that trained the grid point ``point`` on ``split``.

    Reads ``<run_dir>/<tag>_<grid>_seed<seed>_history.csv`` (the name is matched exactly: ``kgat`` never reads the files
    of ``kgat_nokg``). A history of another point, of another split or of a ``-dirty`` commit is left out, as the
    registry rows of such runs are. A model without history files gives an empty list, not an error.
    """
    want = {k: v for k, v in json.loads(point).items() if k not in EXTRA_KEYS}
    frames = []
    if not run_dir.is_dir():
        return frames
    pattern = re.compile(rf"{re.escape(tag)}_\d+_seed\d+_history\.csv")
    for path in sorted(p for p in run_dir.iterdir() if pattern.fullmatch(p.name)):
        h = pd.read_csv(path, dtype={"git_commit": str, "split_sha256": str})
        if h.empty or "recall@20" not in h:
            continue
        first = h.iloc[0]
        if str(first.get("git_commit", "")).endswith("-dirty") or first.get("split_sha256") != split:
            continue
        config = json.loads(first["config"])
        have = {**config.get("model", {}), **config.get("fit", {})}  # the model's and the trainer's settings
        if all(have.get(k) == v for k, v in want.items()):
            frames.append(h[["epoch", "recall@20"]].dropna().assign(seed=int(first["seed"])))
    return frames


def curve_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per epoch: how many seeds reached it, the mean and the sample std (ddof=1) of their val Recall@20."""
    g = pd.concat(frames).groupby("epoch")["recall@20"]
    return pd.DataFrame({"n_seeds": g.count(), "mean": g.mean(), "std": g.std(ddof=1)}).reset_index()


def history_curves(run_dir: Path, chosen: dict[str, ModelRows], split: str | None) -> dict[str, pd.DataFrame]:
    curves = {}
    for tag in CURVE_MODELS:
        if tag in chosen:
            frames = selected_histories(run_dir, tag, chosen[tag].point, split)
            if frames:
                curves[tag] = curve_table(frames)
    return curves


def draw_curves(dataset: str, curves: dict[str, pd.DataFrame], path: Path) -> bool:
    """Val Recall@20 against the epoch, mean over seeds, one line per model; False (nothing written) without a curve.

    A line is solid, with a band of one sample std, while all the seeds of that model still train; after the first
    one stops early the mean is over fewer seeds, so it is dashed and has no band (it would read as a trend).
    """
    if not curves:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for tag, t in curves.items():
        color, n_all = CURVE_COLOR[tag], int(t["n_seeds"].max())
        full, rest = t[t["n_seeds"] == n_all], t[t["n_seeds"] < n_all]
        ax.plot(full["epoch"], full["mean"], color=color, lw=1.8, label=f"{NAME[tag]} ({n_all} seed)")
        if n_all > 1:
            ax.fill_between(
                full["epoch"], full["mean"] - full["std"], full["mean"] + full["std"], color=color, alpha=0.15, lw=0
            )
        if len(rest):
            tail = pd.concat([full.tail(1), rest])
            ax.plot(tail["epoch"], tail["mean"], color=color, lw=1.2, ls="--", alpha=0.7)
    ax.set_title(f"{dataset}: Recall@20 trên val theo epoch (sơ bộ, chưa đóng băng)", fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Recall@20 (val)")
    ax.grid(True, color="#e4e7eb", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.text(
        0.01, 0.005, "nét liền: đủ các seed, dải = ± 1 std mẫu; nét đứt: còn ít seed hơn", fontsize=6, color="#52514e"
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build(path: str | Path, runs_dir: str | Path, datasets) -> dict:
    """Everything the report prints, per dataset."""
    rows, dirty = official_test_rows(path)
    result = {"datasets": {}, "freeze_tag": freeze_tag_exists(), "superseded": {}}
    old = superseded_rows(path)
    for dataset in datasets:
        chosen, split = choose(rows, dataset)
        curves = history_curves(Path(runs_dir) / dataset, chosen, split)
        result["datasets"][dataset] = {
            "chosen": chosen,
            "split": split,
            "pairs": compare(chosen, Path(runs_dir), dataset),
            "dirty_ignored": int((dirty["dataset"] == dataset).sum()),
            "curves": curves,
            "no_history": [NAME[t] for t in CURVE_MODELS if t not in curves],
            "figure": None,  # set by the caller to the figure's path, relative to the markdown file, once it is drawn
        }
        result["superseded"][dataset] = choose(old, dataset)[0]
    return result


# --- output ---------------------------------------------------------------------------------------------------------


def models_frame(result: dict) -> pd.DataFrame:
    out = []
    for dataset, info in result["datasets"].items():
        for tag, label in MODELS:
            m = info["chosen"].get(tag)
            if m is None and tag in OPTIONAL_MODELS:
                continue
            row = {"dataset": dataset, "model": tag, "label": label, "split_sha256": info["split"], "status": NOT_RUN}
            if m is not None:
                row.update(
                    status="ok", point=shown_point(m.point), n_seeds=len(m.rows), n_users=int(m.rows["n_users"].iloc[0])
                )
                for metric in METRICS:
                    mean, std = m.mean_std(metric)
                    row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
            out.append(row)
    return pd.DataFrame(out)


def preliminary_note(freeze_tag_present: bool, subsets=()) -> str:
    if freeze_tag_present:
        status = (
            f"Tag `{FREEZE_TAG}` có trong repo: kiểm tra rằng các dòng KGAT và T-KGAT dưới đây thuộc commit của tag."
        )
    else:
        status = f"Chưa có tag `{FREEZE_TAG}`: các số này chưa đóng băng."
    note = (
        "> **Số của KGAT, KGAT không KG, T-KGAT và T-KGAT hoán vị là kết quả sơ bộ, chưa đóng băng "
        f"(tag `{FREEZE_TAG}`), chạy trước khi cô duyệt giao thức** (t1/t2, quyết định 0002). {status} KGAT và "
        "T-KGAT không tái lập từng bit trên GPU: coi chênh lệch giữa các lần chạy của một seed như nhiễu seed."
    )
    for dataset in subsets:
        note += f"\n>\n> **{SUBSET_NOTE[dataset]}.**"
    return note


def markdown(result: dict) -> str:
    lines = [
        "# Bảng RQ3 (tự sinh bởi `scripts/10_rq3_table.py`)",
        "",
        preliminary_note(result["freeze_tag"], [d for d in result["datasets"] if d in SUBSET_NOTE]),
        "",
        "Nguồn: `results/registry.csv` qua `registry.load`, chỉ dòng `phase = test` không mang nhãn `-dirty`; mỗi mô "
        "hình lấy điểm của lần chạy mới nhất trên split mới nhất. Trung bình ± độ lệch chuẩn **mẫu** (`ddof=1`) qua "
        "các "
        "seed; một seed thì không có độ lệch chuẩn. Kiểm định: Wilcoxon cặp theo từng user (`tkgat.eval.stats`) trên "
        "trung bình các seed chung, Holm trong cùng bộ dữ liệu và metric, họ cố định gồm ba phép so sánh; hai phép của "
        "giai đoạn 2 (nếu có) là một họ riêng. "
        "KG gồm fact Wikidata (bản chụp 2026) và metadata MovieLens (quyết định 0011).",
        "",
    ]
    for dataset, info in result["datasets"].items():
        chosen = info["chosen"]
        users = {int(m.rows["n_users"].iloc[0]) for m in chosen.values()}
        lines += [f"## {dataset}" + (f" (user test: {', '.join(str(u) for u in sorted(users))})" if users else ""), ""]
        if dataset in SUBSET_NOTE:
            lines += [f"> {SUBSET_NOTE[dataset]}.", ""]
        missing = [NAME[t] for t in BASELINES if t not in chosen]
        if len(missing) == len(BASELINES):
            lines += [
                f"> **Baseline của {dataset} chưa chạy:** không có dòng test sạch nào của {', '.join(missing)}; "
                "chưa có baseline để so với KGAT và T-KGAT trên bộ dữ liệu này.",
                "",
            ]
        elif missing:
            lines += [f"> Baseline chưa chạy ở {dataset}: {', '.join(missing)}.", ""]
        if not chosen:
            lines += [f"*{NOT_RUN}: không có dòng test sạch nào của các mô hình này trong registry.*", ""]
        lines += [
            "| Mô hình | Cấu hình | Recall@20 (test) | NDCG@20 (test) | Số seed |",
            "|---|---|---|---|---|",
        ]
        for tag, label in MODELS:
            m = chosen.get(tag)
            if m is None and tag in OPTIONAL_MODELS:
                continue
            star = " [sơ bộ]" if tag in PRELIMINARY else ""
            if m is None:
                lines.append(f"| {label}{star} | — | {NOT_RUN} | {NOT_RUN} | — |")
                continue
            cells = [vn_pm(*m.mean_std(metric)) for metric in METRICS]
            few = " **(chưa đủ 5 seed)**" if len(m.rows) not in (1, 5) else ""
            lines.append(f"| {label}{star} | {shown_point(m.point)} | {cells[0]} | {cells[1]} | {len(m.rows)}{few} |")
        lines += ["", "**So sánh cặp** (hiệu = trước − sau):", ""]
        lines += [
            "| So sánh | Metric | Seed chung | n_pairs | n_nonzero | Hiệu trung bình | p thô | p Holm | Ghi chú |"
        ]
        lines += ["|---|---|---|---|---|---|---|---|---|"]
        for r in info["pairs"]:
            if r["status"] != "ok":
                lines.append(f"| {r['pair']} | {METRIC_NAME[r['metric']]} | — | — | — | — | — | — | {r['status']} |")
                continue
            lines.append(
                f"| {r['pair']} | {METRIC_NAME[r['metric']]} | {r['n_seeds']} | {r['n_pairs']} | {r['n_nonzero']} "
                f"| {vn_dec(r['mean_diff'], 4)} | {vn_p(r['p_raw'])} | {vn_p(r['p_holm'])} | {r['warning'] or '—'} |"
            )
        if info["dirty_ignored"]:
            lines += ["", f"*{info['dirty_ignored']} dòng test mang nhãn `-dirty` bị bỏ qua.*"]
        if info.get("figure"):
            lines += ["", f"![Learning curve val của KGAT, T-KGAT và T-KGAT hoán vị trên {dataset}]({info['figure']})"]
            if info["no_history"]:
                lines += ["", f"*Chưa có history của: {', '.join(info['no_history'])}.*"]
        else:
            lines += ["", "*Chưa có file history của KGAT, T-KGAT hay T-KGAT hoán vị: chưa vẽ learning curve val.*"]
        lines.append("")
    lines += superseded_markdown(result.get("superseded", {}))
    return "\n".join(lines)


def superseded_markdown(superseded: dict) -> list[str]:
    """The appendix of decision 0011: the KG models on the Wikidata-only KG, kept for comparison, never official."""
    if not any(superseded.values()):
        return []
    lines = [
        "## Phụ lục: KG chỉ có Wikidata (bị thay bởi quyết định 0011)",
        "",
        "> Các dòng này chạy trước quyết định 0011, trên KG chưa có metadata MovieLens. Chúng **không phải kết quả "
        "chính thức**; giữ lại để người đọc thấy cả hai, vì KG được đổi sau khi điểm test của chúng đã được xem.",
        "",
        "| Bộ dữ liệu | Mô hình | Cấu hình | Recall@20 (test) | NDCG@20 (test) | Số seed |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, chosen in superseded.items():
        for tag in KG_MODELS:
            m = chosen.get(tag)
            if m is None:
                continue
            cells = [vn_pm(*m.mean_std(metric)) for metric in METRICS]
            lines.append(
                f"| {dataset} | {NAME[tag]} | {shown_point(m.point)} | {cells[0]} | {cells[1]} | {len(m.rows)} |"
            )
    return [*lines, ""]
