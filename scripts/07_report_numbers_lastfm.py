"""Write the numbers of the Last-FM data report (docs/data-report-lastfm-numbers.md).

    python scripts/07_report_numbers_lastfm.py
    python scripts/07_report_numbers_lastfm.py --out /tmp/numbers.md

The Last-FM counterpart of ``scripts/07_report_numbers.py``. Every number of ``docs/data-report-lastfm.md`` must
come from this file (``tests/test_data_report_lastfm.py`` checks it), and every number here is computed from files
written by the pipeline:

  data/processed/<run>/meta.json            the split (script 02): cut-offs, counts, the user subsample
  data/processed/<run>/train.csv            the train timestamps: Delta-t anchor of T-KGAT, time spread of the train
                                            edges (eda.node_time_spread), edges per week around 2014-04-01
  data/processed/<run>/{train,val,test}.csv the last timestamp of the pairs (only the ts column is read)
  data/processed/<run>/kg/<variant>/        kg_meta.json, kg_t1_dates.tsv, dropped.csv (script 06_build_kg_lastfm)
  results/registry.csv                      training hours per device and the range of best epochs, VALIDATION rows
                                            only (``phase == "val"``): the test rows are dropped before anything is
                                            computed, so no test score can reach this file (rule 1 of CLAUDE.md)

A source that does not exist yet gives "chưa có" and is filled in on the next run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkgat import registry  # noqa: E402
from tkgat.analysis import eda, rq3  # noqa: E402
from tkgat.analysis.report import (  # noqa: E402
    CONVENTIONS,
    MISSING,
    Numbers,
    parse_numbers,
    vn_date,
    vn_dec,
    vn_int,
    vn_pct,
)
from tkgat.kg.lastfm import DATE_PREDICATES, FREEBASE_SNAPSHOT, HEAD_NOT_ITEM  # noqa: E402
from tkgat.kg.temporal import VARIANTS  # noqa: E402
from tkgat.pipeline.build import load_config  # noqa: E402

# lastfm-u25 is the official run (decision 0013); lastfm-u10 stays as the small development set
RUNS = {
    "lastfm": "configs/data/lastfm.yaml",
    "lastfm-u25": "configs/data/lastfm_u25.yaml",
    "lastfm-u10": "configs/data/lastfm_u10.yaml",
}
# MovieLens runs whose Delta-t scale the Last-FM one is compared with (same computation, same files)
ML_RUNS = {"ml-100k": "configs/data/ml100k.yaml", "ml-1m-u10": "configs/data/ml1m_u10.yaml"}
# values of docs/data-report-numbers.md that the Last-FM report compares against, by key
COMPARE = [
    "conv.first_window",
    "ml100k.val_kept_share",
    "ml100k.test_kept_share",
    "ml100k.val_users",
    "ml100k.test_users",
    "ml100k.first_24h_share",
    "ml100k.user_spread_under_1h",
    "ml100k.no_date_share",
    "ml100k.filtered_share_t1",
    "ml1m.val_kept_share",
    "ml1m.test_kept_share",
    "ml1m.val_users",
    "ml1m.test_users",
    "ml1m.filtered_share_t1",
]
S1, S2, S3 = "1. Dữ liệu và chia theo thời gian", "2. KG theo thời gian", "3. Thang Delta-t của Stage 2"
S_EDGE = "4. Thời gian của cạnh train (Stage 1)"
S_WEEK = "5. Số cạnh train theo tuần quanh 01/04/2014"
S_RUN = "6. Thời gian chạy và epoch tốt nhất (registry, chỉ dòng val)"
S_ML = "7. Đối chiếu với MovieLens"
YEAR_S = 365.25 * 86400
# The pair set of the KGAT Last-FM data holds only pairs still listened to from 2014-04-01 (decision 0012). The weekly
# count of train edges is compared over the four Monday-started weeks before and the four weeks from that Monday.
WEEK_MARKER = pd.Timestamp("2014-03-31")  # Monday of the week that contains 2014-04-01
WEEKS_BEFORE = (pd.Timestamp("2014-03-03"), WEEK_MARKER)  # [start, end)
WEEKS_AFTER = (WEEK_MARKER, pd.Timestamp("2014-04-21"))  # [start, end]


def day(ts: int) -> str:
    return vn_date(pd.Timestamp(int(ts), unit="s", tz="UTC").date().isoformat())


def add_split(n: Numbers, run: str, meta: dict) -> None:
    """Counts of the split, straight from meta.json (script 02)."""
    src = f"data/processed/{run}/meta.json"
    n.add(S1, f"{run}.pairs", vn_int(meta["n_raw"]), src)
    n.add(S1, f"{run}.t1", day(meta["t1"]), src)
    n.add(S1, f"{run}.t2", day(meta["t2"]), src)
    n.add(S1, f"{run}.train_pos", vn_int(meta["train_pos"]), src)
    n.add(S1, f"{run}.users", vn_int(meta["n_users"]), src)
    n.add(S1, f"{run}.items", vn_int(meta["n_items"]), src)
    for part in ("val", "test"):
        n.add(S1, f"{run}.{part}_pos", vn_int(meta[f"{part}_pos"]), src)
        n.add(S1, f"{run}.{part}_users", vn_int(meta[f"{part}_users"]), src)
        before = meta[f"{part}_pos_before_filter"]
        n.add(S1, f"{run}.{part}_kept_share", vn_pct(meta[f"{part}_pos"] / before), src)
        n.add(
            S1,
            f"{run}.{part}_dropped_cold_user_share",
            vn_pct(meta[f"{part}_dropped_cold_user_interactions"] / before),
            src,
        )
        n.add(
            S1,
            f"{run}.{part}_dropped_cold_item_share",
            vn_pct(meta[f"{part}_dropped_cold_item_interactions"] / before),
            src,
        )
    n.add(S1, f"{run}.train_pos_before_kcore", vn_int(meta["train_pos_before_kcore"]), src)
    n.add(S1, f"{run}.kcore_removed", vn_int(meta["train_pos_before_kcore"] - meta["train_pos"]), src)
    sub = meta.get("clean", {}).get("user_subsample")
    if sub:
        n.add(S1, f"{run}.users_sampled", f"{vn_int(sub['users_kept'])} / {vn_int(sub['users_total'])}", src)
        n.add(S1, f"{run}.user_fraction", vn_pct(sub["user_fraction"], 0), src)
        n.add(S1, f"{run}.user_seed", str(sub["user_seed"]), src)


def add_last_timestamp(n: Numbers, run: str, processed: Path) -> None:
    """The latest timestamp among the pairs that survive the split: the report quotes it as the end of the data.

    It is the maximum over train, val and test, i.e. after the cold-user / cold-item filter, so it is a lower
    bound of the latest listening event of the raw pairs.
    """
    last = max(
        int(pd.read_csv(processed / f"{part}.csv", usecols=["ts"])["ts"].max()) for part in ("train", "val", "test")
    )
    n.add(S1, f"{run}.last_ts", day(last), f"data/processed/{run}/train.csv, val.csv, test.csv, cột ts lớn nhất")


def add_kg(n: Numbers, run: str, metas: dict[str, dict], dropped: pd.DataFrame) -> None:
    """Counts of the three KG variants, from kg_meta.json and dropped.csv (script 06_build_kg_lastfm)."""
    base = metas["default"]
    src = f"data/processed/{run}/kg/default/kg_meta.json"
    n.add(S2, f"{run}.kg.author_triples", vn_int(base["n_author_triples"]), src)
    n.add(S2, f"{run}.kg.item_seeded", vn_int(base["n_item_seeded_triples"]), src)
    n.add(S2, f"{run}.kg.head_seeded", vn_int(base["n_head_seeded_triples"]), src)
    n_drop = base["n_dropped_head_not_train_item"]
    n.add(S2, f"{run}.kg.dropped_not_1hop", vn_int(n_drop), src)
    n.add(S2, f"{run}.kg.dropped_not_1hop_share", vn_pct(n_drop / base["n_item_seeded_triples"]), src)
    n.add(S2, f"{run}.kg.entities", vn_int(base["n_entities"]), src)
    n.add(S2, f"{run}.kg.tail_entities", vn_int(base["n_entities"] - base["n_items"]), src)
    n.add(S2, f"{run}.kg.relations", vn_int(base["n_relations"]), src)
    for variant in VARIANTS:
        m = metas[variant]
        vsrc = f"data/processed/{run}/kg/{variant}/kg_meta.json"
        for cut in ("t1", "t2"):
            n.add(S2, f"{run}.kg.{variant}.{cut}", vn_int(m["n_triples"][cut]), vsrc)
        n.add(S2, f"{run}.kg.{variant}.dated_t1", vn_int(m["n_dated_triples"]["t1"]), vsrc)
    static_t1, default_t1 = metas["static"]["n_triples"]["t1"], metas["default"]["n_triples"]["t1"]
    n.add(S2, f"{run}.kg.filtered_t1", vn_int(static_t1 - default_t1), src)
    n.add(S2, f"{run}.kg.filtered_t1_share", vn_pct((static_t1 - default_t1) / static_t1), src)
    n.add(S2, f"{run}.kg.undated_t1", vn_int(default_t1 - metas["default"]["n_dated_triples"]["t1"]), src)
    n.add(S2, f"{run}.kg.undated_t1_share", vn_pct(1 - metas["default"]["n_dated_triples"]["t1"] / default_t1), src)
    n.add(S2, f"{run}.kg.items_with_facts", vn_int(base["n_items_with_facts"]["t1"]), src)
    n.add(S2, f"{run}.kg.entities_only_in_t2", vn_int(base["n_entities_only_in_t2"]), src)
    # which relations the triples that the 1-hop format cannot carry belong to
    not_1hop = dropped[(dropped["reason"] == HEAD_NOT_ITEM) & (dropped["cutoff"] == "t1")]
    counts = not_1hop["relation"].value_counts()
    dsrc = f"data/processed/{run}/kg/default/dropped.csv"
    for relation, count in counts.items():
        n.add(S2, f"{run}.kg.dropped_not_1hop.{relation}", vn_int(count), dsrc)
    # relations whose facts carry no usable date at all
    rel = pd.DataFrame(base["date_kind_per_relation"])
    for row in rel.sort_values("facts", ascending=False).itertuples(index=False):
        n.add(S2, f"{run}.kg.no_date.{row.relation}", vn_pct(row.no_date), src)
    n.add(S2, f"{run}.kg.relations_fully_undated", vn_int(int((rel["no_date"] == 1).sum())), src)


def add_delta_t(n: Numbers, run: str, processed: Path, metas: dict[str, dict], section: str = S3) -> None:
    """The Delta-t scale of Stage 2: how many KG triples fall inside the normalisation window.

    ``t_ref`` and ``dt_den`` are the constants of ``tkgat.kg.ckg.build_ckg``, both from the train timestamps
    only; ``dt_norm`` is capped at 1 for KG edges (decision 0009, item 10), so a fact older than the window is
    indistinguishable from any other old fact.
    """
    ts = pd.read_csv(processed / "train.csv", usecols=["ts"])["ts"].to_numpy()
    dt_den = float(ts.max() - ts.min())
    dates = pd.read_csv(processed / "kg" / "default" / "kg_t1_dates.tsv", sep="\t")["ts"].to_numpy(dtype=np.float64)
    src = f"data/processed/{run}/train.csv + kg/default/kg_t1_dates.tsv"
    n.add(section, f"{run}.dt.window_years", vn_dec(dt_den / YEAR_S), src)
    n.add(section, f"{run}.dt.t_ref", day(ts.max()), f"data/processed/{run}/train.csv")
    dated = np.isfinite(dates)
    dt_norm = np.clip(float(ts.max()) - dates[dated], 0, None) / dt_den
    total = len(dates)
    n.add(section, f"{run}.dt.undated_share", vn_pct(int((~dated).sum()) / total), src)
    n.add(section, f"{run}.dt.inside_window_share", vn_pct(int((dt_norm < 1).sum()) / total), src)
    n.add(section, f"{run}.dt.capped_share", vn_pct(int((dt_norm >= 1).sum()) / total), src)
    n.add(section, f"{run}.dt.median_dated", vn_dec(float(np.median(dt_norm))), src)
    expected = metas["default"]["n_triples"]["t1"]
    if expected != total:
        raise RuntimeError(f"{run}: kg_t1_dates.tsv has {total} rows, kg_meta.json says {expected}")


def add_edge_time(n: Numbers, run: str, processed: Path, t1: int) -> None:
    """Time spread of the train edges and edges per week: the conditions under which Stage 1 can act.

    Same functions as the MovieLens report (``eda.node_time_spread`` with the t1 anchor, ``eda.share_below``,
    ``eda.weekly_counts``), applied to ``train.csv`` only.
    """
    src = f"data/processed/{run}/train.csv qua eda.node_time_spread (mốc t1)"
    train = pd.read_csv(processed / "train.csv")
    by_user = eda.node_time_spread(train, by="user", ref=t1)
    by_item = eda.node_time_spread(train, by="item", ref=t1)
    share = eda.share_below(by_user["spread_s"], {"1h": 3600})["1h"]
    n.add(S_EDGE, f"{run}.spread.user_under_1h", vn_pct(share, 2), src + ", độ trải < 1 giờ")
    n.add(S_EDGE, f"{run}.spread.user_std_days_median", vn_dec(float(by_user["std_days"].median()), 1), src)
    n.add(S_EDGE, f"{run}.spread.item_std_days_median", vn_dec(float(by_item["std_days"].median()), 1), src)
    weekly = eda.weekly_counts(train)
    wsrc = f"data/processed/{run}/train.csv qua eda.weekly_counts"
    before = weekly[(weekly.index >= WEEKS_BEFORE[0]) & (weekly.index < WEEKS_BEFORE[1])]
    after = weekly[(weekly.index >= WEEKS_AFTER[0]) & (weekly.index <= WEEKS_AFTER[1])]
    n.add(S_WEEK, f"{run}.week.before_mean", vn_int(round(before.mean())), wsrc + ", 4 tuần trước 31/03/2014")
    n.add(S_WEEK, f"{run}.week.after_mean", vn_int(round(after.mean())), wsrc + ", 4 tuần từ 31/03/2014")


def device_slug(label: str) -> str:
    """A key-safe name of a registry device label (``cuda:Tesla T4`` -> ``Tesla_T4``)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label.removeprefix("cuda:")).strip("_")


def add_registry(n: Numbers, run: str, rows: pd.DataFrame) -> None:
    """Training hours per device and the range of best epochs, from the validation rows of the registry.

    ``rows`` are the validation rows of one run: a validation row carries the whole training time of its
    (grid point, seed) and the epoch chosen on validation. The test rows are never passed in.
    """
    src = "results/registry.csv (phase=val)"
    hours = rows.assign(hours=rows["seconds"].astype(float) / 3600)
    for device, total in hours.groupby("device")["hours"].sum().items():
        n.add(S_RUN, f"{run}.hours.device.{device_slug(device)}", vn_dec(total, 1), src + ", tổng cột seconds")
    for (device, model), total in hours.groupby(["device", "model"])["hours"].sum().items():
        n.add(S_RUN, f"{run}.hours.{device_slug(device)}.{model}", vn_dec(total, 1), src + ", tổng cột seconds")
    esrc = src + ", mọi điểm lưới và seed"
    for model, group in rows.groupby("model"):
        n.add(S_RUN, f"{run}.best_epoch.{model}.min", vn_int(group["best_epoch"].min()), esrc)
        n.add(S_RUN, f"{run}.best_epoch.{model}.max", vn_int(group["best_epoch"].max()), esrc)
    # The grid point of each model with the best mean (over seeds) validation Recall@20: the rule of scripts/run.py
    # (``np.argmax(val_means)``), which is what scripts/10_rq3_table.py later reads back as the point of the test rows.
    # Points are grouped with rq3.point_key (``perm_seed`` is the run seed, not a setting). An exact tie would go to the
    # first point in string order here and to the first point of the grid in run.py; it does not occur in this registry.
    keyed = rows.assign(point=rows["config"].map(rq3.point_key))
    best_point = keyed.groupby(["model", "point"])["recall@20"].mean().groupby("model").idxmax()
    csrc = src + ", điểm lưới có trung bình Recall@20 val cao nhất"
    for model, (_, point) in best_point.items():
        group = keyed[(keyed["model"] == model) & (keyed["point"] == point)]
        n.add(S_RUN, f"{run}.best_epoch.{model}.chosen.min", vn_int(group["best_epoch"].min()), csrc)
        n.add(S_RUN, f"{run}.best_epoch.{model}.chosen.max", vn_int(group["best_epoch"].max()), csrc)


def add_comparison(n: Numbers, movielens_numbers: Path) -> None:
    """The MovieLens values the Last-FM report compares against.

    Counts and shares are taken by key from ``docs/data-report-numbers.md`` (written by
    ``scripts/07_report_numbers.py``), never recomputed here, so the two reports can never disagree. The only
    exception is the Stage 2 Delta-t scale, which that script does not publish: it is computed with the very
    same :func:`add_delta_t` on the MovieLens runs, from the same kinds of pipeline file.
    """
    if not movielens_numbers.exists():
        n.add(S_ML, "cmp.ml100k.val_kept_share", MISSING, "docs/data-report-numbers.md")
    else:
        known = parse_numbers(movielens_numbers.read_text(encoding="utf-8"))
        for key in COMPARE:
            if key not in known:
                raise KeyError(f"{movielens_numbers.name} has no key {key!r}: re-run scripts/07_report_numbers.py")
            n.add(S_ML, f"cmp.{key}", known[key][0], f"docs/data-report-numbers.md (`{key}`)")
    for run, config in ML_RUNS.items():
        processed = ROOT / load_config(ROOT / config)["processed_dir"]
        kg_dir = processed / "kg"
        if not (processed / "train.csv").exists() or not (kg_dir / "default" / "kg_meta.json").exists():
            n.add(S_ML, f"{run}.dt.inside_window_share", MISSING, f"data/processed/{run}/kg/default/")
            continue
        metas = {v: json.loads((kg_dir / v / "kg_meta.json").read_text(encoding="utf-8")) for v in VARIANTS}
        add_delta_t(n, run, processed, metas, section=S_ML)


def build() -> Numbers:
    n = Numbers()
    n.add(CONVENTIONS, "conv.freebase_snapshot", vn_date(FREEBASE_SNAPSHOT), "tkgat.kg.lastfm.FREEBASE_SNAPSHOT")
    n.add(CONVENTIONS, "conv.date_predicates", str(len(DATE_PREDICATES)), "tkgat.kg.lastfm.DATE_PREDICATES")
    n.add(CONVENTIONS, "conv.quantiles", "0,8 / 0,9", "configs/data/lastfm.yaml")
    n.add(CONVENTIONS, "conv.kcore", "5", "configs/data/lastfm.yaml")
    n.add(CONVENTIONS, "conv.min_rating", "1", "configs/data/lastfm.yaml")
    kgat = load_config(ROOT / "configs" / "model" / "kgat.yaml")
    n.add(CONVENTIONS, "conv.seeds", vn_int(len(kgat["seeds"])), "configs/model/kgat.yaml")
    n.add(CONVENTIONS, "conv.grid_points", vn_int(len(kgat["grid"])), "configs/model/kgat.yaml")
    n.add(CONVENTIONS, "conv.max_epochs", vn_int(kgat["fixed"]["max_epochs"]), "configs/model/kgat.yaml")
    bprmf = load_config(ROOT / "configs" / "model" / "bprmf.yaml")
    n.add(CONVENTIONS, "conv.max_epochs_bprmf", vn_int(bprmf["fixed"]["max_epochs"]), "configs/model/bprmf.yaml")
    registry_path = ROOT / "results" / "registry.csv"
    # validation rows only, filtered before anything else: the test rows never enter this script
    val_rows = registry.load(registry_path).query("phase == 'val'") if registry_path.exists() else None
    for run, config in RUNS.items():
        processed = ROOT / load_config(ROOT / config)["processed_dir"]
        meta_path = processed / "meta.json"
        if not meta_path.exists():
            n.add(S1, f"{run}.train_pos", MISSING, f"data/processed/{run}/meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        add_split(n, run, meta)
        add_last_timestamp(n, run, processed)
        add_edge_time(n, run, processed, int(meta["t1"]))
        run_rows = None if val_rows is None else val_rows[val_rows["dataset"] == run]
        if run_rows is not None and len(run_rows):
            add_registry(n, run, run_rows)
        kg_dir = processed / "kg"
        if not (kg_dir / "default" / "kg_meta.json").exists():
            n.add(S2, f"{run}.kg.default.t1", MISSING, f"data/processed/{run}/kg/default/kg_meta.json")
            continue
        metas = {v: json.loads((kg_dir / v / "kg_meta.json").read_text(encoding="utf-8")) for v in VARIANTS}
        add_kg(n, run, metas, pd.read_csv(kg_dir / "default" / "dropped.csv"))
        add_delta_t(n, run, processed, metas)
    add_comparison(n, ROOT / "docs" / "data-report-numbers.md")
    # the loop adds the sections run by run; list them in the order of their numbers (the sort is stable)
    order = [CONVENTIONS, S1, S2, S3, S_EDGE, S_WEEK, S_RUN, S_ML]
    n.rows.sort(key=lambda row: order.index(row[0]))
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "data-report-lastfm-numbers.md"))
    args = ap.parse_args(argv)
    numbers = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        numbers.render(
            title="Số liệu của báo cáo dữ liệu Last-FM",
            script="scripts/07_report_numbers_lastfm.py",
            report="docs/data-report-lastfm.md",
            test="tests/test_data_report_lastfm.py",
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"{len(numbers.rows)} numbers -> {out}")
    missing = [k for _, k, v, _ in numbers.rows if v == MISSING]
    if missing:
        print(f'"{MISSING}": ' + ", ".join(missing))


if __name__ == "__main__":
    main()
