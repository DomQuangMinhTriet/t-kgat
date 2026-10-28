"""Write the numbers of the data report (docs/data-report-numbers.md) and its four figures (docs/figures/).

    python scripts/07_report_numbers.py
    python scripts/07_report_numbers.py --out /tmp/numbers.md --figures-dir none

Every number of ``docs/data-report.md`` must come from this file (``tests/test_data_report.py`` checks it), and
every number here is computed from files written by the pipeline:

  data/raw/ + data/checksums.json       data sizes and checksums
  data/processed/<dataset>/             meta.json (split, cleaning), train/val/test.csv, kg/<variant>/*
  data/processed/links/                 linking_report.json
  data/raw/wikidata/facts.tsv           dated Wikidata facts
  results/registry.csv                  baseline scores, only through ``registry.load()``; for each model only the
                                        test rows of the grid point that the latest run selected (``selected`` in
                                        results/runs/<dataset>/<model>_summary.json, decision 0007)
  results/linking_manual_check.json     optional: {"n_checked": int, "n_wrong": int}, written by item 15
  results/linking_manual_check_sample.csv  optional: the sampled films; its counts must agree with the json

Nothing else is read from results/: no per-user test file and no test row of a grid point that was not selected.
A source that does not exist yet gives "chưa có" and is filled in on the next run. The figures only show
aggregates (counts, shares, distributions of time spreads), never rows of the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkgat import registry  # noqa: E402
from tkgat.analysis import eda, kg  # noqa: E402
from tkgat.analysis.eda import DAY, HOUR  # noqa: E402
from tkgat.analysis.report import (  # noqa: E402
    CONVENTIONS,
    MISSING,
    Z95,
    Numbers,
    sample_counts,
    vn_date,
    vn_dec,
    vn_int,
    vn_pct,
    vn_pm,
    wilson_interval,
)
from tkgat.data.clean import clean  # noqa: E402
from tkgat.data.leakage import check_split  # noqa: E402
from tkgat.data.loaders import load  # noqa: E402
from tkgat.data.split import dedupe_pairs  # noqa: E402
from tkgat.pipeline.build import load_config, load_saved  # noqa: E402

# Fixed conventions: the computations below use these constants, and the numbers file lists them, so a
# convention can be typed in the report without being a hand-copied data number.
FIRST_WINDOW_S = DAY  # "the first day" of a user = 24 h after the first rating
SESSION_GAP_S, SESSION_GAP_ALT_S = HOUR, 30 * 60
SPREADS = {"1h": ("1 giờ", HOUR), "1d": ("1 ngày", DAY), "30d": ("30 ngày", 30 * DAY)}  # slug: (label, seconds)
TOP_FRACS = (0.1, 0.2)
POSITIVE_RATING = 4
SEEDS = "2019–2023"
KS = (10, 20)
DATASETS = {"ml100k": "configs/data/ml100k.yaml", "ml1m": "configs/data/ml1m.yaml"}
LABEL = {"ml100k": "ML-100K", "ml1m": "ML-1M"}
TITANIC, TOY_STORY = "m.0dr_4", "m.0dyb1"

S_CONV = CONVENTIONS
S1, S2, S3 = "1. Bộ dữ liệu và nguồn", "2. Làm sạch", "3. Rating theo phiên và Δt"
S4, S5, S6, S7 = "4. Chia theo mốc t1/t2", "5. Liên kết thực thể", "6. KG theo thời gian", "7. Baseline"

# chart tokens (light surface; the blue/orange pair passes the dataviz palette validator)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COLOR = {"ml100k": BLUE, "ml1m": ORANGE}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fmt_param(v) -> str:
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    return str(v)


def fmt_config(cfg: dict) -> str:
    if cfg == {"window_days": None}:
        return "toàn bộ train (không giới hạn cửa sổ)"
    return ", ".join(f"{k} = {fmt_param(v)}" for k, v in sorted(cfg.items()))


# ---------------------------------------------------------------------------------------------------------------


def load_dataset(key: str) -> dict:
    cfg = load_config(ROOT / DATASETS[key])
    inter, items = load(cfg["name"], ROOT / cfg["raw_dir"])
    split, meta = load_saved({**cfg, "processed_dir": str(ROOT / cfg["processed_dir"])})
    proc = ROOT / cfg["processed_dir"]
    metas = {v: json.loads((proc / "kg" / v / "kg_meta.json").read_text(encoding="utf-8")) for v in kg_variants()}
    return {
        "cfg": cfg,
        "run_name": cfg.get("run_name", cfg["name"]),
        "inter": inter,
        "items": items,
        "split": split,
        "meta": meta,
        "metas": metas,
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "dropped": pd.read_csv(proc / "kg/default/dropped.csv", dtype=str),
        "entities": pd.read_csv(proc / "kg/static/entities.csv"),
        "triples": {
            (v, c): pd.read_csv(proc / f"kg/{v}/kg_{c}.tsv", sep="\t")
            for v in ("static", "default")
            for c in ("t1", "t2")
        },
    }


def kg_variants() -> tuple[str, ...]:
    return ("default", "drop_undated", "static")


def add_conventions(n: Numbers) -> None:
    src = "hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính)"
    n.add(S_CONV, "conv.first_window", f"{FIRST_WINDOW_S // HOUR} giờ kể từ rating đầu tiên của user", src)
    n.add(S_CONV, "conv.session_gap", f"phiên kết thúc khi nghỉ quá {SESSION_GAP_S // 60} phút", src)
    n.add(S_CONV, "conv.session_gap_alt", f"ngưỡng nhạy: {SESSION_GAP_ALT_S // 60} phút", src)
    n.add(S_CONV, "conv.spread_thresholds", ", ".join(label for label, _ in SPREADS.values()), src)
    level = 2 * NormalDist().cdf(Z95) - 1  # the two-sided level that Z95 stands for
    n.add(S_CONV, "conv.wilson_level", f"khoảng tin cậy Wilson {round(100 * level)}% (hai phía)", src)
    n.add(S_CONV, "conv.top_item_fractions", " và ".join(f"{round(100 * f)}%" for f in TOP_FRACS), src)
    n.add(S_CONV, "conv.positive_rating", f"rating ≥ {POSITIVE_RATING}", src)
    n.add(S_CONV, "conv.seeds", f"5 seed ({SEEDS})", src)
    n.add(S_CONV, "conv.k", " và ".join(str(k) for k in KS), src)
    n.add(S_CONV, "conv.timezone", "Mọi ngày và mọi mốc thời gian tính theo UTC", src)
    n.add(
        S_CONV,
        "conv.year_only_dates",
        "ngày chỉ có năm được đọc là 31/12 của năm đó; ngày chỉ có tháng là ngày cuối tháng (quyết định 0004 và 0006)",
        "src/tkgat/kg/temporal.py (end_of_period)",
    )
    n.add(
        S_CONV,
        "conv.window_rounding",
        f"Mọi số ngày (khoảng thời gian của dữ liệu và độ dài cửa sổ) = chênh lệch thời gian (giây) chia {DAY}, "
        "làm tròn đến số nguyên gần nhất; val = [t1, t2), test = từ t2 đến rating cuối cùng của dữ liệu gốc",
        src,
    )
    n.add(
        S_CONV,
        "conv.item_bases",
        "có ba cơ sở đếm item: item được rating (dữ liệu gốc); item trong danh mục liên kết (sau làm sạch); "
        "item trong train (sau 5-core)",
        src,
    )


def add_checksums(n: Numbers) -> None:
    path = ROOT / "data" / "checksums.json"
    src = "data/checksums.json và data/raw/ (SHA-256)"
    if not path.exists():
        for k in ("files", "ok", "missing", "mismatch"):
            n.add(S1, f"checksum.{k}", MISSING, src)
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    status = {"ok": 0, "missing": 0, "mismatch": 0}
    for name, digest in expected.items():
        p = ROOT / "data" / "raw" / name
        status["missing" if not p.exists() else "ok" if registry.sha256_file(p) == digest else "mismatch"] += 1
    n.add(S1, "checksum.files", vn_int(len(expected)), src)
    for k, v in status.items():
        n.add(S1, f"checksum.{k}", vn_int(v), src)


def add_overview_and_cleaning(n: Numbers, key: str, d: dict) -> None:
    ov, meta = eda.overview(d["inter"]), d["meta"]
    src = f"data/raw/{d['cfg']['name']} qua eda.overview"
    n.add(S1, f"{key}.users", vn_int(ov["users"]), src + " (cơ sở: user có ít nhất một rating)")
    n.add(S1, f"{key}.rated_items", vn_int(ov["items"]), src + " (cơ sở: item có ít nhất một rating)")
    n.add(S1, f"{key}.ratings", vn_int(ov["ratings"]), src)
    n.add(S1, f"{key}.density", vn_pct(ov["density"]), src + " (rating / (user × item được rating))")
    n.add(
        S1,
        f"{key}.positive_rating_share",
        vn_pct(float((d["inter"]["rating"] >= POSITIVE_RATING).mean())),
        src + f" (rating ≥ {POSITIVE_RATING} trên mọi rating gốc)",
    )
    n.add(S1, f"{key}.first_date", vn_date(ov["first"]), src)
    n.add(S1, f"{key}.last_date", vn_date(ov["last"]), src)
    n.add(S1, f"{key}.span_days", vn_int(ov["span_days"]), src + " (làm tròn đến ngày)")
    n.add(S1, f"{key}.median_ratings_per_user", vn_int(ov["median_ratings_per_user"]), src)
    years = eda.year_shares(d["inter"])
    n.add(S1, f"{key}.peak_year", str(int(years.idxmax())), src + " qua eda.year_shares")
    n.add(S1, f"{key}.peak_year_share", vn_pct(years.max()), src + " qua eda.year_shares")
    n.add(S1, f"{key}.last_year", str(int(years.index.max())), src + " qua eda.year_shares")

    msrc = f"{rel(ROOT / d['cfg']['processed_dir'])}/meta.json"
    clean = meta.get("clean", {})
    if clean:
        n.add(S2, f"{key}.untitled_items", ", ".join(str(i) for i in clean["drop_untitled"]["untitled_items"]), msrc)
        n.add(S2, f"{key}.untitled_ratings", vn_int(clean["drop_untitled"]["interactions_dropped"]), msrc)
        n.add(
            S2, f"{key}.duplicate_title_items", vn_int(clean["merge_duplicate_titles"]["duplicate_title_items"]), msrc
        )
        n.add(S2, f"{key}.remapped_ratings", vn_int(clean["merge_duplicate_titles"]["interactions_remapped"]), msrc)
    else:
        n.add(S2, f"{key}.cleaning", "không làm sạch (config clean: false)", msrc)
    n.add(S2, f"{key}.ratings_after_clean", vn_int(meta["n_raw"]), msrc)
    n.add(S2, f"{key}.duplicate_pairs_removed", vn_int(meta["n_duplicate_pairs_removed"]), msrc)
    n.add(S2, f"{key}.ratings_after_dedupe", vn_int(sum(meta["period_sizes"].values())), msrc)


def add_sessions(n: Numbers, key: str, d: dict, ctx: dict) -> None:
    x, tr, t1 = d["inter"], d["train"], d["meta"]["t1"]
    src = "data/raw qua eda.first_window_share / eda.sessions"
    n.add(S3, f"{key}.first_24h_share", vn_pct(eda.first_window_share(x, FIRST_WINDOW_S)), src)
    n.add(S3, f"{key}.first_calendar_day_share", vn_pct(eda.first_calendar_day_share(x)), src)
    n.add(S3, f"{key}.one_day_users", vn_pct(float((eda.active_days(x) == 1).mean())), src + " (eda.active_days)")
    s60, s30 = eda.sessions(x, SESSION_GAP_S), eda.sessions(x, SESSION_GAP_ALT_S)
    multi = s60[s60["n_ratings"] > 1]
    n.add(S3, f"{key}.sessions_60", vn_int(len(s60)), src)
    n.add(S3, f"{key}.sessions_30", vn_int(len(s30)), src)
    n.add(S3, f"{key}.sessions_per_user_median", vn_dec(float(s60.groupby("user").size().median()), 0), src)
    n.add(S3, f"{key}.session_median_minutes", vn_dec(float(multi["duration_s"].median() / 60), 1), src)

    tsrc = f"{rel(ROOT / d['cfg']['processed_dir'])}/train.csv qua eda.node_time_spread (mốc t1)"
    gaps = eda.consecutive_gaps(tr, by="user")
    n.add(S3, f"{key}.gap_under_1min", vn_pct(float((gaps < 60).mean())), tsrc + ", eda.consecutive_gaps")
    n.add(S3, f"{key}.gap_under_1h", vn_pct(float((gaps < HOUR).mean())), tsrc + ", eda.consecutive_gaps")
    for by in ("user", "item"):
        sp = eda.node_time_spread(tr, by=by, ref=t1)
        ctx[(key, "spread", by)] = sp["spread_s"]
        for slug, (label, thr) in SPREADS.items():
            share = eda.share_below(sp["spread_s"], {label: thr})[label]
            n.add(S3, f"{key}.{by}_spread_under_{slug}", vn_pct(share), tsrc + f", độ trải < {label}")
        med = float(sp["std_days"].median())
        if by == "user":
            n.add(S3, f"{key}.user_std_minutes_median", vn_dec(med * 1440, 0), tsrc)
        else:
            n.add(S3, f"{key}.item_std_days_median", vn_dec(med, 0), tsrc)
    ctx[(key, "weekly")] = eda.weekly_counts(x)


def cold_user_origin(d: dict) -> dict[str, dict[str, float]]:
    """Where the positives dropped because their user is not in train come from.

    A "new" user is one whose first rating (any value, after cleaning and deduplication) is at or after t1, so
    it has no interaction before the cut-off. The split is replayed as ``temporal_split`` does and the count of
    dropped positives is checked against meta.json.
    """
    cfg, m = d["cfg"], d["meta"]
    inter, items = load(cfg["name"], ROOT / cfg["raw_dir"])
    if cfg.get("clean"):
        inter, items, _ = clean(inter, items)
    df = dedupe_pairs(inter[["user", "item", "rating", "ts"]])
    t1, t2 = m["t1"], m["t2"]
    first = df.groupby("user")["ts"].min()
    train_users = set(d["train"]["user"])
    out = {}
    for part, window in (("val", (df["ts"] >= t1) & (df["ts"] < t2)), ("test", df["ts"] >= t2)):
        pos = df[window & (df["rating"] >= m["config"]["min_rating"])]
        cold = pos[~pos["user"].isin(train_users)]
        assert (
            len(cold) == m[f"{part}_dropped_cold_user_interactions"]
        ), f"{part}: cold-user positives differ from meta.json"
        users = cold["user"].unique()
        is_new = first.reindex(cold["user"]).to_numpy() >= t1
        out[part] = {
            "new_share": float(is_new.mean()),
            "users": len(users),
            "new_users": int((first.reindex(users) >= t1).sum()),
        }
    return out


def add_split(n: Numbers, key: str, d: dict) -> None:
    m, cfg = d["meta"], d["meta"]["config"]
    src = f"{rel(ROOT / d['cfg']['processed_dir'])}/meta.json"
    origin = cold_user_origin(d)
    n.add(S4, f"{key}.quantiles", f"{round(100 * cfg['q_val'])}% và {round(100 * cfg['q_test'])}% thời gian", src)
    n.add(S4, f"{key}.kcore", f"{cfg['kcore']}-core, tính trên train", src)
    n.add(S4, f"{key}.t1", vn_date(pd.to_datetime(m["t1"], unit="s").date()), src)
    n.add(S4, f"{key}.t2", vn_date(pd.to_datetime(m["t2"], unit="s").date()), src)
    n.add(S4, f"{key}.val_window_days", vn_int(eda.days(m["t2"] - m["t1"])), src + " (làm tròn đến ngày)")
    last = int(d["inter"]["ts"].max())
    n.add(
        S4,
        f"{key}.test_window_days",
        vn_int(eda.days(last - m["t2"])),
        src + " và ngày cuối của data/raw (làm tròn đến ngày)",
    )
    n.add(S4, f"{key}.train_positives_before_kcore", vn_int(m["train_pos_before_kcore"]), src)
    n.add(S4, f"{key}.train_positives", vn_int(m["train_pos"]), src)
    n.add(S4, f"{key}.train_users", vn_int(m["n_users"]), src + " (cơ sở: user trong train sau 5-core)")
    n.add(S4, f"{key}.train_items", vn_int(m["n_items"]), src + " (cơ sở: item trong train sau 5-core)")
    for part in ("val", "test"):
        n.add(S4, f"{key}.{part}_positives_before_filter", vn_int(m[f"{part}_pos_before_filter"]), src)
        n.add(S4, f"{key}.{part}_dropped_cold_user", vn_int(m[f"{part}_dropped_cold_user_interactions"]), src)
        n.add(S4, f"{key}.{part}_dropped_cold_item", vn_int(m[f"{part}_dropped_cold_item_interactions"]), src)
        n.add(S4, f"{key}.{part}_cold_items", vn_int(m[f"{part}_cold_items"]), src)
        o = origin[part]
        osrc = "tính lại từ data/raw theo temporal_split, đối chiếu meta.json (user mới = rating đầu tiên từ t1)"
        n.add(
            S4,
            f"{key}.{part}_cold_user_new_share",
            vn_pct(o["new_share"]),
            osrc + "; tỷ lệ positive bị loại vì user mới",
        )
        n.add(
            S4,
            f"{key}.{part}_cold_user_old_share",
            vn_pct(1 - o["new_share"]),
            osrc + "; phần thuộc user có rating trước t1",
        )
        n.add(S4, f"{key}.{part}_cold_users", vn_int(o["users"]), osrc + "; số user bị loại")
        n.add(S4, f"{key}.{part}_cold_users_new", vn_int(o["new_users"]), osrc + "; trong đó user mới")
        n.add(S4, f"{key}.{part}_positives", vn_int(m[f"{part}_pos"]), src)
        n.add(S4, f"{key}.{part}_users", vn_int(m[f"{part}_users"]), src)
        n.add(S4, f"{key}.{part}_kept_share", vn_pct(m[f"{part}_pos"] / m[f"{part}_pos_before_filter"]), src)
    n.add(
        S4,
        f"{key}.check_split_problems",
        vn_int(len(check_split(d["split"]))),
        "tkgat.data.leakage.check_split chạy trên split đã lưu",
    )


def add_linking(n: Numbers) -> None:
    path = ROOT / "data" / "processed" / "links" / "linking_report.json"
    src = rel(path)
    rep = json.loads(path.read_text(encoding="utf-8"))
    for key in DATASETS:
        r = rep[key]
        n.add(
            S5,
            f"{key}.link_catalog_items",
            vn_int(r["items"]),
            src + " (cơ sở: danh mục phim sau làm sạch, kể cả item chưa có rating)",
        )
        n.add(
            S5, f"{key}.link_freebase_items", vn_int(r["freebase_items"]), src + " (cơ sở: danh mục phim sau làm sạch)"
        )
        n.add(S5, f"{key}.link_freebase_item_share", vn_pct(r["freebase_item_share"]), src)
        n.add(S5, f"{key}.link_freebase_rating_share", vn_pct(r["freebase_rating_share"]), src)
        n.add(S5, f"{key}.link_yago_item_share", vn_pct(r["yago_item_share"]), src)
    rules = rep["ml100k"]["match_rules"]
    names = {
        "title+year": "title_year",
        "aka_title+year": "aka_title_year",
        "contains+year": "contains_year",
        "title+year±1": "title_year_pm1",
        "<NA>": "unmatched",
    }
    for rule, count in rules.items():
        n.add(S5, f"ml100k.link_rule.{names.get(rule, rule)}", vn_int(count), src + " (match_rules)")
    add_link_check(n, ROOT / "results" / "linking_manual_check.json")


def add_link_check(n: Numbers, check: Path) -> None:
    """Manual check of the entity links (item 15): the json, and the sample csv next to it when it exists."""
    csrc = "results/linking_manual_check.json (việc 15)"
    if check.exists():
        c = json.loads(check.read_text(encoding="utf-8"))
        n.add(S5, "link_check.n_checked", vn_int(c["n_checked"]), csrc)
        n.add(S5, "link_check.n_wrong", vn_int(c["n_wrong"]), csrc)
        n.add(S5, "link_check.error_rate", vn_pct(c["n_wrong"] / c["n_checked"]), csrc)
        low, high = wilson_interval(c["n_wrong"], c["n_checked"])
        wsrc = csrc + ", khoảng tin cậy Wilson 95% của tỉ lệ sai"
        n.add(S5, "link_check.wilson_low", vn_pct(low), wsrc)
        n.add(S5, "link_check.wilson_high", vn_pct(high), wsrc)
        sample = check.with_name("linking_manual_check_sample.csv")
        if sample.exists():
            counts = sample_counts(sample)
            if (counts["n"], counts["n_wrong"]) != (c["n_checked"], c["n_wrong"]):
                raise SystemExit(f"{check} and {sample} disagree: {c} against {counts}")
            ssrc = "results/linking_manual_check_sample.csv (cột needs_human_check)"
            n.add(S5, "link_check.n_human", vn_int(counts["n_human"]), ssrc)
            n.add(S5, "link_check.n_auto", vn_int(counts["n"] - counts["n_human"]), ssrc)
        else:
            for k in ("n_human", "n_auto"):
                n.add(S5, f"link_check.{k}", MISSING, "results/linking_manual_check_sample.csv")
    else:
        for k in ("n_checked", "n_wrong", "error_rate", "wilson_low", "wilson_high", "n_human", "n_auto"):
            n.add(S5, f"link_check.{k}", MISSING, csrc)


def add_kg(n: Numbers, key: str, d: dict, facts: pd.DataFrame, ctx: dict) -> None:
    meta = d["metas"]["default"]
    src = f"{rel(ROOT / d['cfg']['processed_dir'])}/kg/*/kg_meta.json"
    n.add(S6, f"{key}.kg_items", vn_int(meta["n_items"]), src + " (cơ sở: item trong train sau 5-core)")
    n.add(S6, f"{key}.kg_items_linked", vn_int(meta["n_items_linked"]), src + " (cơ sở: item trong train)")
    n.add(
        S6, f"{key}.kg_items_with_facts", vn_int(meta["n_items_with_facts"]["t1"]), src + " (cơ sở: item trong train)"
    )
    kinds = kg.relation_date_kinds(meta)
    total = kinds["facts"].sum()
    stmt = kinds["facts"] * kinds["statement_dated"]
    n.add(S6, f"{key}.kg_facts", vn_int(total), src + " qua kg.relation_date_kinds")
    n.add(S6, f"{key}.statement_dated_share", vn_pct(float(stmt.sum() / total)), src + " qua kg.relation_date_kinds")
    n.add(
        S6,
        f"{key}.award_share_of_statement_dated",
        vn_pct(float(stmt[list(kg.AWARD_RELATIONS)].sum() / stmt.sum())),
        src + " qua kg.relation_date_kinds",
    )
    n.add(
        S6,
        f"{key}.no_date_share",
        vn_pct(float((kinds["facts"] * kinds["no_date"]).sum() / total)),
        src + " qua kg.relation_date_kinds",
    )
    vt = kg.variant_triples(d["metas"])
    for v in kg_variants():
        for c in ("t1", "t2"):
            n.add(S6, f"{key}.triples_{v}_{c}", vn_int(vt.loc[v, c]), src + " qua kg.variant_triples")
    n.add(
        S6,
        f"{key}.filtered_share_t1",
        vn_pct(1 - vt.loc["default", "t1"] / vt.loc["static", "t1"]),
        "1 − default/static tại t1",
    )
    n.add(
        S6,
        f"{key}.filtered_facts_t1",
        vn_int(vt.loc["static", "t1"] - vt.loc["default", "t1"]),
        "static − default tại t1",
    )
    nd = meta["n_dropped"]
    n.add(S6, f"{key}.dropped_future_tail_t1", vn_int(nd["t1"].get("future_tail", 0)), src)
    n.add(S6, f"{key}.dropped_future_statement_t1", vn_int(nd["t1"].get("future_statement", 0)), src)
    n.add(S6, f"{key}.dropped_undated_t1", vn_int(d["metas"]["drop_undated"]["n_dropped"]["t1"]["undated"]), src)
    n.add(S6, f"{key}.entities_only_in_t2", vn_int(meta["n_entities_only_in_t2"]), src)
    drops = kg.dropped_by_relation(meta)
    ctx[(key, "drops")] = drops

    dr = d["dropped"]
    aw = {
        c: dr[(dr["cutoff"] == c) & (dr["relation"] == "award_received") & (dr["reason"] == "future_statement")]
        for c in ("t1", "t2")
    }
    between = aw["t1"].merge(aw["t2"][["freebase_id", "tail"]], on=["freebase_id", "tail"], how="left", indicator=True)
    between = between[between["_merge"] == "left_only"]
    n.add(S6, f"{key}.award_future_t1", vn_int(len(aw["t1"])), "dropped.csv (award_received, future_statement)")
    n.add(S6, f"{key}.award_future_t2", vn_int(len(aw["t2"])), "dropped.csv (award_received, future_statement)")
    n.add(S6, f"{key}.award_between_t1_t2", vn_int(len(between)), "dropped.csv: có ở t1, không có ở t2")

    entities, train = d["entities"], d["train"]
    counts = kg.item_fact_counts(d["triples"][("static", "t1")], entities)
    dropped_t1 = dr[dr["cutoff"] == "t1"]
    for frac in TOP_FRACS:
        c = kg.popular_concentration(counts, dropped_t1, train, frac=frac)
        tag = f"{round(100 * frac)}"
        csrc = "kg.popular_concentration (static t1, dropped.csv, train.csv)"
        n.add(S6, f"{key}.top{tag}_items", vn_int(c["top_items"]), csrc)
        n.add(S6, f"{key}.top{tag}_share_all_facts", vn_pct(c["share_of_all_facts"]), csrc)
        n.add(S6, f"{key}.top{tag}_share_future_award_facts", vn_pct(c["share_of_future_award_facts"]), csrc)
        if frac == TOP_FRACS[0]:
            n.add(S6, f"{key}.future_award_facts_t1", vn_int(c["future_award_facts"]), csrc)
            n.add(S6, f"{key}.future_award_films_t1", vn_int(c["films_losing_award_facts"]), csrc)

    asrc = "kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share"
    shares = {}
    for cut, part in (("t1", "val"), ("t2", "test")):
        items = kg.items_with_filtered_facts(d["triples"][("static", cut)], d["triples"][("default", cut)], entities)
        shares[part] = kg.positive_share(d[part], items)
        if part == "val":
            n.add(S6, f"{key}.affected_items", vn_int(len(items)), asrc)
            n.add(S6, f"{key}.affected_item_share", vn_pct(len(items) / meta["n_items"]), asrc)
        n.add(S6, f"{key}.affected_{part}_positive_share", vn_pct(shares[part]), asrc)
    ctx[(key, "affected")] = shares


def add_kg_global(n: Numbers, facts: pd.DataFrame, d100: dict) -> None:
    fsrc = "data/raw/wikidata/facts.tsv"
    snap = d100["metas"]["default"]["wikidata_snapshot"]
    n.add(S6, "kg.snapshot", vn_date(snap), "kg_meta.json (wikidata_snapshot)")
    n.add(S6, "kg.facts_total", vn_int(len(facts)), fsrc)
    n.add(S6, "kg.relations", vn_int(facts["relation"].nunique()), fsrc)
    n.add(S6, "kg.films_with_facts", vn_int(facts["freebase_id"].nunique()), fsrc)
    by_prec, jan1 = kg.date_precision(facts)
    n.add(S6, "kg.point_in_time_count", vn_int(by_prec.sum()), fsrc + " qua kg.date_precision")
    n.add(S6, "kg.point_in_time_year_only_share", vn_pct(float(by_prec.get("year", 0) / by_prec.sum())), fsrc)
    for col in ("film_publication", "tail_inception", "tail_birth"):
        n.add(S6, f"kg.{col}_jan1_share", vn_pct(float(jan1.loc[col, "share_jan1"])), fsrc + " (đuôi -01-01)")

    dropped = d100["dropped"]
    links = pd.read_csv(ROOT / "data/processed/links/ml100k_links.csv", dtype={"freebase_id": str})
    title = dict(zip(links["freebase_id"], links["title"], strict=True))
    tit = kg.film_fact_status(facts[facts["freebase_id"] == TITANIC], dropped)
    aw = tit[tit["relation"].isin(kg.AWARD_RELATIONS)]
    added = aw[aw["status"] == "added_t2"]
    both = aw[aw["status"] == "dropped_both"]
    tsrc = "kg.film_fact_status (facts.tsv, dropped.csv của ML-100K)"
    n.add(S6, "titanic.title", title[TITANIC], "ml100k_links.csv")
    n.add(S6, "titanic.facts", vn_int(len(tit)), tsrc)
    n.add(S6, "titanic.missing_t1", vn_int((tit["status"] != "kept_t1").sum()), tsrc)
    n.add(S6, "titanic.awards_added_t2", vn_int(len(added)), tsrc)
    oscars = int(added["tail_label"].str.startswith("Academy Award").sum())
    n.add(S6, "titanic.oscars_added_t2", vn_int(oscars), tsrc)
    n.add(
        S6,
        "titanic.awards_date",
        ", ".join(vn_date(x) for x in sorted({str(x) for x in added["statement_date"]})),
        tsrc,
    )
    dates = both["statement_date"].dropna().astype(str)
    missing = tit[tit["status"] != "kept_t1"]
    other = missing[~missing["relation"].isin(kg.AWARD_RELATIONS)]
    assert len(missing) == len(added) + len(both) + len(other), "Titanic: facts missing at t1 do not add up"
    n.add(S6, "titanic.missing_t1_other", vn_int(len(other)), tsrc + " (fact không thuộc award_received/nominated_for)")
    n.add(
        S6,
        "titanic.missing_t1_other_facts",
        "; ".join(
            f"{r.relation} → {r.tail_label} (ngày tồn tại của đích: {vn_date(r.tail_date)})"
            if r.tail_date
            else f"{r.relation} → {r.tail_label}"
            for r in other.itertuples()
        ),
        tsrc,
    )
    n.add(S6, "titanic.dropped_both", vn_int(len(both)), tsrc)
    n.add(S6, "titanic.dropped_both_year_only", vn_int(dates.str.endswith("-12-31").sum()), tsrc)
    toy = kg.film_fact_status(facts[facts["freebase_id"] == TOY_STORY], dropped)
    seq = toy[(toy["status"] != "kept_t1") & (toy["relation"] == "followed_by")].iloc[0]
    n.add(S6, "toystory.title", title[TOY_STORY], "ml100k_links.csv")
    n.add(S6, "toystory.facts_lost_t1", vn_int((toy["status"] != "kept_t1").sum()), tsrc)
    n.add(S6, "toystory.sequel", seq["tail_label"], tsrc)
    n.add(S6, "toystory.sequel_date", vn_date(seq["tail_date"]), tsrc)


def add_baselines(n: Numbers, reg: pd.DataFrame, key: str, d: dict | None, run_name: str) -> None:
    for tag in ("mostpop", "mostpop_recent", "bprmf"):
        src = f"results/registry.csv qua registry.load(), {tag}, dataset = {run_name}"
        summary_path = ROOT / "results" / "runs" / run_name / f"{tag}_summary.json"
        keys = ("selected", "test_recall20", "test_ndcg20", "n_seeds", "val_users", "test_users")
        rows = reg[(reg["dataset"] == run_name) & (reg["model"] == tag)]
        if rows.empty or not summary_path.exists():
            for k in keys:
                n.add(S7, f"{key}.{tag}.{k}", MISSING, src)
            continue
        selected = json.loads(summary_path.read_text(encoding="utf-8"))["selected"]
        cfg_json = json.dumps(selected, sort_keys=True)
        test = rows[(rows["phase"] == "test") & (rows["config"] == cfg_json)]
        val = rows[(rows["phase"] == "val") & (rows["config"] == cfg_json)]
        ssrc = src + "; cấu hình chọn = trường selected của summary (decision 0007)"
        n.add(S7, f"{key}.{tag}.selected", fmt_config(selected), ssrc)
        for metric, name in (("recall@20", "test_recall20"), ("ndcg@20", "test_ndcg20")):
            std = float(test[metric].std(ddof=1)) if len(test) > 1 else None
            n.add(S7, f"{key}.{tag}.{name}", vn_pm(float(test[metric].mean()), std), ssrc + " (sample std, ddof=1)")
        n.add(S7, f"{key}.{tag}.n_seeds", vn_int(len(test)), ssrc)
        n.add(S7, f"{key}.{tag}.val_users", vn_int(val["n_users"].iloc[0]), ssrc)
        n.add(S7, f"{key}.{tag}.test_users", vn_int(test["n_users"].iloc[0]), ssrc)
        if tag == "mostpop_recent":
            grid = [
                p["window_days"]
                for p in yaml.safe_load((ROOT / "configs/model/mostpop_recent.yaml").read_text())["grid"]
            ]
            w = selected["window_days"]
            n.add(
                S7,
                f"{key}.mostpop_recent.grid",
                ", ".join(str(g) for g in grid) + " ngày",
                "configs/model/mostpop_recent.yaml",
            )
            for g in grid:
                v = rows[(rows["phase"] == "val") & (rows["config"] == json.dumps({"window_days": g}))]
                n.add(
                    S7,
                    f"{key}.mostpop_recent.val_recall20.w{g}",
                    vn_dec(float(v["recall@20"].iloc[0]), 4),
                    src + " (phase val)",
                )
            edge = (
                "có (nhỏ nhất của grid)" if w == min(grid) else "có (lớn nhất của grid)" if w == max(grid) else "không"
            )
            n.add(S7, f"{key}.mostpop_recent.on_grid_edge", edge, "so selected với configs/model/mostpop_recent.yaml")


def add_registry_notes(n: Numbers, reg_all: pd.DataFrame, reg: pd.DataFrame) -> None:
    src = "results/registry.csv qua registry.load()"
    n.add(S7, "registry.rows_all", vn_int(len(reg_all)), src + "(latest=False)")
    n.add(S7, "registry.rows_latest", vn_int(len(reg)), src)
    n.add(S7, "registry.rows_dirty_label", vn_int(reg_all["git_commit"].astype(str).str.endswith("-dirty").sum()), src)
    n.add(S7, "registry.key", "dataset, split_sha256, model, config, seed, phase", "registry.RUN_KEY")
    n.add(S7, "registry.splits", vn_int(reg["split_sha256"].nunique()), src)


# ---------------------------------------------------------------------------------------------------------------


def style_axes() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
            "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
            "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
            "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 2,
            "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
        }
    )  # fmt: skip


def draw_figures(out: Path, data: dict[str, dict], ctx: dict) -> list[Path]:
    style_axes()
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # Figure 1: ratings per week with t1 and t2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, key in zip(axes, DATASETS, strict=True):
        w = ctx[(key, "weekly")]
        t1, t2 = (pd.to_datetime(data[key]["meta"][k], unit="s") for k in ("t1", "t2"))
        ax.axvspan(t1, t2, color=GRID, alpha=0.9, lw=0)
        ax.axvspan(t2, w.index.max() + pd.Timedelta(days=7), color=GRID, alpha=0.45, lw=0)
        ax.plot(w.index, w.values, color=COLOR[key])
        if key == "ml1m":
            ax.set_yscale("log")  # one spike around t1 would flatten the rest on a linear axis
        top = ax.get_ylim()[1]
        for t, lab, ha in ((t1, "t1 ", "right"), (t2, " t2", "left")):
            ax.axvline(t, color=INK2, lw=1, ls="--")
            ax.text(t, top * 0.97, lab, color=INK2, ha=ha, va="top", fontsize=8)
        ax.set_title(f"{LABEL[key]}: số rating mỗi tuần")
        ax.set_ylabel("rating / tuần" + (" (thang log)" if key == "ml1m" else ""))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    paths.append(out / "fig1_rating_theo_tuan.png")
    fig.savefig(paths[-1], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: time spread of the train edges of one user and of one item
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    floor = 1 / 1440
    for ax, by, title in ((axes[0], "user", "Các cạnh của một user"), (axes[1], "item", "Các cạnh của một item")):
        for key in DATASETS:
            v = np.sort(np.maximum(ctx[(key, "spread", by)].to_numpy() / DAY, floor))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLOR[key], label=LABEL[key])
        for t, lab in ((thr / DAY, label) for label, thr in SPREADS.values()):
            ax.axvline(t, color=MUTED, lw=0.8, ls=":")
            ax.text(
                t,
                0.03 if by == "user" else 0.5,
                f" {lab} ",
                color=MUTED,
                fontsize=7,
                ha="left" if by == "user" else "right",
            )
        ax.set_xscale("log")
        ax.set_xlim(floor, 1500)
        ax.set_xlabel("độ trải max(ts) − min(ts) (ngày, log)")
        ax.set_title(title)
    axes[0].set_ylabel("tỷ lệ node có độ trải ≤ x")
    axes[0].legend(loc="center right")
    fig.tight_layout()
    paths.append(out / "fig2_do_trai_thoi_gian.png")
    fig.savefig(paths[-1], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # Figure 3: val/test positives kept or dropped by the split
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rows, labels = [], []
    for key in DATASETS:
        for part in ("val", "test"):
            m = data[key]["meta"]
            tot = m[f"{part}_pos_before_filter"]
            rows.append(
                (
                    m[f"{part}_pos"] / tot,
                    m[f"{part}_dropped_cold_user_interactions"] / tot,
                    m[f"{part}_dropped_cold_item_interactions"] / tot,
                )
            )
            labels.append(f"{LABEL[key]} · {part}")
    left = np.zeros(len(rows))
    for i, (color, name) in enumerate(((AQUA, "còn lại"), (YELLOW, "loại: user mới"), (PINK, "loại: item mới"))):
        share = np.array([r[i] for r in rows])
        ax.barh(labels, share, left=left, color=color, label=name, height=0.55, edgecolor=SURFACE, linewidth=2)
        for y, (s, l0) in enumerate(zip(share, left, strict=True)):
            if s >= 0.06:
                ax.text(l0 + s / 2, y, vn_pct(s, 0), ha="center", va="center", fontsize=8, color=INK)
        left += share
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("tỷ lệ positive của cửa sổ (trước khi lọc)")
    ax.set_title("Positive val/test còn lại và bị loại bởi user hoặc item mới")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    paths.append(out / "fig3_positive_con_lai.png")
    fig.savefig(paths[-1], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # Figure 4: facts dropped by the time filter (ML-100K) and positives on the items that lose facts
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    t = ctx[("ml100k", "drops")]
    t = t[(t["future_statement_t1"] + t["future_tail_t1"]) > 0]
    t = t.assign(total=t["future_statement_t1"] + t["future_tail_t1"]).sort_values("total")
    ax.barh(
        t.index, t["future_tail_t1"], color=BLUE, label="đích chưa tồn tại", height=0.65, edgecolor=SURFACE, linewidth=2
    )
    ax.barh(
        t.index,
        t["future_statement_t1"],
        left=t["future_tail_t1"],
        color=ORANGE,
        label="ngày statement sau mốc",
        height=0.65,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for y, v in enumerate(t["total"]):
        ax.text(v, y, f" {v}", va="center", color=INK2, fontsize=7)
    ax.set_title("ML-100K: fact bị loại tại t1")
    ax.set_xlabel("số fact")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    labels, vals = [], []
    for key in DATASETS:
        for part in ("val", "test"):
            labels.append(f"{LABEL[key]}\n{part}")
            vals.append(ctx[(key, "affected")][part])
    ax.bar(labels, vals, color=INK2, width=0.55)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.01, vn_pct(v, 0), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("tỷ lệ positive")
    ax.set_title("Positive trên item bị lọc")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    paths.append(out / "fig4_fact_bi_loai.png")
    fig.savefig(paths[-1], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------------------------------------------


def build(root_registry: Path) -> tuple[Numbers, dict, dict]:
    n, ctx = Numbers(), {}
    add_conventions(n)
    add_checksums(n)
    data = {key: load_dataset(key) for key in DATASETS}
    facts = pd.read_csv(
        ROOT / "data/raw/wikidata/facts.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
    )
    for key, d in data.items():
        add_overview_and_cleaning(n, key, d)
        add_sessions(n, key, d, ctx)
        add_split(n, key, d)
    add_linking(n)
    add_kg_global(n, facts, data["ml100k"])
    for key, d in data.items():
        add_kg(n, key, d, facts, ctx)
    reg_all = registry.load(root_registry, latest=False)
    reg = registry.load(root_registry)
    for key, d in data.items():
        add_baselines(n, reg, key, d, d["run_name"])
    add_registry_notes(n, reg_all, reg)
    return n, data, ctx


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "data-report-numbers.md"))
    ap.add_argument("--figures-dir", default=str(ROOT / "docs" / "figures"), help='a folder, or "none"')
    ap.add_argument("--registry", default=str(ROOT / "results" / "registry.csv"))
    args = ap.parse_args(argv)

    numbers, data, ctx = build(Path(args.registry))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(numbers.render(), encoding="utf-8", newline="\n")
    print(f"{len(numbers.rows)} numbers -> {out}")
    missing = [k for _, k, v, _ in numbers.rows if v == MISSING]
    if missing:
        print(f'"{MISSING}": ' + ", ".join(missing))
    if args.figures_dir != "none":
        for p in draw_figures(Path(args.figures_dir), data, ctx):
            print("figure ->", p)


if __name__ == "__main__":
    main()
