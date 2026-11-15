"""Write the numbers of the ``lastfm-u25-latest`` dataset (docs/data-report-lastfm-latest-numbers.md).

    python scripts/11_numbers_latest_lastfm.py --registry <merged registry of the five result zips>
    python scripts/11_numbers_latest_lastfm.py --processed data/processed/lastfm-u25-latest --out /tmp/latest.md

``lastfm-u25-latest`` is Phon's dataset: the same users, 5-core, t1/t2 quantiles and KG pipeline as ``lastfm-u25``,
but the timestamp of a (user, item) pair is the LATEST listen, not the earliest (draft decision 0015, not accepted
yet). Its code and its registry rows are not in this repository (they carry the commit ``873ca7f`` of a separate
export), so this script is small and separate from ``scripts/07_report_numbers_lastfm.py``: it re-uses that script's
functions for everything that does not need the three KG variants (the Kaggle payload only holds ``default``), and it
never reads the ``results/registry.csv`` of this repository.

Inputs (all outside git):

  data/processed/lastfm-u25-latest/   meta.json, train/val/test.csv, kg/default/ (from the Kaggle payload zip)
  --registry                          the merged registry of the result zips (rows of ``dataset = lastfm-u25-latest``);
                                      only ``phase == "val"`` rows are read, so no test score reaches this file
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkgat import registry  # noqa: E402
from tkgat.analysis import eda  # noqa: E402
from tkgat.analysis.report import CONVENTIONS, MISSING, Numbers, vn_date, vn_dec, vn_int, vn_pct  # noqa: E402
from tkgat.kg.lastfm import FREEBASE_SNAPSHOT  # noqa: E402

RUN = "lastfm-u25-latest"
DAY = 86400
S_EDGE_L = "4. Thời gian của cạnh train (Stage 1) của bản latest"
# the source label of the registry rows written by scripts/07 (this repository's registry) and the true one here
REGISTRY_LABEL = "results/registry.csv (phase=val)"
REGISTRY_HERE = "registry gộp của các zip kết quả của Phôn (phase=val)"


def _script_07():
    """``scripts/07_report_numbers_lastfm.py`` as a module (its file name is not an identifier)."""
    spec = importlib.util.spec_from_file_location("report07", ROOT / "scripts" / "07_report_numbers_lastfm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def day_vn(ts: float) -> str:
    return vn_date(pd.Timestamp(int(ts), unit="s", tz="UTC").date().isoformat())


def add_kg_default(n: Numbers, run: str, kg_meta: dict, section: str) -> None:
    """The counts of the ``default`` KG variant only (the Kaggle payload holds no other variant)."""
    src = f"data/processed/{run}/kg/default/kg_meta.json"
    n.add(section, f"{run}.kg.entities", vn_int(kg_meta["n_entities"]), src)
    n.add(section, f"{run}.kg.relations", vn_int(kg_meta["n_relations"]), src)
    n.add(section, f"{run}.kg.default.t1", vn_int(kg_meta["n_triples"]["t1"]), src)
    n.add(section, f"{run}.kg.default.dated_t1", vn_int(kg_meta["n_dated_triples"]["t1"]), src)
    undated = 1 - kg_meta["n_dated_triples"]["t1"] / kg_meta["n_triples"]["t1"]
    n.add(section, f"{run}.kg.undated_t1_share", vn_pct(undated), src)


def add_edges_latest(n: Numbers, run: str, train: pd.DataFrame, t1: int) -> None:
    """Delta-t of the train edges (Stage 1) and the edges per week of the train window.

    ``dt_norm = (t_ref - ts) / dt_den`` with ``t_ref`` the latest train timestamp and ``dt_den`` the length of the
    train window, as ``tkgat.kg.ckg.build_ckg`` does.
    """
    src = f"data/processed/{run}/train.csv"
    ts = train["ts"].to_numpy(dtype=np.float64)
    t_ref, dt_den = float(ts.max()), float(ts.max() - ts.min())
    n.add(S_EDGE_L, f"{run}.edge.window_days", vn_dec(dt_den / DAY, 0), src + ", max - min của ts")
    n.add(S_EDGE_L, f"{run}.edge.first_day", day_vn(ts.min()), src + ", ts nhỏ nhất")
    dt_norm = np.clip(t_ref - ts, 0, None) / dt_den
    dsrc = src + ", dt_norm = (t_ref - ts) / dt_den"
    n.add(S_EDGE_L, f"{run}.edge.dt_norm_median", vn_dec(float(np.median(dt_norm))), dsrc)
    n.add(S_EDGE_L, f"{run}.edge.dt_norm_below_0_25", vn_pct(float((dt_norm < 0.25).mean())), dsrc)
    n.add(S_EDGE_L, f"{run}.edge.last_7d_share", vn_pct(float(((t_ref - ts) < 7 * DAY).mean()), 1), src)
    n.add(S_EDGE_L, f"{run}.edge.last_30d_share", vn_pct(float(((t_ref - ts) < 30 * DAY).mean()), 1), src)
    weekly = eda.weekly_counts(train)
    full = weekly.iloc[1:-1] if len(weekly) > 2 else weekly  # the first and last week of the window are partial
    wsrc = src + " qua eda.weekly_counts, bỏ tuần đầu và tuần cuối (không đủ)"
    n.add(S_EDGE_L, f"{run}.week.first_full", vn_int(int(full.iloc[0])), wsrc)
    n.add(S_EDGE_L, f"{run}.week.last_full", vn_int(int(full.iloc[-1])), wsrc)
    by_user = eda.node_time_spread(train, by="user", ref=t1)
    share = eda.share_below(by_user["spread_s"], {"1h": 3600})["1h"]
    n.add(S_EDGE_L, f"{run}.spread.user_under_1h", vn_pct(share, 2), src + " qua eda.node_time_spread, trải < 1 giờ")
    n.add(S_EDGE_L, f"{run}.spread.user_std_days_median", vn_dec(float(by_user["std_days"].median()), 1), src)


def build(processed: Path, registry_path: Path | None) -> Numbers:
    r07 = _script_07()
    n = Numbers()
    n.add(CONVENTIONS, "conv.freebase_snapshot", vn_date(FREEBASE_SNAPSHOT), "tkgat.kg.lastfm.FREEBASE_SNAPSHOT")
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    r07.add_split(n, RUN, meta)
    r07.add_last_timestamp(n, RUN, processed)
    add_edges_latest(n, RUN, pd.read_csv(processed / "train.csv"), int(meta["t1"]))
    kg_meta = json.loads((processed / "kg" / "default" / "kg_meta.json").read_text(encoding="utf-8"))
    add_kg_default(n, RUN, kg_meta, r07.S2)
    r07.add_delta_t(n, RUN, processed, {"default": kg_meta})
    if registry_path is not None and registry_path.exists():
        # validation rows only, filtered first: no test score can reach this file
        rows = registry.load(registry_path).query("phase == 'val'")
        rows = rows[rows["dataset"] == RUN]
        if len(rows):
            r07.add_registry(n, RUN, rows)
            n.rows = [(a, b, c, d.replace(REGISTRY_LABEL, REGISTRY_HERE)) for a, b, c, d in n.rows]
    order = [CONVENTIONS, r07.S1, r07.S2, r07.S3, S_EDGE_L, r07.S_RUN]
    n.rows.sort(key=lambda row: order.index(row[0]))
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed" / RUN))
    ap.add_argument("--registry", default=None, help="merged registry.csv of Phon's result zips")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data-report-lastfm-latest-numbers.md"))
    args = ap.parse_args(argv)
    numbers = build(Path(args.processed), None if args.registry is None else Path(args.registry))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        numbers.render(
            title="Số liệu của bản Last-FM `lastfm-u25-latest` (của Phôn)",
            script="scripts/11_numbers_latest_lastfm.py",
            report="notebooks/00_present_lastfm.ipynb",
            test="tests/test_latest_numbers.py",
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
