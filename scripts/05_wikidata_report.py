"""Summarise the Wikidata facts file: coverage per relation and how many facts carry a date.

    python scripts/05_wikidata_report.py
    python scripts/05_wikidata_report.py --facts data/raw/wikidata/facts.tsv --out data/processed/wikidata_report.json

The report answers the questions decision 0004 needs before the time-consistent KG is built:
how many linked films were found on Wikidata, which relations are dense enough to keep, and which
share of facts can be placed in time (and with what precision).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

STATEMENT_DATES = ["point_in_time", "start_time"]  # when the statement itself became true
TAIL_DATES = ["tail_publication", "tail_inception", "tail_birth"]  # when the target entity came into existence
PRECISION = {"8": "decade", "9": "year", "10": "month", "11": "day"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def requested_films(link_files: list[str]) -> set[str]:
    mids: set[str] = set()
    for p in link_files:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                mids |= {r["freebase_id"] for r in csv.DictReader(f) if r.get("freebase_id")}
    return mids


def build_report(facts: pd.DataFrame, requested: set[str]) -> dict:
    facts = facts.fillna("")
    facts["statement_dated"] = facts[STATEMENT_DATES].ne("").any(axis=1)
    facts["tail_dated"] = facts[TAIL_DATES].ne("").any(axis=1)
    facts["dated"] = facts["statement_dated"] | facts["tail_dated"]
    films = set(facts["freebase_id"])
    per_rel = (
        facts.groupby("relation")
        .agg(facts=("tail", "size"), films=("freebase_id", "nunique"), tails=("tail", "nunique"),
             statement_dated=("statement_dated", "mean"), tail_dated=("tail_dated", "mean"),
             any_date=("dated", "mean"))
        .sort_values("facts", ascending=False)
    )
    per_rel["film_coverage"] = per_rel["films"] / max(len(films), 1)
    prec = facts.loc[facts["point_in_time"] != "", "point_in_time_precision"].astype(str)
    return {
        "n_facts": int(len(facts)),
        "n_films_with_facts": len(films),
        "n_films_requested": len(requested) or None,
        "films_without_facts": sorted(requested - films)[:50] if requested else [],
        "n_films_without_facts": len(requested - films) if requested else None,
        "dated_share_overall": round(float(facts["dated"].mean()), 4),
        "statement_dated_share": round(float(facts["statement_dated"].mean()), 4),
        "film_publication_known_share": round(float(facts.groupby("freebase_id")["film_publication"]
                                                   .first().ne("").mean()), 4),
        "point_in_time_precision": {PRECISION.get(k, k): int(v) for k, v in prec.value_counts().items()},
        "per_relation": per_rel.round(4).reset_index().to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/raw/wikidata/facts.tsv")
    ap.add_argument("--links", nargs="+", default=["data/processed/links/ml100k_links.csv",
                                                   "data/processed/links/ml1m_links.csv"])
    ap.add_argument("--out", default="data/processed/wikidata_report.json")
    args = ap.parse_args()

    path = Path(args.facts)
    facts = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    report = {"facts_file": str(path), "md5": md5(path), **build_report(facts, requested_films(args.links))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_relation", "films_without_facts")},
                     indent=2))
    print(pd.DataFrame(report["per_relation"]).to_string(index=False))


if __name__ == "__main__":
    main()
