"""Fetch dated knowledge-graph facts from Wikidata for the linked MovieLens items.

Films are found through their Freebase id (Wikidata property P646), which KB4Rec provides.
For every film the script downloads a fixed set of relations together with the dates needed by
decision 0004: the qualifier dates of the statement (point in time with its precision, start
time) and the
existence date of the target entity (publication date, inception, birth date).

Standard library only, so it runs on any machine with internet access:

    python scripts/04_fetch_wikidata.py                 # all linked ML-100K and ML-1M films
    python scripts/04_fetch_wikidata.py --limit 20      # quick check

Output: data/raw/wikidata/facts.tsv (+ fetch_log.json).  Re-running skips batches already saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "t-kgat-thesis/0.1 (HCMUS graduation thesis; contact: repository owner)"

# relation id -> short name used in the KG
RELATIONS = {
    "P57": "director",
    "P58": "screenwriter",
    "P161": "cast_member",
    "P162": "producer",
    "P86": "composer",
    "P344": "director_of_photography",
    "P136": "genre",
    "P495": "country_of_origin",
    "P364": "original_language",
    "P272": "production_company",
    "P750": "distributed_by",
    "P144": "based_on",
    "P155": "follows",
    "P156": "followed_by",
    "P166": "award_received",
    "P1411": "nominated_for",
    "P840": "narrative_location",
    "P921": "main_subject",
}

QUERY = """
SELECT ?fb ?pid ?val ?valLabel (MIN(?pit) AS ?point_in_time) (MIN(?pitPrec) AS ?point_in_time_precision)
       (MIN(?start) AS ?start_time)
       (MIN(?valPub) AS ?val_publication) (MIN(?valInception) AS ?val_inception) (MIN(?valBorn) AS ?val_birth)
       (MIN(?filmPub) AS ?film_publication)
WHERE {
  VALUES ?fb { %(fbs)s }
  VALUES (?claim ?ps ?pid) { %(pids)s }
  ?film wdt:P646 ?fb .
  ?film ?claim ?st . ?st ?ps ?val .
  OPTIONAL { ?st pqv:P585 [ wikibase:timeValue ?pit ; wikibase:timePrecision ?pitPrec ] }
  OPTIONAL { ?st pq:P580 ?start }
  OPTIONAL { ?val wdt:P577 ?valPub }
  OPTIONAL { ?val wdt:P571 ?valInception }
  OPTIONAL { ?val wdt:P569 ?valBorn }
  OPTIONAL { ?film wdt:P577 ?filmPub }
  OPTIONAL { ?val rdfs:label ?valLabel . FILTER(LANG(?valLabel) = "en") }
}
GROUP BY ?fb ?pid ?val ?valLabel
"""

FIELDS = [
    "freebase_id",
    "relation",
    "tail",
    "tail_label",
    "point_in_time",
    "point_in_time_precision",
    "start_time",
    "tail_publication",
    "tail_inception",
    "tail_birth",
    "film_publication",
]


def to_wd_freebase(mid: str) -> str:
    # KB4Rec writes "m.0dyb1"; Wikidata stores "/m/0dyb1"
    return "/" + mid.replace(".", "/", 1)


def build_query(mids: list[str]) -> str:
    fbs = " ".join(f'"{to_wd_freebase(m)}"' for m in mids)
    pids = " ".join(f'(p:{p} ps:{p} "{p}")' for p in RELATIONS)
    return QUERY % {"fbs": fbs, "pids": pids}


def run_query(q: str, retries: int = 4) -> list[dict]:
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            ENDPOINT, data=data, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:  # noqa: BLE001 - network errors, 429, 5xx
            wait = 10 * (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("Wikidata query failed repeatedly")


def rows_from(bindings: list[dict]) -> list[dict]:
    g = lambda b, k: b.get(k, {}).get("value", "")  # noqa: E731
    out = []
    for b in bindings:
        out.append(
            {
                "freebase_id": g(b, "fb").lstrip("/").replace("/", ".", 1),
                "relation": RELATIONS[g(b, "pid")],
                "tail": g(b, "val").rsplit("/", 1)[-1],
                "tail_label": g(b, "valLabel"),
                "point_in_time": g(b, "point_in_time")[:10],
                # Wikidata precision: 9 = year, 10 = month, 11 = day
                "point_in_time_precision": g(b, "point_in_time_precision"),
                "start_time": g(b, "start_time")[:10],
                "tail_publication": g(b, "val_publication")[:10],
                "tail_inception": g(b, "val_inception")[:10],
                "tail_birth": g(b, "val_birth")[:10],
                "film_publication": g(b, "film_publication")[:10],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--links", nargs="+", default=["data/processed/links/ml100k_links.csv", "data/processed/links/ml1m_links.csv"]
    )
    ap.add_argument("--out", default="data/raw/wikidata")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mids: list[str] = []
    for path in args.links:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("freebase_id"):
                    mids.append(row["freebase_id"])
    mids = sorted(set(mids))
    if args.limit:
        mids = mids[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facts, log_path = out / "facts.tsv", out / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"done_batches": []}
    new_file = not facts.exists()
    with facts.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new_file:
            w.writeheader()
        for start in range(0, len(mids), args.batch):
            if start in log["done_batches"]:
                continue
            chunk = mids[start : start + args.batch]
            rows = rows_from(run_query(build_query(chunk)))
            w.writerows(rows)
            f.flush()
            log["done_batches"].append(start)
            log["n_films_requested"] = len(mids)
            log_path.write_text(json.dumps(log))
            print(f"batch {start // args.batch + 1}/{-(-len(mids) // args.batch)}: {len(rows)} facts")
            time.sleep(1.0)  # be polite to the public endpoint
    print(f"done -> {facts}")


if __name__ == "__main__":
    main()
