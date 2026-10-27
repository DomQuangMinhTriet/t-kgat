"""Enrich the generated MovieLens-100K KGAT datasets with Wikidata facts.

Inputs (defaults are relative to the repository root):

* ``knowledge_graph_attention_network/Data/ml-100k-chrono``;
* ``knowledge_graph_attention_network/Data/ml-100k-chrono-timestamp``;
* MovieLens-1M ``movies.dat``;
* KB4Rec ``ml2fb.txt``;
* the frozen Wikidata ``facts.tsv`` snapshot.

The script writes two new datasets instead of mutating the generated base data:

* ``ml-100k-chrono-wikidata``;
* ``ml-100k-chrono-timestamp-wikidata``.

MovieLens-100K items are matched to MovieLens-1M by title and year, then to a
Freebase id through KB4Rec. Wikidata facts are selected by that Freebase id.
The existing item/entity/relation ids are preserved; Wikidata relations and
tail entities are appended deterministically.

``kg_final.txt`` contains the MovieLens metadata KG plus Wikidata facts valid
strictly before the validation cut-off. This is the training-safe KG consumed
by the original KGAT loader. ``kg_test.txt`` contains the corresponding graph
at the test cut-off for later experiments with a time-varying KG. Undated facts
are kept by default; pass ``--drop-undated`` for the sensitivity variant.

The two enriched datasets are checked to be byte-identical for every KG and
mapping file. Rating and timestamp files are copied unchanged.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

WIKIDATA_RELATIONS = (
    "director",
    "screenwriter",
    "cast_member",
    "producer",
    "composer",
    "director_of_photography",
    "genre",
    "country_of_origin",
    "original_language",
    "production_company",
    "distributed_by",
    "based_on",
    "follows",
    "followed_by",
    "award_received",
    "nominated_for",
    "narrative_location",
    "main_subject",
)
TAIL_DATE_FIELDS = ("tail_publication", "tail_inception", "tail_birth")
VALID_PRECISIONS = {8, 9, 10, 11}
ARTICLES = r"The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El|Los|Las|Det|Den|Une|Un"
TRAIL_ARTICLE = re.compile(rf"^(.*), ({ARTICLES})$")
YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
AKA = re.compile(r"\s*\(([^()]*)\)\s*$")

ENRICHED_COMMON_FILES = (
    "entity_list.txt",
    "relation_list.txt",
    "item_list.txt",
    "kg_movielens.txt",
    "kg_wikidata_train.txt",
    "kg_wikidata_test.txt",
    "kg_final.txt",
    "kg_test.txt",
    "item_wikidata_links.tsv",
    "wikidata_entity_labels.tsv",
    "wikidata_fact_details.tsv",
    "wikidata_enrichment_meta.json",
)


@dataclass(frozen=True)
class Movie:
    item: int
    title: str
    year: int | None
    key: str
    key_full: str
    key_aka: str


@dataclass(frozen=True)
class Link:
    raw_item: int
    item: int
    title: str
    year: int | None
    ml1m_item: int | None
    freebase_id: str
    rule: str


@dataclass(frozen=True)
class Fact:
    item: int
    freebase_id: str
    relation: str
    tail: str
    tail_label: str
    statement_date: dt.date | None
    tail_date: dt.date | None
    raw: dict[str, str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("knowledge_graph_attention_network/Data")
    parser.add_argument("--chrono", type=Path, default=root / "ml-100k-chrono")
    parser.add_argument("--timestamp", type=Path, default=root / "ml-100k-chrono-timestamp")
    parser.add_argument("--chrono-out", type=Path, default=root / "ml-100k-chrono-wikidata")
    parser.add_argument(
        "--timestamp-out",
        type=Path,
        default=root / "ml-100k-chrono-timestamp-wikidata",
    )
    parser.add_argument(
        "--ml1m-movies",
        type=Path,
        default=Path("data-for-Phon (1)/data/raw/ml-1m/movies.dat"),
    )
    parser.add_argument(
        "--kb4rec",
        type=Path,
        default=Path("data-for-Phon (1)/data/raw/kb4rec/ml2fb.txt"),
    )
    parser.add_argument(
        "--facts",
        type=Path,
        default=Path("data-for-Phon (1)/data/raw/wikidata/facts.tsv"),
    )
    parser.add_argument(
        "--drop-undated",
        action="store_true",
        help="Drop facts with neither a statement date nor a target-existence date.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into non-empty enriched output directories.",
    )
    return parser.parse_args(argv)


def split_title(title: str) -> tuple[str, int | None]:
    match = YEAR.search(title)
    year = int(match.group(1)) if match else None
    return (YEAR.sub("", title) if match else title).strip(), year


def norm_title(title: str) -> str:
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    value = AKA.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    match = TRAIL_ARTICLE.match(value)
    if match:
        value = f"{match.group(2)} {match.group(1)}"
    return re.sub(r"[^a-z0-9]", "", value.lower())


def movie_record(item: int, title: str) -> Movie:
    title_without_year, year = split_title(title)
    aka = AKA.search(title_without_year)
    return Movie(
        item=item,
        title=title,
        year=year,
        key=norm_title(title_without_year),
        key_full=re.sub(r"[^a-z0-9]", "", title_without_year.lower()),
        key_aka=norm_title(aka.group(1)) if aka else "",
    )


def read_ml1m_movies(path: Path) -> list[Movie]:
    movies = []
    with path.open(encoding="latin-1") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("::", 2)
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected movie_id::title::genres")
            movies.append(movie_record(int(fields[0]), fields[1]))
    return movies


def unique_item(candidates: list[Movie]) -> int | None:
    return candidates[0].item if len(candidates) == 1 else None


def match_movie(source: Movie, target: list[Movie]) -> tuple[int | None, str]:
    by_key = [movie for movie in target if movie.key == source.key]
    hit = unique_item([movie for movie in by_key if movie.year == source.year])
    if hit is not None:
        return hit, "title+year"
    if source.year is not None:
        hit = unique_item([movie for movie in by_key if movie.year is not None and abs(movie.year - source.year) <= 1])
        if hit is not None:
            return hit, "title+year±1"
    by_full = [movie for movie in target if movie.key_full == source.key_full and movie.year == source.year]
    hit = unique_item(by_full)
    if hit is not None:
        return hit, "full_title+year"
    by_aka = [movie for movie in target if movie.key_aka == source.key and movie.year == source.year]
    hit = unique_item(by_aka)
    if hit is not None:
        return hit, "aka_title+year"
    if len(source.key) >= 6:
        contained = [
            movie
            for movie in target
            if movie.year == source.year
            and (source.key in movie.key or (len(movie.key) >= 6 and movie.key in source.key))
        ]
        hit = unique_item(contained)
        if hit is not None:
            return hit, "contains+year"
    return None, "unmatched"


def read_space_table(path: Path, header: list[str]) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split() != header:
        raise ValueError(f"unexpected header in {path}; expected {' '.join(header)}")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"unexpected column count in {path}")
    return rows


def read_kb4rec(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected two columns")
            mapping[int(fields[0])] = fields[1]
    return mapping


def build_links(dataset: Path, ml1m_path: Path, kb4rec_path: Path) -> list[Link]:
    item_rows = read_space_table(dataset / "item_list.txt", ["org_id", "remap_id", "freebase_id"])
    title_by_raw: dict[int, str] = {}
    with (dataset / "movie_metadata.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            title_by_raw[int(row["raw_item"])] = row["title"]

    targets = read_ml1m_movies(ml1m_path)
    freebase_by_item = read_kb4rec(kb4rec_path)
    links = []
    for raw_value, remap_value, _local_entity in item_rows:
        raw_item, item = int(raw_value), int(remap_value)
        source = movie_record(raw_item, title_by_raw[raw_item])
        ml1m_item, rule = match_movie(source, targets)
        freebase_id = freebase_by_item.get(ml1m_item, "") if ml1m_item is not None else ""
        if ml1m_item is not None and not freebase_id:
            rule = f"{rule};no_kb4rec"
        links.append(Link(raw_item, item, source.title, source.year, ml1m_item, freebase_id, rule))
    return links


def parse_precision(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def end_of_period(value: str, precision: int | None = None) -> dt.date | None:
    if not value:
        return None
    try:
        date = dt.date.fromisoformat(value[:10])
    except ValueError:
        return None
    if precision is None:
        precision = 9 if (date.month, date.day) == (1, 1) else 11
    if precision not in VALID_PRECISIONS:
        return None
    if precision == 11:
        return date
    if precision == 10:
        return date.replace(day=calendar.monthrange(date.year, date.month)[1])
    if precision == 9:
        return dt.date(date.year, 12, 31)
    return dt.date(min(date.year // 10 * 10 + 9, 9999), 12, 31)


def annotate_fact(item: int, row: dict[str, str]) -> Fact:
    point_in_time = row.get("point_in_time", "")
    precision = parse_precision(row.get("point_in_time_precision", ""))
    statement_date = end_of_period(point_in_time, precision if precision is not None else -1)
    if statement_date is None:
        statement_date = end_of_period(row.get("start_time", ""))
    tail_dates = [end_of_period(row.get(field, "")) for field in TAIL_DATE_FIELDS]
    tail_date = min((date for date in tail_dates if date is not None), default=None)
    return Fact(
        item=item,
        freebase_id=row["freebase_id"],
        relation=row["relation"],
        tail=row["tail"],
        tail_label=row.get("tail_label", ""),
        statement_date=statement_date,
        tail_date=tail_date,
        raw=row,
    )


def read_facts(path: Path, links: list[Link]) -> list[Fact]:
    items_by_freebase: dict[str, list[int]] = defaultdict(list)
    for link in links:
        if link.freebase_id:
            items_by_freebase[link.freebase_id].append(link.item)
    facts: dict[tuple[int, str, str], Fact] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"freebase_id", "relation", "tail"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} is missing required columns {sorted(required)}")
        for row in reader:
            items = items_by_freebase.get(row["freebase_id"], [])
            if not items or row["relation"] not in WIKIDATA_RELATIONS or not row["tail"]:
                continue
            for item in items:
                fact = annotate_fact(item, row)
                facts.setdefault((fact.item, fact.relation, fact.tail), fact)
    return sorted(facts.values(), key=lambda fact: (fact.item, fact.relation, fact.tail))


def cutoff_day(timestamp: int) -> dt.date:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date()


def drop_reason(fact: Fact, cutoff: dt.date, drop_undated: bool) -> str:
    if fact.statement_date is not None and fact.statement_date >= cutoff:
        return "future_statement"
    if fact.tail_date is not None and fact.tail_date >= cutoff:
        return "future_tail"
    if drop_undated and fact.statement_date is None and fact.tail_date is None:
        return "undated"
    return ""


def read_cutoffs(dataset: Path) -> tuple[int, int]:
    meta = json.loads((dataset / "dataset_meta.json").read_text(encoding="utf-8"))
    return int(meta["splits"]["val"]["first_timestamp"]), int(meta["splits"]["test"]["first_timestamp"])


def read_integer_triples(path: Path) -> list[tuple[int, int, int]]:
    triples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected three integer columns")
        triples.append(tuple(int(value) for value in fields))
    return triples


def write_integer_triples(path: Path, triples: set[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for head, relation, tail in sorted(triples):
            handle.write(f"{head} {relation} {tail}\n")


def clone_dataset(source: Path, output: Path, overwrite: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to replace enriched files")
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)


def write_links(path: Path, links: list[Link], fact_counts: Counter[str]) -> None:
    fields = [
        "raw_item",
        "item",
        "title",
        "year",
        "ml1m_item",
        "freebase_id",
        "match_rule",
        "wikidata_facts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for link in links:
            writer.writerow(
                {
                    "raw_item": link.raw_item,
                    "item": link.item,
                    "title": link.title,
                    "year": "" if link.year is None else link.year,
                    "ml1m_item": "" if link.ml1m_item is None else link.ml1m_item,
                    "freebase_id": link.freebase_id,
                    "match_rule": link.rule,
                    "wikidata_facts": fact_counts[link.freebase_id],
                }
            )


def write_space_table(path: Path, header: list[str], rows: list[list[str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(" ".join(header) + "\n")
        for row in rows:
            handle.write(" ".join(str(value) for value in row) + "\n")


def write_fact_details(
    path: Path,
    facts: list[Fact],
    train_cutoff: dt.date,
    test_cutoff: dt.date,
    drop_undated: bool,
    train_relation_names: set[str],
    train_tails: set[str],
) -> None:
    fields = [
        "item",
        "freebase_id",
        "relation",
        "tail",
        "tail_label",
        "statement_date",
        "tail_date",
        "train_status",
        "test_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for fact in facts:
            train_reason = drop_reason(fact, train_cutoff, drop_undated)
            test_reason = drop_reason(fact, test_cutoff, drop_undated)
            if not test_reason and fact.relation not in train_relation_names:
                test_reason = "relation_absent_from_train"
            if not test_reason and fact.tail not in train_tails:
                test_reason = "tail_absent_from_train"
            writer.writerow(
                {
                    "item": fact.item,
                    "freebase_id": fact.freebase_id,
                    "relation": fact.relation,
                    "tail": fact.tail,
                    "tail_label": fact.tail_label,
                    "statement_date": fact.statement_date or "",
                    "tail_date": fact.tail_date or "",
                    "train_status": train_reason or "kept",
                    "test_status": test_reason or "kept",
                }
            )


def enrich(
    source: Path,
    output: Path,
    links: list[Link],
    facts: list[Fact],
    facts_path: Path,
    train_cutoff_ts: int,
    test_cutoff_ts: int,
    drop_undated: bool,
    overwrite: bool,
) -> dict:
    clone_dataset(source, output, overwrite)
    train_cutoff, test_cutoff = cutoff_day(train_cutoff_ts), cutoff_day(test_cutoff_ts)
    train_facts = [fact for fact in facts if not drop_reason(fact, train_cutoff, drop_undated)]
    test_candidates = [fact for fact in facts if not drop_reason(fact, test_cutoff, drop_undated)]

    relations = read_space_table(source / "relation_list.txt", ["org_id", "remap_id"])
    relation_rows: list[list[str | int]] = [[row[0], int(row[1])] for row in relations]
    train_relation_names = {fact.relation for fact in train_facts}
    external_names = [name for name in WIKIDATA_RELATIONS if name in train_relation_names]
    relation_id = {row[0]: int(row[1]) for row in relations}
    for name in external_names:
        key = f"wikidata.{name}"
        relation_id[key] = len(relation_rows)
        relation_rows.append([key, relation_id[key]])

    train_facts = [fact for fact in train_facts if fact.relation in train_relation_names]
    train_tails = {fact.tail for fact in train_facts}
    test_facts = [
        fact for fact in test_candidates if fact.relation in train_relation_names and fact.tail in train_tails
    ]

    entities = read_space_table(source / "entity_list.txt", ["org_id", "remap_id"])
    entity_rows: list[list[str | int]] = [[row[0], int(row[1])] for row in entities]
    entity_id = {row[0]: int(row[1]) for row in entities}
    labels: dict[str, str] = {}
    for fact in train_facts:
        labels.setdefault(fact.tail, fact.tail_label)
    for tail in sorted(labels):
        key = f"wikidata.{tail}"
        entity_id[key] = len(entity_rows)
        entity_rows.append([key, entity_id[key]])

    def fact_triples(selected: list[Fact]) -> set[tuple[int, int, int]]:
        return {
            (
                fact.item,
                relation_id[f"wikidata.{fact.relation}"],
                entity_id[f"wikidata.{fact.tail}"],
            )
            for fact in selected
        }

    wiki_train, wiki_test = fact_triples(train_facts), fact_triples(test_facts)
    local_triples = set(read_integer_triples(source / "kg_final.txt"))
    write_integer_triples(output / "kg_movielens.txt", local_triples)
    write_integer_triples(output / "kg_wikidata_train.txt", wiki_train)
    write_integer_triples(output / "kg_wikidata_test.txt", wiki_test)
    write_integer_triples(output / "kg_final.txt", local_triples | wiki_train)
    write_integer_triples(output / "kg_test.txt", local_triples | wiki_test)

    write_space_table(output / "entity_list.txt", ["org_id", "remap_id"], entity_rows)
    write_space_table(output / "relation_list.txt", ["org_id", "remap_id"], relation_rows)

    item_rows = read_space_table(source / "item_list.txt", ["org_id", "remap_id", "freebase_id"])
    freebase_by_item = {link.item: link.freebase_id for link in links}
    enriched_items = [
        [raw_item, remap_item, freebase_by_item.get(int(remap_item), "") or local_entity]
        for raw_item, remap_item, local_entity in item_rows
    ]
    write_space_table(
        output / "item_list.txt",
        ["org_id", "remap_id", "freebase_id"],
        enriched_items,
    )

    with (output / "wikidata_entity_labels.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["entity", "wikidata_id", "label"])
        for tail in sorted(labels):
            writer.writerow([entity_id[f"wikidata.{tail}"], tail, labels[tail]])

    fact_counts = Counter(fact.freebase_id for fact in facts)
    write_links(output / "item_wikidata_links.tsv", links, fact_counts)
    write_fact_details(
        output / "wikidata_fact_details.tsv",
        facts,
        train_cutoff,
        test_cutoff,
        drop_undated,
        train_relation_names,
        train_tails,
    )

    per_relation = []
    for name in external_names:
        source_facts = [fact for fact in facts if fact.relation == name]
        per_relation.append(
            {
                "relation": name,
                "source": len(source_facts),
                "train_kept": sum(fact.relation == name for fact in train_facts),
                "test_kept": sum(fact.relation == name for fact in test_facts),
                "train_future_statement": sum(
                    fact.relation == name and drop_reason(fact, train_cutoff, drop_undated) == "future_statement"
                    for fact in facts
                ),
                "train_future_tail": sum(
                    fact.relation == name and drop_reason(fact, train_cutoff, drop_undated) == "future_tail"
                    for fact in facts
                ),
            }
        )

    meta = {
        "wikidata_snapshot": str(facts_path.as_posix()),
        "snapshot_limitation": "Wikidata is a 2026 snapshot; undated facts may still encode later knowledge.",
        "drop_undated": drop_undated,
        "train_cutoff_timestamp": train_cutoff_ts,
        "train_cutoff_day_utc": train_cutoff.isoformat(),
        "test_cutoff_timestamp": test_cutoff_ts,
        "test_cutoff_day_utc": test_cutoff.isoformat(),
        "items": len(links),
        "items_matched_to_ml1m": sum(link.ml1m_item is not None for link in links),
        "items_with_freebase": sum(bool(link.freebase_id) for link in links),
        "items_with_wikidata_facts": len({fact.item for fact in facts}),
        "source_wikidata_facts": len(facts),
        "wikidata_relations": len(external_names),
        "wikidata_entities": len(labels),
        "wikidata_train_triples": len(wiki_train),
        "wikidata_test_triples": len(wiki_test),
        "test_facts_with_unseen_tail_dropped": sum(
            fact.relation in train_relation_names and fact.tail not in train_tails for fact in test_candidates
        ),
        "local_triples": len(local_triples),
        "kg_final_triples": len(local_triples | wiki_train),
        "kg_test_triples": len(local_triples | wiki_test),
        "per_relation": per_relation,
    }
    (output / "wikidata_enrichment_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dataset_meta_path = output / "dataset_meta.json"
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    dataset_meta["wikidata_enrichment"] = meta
    dataset_meta_path.write_text(json.dumps(dataset_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def validate_output(output: Path) -> None:
    entities = read_space_table(output / "entity_list.txt", ["org_id", "remap_id"])
    relations = read_space_table(output / "relation_list.txt", ["org_id", "remap_id"])
    if [int(row[1]) for row in entities] != list(range(len(entities))):
        raise AssertionError("entity remap ids are not contiguous")
    if [int(row[1]) for row in relations] != list(range(len(relations))):
        raise AssertionError("relation remap ids are not contiguous")
    for name in ("kg_movielens.txt", "kg_wikidata_train.txt", "kg_wikidata_test.txt", "kg_final.txt", "kg_test.txt"):
        triples = read_integer_triples(output / name)
        if len(triples) != len(set(triples)):
            raise AssertionError(f"{name} contains duplicate triples")
        for head, relation, tail in triples:
            if not (0 <= head < len(entities) and 0 <= relation < len(relations) and 0 <= tail < len(entities)):
                raise AssertionError(f"{name} contains an id outside the mappings")


def validate_pair(chrono: Path, timestamp: Path) -> None:
    for name in ENRICHED_COMMON_FILES:
        if (chrono / name).read_bytes() != (timestamp / name).read_bytes():
            raise AssertionError(f"enriched datasets differ in {name}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    required = [args.chrono, args.timestamp, args.ml1m_movies, args.kb4rec, args.facts]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {', '.join(missing)}")

    base_common = ("item_list.txt", "entity_list.txt", "relation_list.txt", "kg_final.txt", "movie_metadata.tsv")
    for name in base_common:
        if (args.chrono / name).read_bytes() != (args.timestamp / name).read_bytes():
            raise AssertionError(f"base datasets differ in {name}")
    chrono_cutoffs = read_cutoffs(args.chrono)
    timestamp_cutoffs = read_cutoffs(args.timestamp)
    if chrono_cutoffs != timestamp_cutoffs:
        raise AssertionError("base datasets use different chronological cut-offs")

    links = build_links(args.chrono, args.ml1m_movies, args.kb4rec)
    facts = read_facts(args.facts, links)
    chrono_meta = enrich(
        args.chrono,
        args.chrono_out,
        links,
        facts,
        args.facts,
        *chrono_cutoffs,
        args.drop_undated,
        args.overwrite,
    )
    timestamp_meta = enrich(
        args.timestamp,
        args.timestamp_out,
        links,
        facts,
        args.facts,
        *timestamp_cutoffs,
        args.drop_undated,
        args.overwrite,
    )
    if chrono_meta != timestamp_meta:
        raise AssertionError("enrichment summaries differ between paired datasets")
    validate_output(args.chrono_out)
    validate_output(args.timestamp_out)
    validate_pair(args.chrono_out, args.timestamp_out)
    print(json.dumps(chrono_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
