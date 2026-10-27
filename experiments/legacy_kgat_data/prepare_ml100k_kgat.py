"""Prepare chronological MovieLens-100K data for the original KGAT code.

This is a standalone, experimental data builder.  It does not use or modify the
existing tkgat preprocessing pipeline.

The script creates two datasets under ``knowledge_graph_attention_network/Data``:

``ml-100k-chrono``
    The first 10,000 interactions after a stable global timestamp sort, split
    into 8,000 train, 1,000 validation and 1,000 test interactions.  The KGAT
    ``train.txt``/``val.txt``/``test.txt`` files do not contain timestamps.
    Rating labels are stored separately as ``*_rating.txt`` using the format
    ``user item:rating item:rating ...``.

``ml-100k-chrono-timestamp``
    The exact same users, items, split and knowledge graph.  It additionally
    writes ``*_timestamp.txt`` using the aligned format
    ``user item:timestamp item:timestamp ...`` so a custom KGAT loader can
    consume time without mistaking a Unix timestamp for an item id.

The original KGAT loader only understands ``train.txt``, ``test.txt`` and
``kg_final.txt``.  It must be extended to read ``val.txt`` for early stopping
and the ``*_timestamp.txt`` sidecars for a time-aware experiment.

MovieLens-100K metadata represented in ``kg_final.txt``:

* title and IMDb reference identifiers;
* every genre flag from ``u.genre``/``u.item`` through ``has_genre``;
* exact release date plus release year, month and decade;
* exact video-release date plus year, month and decade when that field is present.

User age, age bucket, gender, occupation, ZIP code and ZIP prefix are represented
in ``user_kg_final.txt`` and retained in readable form in ``user_metadata.tsv``.
They are deliberately kept out of ``kg_final.txt`` because the original KGAT
loader has a separate user namespace; a custom loader must attach these triples
to the collaborative-filtering user nodes instead of creating duplicate users.

By default, every MovieLens rating is treated as an observed interaction so the
requested split contains exactly 8,000/1,000/1,000 rows.  Use ``--min-rating 4``
if the experiment should treat only ratings >= 4 as positive feedback; the
script will then select the earliest 10,000 qualifying interactions.

With the default selection, ``train.txt`` retains all 8,000 observed training
events so their ratings can later determine edge weights.  ``val.txt`` and
``test.txt`` contain only rating 4--5 items as recommendation ground truth.
Their ``*_rating.txt`` sidecars still retain all 1,000 raw events so rating 1--3
items can be masked or otherwise handled by the custom evaluator.

The script refuses to write into a non-empty output directory unless
``--overwrite`` is provided.  It never deletes directories and only overwrites
the files that it owns.

Example (do not run until the protocol has been reviewed):

    python scripts/prepare_ml100k_kgat.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

GENRE_COLUMN_START = 5
SPLIT_NAMES = ("train", "val", "test")
EVAL_MIN_RATING = 4
COMMON_OUTPUT_FILES = (
    "train.txt",
    "val.txt",
    "test.txt",
    "user_list.txt",
    "item_list.txt",
    "entity_list.txt",
    "relation_list.txt",
    "kg_final.txt",
    "user_kg_final.txt",
    "movie_metadata.tsv",
    "user_metadata.tsv",
    "entity_metadata.tsv",
    "train_rating.txt",
    "val_rating.txt",
    "test_rating.txt",
)


@dataclass(frozen=True)
class Interaction:
    raw_user: int
    raw_item: int
    rating: int
    timestamp: int


@dataclass(frozen=True)
class Movie:
    raw_item: int
    title: str
    release_date: str
    video_release_date: str
    imdb_url: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class User:
    raw_user: int
    age: int
    gender: str
    occupation: str
    zip_code: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data-for-Phon (1)/data/raw/ml-100k"),
        help="Directory containing u.data, u.item, u.user and u.genre.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("knowledge_graph_attention_network/Data"),
        help="Parent directory for ml-100k-chrono and ml-100k-chrono-timestamp.",
    )
    parser.add_argument("--n-interactions", type=int, default=10_000)
    parser.add_argument("--n-train", type=int, default=8_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=1_000)
    parser.add_argument(
        "--min-rating",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Minimum rating treated as an observed KGAT interaction (default: keep all ratings).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files owned by this script when output directories are non-empty.",
    )
    args = parser.parse_args(argv)

    requested = args.n_train + args.n_val + args.n_test
    if requested != args.n_interactions:
        parser.error(
            "n-train + n-val + n-test must equal n-interactions "
            f"({args.n_train} + {args.n_val} + {args.n_test} != {args.n_interactions})"
        )
    if min(args.n_interactions, args.n_train, args.n_val, args.n_test) <= 0:
        parser.error("interaction and split sizes must be positive")
    return args


def read_genres(path: Path) -> list[str]:
    genres: dict[int, str] = {}
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            name, raw_index = line.rsplit("|", 1)
            genres[int(raw_index)] = name
    if not genres:
        raise ValueError(f"no genres found in {path}")
    expected = list(range(max(genres) + 1))
    if sorted(genres) != expected:
        raise ValueError(f"genre indices in {path} are not contiguous from zero")
    return [genres[index] for index in expected]


def read_interactions(path: Path, min_rating: int) -> list[Interaction]:
    interactions: list[Interaction] = []
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected 4 fields, got {len(fields)}")
            interaction = Interaction(*(int(value) for value in fields))
            if interaction.rating >= min_rating:
                interactions.append(interaction)

    # Timestamp is the protocol key; user/item provide a deterministic order for ties.
    interactions.sort(key=lambda row: (row.timestamp, row.raw_user, row.raw_item))
    return interactions


def read_movies(path: Path, genre_names: list[str]) -> dict[int, Movie]:
    movies: dict[int, Movie] = {}
    with path.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for line_number, fields in enumerate(reader, start=1):
            expected = GENRE_COLUMN_START + len(genre_names)
            if len(fields) != expected:
                raise ValueError(f"{path}:{line_number}: expected {expected} fields, got {len(fields)}")
            raw_item = int(fields[0])
            flags = [int(value) for value in fields[GENRE_COLUMN_START:]]
            genres = tuple(name for name, flag in zip(genre_names, flags, strict=True) if flag)
            movies[raw_item] = Movie(
                raw_item=raw_item,
                title=fields[1],
                release_date=fields[2],
                video_release_date=fields[3],
                imdb_url=fields[4],
                genres=genres,
            )
    return movies


def read_users(path: Path) -> dict[int, User]:
    users: dict[int, User] = {}
    with path.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for line_number, fields in enumerate(reader, start=1):
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
            raw_user = int(fields[0])
            users[raw_user] = User(raw_user, int(fields[1]), fields[2], fields[3], fields[4])
    return users


def parse_movielens_date(value: str) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%d-%b-%Y").date()
    except ValueError as exc:
        raise ValueError(f"invalid MovieLens date {value!r}") from exc


def safe_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return key or "unknown"


def split_rows(rows: list[Interaction], n_train: int, n_val: int) -> dict[str, list[Interaction]]:
    train_end = n_train
    val_end = n_train + n_val
    return {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }


def check_chronology(parts: dict[str, list[Interaction]]) -> None:
    all_rows = [row for name in SPLIT_NAMES for row in parts[name]]
    timestamps = [row.timestamp for row in all_rows]
    if timestamps != sorted(timestamps):
        raise AssertionError("interactions are not globally ordered by non-decreasing timestamp")
    if not (
        parts["train"][-1].timestamp
        <= parts["val"][0].timestamp
        <= parts["val"][-1].timestamp
        <= parts["test"][0].timestamp
    ):
        raise AssertionError("train/validation/test time ranges are not chronological")


def make_id_maps(rows: list[Interaction]) -> tuple[dict[int, int], dict[int, int]]:
    # Sorted raw ids make the mapping deterministic and easy to audit.
    raw_users = sorted({row.raw_user for row in rows})
    raw_items = sorted({row.raw_item for row in rows})
    return (
        {raw_id: remap_id for remap_id, raw_id in enumerate(raw_users)},
        {raw_id: remap_id for remap_id, raw_id in enumerate(raw_items)},
    )


def add_entity(
    entity_key: str,
    entity_kind: str,
    entities: list[tuple[str, int, str]],
    entity_ids: dict[str, int],
) -> int:
    if entity_key not in entity_ids:
        entity_id = len(entity_ids)
        entity_ids[entity_key] = entity_id
        entities.append((entity_key, entity_id, entity_kind))
    return entity_ids[entity_key]


def build_metadata_kg(
    item_map: dict[int, int],
    movies: dict[int, Movie],
    user_map: dict[int, int],
    users: dict[int, User],
) -> tuple[
    list[tuple[str, int, str]],
    list[tuple[str, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    # User relations come first so the final item relation still gives the original KGAT loader the full
    # relation count through max(kg_final[:, 1]) + 1. User triples are loaded separately by a custom loader.
    user_relations = [
        "has_age",
        "has_age_bucket",
        "has_gender",
        "has_occupation",
        "has_zip_code",
        "has_zip_prefix",
    ]
    item_relations = [
        "has_title",
        "has_imdb_reference",
        "has_genre",
        "released_on_date",
        "released_in_year",
        "released_in_month",
        "released_in_decade",
    ]
    if any(movies[raw_item].video_release_date for raw_item in item_map):
        item_relations.extend(
            [
                "video_released_on_date",
                "video_released_in_year",
                "video_released_in_month",
                "video_released_in_decade",
            ]
        )
    relations = [*user_relations, *item_relations]
    relation_ids = {name: index for index, name in enumerate(relations)}

    # KGAT assumes item remap ids are the first entity ids and are identical.
    entities: list[tuple[str, int, str]] = []
    entity_ids: dict[str, int] = {}
    for raw_item, remap_item in sorted(item_map.items(), key=lambda pair: pair[1]):
        key = f"ml100k.item.{raw_item}"
        if remap_item != len(entity_ids):
            raise AssertionError("item remap ids must be contiguous and lead the entity vocabulary")
        entity_ids[key] = remap_item
        entities.append((key, remap_item, "item"))

    user_triples: set[tuple[int, int, int]] = set()
    for raw_user, head in user_map.items():
        profile = users.get(raw_user)
        if profile is None:
            raise ValueError(f"user {raw_user} appears in interactions but is missing from u.user")
        age_start = profile.age // 10 * 10
        attributes = [
            ("has_age", f"user.age.{profile.age}", "user_age"),
            ("has_age_bucket", f"user.age_bucket.{age_start}_{age_start + 9}", "user_age_bucket"),
            ("has_gender", f"user.gender.{safe_key(profile.gender)}", "user_gender"),
            ("has_occupation", f"user.occupation.{safe_key(profile.occupation)}", "user_occupation"),
            ("has_zip_code", f"user.zip.{safe_key(profile.zip_code)}", "user_zip_code"),
            ("has_zip_prefix", f"user.zip_prefix.{safe_key(profile.zip_code[:3])}", "user_zip_prefix"),
        ]
        for relation, entity_key, entity_kind in attributes:
            tail = add_entity(entity_key, entity_kind, entities, entity_ids)
            user_triples.add((head, relation_ids[relation], tail))

    item_triples: set[tuple[int, int, int]] = set()
    for raw_item, head in item_map.items():
        movie = movies.get(raw_item)
        if movie is None:
            raise ValueError(f"item {raw_item} appears in interactions but is missing from u.item")

        title = add_entity(f"movie.title.item_{raw_item}", "movie_title", entities, entity_ids)
        item_triples.add((head, relation_ids["has_title"], title))
        if movie.imdb_url:
            imdb = add_entity(f"movie.imdb.item_{raw_item}", "imdb_reference", entities, entity_ids)
            item_triples.add((head, relation_ids["has_imdb_reference"], imdb))

        for genre in movie.genres:
            tail = add_entity(f"genre.{safe_key(genre)}", "genre", entities, entity_ids)
            item_triples.add((head, relation_ids["has_genre"], tail))

        for prefix, raw_date in (
            ("released", movie.release_date),
            ("video_released", movie.video_release_date),
        ):
            date = parse_movielens_date(raw_date)
            if date is None:
                continue
            values = {
                "date": date.isoformat(),
                "year": str(date.year),
                "month": f"{date.month:02d}",
                "decade": f"{date.year // 10 * 10}s",
            }
            for granularity, value in values.items():
                relation = f"{prefix}_on_date" if granularity == "date" else f"{prefix}_in_{granularity}"
                tail_key = f"{prefix}.{granularity}.{value}"
                tail = add_entity(tail_key, f"{prefix}_{granularity}", entities, entity_ids)
                item_triples.add((head, relation_ids[relation], tail))

    relation_rows = [(name, relation_id) for name, relation_id in relation_ids.items()]
    return entities, relation_rows, sorted(item_triples), sorted(user_triples)


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite to replace this script's files")
    path.mkdir(parents=True, exist_ok=True)


def write_grouped_interactions(
    path: Path,
    rows: list[Interaction],
    user_map: dict[int, int],
    item_map: dict[int, int],
    min_rating: int = 1,
) -> None:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row.rating < min_rating:
            continue
        grouped[user_map[row.raw_user]].append((row.timestamp, item_map[row.raw_item]))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user in sorted(grouped):
            # Keep per-user items chronological even though the original loader later converts them to a set.
            item_ids = [item for _, item in sorted(grouped[user])]
            handle.write(" ".join(str(value) for value in [user, *item_ids]) + "\n")


def write_interaction_events(
    path: Path,
    parts: dict[str, list[Interaction]],
    user_map: dict[int, int],
    item_map: dict[int, int],
    include_timestamp: bool,
) -> None:
    fieldnames = ["ordinal", "split", "user", "item", "rating"]
    if include_timestamp:
        fieldnames.append("timestamp")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        ordinal = 0
        for split in SPLIT_NAMES:
            for row in parts[split]:
                output = {
                    "ordinal": ordinal,
                    "split": split,
                    "user": user_map[row.raw_user],
                    "item": item_map[row.raw_item],
                    "rating": row.rating,
                }
                if include_timestamp:
                    output["timestamp"] = row.timestamp
                writer.writerow(output)
                ordinal += 1


def write_value_sidecars(
    output: Path,
    parts: dict[str, list[Interaction]],
    user_map: dict[int, int],
    item_map: dict[int, int],
    field: str,
) -> None:
    if field not in {"rating", "timestamp"}:
        raise ValueError(f"unsupported interaction label {field!r}")
    for split in SPLIT_NAMES:
        grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for row in parts[split]:
            grouped[user_map[row.raw_user]].append((row.timestamp, item_map[row.raw_item], getattr(row, field)))
        path = output / f"{split}_{field}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for user in sorted(grouped):
                ordered = sorted(grouped[user], key=lambda entry: (entry[0], entry[1]))
                labels = [f"{item}:{value}" for _, item, value in ordered]
                handle.write(" ".join([str(user), *labels]) + "\n")


def write_mapping_files(
    output: Path,
    user_map: dict[int, int],
    item_map: dict[int, int],
    entities: list[tuple[str, int, str]],
    relations: list[tuple[str, int]],
) -> None:
    with (output / "user_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("org_id remap_id\n")
        for raw_id, remap_id in sorted(user_map.items(), key=lambda pair: pair[1]):
            handle.write(f"{raw_id} {remap_id}\n")

    with (output / "item_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        # Keep the original KGAT three-column header.  The value is a stable local entity key,
        # not a real Freebase id, because MovieLens-100K itself does not contain Freebase ids.
        handle.write("org_id remap_id freebase_id\n")
        for raw_id, remap_id in sorted(item_map.items(), key=lambda pair: pair[1]):
            handle.write(f"{raw_id} {remap_id} ml100k.item.{raw_id}\n")

    with (output / "entity_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("org_id remap_id\n")
        for key, remap_id, _kind in sorted(entities, key=lambda row: row[1]):
            handle.write(f"{key} {remap_id}\n")

    with (output / "entity_metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["org_id", "remap_id", "kind"], delimiter="\t")
        writer.writeheader()
        for key, remap_id, kind in sorted(entities, key=lambda row: row[1]):
            writer.writerow({"org_id": key, "remap_id": remap_id, "kind": kind})

    with (output / "relation_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("org_id remap_id\n")
        for name, remap_id in sorted(relations, key=lambda row: row[1]):
            handle.write(f"{name} {remap_id}\n")


def write_kg(path: Path, triples: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for head, relation, tail in triples:
            handle.write(f"{head} {relation} {tail}\n")


def write_metadata(
    output: Path,
    item_map: dict[int, int],
    user_map: dict[int, int],
    movies: dict[int, Movie],
    users: dict[int, User],
) -> None:
    movie_fields = [
        "raw_item",
        "item",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "genres",
    ]
    with (output / "movie_metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=movie_fields, delimiter="\t")
        writer.writeheader()
        for raw_item, item in sorted(item_map.items(), key=lambda pair: pair[1]):
            movie = movies[raw_item]
            writer.writerow(
                {
                    "raw_item": raw_item,
                    "item": item,
                    "title": movie.title,
                    "release_date": movie.release_date,
                    "video_release_date": movie.video_release_date,
                    "imdb_url": movie.imdb_url,
                    "genres": "|".join(movie.genres),
                }
            )

    user_fields = ["raw_user", "user", "age", "gender", "occupation", "zip_code"]
    with (output / "user_metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=user_fields, delimiter="\t")
        writer.writeheader()
        for raw_user, user in sorted(user_map.items(), key=lambda pair: pair[1]):
            profile = users.get(raw_user)
            if profile is None:
                raise ValueError(f"user {raw_user} appears in interactions but is missing from u.user")
            writer.writerow(
                {
                    "raw_user": raw_user,
                    "user": user,
                    "age": profile.age,
                    "gender": profile.gender,
                    "occupation": profile.occupation,
                    "zip_code": profile.zip_code,
                }
            )


def cold_start_stats(parts: dict[str, list[Interaction]]) -> dict[str, int]:
    train_users = {row.raw_user for row in parts["train"]}
    train_items = {row.raw_item for row in parts["train"]}
    result: dict[str, int] = {}
    for split in ("val", "test"):
        result[f"{split}_cold_users"] = len({row.raw_user for row in parts[split]} - train_users)
        result[f"{split}_cold_items"] = len({row.raw_item for row in parts[split]} - train_items)
        result[f"{split}_cold_user_interactions"] = sum(row.raw_user not in train_users for row in parts[split])
        result[f"{split}_cold_item_interactions"] = sum(row.raw_item not in train_items for row in parts[split])
    return result


def build_meta(
    args: argparse.Namespace,
    parts: dict[str, list[Interaction]],
    user_map: dict[int, int],
    item_map: dict[int, int],
    entities: list[tuple[str, int, str]],
    relations: list[tuple[str, int]],
    triples: list[tuple[int, int, int]],
    user_triples: list[tuple[int, int, int]],
    timestamp_labelled: bool,
) -> dict:
    split_meta = {}
    for split in SPLIT_NAMES:
        rows = parts[split]
        split_meta[split] = {
            "interactions": len(rows),
            "kgat_interactions": sum(row.rating >= (1 if split == "train" else EVAL_MIN_RATING) for row in rows),
            "first_timestamp": rows[0].timestamp,
            "last_timestamp": rows[-1].timestamp,
            "users": len({row.raw_user for row in rows}),
            "items": len({row.raw_item for row in rows}),
        }
    return {
        "source": str(args.raw_dir.as_posix()),
        "selection": (
            f"earliest {args.n_interactions} interactions after filtering rating >= {args.min_rating}; "
            "stable global order (timestamp, raw_user, raw_item)"
        ),
        "timestamp_labelled": timestamp_labelled,
        "min_rating": args.min_rating,
        "evaluation_ground_truth_min_rating": EVAL_MIN_RATING,
        "n_interactions": args.n_interactions,
        "n_users": len(user_map),
        "n_items": len(item_map),
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_item_kg_triples": len(triples),
        "n_user_kg_triples": len(user_triples),
        "splits": split_meta,
        "boundary_ties": {
            "train_val_same_timestamp": parts["train"][-1].timestamp == parts["val"][0].timestamp,
            "val_test_same_timestamp": parts["val"][-1].timestamp == parts["test"][0].timestamp,
        },
        "cold_start": cold_start_stats(parts),
        "compatibility_notes": [
            "The original KGAT loader does not read val.txt; add validation support before training.",
            "train.txt retains all observed ratings; a custom weighted loader must not treat rating 1-3 "
            "as equally positive.",
            "val.txt and test.txt contain only rating 4-5 ground truth; their rating sidecars retain all events.",
            "The timestamp sidecars require a custom loader; do not append timestamps to KGAT train.txt.",
            "User metadata relations are stored in user_kg_final.txt and require a custom KGAT loader.",
            "Exact 8000/1000/1000 row boundaries are kept even when equal timestamps straddle a boundary.",
        ],
    }


def write_dataset(
    output: Path,
    args: argparse.Namespace,
    parts: dict[str, list[Interaction]],
    user_map: dict[int, int],
    item_map: dict[int, int],
    movies: dict[int, Movie],
    users: dict[int, User],
    entities: list[tuple[str, int, str]],
    relations: list[tuple[str, int]],
    triples: list[tuple[int, int, int]],
    user_triples: list[tuple[int, int, int]],
    timestamp_labelled: bool,
) -> None:
    ensure_output_directory(output, args.overwrite)
    for split in SPLIT_NAMES:
        ground_truth_min_rating = 1 if split == "train" else EVAL_MIN_RATING
        write_grouped_interactions(
            output / f"{split}.txt",
            parts[split],
            user_map,
            item_map,
            min_rating=ground_truth_min_rating,
        )
    write_mapping_files(output, user_map, item_map, entities, relations)
    write_kg(output / "kg_final.txt", triples)
    write_kg(output / "user_kg_final.txt", user_triples)
    write_metadata(output, item_map, user_map, movies, users)
    write_interaction_events(
        output / "interactions_chronological.tsv",
        parts,
        user_map,
        item_map,
        include_timestamp=timestamp_labelled,
    )
    write_value_sidecars(output, parts, user_map, item_map, field="rating")
    if timestamp_labelled:
        write_value_sidecars(output, parts, user_map, item_map, field="timestamp")
    meta = build_meta(
        args,
        parts,
        user_map,
        item_map,
        entities,
        relations,
        triples,
        user_triples,
        timestamp_labelled,
    )
    (output / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def read_space_table(path: Path, expected_header: list[str]) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split() != expected_header:
        raise AssertionError(f"{path.name} has an incompatible KGAT header")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if any(len(row) != len(expected_header) for row in rows):
        raise AssertionError(f"{path.name} has an incompatible KGAT column count")
    return rows


def validate_kgat_output(output: Path, timestamp_labelled: bool) -> None:
    """Validate files against the formats documented and parsed by the original KGAT source."""
    user_rows = read_space_table(output / "user_list.txt", ["org_id", "remap_id"])
    item_rows = read_space_table(output / "item_list.txt", ["org_id", "remap_id", "freebase_id"])
    entity_rows = read_space_table(output / "entity_list.txt", ["org_id", "remap_id"])
    relation_rows = read_space_table(output / "relation_list.txt", ["org_id", "remap_id"])

    for name, rows in (
        ("user", user_rows),
        ("item", item_rows),
        ("entity", entity_rows),
        ("relation", relation_rows),
    ):
        ids = [int(row[1]) for row in rows]
        if ids != list(range(len(ids))):
            raise AssertionError(f"{name} remap ids are not contiguous from zero")

    n_users, n_items = len(user_rows), len(item_rows)
    n_entities, n_relations = len(entity_rows), len(relation_rows)
    if [int(row[1]) for row in item_rows] != [int(row[1]) for row in entity_rows[:n_items]]:
        raise AssertionError("item ids are not the leading entity ids required by KGAT")

    loader_users: set[int] = set()
    loader_items: set[int] = set()
    for split in SPLIT_NAMES:
        path = output / f"{split}.txt"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) < 2:
                raise AssertionError(f"{path.name}:{line_number} must contain a user and at least one item")
            values = [int(value) for value in fields]
            if not 0 <= values[0] < n_users or any(not 0 <= item < n_items for item in values[1:]):
                raise AssertionError(f"{path.name}:{line_number} contains an id outside its mapping")
            if split in {"train", "test"}:  # these are the only files read by the unmodified KGAT loader
                loader_users.add(values[0])
                loader_items.update(values[1:])
    if max(loader_users) + 1 != n_users or max(loader_items) + 1 != n_items:
        raise AssertionError("the original KGAT loader would infer user/item counts inconsistent with mappings")

    kg_lines = (output / "kg_final.txt").read_text(encoding="utf-8").splitlines()
    if not kg_lines:
        raise AssertionError("kg_final.txt is empty")
    used_relations: set[int] = set()
    for line_number, line in enumerate(kg_lines, start=1):
        fields = line.split()
        if len(fields) != 3:
            raise AssertionError(f"kg_final.txt:{line_number} must contain exactly three integers")
        head, relation, tail = (int(value) for value in fields)
        if not (0 <= head < n_items and 0 <= tail < n_entities and 0 <= relation < n_relations):
            raise AssertionError(f"kg_final.txt:{line_number} contains an id outside its mapping")
        used_relations.add(relation)
    if max(used_relations) + 1 != n_relations:
        raise AssertionError("the original KGAT loader would infer a relation count inconsistent with mappings")

    user_kg_lines = (output / "user_kg_final.txt").read_text(encoding="utf-8").splitlines()
    if not user_kg_lines:
        raise AssertionError("user_kg_final.txt is empty")
    for line_number, line in enumerate(user_kg_lines, start=1):
        fields = line.split()
        if len(fields) != 3:
            raise AssertionError(f"user_kg_final.txt:{line_number} must contain exactly three integers")
        head, relation, tail = (int(value) for value in fields)
        if not (0 <= head < n_users and 0 <= tail < n_entities and 0 <= relation < n_relations):
            raise AssertionError(f"user_kg_final.txt:{line_number} contains an id outside its mapping")
        used_relations.add(relation)
    if used_relations != set(range(n_relations)):
        raise AssertionError("relation_list.txt contains relation ids unused by item/user metadata triples")

    for split in SPLIT_NAMES:
        rating_path = output / f"{split}_rating.txt"
        if not rating_path.is_file():
            raise AssertionError(f"missing {rating_path.name}")
        timestamp_path = output / f"{split}_timestamp.txt"
        if timestamp_path.exists() != timestamp_labelled:
            raise AssertionError(f"unexpected timestamp-file state for {timestamp_path.name}")


def check_paired_outputs(chrono: Path, timestamped: Path) -> None:
    """Ensure the timestamp variant differs only by its timestamp labels and metadata note."""
    for name in COMMON_OUTPUT_FILES:
        plain_bytes = (chrono / name).read_bytes()
        timestamped_bytes = (timestamped / name).read_bytes()
        if plain_bytes != timestamped_bytes:
            raise AssertionError(f"paired datasets differ in common file {name}")

    with (chrono / "interactions_chronological.tsv").open(encoding="utf-8", newline="") as plain_handle:
        plain_rows = list(csv.DictReader(plain_handle, delimiter="\t"))
    with (timestamped / "interactions_chronological.tsv").open(encoding="utf-8", newline="") as timestamped_handle:
        timestamped_rows = list(csv.DictReader(timestamped_handle, delimiter="\t"))

    if len(plain_rows) != len(timestamped_rows):
        raise AssertionError("paired datasets contain a different number of chronological interactions")
    for ordinal, (plain_row, timestamped_row) in enumerate(zip(plain_rows, timestamped_rows, strict=True)):
        without_timestamp = {key: value for key, value in timestamped_row.items() if key != "timestamp"}
        if plain_row != without_timestamp:
            raise AssertionError(f"paired datasets differ at chronological interaction {ordinal}")
        if not timestamped_row.get("timestamp"):
            raise AssertionError(f"timestamp label is missing at chronological interaction {ordinal}")

    for split in SPLIT_NAMES:
        rating_lines = (timestamped / f"{split}_rating.txt").read_text(encoding="utf-8").splitlines()
        timestamp_lines = (timestamped / f"{split}_timestamp.txt").read_text(encoding="utf-8").splitlines()
        if len(rating_lines) != len(timestamp_lines):
            raise AssertionError(f"rating/timestamp user counts differ in {split}")
        for line_number, (rating_line, timestamp_line) in enumerate(
            zip(rating_lines, timestamp_lines, strict=True), start=1
        ):
            rating_fields = rating_line.split()
            timestamp_fields = timestamp_line.split()
            rating_keys = [field.split(":", 1)[0] for field in rating_fields[1:]]
            timestamp_keys = [field.split(":", 1)[0] for field in timestamp_fields[1:]]
            if rating_fields[0] != timestamp_fields[0] or rating_keys != timestamp_keys:
                raise AssertionError(f"rating/timestamp labels are not aligned in {split}, line {line_number}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    required = ["u.data", "u.item", "u.user", "u.genre"]
    missing = [name for name in required if not (args.raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing MovieLens files under {args.raw_dir}: {', '.join(missing)}")

    genres = read_genres(args.raw_dir / "u.genre")
    interactions = read_interactions(args.raw_dir / "u.data", args.min_rating)
    if len(interactions) < args.n_interactions:
        raise ValueError(
            f"only {len(interactions)} interactions satisfy rating >= {args.min_rating}; "
            f"cannot select {args.n_interactions}"
        )
    selected = interactions[: args.n_interactions]
    parts = split_rows(selected, args.n_train, args.n_val)
    check_chronology(parts)

    movies = read_movies(args.raw_dir / "u.item", genres)
    users = read_users(args.raw_dir / "u.user")
    user_map, item_map = make_id_maps(selected)
    entities, relations, triples, user_triples = build_metadata_kg(item_map, movies, user_map, users)

    shared = {
        "args": args,
        "parts": parts,
        "user_map": user_map,
        "item_map": item_map,
        "movies": movies,
        "users": users,
        "entities": entities,
        "relations": relations,
        "triples": triples,
        "user_triples": user_triples,
    }
    chrono_output = args.output_root / "ml-100k-chrono"
    timestamp_output = args.output_root / "ml-100k-chrono-timestamp"
    write_dataset(chrono_output, timestamp_labelled=False, **shared)
    write_dataset(timestamp_output, timestamp_labelled=True, **shared)
    validate_kgat_output(chrono_output, timestamp_labelled=False)
    validate_kgat_output(timestamp_output, timestamp_labelled=True)
    check_paired_outputs(chrono_output, timestamp_output)

    print(json.dumps({"arguments": vars(args), "outputs": [str(args.output_root)]}, default=str, indent=2))


if __name__ == "__main__":
    main()
