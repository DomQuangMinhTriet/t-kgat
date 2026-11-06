"""Generate a new temporal Last-FM dataset from the published KGAT data.

The author dataset is never modified. Interactions use the author's user/item
universe, keep the earliest timestamp per pair, and follow the project's
leakage-safe 0.8/0.1/0.1 temporal protocol.

The original KG has no dates, so ``--kg-policy static`` is the default: only the
interactions are made temporal and the KG is copied unchanged. ``--kg-policy strict``
needs ``--kg-fact-times``, a four-column file ``head relation tail timestamp`` in the
original KGAT IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import pandas as pd

from tkgat.data.leakage import check_split
from tkgat.data.split import SplitConfig, TemporalSplit, temporal_split

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_DEFAULT = ROOT / "knowledge_graph_attention_network" / "Data" / "last-fm"
INTERACTIONS_DEFAULT = ROOT / "data-last-fm" / "lfm1b-tracks" / "lfm1b-tracks.inter"
CACHE_DEFAULT = ROOT / "data-last-fm" / "lfm1b-tracks" / "kgat_author_earliest.tsv"
OUTPUT_DEFAULT = ROOT / "knowledge_graph_attention_network" / "Data" / "last-fm-temporal"
SHIFT = 32
MASK = (1 << SHIFT) - 1


@dataclass(frozen=True)
class Mappings:
    users: dict[int, str]  # old remap -> raw id
    items: dict[int, str]
    freebase: dict[int, str]
    entities: dict[int, str]
    relations: dict[int, str]


def pair_key(user: int, item: int) -> int:
    if user < 0 or item < 0 or user > MASK or item > MASK:
        raise ValueError(f"ID outside unsigned 32-bit range: ({user}, {item})")
    return (user << SHIFT) | item


def decode_pair(key: int) -> tuple[int, int]:
    return key >> SHIFT, key & MASK


def read_mapping(path: Path, width: int) -> list[list[str]]:
    with path.open(encoding="utf-8") as handle:
        if not handle.readline():
            raise ValueError(f"empty mapping: {path}")
        rows = []
        for line_no, line in enumerate(handle, 2):
            row = line.split()
            if len(row) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} columns")
            rows.append(row)
    return rows


def load_mappings(author: Path) -> Mappings:
    users = read_mapping(author / "user_list.txt", 2)
    items = read_mapping(author / "item_list.txt", 3)
    entities = read_mapping(author / "entity_list.txt", 2)
    relations = read_mapping(author / "relation_list.txt", 2)
    return Mappings(
        users={int(new): raw for raw, new in users},
        items={int(new): raw for raw, new, _ in items},
        freebase={int(new): fb for _, new, fb in items},
        entities={int(new): raw for raw, new in entities},
        relations={int(new): raw for raw, new in relations},
    )


def load_author_pairs(author: Path, maps: Mappings) -> tuple[set[int], dict]:
    pairs: set[int] = set()
    tokens = 0
    for filename in ("train.txt", "test.txt"):
        with (author / filename).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                values = line.split()
                if len(values) < 2:
                    raise ValueError(f"{filename}:{line_no}: missing items")
                user = int(values[0])
                if user not in maps.users:
                    raise ValueError(f"{filename}:{line_no}: unknown user {user}")
                for value in values[1:]:
                    item = int(value)
                    if item not in maps.items:
                        raise ValueError(f"{filename}:{line_no}: unknown item {item}")
                    pairs.add(pair_key(user, item))
                    tokens += 1
    return pairs, {
        "author_tokens": tokens,
        "author_unique_pairs": len(pairs),
        "author_duplicates": tokens - len(pairs),
    }


def load_cache(path: Path, expected: set[int]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype="int64")
    if list(frame.columns) != ["user", "item", "ts"]:
        raise ValueError(f"unexpected cache columns: {list(frame.columns)}")
    if frame.duplicated(["user", "item"]).any():
        raise ValueError("timestamp cache contains duplicate pairs")
    actual = {pair_key(int(u), int(i)) for u, i in zip(frame.user, frame.item, strict=True)}
    if actual != expected:
        raise ValueError(f"cache pair universe differs: cache={len(actual):,}, expected={len(expected):,}")
    frame["rating"] = pd.Series(1, index=frame.index, dtype="int8")
    return frame


def scan_interactions(path: Path, expected: set[int], maps: Mappings, progress_every: int) -> tuple[pd.DataFrame, dict]:
    raw_users = {raw: old for old, raw in maps.users.items()}
    raw_items = {raw: old for old, raw in maps.items.items()}
    earliest: dict[int, int] = {}
    rows = matched = 0
    started = time.monotonic()
    with path.open(encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
        header = handle.readline().rstrip().split("\t")
        if len(header) < 3 or not header[2].startswith("timestamp"):
            raise ValueError(f"unexpected interaction header: {header}")
        for line_no, line in enumerate(handle, 2):
            rows += 1
            if progress_every and rows % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"scanned={rows:,} matched_pairs={len(earliest):,}/{len(expected):,} "
                    f"rate={rows / elapsed:,.0f} rows/s",
                    flush=True,
                )
            try:
                raw_user, raw_item, raw_ts = line.rstrip("\r\n").split("\t", 2)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: expected three columns") from exc
            user = raw_users.get(raw_user)
            item = raw_items.get(raw_item)
            if user is None or item is None:
                continue
            key = pair_key(user, item)
            if key not in expected:
                continue
            matched += 1
            ts = int(raw_ts)
            if key not in earliest or ts < earliest[key]:
                earliest[key] = ts
    missing = expected - earliest.keys()
    if missing:
        raise RuntimeError(f"{len(missing):,} author pairs have no timestamp; examples={list(missing)[:10]}")
    frame = pd.DataFrame(
        [(*decode_pair(key), ts, 1) for key, ts in earliest.items()],
        columns=["user", "item", "ts", "rating"],
    ).astype({"user": "int64", "item": "int64", "ts": "int64", "rating": "int8"})
    return frame, {
        "lfm_rows_scanned": rows,
        "lfm_candidate_rows": matched,
        "pairs_timestamped": len(frame),
        "scan_seconds": round(time.monotonic() - started, 3),
    }


def save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["user", "item", "ts"]].sort_values(["user", "item"], kind="mergesort").to_csv(
        path, sep="\t", index=False, lineterminator="\n"
    )


def make_split(frame: pd.DataFrame, q_valid: float, q_test: float, kcore: int) -> TemporalSplit:
    split = temporal_split(frame, SplitConfig(q_val=q_valid, q_test=q_test, min_rating=1, kcore=kcore))
    problems = check_split(split)
    if problems:
        raise RuntimeError("interaction leakage:\n- " + "\n- ".join(problems))
    for part in (split.train, split.val, split.test):
        part.sort_values(["ts", "user", "item"], kind="mergesort", inplace=True, ignore_index=True)
    return split


def read_kg(path: Path) -> list[tuple[int, int, int]]:
    triples = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = line.split()
            if len(row) != 3:
                raise ValueError(f"{path}:{line_no}: expected h r t")
            triples.append(tuple(map(int, row)))
    if len(set(triples)) != len(triples):
        raise ValueError("author KG contains duplicate triples")
    return triples


def read_fact_times(path: Path) -> dict[tuple[int, int, int], int]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = line.split()
            if not row or (line_no == 1 and row[0].lower() in {"head", "h"}):
                continue
            if len(row) != 4:
                raise ValueError(f"{path}:{line_no}: expected head relation tail timestamp")
            h, r, t, ts = map(int, row)
            triple = (h, r, t)
            if triple in result and result[triple] != ts:
                raise ValueError(f"conflicting fact timestamps: {triple}")
            result[triple] = ts
    return result


def kg_snapshots(
    author: Path,
    train_items: set[int],
    t1: int,
    t2: int,
    policy: str,
    fact_path: Path | None,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], dict, dict]:
    source = read_kg(author / "kg_final.txt")
    seeded = [triple for triple in source if triple[0] in train_items or triple[2] in train_items]
    if policy == "static":
        return (
            seeded,
            seeded.copy(),
            {},
            {
                "policy": "static",
                "time_consistent": False,
                "warning": "Author KG copied without fact dates; reference only, not leakage-free.",
                "source_triples": len(source),
                "train_item_seeded_triples": len(seeded),
            },
        )
    if fact_path is None:
        raise ValueError("strict KG policy requires --kg-fact-times")
    times = read_fact_times(fact_path)
    unknown = set(times) - set(source)
    if unknown:
        raise ValueError(f"fact-time file has {len(unknown)} triples absent from author KG")
    dated = [(triple, times[triple]) for triple in seeded if triple in times]
    kg_t1 = [triple for triple, ts in dated if ts < t1]
    kg_t2 = [triple for triple, ts in dated if ts < t2]
    if not set(kg_t1) <= set(kg_t2):
        raise RuntimeError("KG t1 is not contained in KG t2")
    return (
        kg_t1,
        kg_t2,
        times,
        {
            "policy": "strict_drop_undated",
            "time_consistent": True,
            "source_triples": len(source),
            "train_item_seeded_triples": len(seeded),
            "dated_seeded_triples": len(dated),
            "undated_seeded_triples_dropped": len(seeded) - len(dated),
            "triples_t1": len(kg_t1),
            "triples_t2": len(kg_t2),
        },
    )


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def grouped(frame: pd.DataFrame, users: dict[int, int], items: dict[int, int]) -> dict[int, list[tuple[int, int]]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for row in frame.itertuples(index=False):
        result.setdefault(users[int(row.user)], []).append((items[int(row.item)], int(row.ts)))
    return result


def write_part(
    out: Path, name: str, frame: pd.DataFrame, users: dict[int, int], items: dict[int, int]
) -> list[tuple[int, int, int]]:
    rows = grouped(frame, users, items)
    write_lines(
        out / f"{name}.txt",
        (" ".join([str(user), *[str(item) for item, _ in rows[user]]]) for user in sorted(rows)),
    )
    write_lines(
        out / f"{name}_timestamp.txt",
        (" ".join([str(user), *[f"{item}:{ts}" for item, ts in rows[user]]]) for user in sorted(rows)),
    )
    events = sorted((int(row.ts), users[int(row.user)], items[int(row.item)]) for row in frame.itertuples(index=False))
    write_lines(
        out / f"{name}_events.tsv",
        chain(["user\titem\ttimestamp"], (f"{user}\t{item}\t{ts}" for ts, user, item in events)),
    )
    return events


def write_mask(path: Path, frame: pd.DataFrame, users: dict[int, int], items: dict[int, int]) -> None:
    known = frame[frame.user.isin(users) & frame.item.isin(items)].sort_values(["ts", "user", "item"], kind="mergesort")
    rows = grouped(known, users, items)
    write_lines(
        path,
        (" ".join([str(user), *[str(item) for item, _ in rows[user]]]) for user in sorted(rows)),
    )


def export(
    out: Path,
    maps: Mappings,
    split: TemporalSplit,
    kg_t1: list[tuple[int, int, int]],
    kg_t2: list[tuple[int, int, int]],
    fact_times: dict,
    source_stats: dict,
    kg_stats: dict,
) -> dict:
    old_users = sorted(map(int, split.train.user.unique()))
    old_items = sorted(map(int, split.train.item.unique()))
    users = {old: new for new, old in enumerate(old_users)}
    items = {old: new for new, old in enumerate(old_items)}
    universe = set(old_items)
    for h, _, t in kg_t2:
        universe.update((h, t))
    extras = sorted(universe - set(old_items))
    entities = {**items, **{old: len(items) + i for i, old in enumerate(extras)}}
    if set(entities) - maps.entities.keys():
        raise ValueError("KG contains entities absent from entity_list.txt")

    def remap(triples: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        return sorted((entities[h], r, entities[t]) for h, r, t in triples)

    new_t1, new_t2 = remap(kg_t1), remap(kg_t2)
    if not set(new_t1) <= set(new_t2):
        raise RuntimeError("remapped KG t1 is not contained in KG t2")
    out.mkdir(parents=True)
    events = {
        name: write_part(out, name, frame, users, items)
        for name, frame in (("train", split.train), ("valid", split.val), ("test", split.test))
    }
    for suffix in (".txt", "_timestamp.txt", "_events.tsv"):
        shutil.copyfile(out / f"valid{suffix}", out / f"val{suffix}")
    write_mask(out / "valid_mask.txt", split.seen_before_val, users, items)
    shutil.copyfile(out / "valid_mask.txt", out / "val_mask.txt")
    write_mask(out / "test_mask.txt", split.seen_before_test, users, items)
    with (out / "interactions_chronological.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ordinal\tsplit\tuser\titem\ttimestamp\n")
        ordinal = 0
        for name in ("train", "valid", "test"):
            for ts, user, item in events[name]:
                handle.write(f"{ordinal}\t{name}\t{user}\t{item}\t{ts}\n")
                ordinal += 1
    write_lines(
        out / "user_list.txt",
        ["org_id remap_id", *[f"{maps.users[old]} {users[old]}" for old in old_users]],
    )
    write_lines(
        out / "item_list.txt",
        [
            "org_id remap_id freebase_id",
            *[f"{maps.items[old]} {items[old]} {maps.freebase[old]}" for old in old_items],
        ],
    )
    entity_rows = sorted(entities.items(), key=lambda row: row[1])
    write_lines(
        out / "entity_list.txt",
        ["org_id remap_id", *[f"{maps.entities[old]} {new}" for old, new in entity_rows]],
    )
    write_lines(
        out / "relation_list.txt",
        ["org_id remap_id", *[f"{maps.relations[old]} {old}" for old in sorted(maps.relations)]],
    )
    write_lines(out / "kg_final.txt", (f"{h} {r} {t}" for h, r, t in new_t1))
    shutil.copyfile(out / "kg_final.txt", out / "kg_valid.txt")
    write_lines(out / "kg_test.txt", (f"{h} {r} {t}" for h, r, t in new_t2))
    if fact_times:
        reverse = {new: old for old, new in entities.items()}

        def dated(triples: list[tuple[int, int, int]]) -> Iterable[str]:
            for h, r, t in triples:
                yield f"{h} {r} {t} {fact_times[(reverse[h], r, reverse[t])]}"

        write_lines(out / "kg_final_timestamp.txt", dated(new_t1))
        shutil.copyfile(out / "kg_final_timestamp.txt", out / "kg_valid_timestamp.txt")
        write_lines(out / "kg_test_timestamp.txt", dated(new_t2))
    counts = {"train": len(split.train), "valid": len(split.val), "test": len(split.test)}
    total = sum(counts.values())
    report = {
        "protocol": (
            "author KGAT Last-FM user/item/pair universe; earliest event per pair; global 0.8/0.9 temporal split; "
            "train-only k-core; validation/test restricted to train users and items"
        ),
        "cutoffs": {"t1": int(split.t1), "t2": int(split.t2)},
        "config": {
            "q_valid": split.config.q_val,
            "q_test": split.config.q_test,
            "kcore": split.config.kcore,
            "implicit_min_rating": 1,
        },
        "n_users": len(users),
        "n_items": len(items),
        "n_entities": len(entities),
        "n_relations": len(maps.relations),
        "interactions": counts,
        "final_interaction_ratios": {name: count / total for name, count in counts.items()},
        "split_stats": split.summary(),
        "interaction_leakage_problems": check_split(split),
        "kg": kg_stats,
        "source": source_stats,
    }
    if report["interaction_leakage_problems"]:
        raise RuntimeError("refusing to write a dataset with interaction leakage")
    (out / "dataset_meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-data", type=Path, default=AUTHOR_DEFAULT)
    parser.add_argument("--interactions", type=Path, default=INTERACTIONS_DEFAULT)
    parser.add_argument("--timestamp-cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--q-valid", type=float, default=0.8)
    parser.add_argument("--q-test", type=float, default=0.9)
    parser.add_argument("--kcore", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--kg-policy", choices=("strict", "static"), default="static")
    parser.add_argument("--kg-fact-times", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {args.output}")
    if not args.author_data.is_dir() or not args.interactions.is_file():
        raise FileNotFoundError("author dataset or LFM interaction file is missing")
    if args.kg_policy == "strict":
        if args.kg_fact_times is None or not args.kg_fact_times.is_file():
            raise FileNotFoundError("strict KG requires an existing --kg-fact-times file")
        read_fact_times(args.kg_fact_times)  # fail before an expensive scan
    maps = load_mappings(args.author_data)
    pairs, author_stats = load_author_pairs(args.author_data, maps)
    if args.timestamp_cache.is_file():
        print(f"loading timestamp cache: {args.timestamp_cache}", flush=True)
        interactions = load_cache(args.timestamp_cache, pairs)
        scan_stats = {"timestamp_cache": str(args.timestamp_cache), "timestamp_cache_hit": True}
    else:
        interactions, scan_stats = scan_interactions(args.interactions, pairs, maps, args.progress_every)
        save_cache(args.timestamp_cache, interactions)
        scan_stats.update(timestamp_cache=str(args.timestamp_cache), timestamp_cache_hit=False)
        print(f"saved timestamp cache: {args.timestamp_cache}", flush=True)
    split = make_split(interactions, args.q_valid, args.q_test, args.kcore)
    kg_t1, kg_t2, fact_times, kg_stats = kg_snapshots(
        args.author_data,
        set(map(int, split.train.item.unique())),
        split.t1,
        split.t2,
        args.kg_policy,
        args.kg_fact_times,
    )
    report = export(
        args.output,
        maps,
        split,
        kg_t1,
        kg_t2,
        fact_times,
        {**author_stats, **scan_stats},
        kg_stats,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
