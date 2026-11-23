"""Disk-backed raw Last-FM splitting and streaming KGAT interaction export."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tkgat.data.lastfm_raw import write_json
from tkgat.pipeline.build import PARTS
from tkgat.registry import sha256_file


def literal(value) -> str:
    """Quote a SQL value, including paths; never interpolate identifiers from inputs."""
    return "'" + str(value).replace("'", "''") + "'"


class DiskSplit:
    def __init__(self, db, pairs, config):
        self.db, self.config = db, config
        self.cut_meta = json.loads(Path(str(pairs) + ".meta.json").read_text(encoding="utf-8"))
        meta = self.cut_meta
        if meta.get("pair_universe") != "raw_events" or meta.get("timestamp_policy") != "earliest_in_window":
            raise ValueError("Expected a raw-event earliest-pair cut manifest")
        self.log("Verifying cut SHA-256")
        if sha256_file(pairs) != meta["sha256"]:
            raise ValueError("Cut file SHA-256 does not match its manifest; rebuild the cut")
        self._import(pairs)
        n = self.count("raw")
        if n != meta["pairs"] or n == 0:
            raise ValueError("Cut rows do not match the manifest")
        self.log("Checking global pair uniqueness on disk")
        if self.one("SELECT 1 FROM raw GROUP BY u,i HAVING count(*) > 1 LIMIT 1") is not None:
            raise ValueError("Expected unique raw user,item,ts pairs")
        self.t1, self.t2 = (self.quantile(n, q) for q in (config.q_val, config.q_test))
        if self.t1 >= self.t2:
            raise ValueError("degenerate temporal cutoffs")
        self.log(f"Cutoffs: t1={self.t1}, t2={self.t2}; computing train-only {config.kcore}-core")
        db.execute(f"CREATE TABLE train AS SELECT * FROM raw WHERE ts < {self.t1}")
        before = self.count("train")
        trace = []
        if config.kcore > 1:
            while True:
                # Each removed vertex has degree < k in this train iteration.
                db.execute(f"CREATE OR REPLACE TABLE good_users AS SELECT u FROM train "
                           f"GROUP BY u HAVING count(*) >= {config.kcore}")
                db.execute(f"CREATE OR REPLACE TABLE good_items AS SELECT i FROM train "
                           f"GROUP BY i HAVING count(*) >= {config.kcore}")
                db.execute("CREATE TABLE next_train AS SELECT * FROM train "
                           "SEMI JOIN good_users USING(u) SEMI JOIN good_items USING(i)")
                after = self.count("next_train")
                trace.append({"before": before, "after": after})
                self.log(f"Core iteration {len(trace)}: {before:,} -> {after:,} pairs")
                db.execute("DROP TABLE train")
                db.execute("ALTER TABLE next_train RENAME TO train")
                if after == before:
                    break
                before = after
        if not self.count("train"):
            raise ValueError("empty training set after train-only k-core")
        db.execute("CREATE TABLE users AS SELECT u, row_number() OVER (ORDER BY u)-1 AS uid "
                   "FROM (SELECT DISTINCT u FROM train)")
        db.execute("CREATE TABLE items AS SELECT i, row_number() OVER (ORDER BY i)-1 AS iid "
                   "FROM (SELECT DISTINCT i FROM train)")
        for name, condition in (("val", f"ts >= {self.t1} AND ts < {self.t2}"),
                                ("test", f"ts >= {self.t2}")):
            db.execute(f"CREATE TABLE {name} AS SELECT r.* FROM raw r "
                       f"SEMI JOIN users USING(u) SEMI JOIN items USING(i) WHERE {condition}")
        for name, cutoff in (("seen_before_val", self.t1), ("seen_before_test", self.t2)):
            db.execute(f"CREATE VIEW {name} AS SELECT * FROM raw WHERE ts < {cutoff}")
        self.log("Auditing complete core, split boundaries, masks and pair accounting")
        self._audit(n, trace)
        self.enc = SimpleNamespace(
            n_users=self.count("users"), n_items=self.count("items"),
            user_ids=db.execute("SELECT u FROM users ORDER BY uid").fetchnumpy()["u"],
            item_ids=db.execute("SELECT i FROM items ORDER BY iid").fetchnumpy()["i"],
        )
        self.split = SimpleNamespace(t1=self.t1, t2=self.t2)

    @staticmethod
    def log(message):
        print(f"[02b] {message}", flush=True)

    def one(self, sql):
        row = self.db.execute(sql).fetchone()
        return row[0] if row else None

    def count(self, name):
        return self.one(f"SELECT count(*) FROM {name}")

    def _import(self, pairs):
        self.db.execute("CREATE TABLE raw(u BIGINT, i BIGINT, ts BIGINT, ordinal BIGINT)")
        self.log("Importing chronological pairs in batches")
        offset, previous = 0, None
        for batch in pd.read_csv(pairs, sep="\t", dtype="int64", chunksize=250_000, skip_blank_lines=False):
            if (list(batch.columns) != ["user", "item", "ts"] or not isinstance(batch.index, pd.RangeIndex)
                    or batch.isna().any().any() or (batch < 0).any().any()):
                raise ValueError("Expected nonnegative integer user,item,ts cut rows")
            keys = batch[["ts", "user", "item"]]
            if not keys.equals(keys.sort_values(["ts", "user", "item"], kind="mergesort")):
                raise ValueError("Cut rows must be chronological")
            first, last = tuple(keys.iloc[0]), tuple(keys.iloc[-1])
            if previous is not None and first < previous:
                raise ValueError("Cut rows must be chronological across batches")
            previous = last
            chunk = batch.rename(columns={"user": "u", "item": "i"})
            chunk["ordinal"] = np.arange(offset, offset + len(chunk), dtype="int64")
            self.db.register("input_batch", chunk)
            self.db.execute("INSERT INTO raw SELECT u,i,ts,ordinal FROM input_batch")
            self.db.unregister("input_batch")
            offset += len(batch)
            if offset % 4_000_000 == 0:
                self.log(f"Imported {offset:,} pairs")
        self.log(f"Imported {offset:,} pairs; no sampling")

    def quantile(self, n, q):
        # Match NumPy's default linear interpolation and integer truncation exactly.
        position = (n - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        values = self.db.execute(f"SELECT ts FROM raw WHERE ordinal IN ({lower},{upper}) "
                                 "ORDER BY ordinal").fetchall()
        return int(np.quantile(np.array([row[0] for row in values], dtype="int64"), position - lower))

    def _audit(self, n, trace):
        cfg, db = self.config, self.db
        if cfg.kcore > 1:
            for key in ("u", "i"):
                if self.one(f"SELECT 1 FROM train GROUP BY {key} HAVING count(*) < {cfg.kcore} LIMIT 1"):
                    raise ValueError("Train-only core has vertices below the minimum degree")
        # A core must include every source edge between its surviving vertices.
        induced = f"SELECT r.* FROM raw r SEMI JOIN users USING(u) SEMI JOIN items USING(i) WHERE ts < {self.t1}"
        if self.one(f"SELECT 1 FROM (({induced}) EXCEPT SELECT * FROM train) LIMIT 1"):
            raise ValueError("Incomplete train core")
        db.execute(f"CREATE VIEW classified AS SELECT r.*, "
                   f"CASE WHEN ts < {self.t1} THEN 'train' WHEN ts < {self.t2} THEN 'val' ELSE 'test' END AS split, "
                   f"CASE WHEN ts < {self.t1} AND (uid IS NULL OR iid IS NULL) THEN 'train_kcore' "
                   "WHEN uid IS NULL THEN 'cold_user' WHEN iid IS NULL THEN 'cold_item' ELSE '' END AS reason "
                   "FROM raw r LEFT JOIN users USING(u) LEFT JOIN items USING(i)")
        kept = {name: self.count(name) for name in ("train", "val", "test")}
        reasons = {name: dict.fromkeys(("train_kcore", "cold_user", "cold_item"), 0) for name in kept}
        for name, reason, total in db.execute(
            "SELECT split,reason,count(*) FROM classified WHERE reason <> '' GROUP BY split,reason"
        ).fetchall():
            reasons[name][reason] = total
        dropped = sum(sum(values.values()) for values in reasons.values())
        if sum(kept.values()) + dropped != n:
            raise ValueError("Input pair accounting does not balance")
        for name in kept:
            condition = f"split = '{name}' AND reason = ''"
            if self.one(f"SELECT count(*) FROM classified WHERE {condition}") != kept[name]:
                raise ValueError(f"Unexplained interactions in {name}")
            expected = f"SELECT u,i,ts,ordinal FROM classified WHERE {condition}"
            if self.one(f"SELECT 1 FROM (({expected}) EXCEPT SELECT * FROM {name}) LIMIT 1"):
                raise ValueError(f"Missing or changed interactions in {name}")
        # Masks are unfiltered source views; their cutoffs cannot use future rows.
        self.audit = {
            "input_pairs": n, "kept": kept, "dropped_by_reason": reasons, "dropped_pairs": dropped,
            "accounted_pairs": n, "unexplained_pairs": 0,
            "checks": dict.fromkeys(("global_quantiles", "split_boundaries", "train_only_kcore",
                                     "known_users_items", "complete_past_only_masks",
                                     "no_unexplained_interaction_loss"), True),
            "core_iterations": trace,
        }
        self.stats = {
            "n_raw": n, "n_duplicate_pairs_removed": 0,
            "period_sizes": {name: self.one(f"SELECT count(*) FROM classified WHERE split='{name}'") for name in kept},
            "train_pos_before_kcore": self.count("seen_before_val"), "train_pos": kept["train"],
            "n_users": self.count("users"), "n_items": self.count("items"),
        }
        for name in ("val", "test"):
            self.stats.update({
                f"{name}_pos": kept[name], f"{name}_users": self.one(f"SELECT count(DISTINCT u) FROM {name}"),
                f"{name}_pos_before_filter": self.stats["period_sizes"][name],
                f"{name}_dropped_cold_user_interactions": reasons[name]["cold_user"],
                f"{name}_dropped_cold_item_interactions": reasons[name]["cold_item"],
                f"{name}_cold_items": self.one(f"SELECT count(*) FROM (SELECT i FROM raw ANTI JOIN items USING(i) "
                                              f"WHERE ts >= {self.t1 if name == 'val' else self.t2} "
                                              + (f"AND ts < {self.t2} " if name == "val" else "")
                                              + "GROUP BY i)"),
            })

    def copy(self, query, path, separator=","):
        self.db.execute(f"COPY ({query}) TO {literal(path)} (HEADER true, DELIMITER {literal(separator)})")

    def save(self, output, source_info):
        output.mkdir(parents=True, exist_ok=True)
        for part in PARTS:
            self.log(f"Writing processed {part}.csv")
            rating = "1 AS rating," if part in ("train", "val", "test") else ""
            self.copy(f'SELECT u AS "user",i AS item,{rating}ts FROM {part} ORDER BY ts,u,i', output / f"{part}.csv")
        self.copy('SELECT u AS "user",i AS item,1 AS rating,ts,split,reason FROM classified '
                  "WHERE reason <> '' ORDER BY ts,u,i", output / "dropped_interactions.csv")
        write_json(output / "meta.json", {
            "dataset": "lastfm", "config": asdict(self.config), "t1": self.t1, "t2": self.t2,
            **self.stats, "clean": {"raw_lfm1b": source_info},
            "sha256": {part: sha256_file(output / f"{part}.csv") for part in PARTS},
        })

    def rows(self, query):
        cursor = self.db.execute(query)
        while batch := cursor.fetchmany(50_000):
            yield from batch

    def grouped(self, out, name, table, labels):
        query = f"SELECT uid,iid,ts FROM {table} JOIN users USING(u) JOIN items USING(i) ORDER BY uid,ts,iid"
        with ExitStack() as stack:
            files = [stack.enter_context((out / f"{name}{suffix}.txt").open(
                "w", encoding="utf-8", newline="\n",
            )) for suffix in (["", "_rating", "_timestamp"] if labels else [""])]
            previous, count = None, 0
            buffers = [[] for _ in files]
            for user, item, ts in self.rows(query):
                prefix = ("" if previous is None else "\n") + str(user) if user != previous else ""
                values = [str(item), f"{item}:1", f"{item}:{ts}"]
                for buffer, value in zip(buffers, values[:len(files)], strict=True):
                    buffer.append(prefix + " " + value)
                previous, count = user, count + 1
                if count % 50_000 == 0:
                    for handle, buffer in zip(files, buffers, strict=True):
                        handle.write("".join(buffer))
                        buffer.clear()
                if count % 5_000_000 == 0:
                    self.log(f"Exported {name}: {count:,} pairs")
            for handle, buffer in zip(files, buffers, strict=True):
                handle.write("".join(buffer) + ("\n" if previous is not None else ""))

    def export(self, out, split, enc, entities, relations, kg_t1, kg_t2, links, timestamps, info, **kwargs):
        """Same KGAT layout as export_kgat; interaction tables never enter Python RAM."""
        out.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            self.log(f"Writing KGAT {name} and aligned timestamps")
            self.grouped(out, name, name, True)
        for name in ("val", "test"):
            self.log(f"Writing complete {name} mask")
            self.grouped(out, f"{name}_mask", f"seen_before_{name}", False)
        self.log("Writing chronological interactions")
        with (out / "interactions_chronological.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["ordinal", "split", "user", "item", "rating", "timestamp"])
            ordinal = 0
            for name in ("train", "val", "test"):
                query = f"SELECT uid,iid,ts FROM {name} JOIN users USING(u) JOIN items USING(i) ORDER BY ts,uid,iid"
                for user, item, ts in self.rows(query):
                    writer.writerow([ordinal, name, user, item, 1, ts])
                    ordinal += 1
        with (out / "user_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("org_id remap_id\n")
            for raw, uid in self.rows("SELECT u,uid FROM users ORDER BY uid"):
                handle.write(f"{raw} {uid}\n")
        with (out / "item_list.txt").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("org_id remap_id freebase_id\n")
            for raw, iid in self.rows("SELECT i,iid FROM items ORDER BY iid"):
                handle.write(f"{raw} {iid} {links.get(raw, 'NA')}\n")
        # The common finalizer writes canonical entity/relation metadata next.
        for name, graph in (("kg_final", kg_t1), ("kg_test", kg_t2)):
            graph.to_csv(out / f"{name}.txt", sep=" ", header=False, index=False, lineterminator="\n")
        return {
            **info, "timestamp_variant": True, "cutoffs": {"t1": self.t1, "t2": self.t2},
            "n_users": enc.n_users, "n_items": enc.n_items, "n_entities": len(entities),
            "n_relations": len(relations), "interactions": self.audit["kept"],
            "kg_triples": {"kg_final.txt (t1)": len(kg_t1), "kg_test.txt (t2)": len(kg_t2)},
            "mask_rows": {name: self.count(f"seen_before_{name}") for name in ("val", "test")},
        }


@contextmanager
def disk_split(pairs, config, scratch_dir, memory_limit):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError('Install disk-backed support: python -m pip install -e ".[drive]"') from exc
    root = Path(scratch_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lfm1b-build-", dir=root) as temporary:
        with duckdb.connect(str(Path(temporary) / "split.duckdb"), config={
            "memory_limit": memory_limit, "threads": 2,
            "temp_directory": str(Path(temporary) / "spill"), "preserve_insertion_order": True,
        }) as db:
            yield DiskSplit(db, pairs, config)
