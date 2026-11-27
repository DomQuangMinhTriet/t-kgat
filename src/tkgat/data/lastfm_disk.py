"""Disk-backed raw Last-FM splitting and streaming KGAT interaction export."""

from __future__ import annotations

import csv
import hashlib
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
    def __init__(self, db, pairs, config, *, target_interactions=None, user_seed=2019):
        self.db, self.config = db, config
        self.sampling = None
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
        trace = self._prune_core()
        if not self.count("train"):
            raise ValueError("empty training set after train-only k-core")
        if target_interactions is not None:
            trace = self._sample_train(target_interactions, user_seed)
        db.execute("CREATE VIEW selected_source AS SELECT r.* FROM raw r" + (
            " SEMI JOIN sampled_users USING(u)" if self.sampling is not None else ""
        ))
        db.execute("CREATE TABLE users AS SELECT u, row_number() OVER (ORDER BY u)-1 AS uid "
                   "FROM (SELECT DISTINCT u FROM train)")
        db.execute("CREATE TABLE items AS SELECT i, row_number() OVER (ORDER BY i)-1 AS iid "
                   "FROM (SELECT DISTINCT i FROM train)")
        for name, condition in (("val", f"ts >= {self.t1} AND ts < {self.t2}"),
                                ("test", f"ts >= {self.t2}")):
            db.execute(f"CREATE TABLE {name} AS SELECT r.* FROM raw r "
                       f"SEMI JOIN users USING(u) SEMI JOIN items USING(i) WHERE {condition}")
        for name, cutoff in (("seen_before_val", self.t1), ("seen_before_test", self.t2)):
            db.execute(f"CREATE VIEW {name} AS SELECT * FROM selected_source WHERE ts < {cutoff}")
        self.log("Auditing complete core, split boundaries, masks and pair accounting")
        self._audit(n, trace)
        self.enc = SimpleNamespace(
            n_users=self.count("users"), n_items=self.count("items"),
            user_ids=db.execute("SELECT u FROM users ORDER BY uid").fetchnumpy()["u"],
            item_ids=db.execute("SELECT i FROM items ORDER BY iid").fetchnumpy()["i"],
        )
        self.split = SimpleNamespace(t1=self.t1, t2=self.t2)

    def _prune_core(self, table="train"):
        """Iterative core on a training-only table; identifiers are internal constants."""
        assert table in {"train", "audit_train"}
        trace, before = [], self.count(table)
        if self.config.kcore > 1:
            while True:
                for key, good in (("u", "good_users"), ("i", "good_items")):
                    self.db.execute(f"CREATE OR REPLACE TABLE {good} AS SELECT {key} FROM {table} "
                                    f"GROUP BY {key} HAVING count(*) >= {self.config.kcore}")
                self.db.execute(f"CREATE TABLE next_train AS SELECT * FROM {table} "
                                "SEMI JOIN good_users USING(u) SEMI JOIN good_items USING(i)")
                after = self.count("next_train")
                trace.append({"before": before, "after": after})
                self.log(f"{table} core iteration {len(trace)}: {before:,} -> {after:,}")
                self.db.execute(f"DROP TABLE {table}")
                self.db.execute(f"ALTER TABLE next_train RENAME TO {table}")
                if after == before:
                    break
                before = after
        return trace

    def _sample_train(self, target, seed):
        """Closest nonempty core of a seeded user prefix, without reading val/test counts."""
        if target < 1 or seed < 0:
            raise ValueError("target_interactions must be positive and user_seed nonnegative")
        db = self.db
        target_train = max(1, int(target * self.config.q_val))
        users = [int(row[0]) for row in db.execute("SELECT DISTINCT u FROM train").fetchall()]
        users.sort(key=lambda user: (hashlib.sha256(f"{seed}:{user}".encode("ascii")).digest(), user))
        ranked = pd.DataFrame({"u": users, "user_rank": np.arange(1, len(users) + 1, dtype="int64")})
        db.register("ranked_input", ranked)
        db.execute("CREATE TABLE ranked_users AS SELECT * FROM ranked_input")
        db.unregister("ranked_input")
        full_count = self.count("train")
        db.execute("ALTER TABLE train RENAME TO full_train")
        probes = []

        def probe(prefix):
            db.execute("DROP TABLE IF EXISTS train")
            db.execute(f"CREATE TABLE train AS SELECT f.* FROM full_train f SEMI JOIN "
                       f"(SELECT u FROM ranked_users WHERE user_rank <= {prefix}) s USING(u)")
            trace = self._prune_core()
            count = self.count("train")
            probes.append({"prefix_users": prefix, "train_pairs": count})
            self.log(f"User sampling: prefix={prefix:,}, {self.config.kcore}-core={count:,}; target={target_train:,}")
            return count, trace

        if target_train >= full_count:
            chosen = len(users)
        else:
            lo, lo_count = 0, 0
            hi = min(len(users), max(1, math.ceil(len(users) * target_train / full_count)))
            hi_count, _ = probe(hi)
            while hi_count < target_train and hi < len(users):
                lo, lo_count = hi, hi_count
                hi = min(len(users), hi * 2)
                hi_count, _ = probe(hi)
            while hi - lo > 1:
                mid = (hi + lo) // 2
                count, _ = probe(mid)
                if count <= target_train:
                    lo, lo_count = mid, count
                else:
                    hi, hi_count = mid, count
            candidates = [(hi, hi_count)] + ([(lo, lo_count)] if lo_count > 0 else [])
            chosen = min(candidates, key=lambda pair: (abs(pair[1] - target_train), pair[0]))[0]
        final_count, _ = probe(chosen)
        if not final_count:
            raise ValueError("No nonempty sampled training core")
        db.execute(f"CREATE TABLE sampled_users AS SELECT u,user_rank FROM ranked_users WHERE user_rank <= {chosen}")
        # Recompute from the original sampled histories, independently of the full core used for probes.
        db.execute(f"CREATE TABLE audit_train AS SELECT r.* FROM raw r SEMI JOIN sampled_users USING(u) "
                   f"WHERE ts < {self.t1}")
        source_trace = self._prune_core("audit_train")
        if self.one("SELECT 1 FROM ((SELECT * FROM audit_train EXCEPT SELECT * FROM train) "
                    "UNION ALL (SELECT * FROM train EXCEPT SELECT * FROM audit_train)) LIMIT 1"):
            raise ValueError("Sampled train differs from the complete core of the selected histories")
        db.execute("DROP TABLE audit_train")
        self.sampling = {
            "method": "sha256(seed:user) prefix, closest nonempty train-only core",
            "seed": seed, "target_total_interactions": target, "target_train_pairs": target_train,
            "actual_train_pairs": final_count, "eligible_users": len(users), "sampled_users": chosen,
            "selection_reference": "full-source t1 training core only; no val/test counts or scores",
            "cutoff_reference": "global quantiles of the full original pair file, frozen before sampling",
            "probes": probes, "full_core_train_pairs": full_count,
            "note": "Approximate total; whole selected user histories retained, not a paper reconstruction.",
        }
        db.execute("DROP TABLE full_train")
        return source_trace

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
        self.log(f"Imported {offset:,} source pairs")

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
        sampling_join = " LEFT JOIN sampled_users s USING(u)" if self.sampling is not None else ""
        sampling_reason = "WHEN s.user_rank IS NULL THEN 'outside_user_sample' " if self.sampling is not None else ""
        db.execute(f"CREATE VIEW classified AS SELECT r.*, "
                   f"CASE WHEN ts < {self.t1} THEN 'train' WHEN ts < {self.t2} THEN 'val' ELSE 'test' END AS split, "
                   f"CASE {sampling_reason}WHEN ts < {self.t1} AND (uid IS NULL OR iid IS NULL) THEN 'train_kcore' "
                   "WHEN uid IS NULL THEN 'cold_user' WHEN iid IS NULL THEN 'cold_item' ELSE '' END AS reason "
                   "FROM raw r LEFT JOIN users USING(u) LEFT JOIN items USING(i)" + sampling_join)
        kept = {name: self.count(name) for name in ("train", "val", "test")}
        reasons = {name: dict.fromkeys(("train_kcore", "cold_user", "cold_item"), 0) for name in kept}
        if self.sampling is not None:
            for values in reasons.values():
                values["outside_user_sample"] = 0
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
        if self.sampling is not None:
            self.sampling["actual_interactions"] = kept
            self.sampling["actual_total_interactions"] = sum(kept.values())
            self.audit["user_sampling"] = self.sampling
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
                f"{name}_cold_items": self.one("SELECT count(*) FROM (SELECT i FROM selected_source "
                                              "ANTI JOIN items USING(i) "
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
                  "WHERE reason <> '' AND reason <> 'outside_user_sample' ORDER BY ts,u,i",
                  output / "dropped_interactions.csv")
        if self.sampling is not None:
            self.copy('SELECT u AS "user",user_rank FROM sampled_users ORDER BY user_rank',
                      output / "sampled_users.tsv", "\t")
            write_json(output / "sampling_report.json", {
                **self.sampling, "source_pairs_sha256": self.cut_meta["sha256"],
                "sampled_users_sha256": sha256_file(output / "sampled_users.tsv"),
                "excluded_user_rows": "counted by split in validation_report.json; reconstruct from sampled_users.tsv",
            })
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
def disk_split(pairs, config, scratch_dir, memory_limit, *, target_interactions=None, user_seed=2019):
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
            yield DiskSplit(db, pairs, config, target_interactions=target_interactions, user_seed=user_seed)
