"""Extract a reproducible two-hop cache directly from a streamed Freebase RDF dump."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from tkgat.data.drive import _ChunkReader, _range_info
from tkgat.data.lastfm_raw import local_source_path, source_sha256, write_json
from tkgat.kg.lastfm import DATE_PREDICATES
from tkgat.kg.lastfm_freebase import PAPER_RELATIONS, freebase_id
from tkgat.registry import sha256_file

DEFAULT_DUMP = "https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz"
DEFAULT_SIZE = 31_305_093_084
DEFAULT_SHA1 = "d1950f3c966932358a79a02cd1d56ae3165b26f4"
DEFAULT_DATE = "2015-04-16"
DEFAULT_EXTRACT = "data/raw/lastfm-v02-freebase/freebase.tsv"
NS = "<http://rdf.freebase.com/ns/"
NT_LINE = re.compile(r"^\s*(<[^>]+>)\s+(<[^>]+>)\s+(.+)\s+\.\s*$")


def http_chunks(url):
    """Resume compressed bytes without resetting the live gzip decoder."""
    import requests

    if urlsplit(url).scheme != "https":
        raise ValueError("Remote dump sources must use HTTPS")
    offset, retries, total, validator = 0, 0, None, None
    with requests.Session() as session:
        while True:
            response = None
            try:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                    if validator:
                        headers["If-Range"] = validator[1]
                response = session.get(url, headers=headers, stream=True, timeout=(15, 60))
                response.raise_for_status()
                total, end, validator = _range_info(response, offset, total, validator)
                if url == DEFAULT_DUMP and total is not None and total != DEFAULT_SIZE:
                    raise ValueError("Freebase mirror size does not match the pinned source")
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    if end is not None and offset + len(chunk) > end:
                        raise RuntimeError("Dump response exceeds its announced range")
                    offset += len(chunk)
                    yield chunk
                if end is not None and offset < end:
                    raise requests.exceptions.ChunkedEncodingError("Short dump response")
                if total is None or offset == total:
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError, requests.exceptions.HTTPError) as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Freebase dump rejected the request: HTTP {status}") from exc
                if retries >= 5:
                    raise RuntimeError(f"Freebase stream failed at byte {offset}; current pass is incomplete") from exc
                retries += 1
                print(f"[01c] Reconnecting dump at byte {offset:,}, retry {retries}/5", flush=True)
                time.sleep(min(2 ** (retries - 1), 16))
            finally:
                if response is not None:
                    response.close()


@contextmanager
def dump_text(source, fingerprint):
    """Hash the actual compressed/raw input, with CRC validation by GzipFile."""
    with closing(http_chunks(source)) if str(source).startswith("https://") else open(source, "rb") as binary:
        chunks = binary if str(source).startswith("https://") else iter(lambda: binary.read(1024 * 1024), b"")
        sha1, sha256, size = hashlib.sha1(), hashlib.sha256(), 0

        def hashed():
            nonlocal size
            for chunk in chunks:
                sha1.update(chunk)
                sha256.update(chunk)
                size += len(chunk)
                yield chunk

        with io.BufferedReader(_ChunkReader(hashed())) as buffered:
            compressed = str(source).split("?", 1)[0].endswith(".gz")
            with gzip.GzipFile(fileobj=buffered) if compressed else io.BufferedReader(buffered) as decoded:
                with io.TextIOWrapper(decoded, encoding="utf-8") as handle:
                    yield handle
        fingerprint.update(size=size, sha1=sha1.hexdigest(), sha256=sha256.hexdigest())


def rdf_rows(handle):
    """Parse the official tab-delimited N-Triples, preserving exact object terms."""
    for number, line in enumerate(handle, 1):
        value = line.rstrip("\r\n")
        if not value.strip() or value.lstrip().startswith("#"):
            continue
        fields = value.split("\t")
        if len(fields) == 4 and fields[-1].strip() == ".":
            subject, predicate, obj = fields[:3]
        else:
            match = NT_LINE.fullmatch(value)
            if match is None:
                raise ValueError(f"Malformed Freebase N-Triples at line {number}")
            subject, predicate, obj = match.groups()
        if not subject.startswith("<") or not predicate.startswith("<"):
            raise ValueError(f"Expected RDF URI subject/predicate at line {number}")
        yield subject.strip(), predicate.strip(), obj.strip()


def resource(term):
    return freebase_id(term) if term.startswith(NS) and term.endswith(">") else None


def train_seed_mids(pairs, links, *, scratch_dir=None, memory_limit="1GB"):
    """Derive extraction seeds from the exact protocol-v2 train core, on disk."""
    from tkgat.data.lastfm_disk import disk_split
    from tkgat.data.protocol_v2 import KCORE, Q_TEST, Q_VAL, check_cut_manifest
    from tkgat.data.split import SplitConfig

    meta = json.loads(Path(str(pairs) + ".meta.json").read_text(encoding="utf-8"))
    check_cut_manifest(meta)
    with disk_split(pairs, SplitConfig(Q_VAL, Q_TEST, 1, KCORE),
                    scratch_dir or Path(pairs).parent, memory_limit) as split:
        seeds = {links[int(raw)] for raw in split.enc.item_ids if int(raw) in links}
    return seeds, meta["sha256"]


def extract_dump(source, item_links, output, *, snapshot_date=None, overwrite=False,
                 train_pairs=None, scratch_dir=None, memory_limit="1GB"):
    """Cache two-hop edges, optionally restricted to the protocol-v2 train items."""
    from tkgat.pipeline.lastfm_raw import read_item_links

    started = time.perf_counter()
    output = Path(output)
    manifest = Path(str(output) + ".meta.json")
    for path in (source, item_links, train_pairs, str(train_pairs) + ".meta.json" if train_pairs else None):
        if path is None:
            continue
        local = local_source_path(path) if not str(path).startswith("https://") else None
        if local in {output.resolve(), manifest.resolve()}:
            raise ValueError("Dump/mapping sources and output must differ")
    if not overwrite and (output.exists() or manifest.exists()):
        raise FileExistsError(f"Extraction exists: {output}; use another path or --overwrite")
    links = read_item_links(item_links)
    mapping_sha256 = source_sha256(item_links)
    seeds = set(links.values())
    seed_pairs_sha256 = None
    if train_pairs is not None:
        seeds, seed_pairs_sha256 = train_seed_mids(train_pairs, links, scratch_dir=scratch_dir,
                                                  memory_limit=memory_limit)
    if not seeds and train_pairs is None:
        raise ValueError("No explicit track-to-Freebase links; cannot seed dump extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, evidence_predicates = set(PAPER_RELATIONS), {*DATE_PREDICATES, "type.object.name"}
    reports, first_fingerprint = [], None
    with tempfile.TemporaryDirectory(prefix="freebase-extract-", dir=output.parent) as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "facts.sqlite"))) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA cache_size=-65536")
            db.execute("CREATE TABLE edges(h TEXT,r TEXT,t TEXT,obj TEXT,PRIMARY KEY(h,r,t)) WITHOUT ROWID")
            db.execute("CREATE TABLE evidence(s TEXT,p TEXT,o TEXT,PRIMARY KEY(s,p,o)) WITHOUT ROWID")
            frontier = seeds
            for step in (1, 2, 3):
                pass_started = time.perf_counter()
                print(f"[01c] Pass {step}/3: {'date/name evidence' if step == 3 else 'hop ' + str(step)}; "
                      f"{len(frontier):,} frontier entities", flush=True)
                report = {"pass": step, "rdf_rows": 0, "matched_rows": 0, "new_unique_rows": 0,
                          "excluded_by_reason": {}}
                fingerprint, pending = {}, []

                def flush(pending=pending, report=report, step=step):
                    if not pending:
                        return
                    previous = db.total_changes
                    table = "evidence" if step == 3 else "edges"
                    marks = "?,?,?" if step == 3 else "?,?,?,?"
                    db.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({marks})", pending)
                    db.commit()
                    report["new_unique_rows"] += db.total_changes - previous
                    pending.clear()

                with dump_text(str(source), fingerprint) as handle:
                    for subject, predicate, obj in rdf_rows(handle):
                        report["rdf_rows"] += 1
                        head, rel = resource(subject), freebase_id(predicate)
                        if step == 3:
                            if head in frontier and rel in evidence_predicates:
                                pending.append((head, rel, obj))
                                report["matched_rows"] += 1
                            else:
                                key = "not_required_evidence"
                                report["excluded_by_reason"][key] = report["excluded_by_reason"].get(key, 0) + 1
                        elif rel in selected:
                            tail = resource(obj)
                            if head and tail and (head in frontier or tail in frontier):
                                pending.append((head, rel, tail, obj))
                                report["matched_rows"] += 1
                            else:
                                key = "non_freebase_resource_endpoint" if not head or not tail else "outside_frontier"
                                report["excluded_by_reason"][key] = report["excluded_by_reason"].get(key, 0) + 1
                        else:
                            key = "relation_not_selected"
                            report["excluded_by_reason"][key] = report["excluded_by_reason"].get(key, 0) + 1
                        if len(pending) >= 10_000:
                            flush()
                        if report["rdf_rows"] % 5_000_000 == 0:
                            print(f"[01c] Pass {step}: scanned={report['rdf_rows']:,} "
                                  f"matched={report['matched_rows']:,}", flush=True)
                flush()
                if str(source) == DEFAULT_DUMP and (
                    fingerprint["size"] != DEFAULT_SIZE or fingerprint["sha1"] != DEFAULT_SHA1
                ):
                    raise ValueError("Freebase mirror does not match its pinned size/SHA-1")
                if first_fingerprint is None:
                    first_fingerprint = fingerprint
                elif fingerprint != first_fingerprint:
                    raise ValueError("Freebase source changed between extraction passes")
                report["fingerprint"] = fingerprint
                report["duplicates_collapsed"] = report["matched_rows"] - report["new_unique_rows"]
                if report["rdf_rows"] != report["matched_rows"] + sum(report["excluded_by_reason"].values()):
                    raise RuntimeError("Freebase extraction row accounting does not balance")
                reports.append(report)
                # Freeze the frontier between passes; never grow it during a pass.
                frontier = seeds | {row[0] for row in db.execute("SELECT h FROM edges UNION SELECT t FROM edges")}
                if step <= 2:
                    n_entities, max_degree = db.execute(
                        "SELECT count(*),coalesce(max(degree),0) FROM ("
                        "SELECT node,count(*) AS degree FROM ("
                        "SELECT h AS node FROM edges UNION ALL SELECT t AS node FROM edges"
                        ") GROUP BY node)"
                    ).fetchone()
                    report["graph_stats"] = {
                        "triples": db.execute("SELECT count(*) FROM edges").fetchone()[0],
                        "entities": n_entities, "max_degree": max_degree,
                    }
                    print(f"[01c] Hop {step} cumulative, before time filtering: "
                          f"triples={report['graph_stats']['triples']:,} entities={n_entities:,} "
                          f"max_degree={max_degree:,}", flush=True)
                report["elapsed_seconds"] = time.perf_counter() - pass_started
                print(f"[01c] Pass {step} complete: {report['rdf_rows']:,} RDF rows; "
                      f"elapsed={report['elapsed_seconds']:.2f}s", flush=True)
            edge_count = db.execute("SELECT count(*) FROM edges").fetchone()[0]
            if source_sha256(item_links) != mapping_sha256:
                raise ValueError("Mapping changed during dump extraction; no extraction was published")
            evidence_count = db.execute("SELECT count(*) FROM evidence").fetchone()[0]
            pending_output = Path(temporary) / "freebase.tsv"
            with pending_output.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("subject\tpredicate\tobject\n")
                for row in db.execute("SELECT h,r,obj FROM edges UNION SELECT s,p,o FROM evidence ORDER BY 1,2,3"):
                    handle.write("\t".join(row) + "\n")
            meta = {
                "source": str(source), "source_snapshot": snapshot_date or
                (DEFAULT_DATE if str(source) == DEFAULT_DUMP else None),
                "source_fingerprint": first_fingerprint, "source_is_paper_kg": False,
                "scope": "two-hop", "relations": list(PAPER_RELATIONS),
                "seed_mids_sha256": hashlib.sha256("\n".join(sorted(seeds)).encode()).hexdigest(),
                "snapshot_date_basis": "mirror metadata date; not verified as the August dump",
                "item_links": str(item_links), "linked_tracks": len(links), "seed_mids": len(seeds),
                "item_links_sha256": mapping_sha256,
                "seed_scope": "protocol-v2 train items" if train_pairs is not None else
                "all explicit supplied links; 02b re-seeds from train only",
                "seed_pairs_sha256": seed_pairs_sha256,
                "hops": 2, "directions": "either endpoint, original RDF direction retained",
                "passes": reports, "resource_facts": edge_count, "evidence_rows": evidence_count,
                "degree_definition": "in_degree + out_degree over distinct RDF triples; self-loops count twice",
                "sha256": sha256_file(pending_output), "source_dump_saved_locally": False,
                "timing_seconds": {"total": time.perf_counter() - started},
            }
            # The full dump was streamed; only this extracted cache is published.
            os.replace(pending_output, output)
            write_json(manifest, meta)
    print(f"[01c] Extraction complete: elapsed={meta['timing_seconds']['total']:.2f}s", flush=True)
    return meta


def validate_extract(path):
    """Refuse unverified/incomplete caches before expensive interaction processing."""
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    if meta.get("scope") != "two-hop" or meta.get("source_is_paper_kg") is not False:
        raise ValueError("Expected the Freebase dump extraction from 01c, never a prebuilt paper KG")
    if sha256_file(path) != meta["sha256"]:
        raise ValueError("Extracted Freebase SHA-256 does not match its manifest")
    return meta
