# 0018 — Extract Last-FM KG facts from a Freebase dump

- Status: preparation explicitly requested by Phôn on 2026-11-22; no official model run approved.
- Supersedes: 0017's Drive-only fact source and one-hop implementation for the new preparation.
- Authorization: Phôn explicitly allowed a complete Freebase dump outside Drive.

## Decision

1. Interactions remain the completed raw Drive cut. No interaction, split or KG
   triple from KGAT's published dataset is an input. Preserve all existing split,
   train-only 5-core, mask, timestamp and accounting rules.
2. Extract facts from Freebase itself. Follow the bidirectional neighborhood
   extraction in [KGAT section 4.1](https://hexiangnan.github.io/papers/kdd19-KGAT.pdf)
   and [KB4Rec](https://github.com/RUCDM/KB4Rec#how-to-get-freebase-subgraph-with-our-linkage):
   first-hop triples have either endpoint in the seed set; second-hop triples
   have either endpoint in the first-hop entity set. Keep original directions.
3. Retain the previously requested nine-relation schema. This is an explicit
   restriction on traversal, not a claim that KGAT selected exactly nine relations
   before extraction. Do not fabricate missing facts or relabel predicates.
4. Use the explicit raw-track-to-MID map supplied in Drive. It has paper lineage
   and partial coverage; this work does not rebuild title matching for all tracks.
   Unlinked tracks retain interactions. The extraction may use all linked MIDs to
   cache a superset; the final KG is seeded only by train items after 5-core.
5. KGAT reports entity occurrence >=10 and relation occurrence >=50 but does not
   fully specify filtering order/iteration. Use a documented single entity pass
   followed by a relation pass, counting distinct source triples and both entity
   endpoints. Freeze eligibility from the temporally valid, train-seeded t1 graph
   and reuse it at t2, so future facts cannot select entities or relations.
6. Apply the temporal cutoff before traversal. Recompute <=2-hop reachability
   after quality filtering. Date evidence for both endpoints is required for
   incoming edges: use the later available endpoint-existence proxy, retaining
   blanks when neither endpoint has a usable date. This is not a statement date.
7. Process complete RDF dumps as streams. Three passes collect first-hop edges,
   second-hop edges, then names/date evidence for the final nodes. Keep extracted
   data and provenance on disk, not the complete source gzip. Validate compressed
   stream hashes across passes; malformed/truncated sources must fail.
8. The Google source returned HTTP 403 in the environment. A readable
   [Internet Archive Freebase mirror](https://archive.org/details/freebase-rdf-latest)
   was verified by a bounded gzip probe. Its metadata date is **2015-04-16**;
   never mislabel it as the previous Drive subset's 2015-08-09 snapshot. Record
   its source URL, pinned compressed size/SHA-1, extracted SHA-256 and source date.

## Consequences

- The workflow is 01b (already completed), 01c (Freebase extraction), then 02b
  (temporal split and full KGAT export). Do not rerun 01b for this source change.
- Three scans of a large compressed dump require substantial network/CPU time.
  Missing raw track titles still limit entity-link coverage. No identical paper
  graph, complete track coverage, or completed full extraction is claimed.
- Two-hop KG heads/tails may both be items or non-item entities. Support this with
  an explicit graph-scope metadata field and separate checks; keep legacy one-hop
  validation intact. Test-time entity availability checks must cover both ends.
- No scores, prior test results or model performance determine this preparation.
