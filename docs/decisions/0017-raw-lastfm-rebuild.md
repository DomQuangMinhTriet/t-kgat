# 0017 — Standalone Last-FM rebuild from the complete raw history

- Status: implementation scope requested by Phôn, 2026-11-21; not approval of an official training run.
- Context: Phôn explicitly requested a KGAT-format rebuild using raw LFM-1b
  interactions, independent of the paper's pairs, and confirmed **complete history**
  rather than a last-N-month window. This extends the preparation work of proposed
  0016; accepted historical decisions and old datasets are left intact.
- Disclosure: this new data preparation is requested after the first-protocol
  test results were seen. No model or score is used to choose a dataset here.

## Decision

1. The new, separate `lastfm-raw` preparation reads all raw users/tracks, deduplicates
   by earliest timestamp and defaults to no window, user sample or event threshold.
   The published interaction files, user map and pair cache are never inputs.
2. Cutting is separate from splitting. The disk-backed cut writes chronological
   raw-id pairs and a SHA-256 manifest; the builder verifies it before splitting.
3. Keep the existing global 0.8/0.9 pair-time quantiles, train-only 5-core,
   cold-start restrictions and ranking masks. Count exclusions; preserve existing
   leakage checks. The cut file itself contains every raw pair.
4. The raw interaction file and the Freebase TSV in the two supplied Drive ZIPs
   are primary. The support ZIP's already generated item map supplies numeric
   track-to-MID links, never an interaction whitelist. Missing links do not remove
   interactions. Its already generated KG is not used for facts.
5. Reuse 0006/0012 tail-date rules. Export timestamps and their date evidence as
   sidecars, disclosing **tail-existence proxy** semantics. Missing dates stay null.
   The supplied Freebase subset is not a full dump or an all-track mapping.
6. Export KGAT's original text layouts and a shared item/entity namespace, alongside
   the project's processed split/one-hop KG files. Preserve original data and scripts.
7. Phôn's subsequent explicit validation checklist requires complete raw-event and
   pair accounting: repeated listens are counted as collapsed, and every split
   exclusion has a recorded k-core/cold-user/cold-item reason. Validate complete
   past-only masks against the input, mapping/date alignment, and repeat-build
   file hashes. Malformed raw rows fail the build rather than being silently skipped.
8. Phôn clarified the source boundary: **only the supplied Drive data**, not the
   official website or extra LFM catalogs. An earlier native-catalog implementation
   was outside that scope and is removed. Freebase supplies facts as well as dates:
   only exact predicates from KGAT Last-FM's nine-relation schema are retained
   before temporal filtering. Other predicates supply labels/date evidence only.
   Literal terms, if any, get stable nodes; no content is relabelled as a validity
   timestamp. The explicit item map is already inside the support ZIP.
9. The final source clarification supersedes the earlier permission for KG
   fallback: **no prebuilt/paper KG facts may fill missing fields**. Keep the nine
   original relation ids, including zero-fact relations, and rebuild facts solely
   from the supplied Freebase TSV. Do not infer or relabel other predicates into
   the schema. Report source-wide relation presence separately from linked train
   coverage, count every exclusion, and preserve per-fact source/path/raw object.
10. The completed raw cut has hundreds of millions of pairs on a 16GB-RAM laptop.
    This changes execution, not selection rules: 02b automatically uses disk-backed
    DuckDB for at least 2 million pairs, with bounded import/export batches and
    external SQL aggregation/sorting. Preserve exact quantiles, iterative train
    core, complete masks, exclusion reasons, ids, timestamps and deterministic
    output. Compare generated file hashes against the pandas reference. The
    supplied cut is reused, so this does not reread Drive or resample users/items.

## Consequences

- Complete-history preparation differs from 0016's proposed windowed development
  set. This record does not accept or alter its training/model proposals.
- Keeping all raw tracks increases size and reduces KG coverage. A full rebuild
  cannot have complete KG links from the supplied partial sources alone.
- Cutting and the large-data 02b path use disk-backed aggregation. Train item
  vocabularies/KG metadata still require RAM. The pandas path and training loaders
  retain their original in-memory behavior; no full-scale training fit is claimed.
- An unbounded raw history retains nonnegative integer timestamp outliers. Bounds
  are explicit, configurable and counted rather than inferred from the raw maximum.
- No official result, training run, decision approval, commit or publication is
  part of this preparation. See `docs/lastfm-raw-rebuild.md` for commands and limits.
