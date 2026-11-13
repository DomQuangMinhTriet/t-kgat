# 0015 — Last-FM latest listening timestamp per pair

- Status: implementation requested by Phôn on 2026-11-10; not run on real data.
- Context: Phôn requested that repeated Last-FM (user, item) events retain the
  **largest** listening timestamp instead of the smallest. The shared support ZIP
  contains only `kgat_author_earliest.tsv`; later timestamps cannot be recovered
  from that cache. Accepted records 0002, 0012 and 0013 remain unchanged.
- Change, scoped to the new Last-FM variant:
  1. Compute `max(timestamp)` over all raw events of each published KGAT pair.
     Input order does not affect the timestamp. Keep the published user/item/pair
     universe, the seeded 25% user sample, and the existing temporal split,
     train-only k-core, cold-start filtering and boundary checks.
  2. Use `lastfm-u25-latest`, a separate processed/export directory, and the
     `kgat_author_latest.tsv` cache when a cache is written by the local generator.
     A latest cache must carry matching timestamp-policy metadata. Never reuse
     or overwrite the earliest-only cache as if it contained last-listen dates.
  3. Read raw events from the repository-documented `data-last-fm.zip` on Drive
     (`1tXtohvT8RAFmz2vJo4vH90kkrpwD75jW`), linked from
     `docs/lastfm-temporal-setup.md` under folder
     `1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc`. Phôn explicitly selected this source
     on 2026-11-10. Sequentially decompress the `lfm1b-tracks.inter` member
     without saving the compressed ZIP or raw file. Retain the existing support
     ZIP as the source of Freebase facts.
  4. Do not compare latest-derived split hashes against the earliest reference.
     Existing earliest configs remain available for reproduction. MovieLens and
     the shared default pair-deduplication function are unchanged.
- Consequences and limitations:
  - The pair timestamp now means **last observed listen**, not first observed
    listen. Cut-offs, split membership, k-core survivors and hashes can change.
  - Choosing a pair's last event over the full history discards its earlier
    listens. A pair listened to before and after a cut-off may occur only in the
    later split; generated masks then describe the retained last-event records,
    not the user's complete pre-cut-off listening history. Passing the existing
    boundary checks does not remove this future-conditioned selection or prove
    that complete-history evaluation is leakage-free.
  - The original KGAT pair-universe selection bias and Freebase date limitations
    still apply. This is not the separate proposed window-free pair dataset.
  - Scanning the original raw history is necessary; reading the small
    earliest-only cache cannot implement this change. Network transfer and CPU
    time will therefore be much larger. No model results are claimed.
