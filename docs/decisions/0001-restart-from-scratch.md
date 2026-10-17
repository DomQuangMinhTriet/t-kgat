# 0001 — Restart from scratch in a new repository

- Status: accepted (2026-10-17)
- Context: at the 2026-10-17 review the advisor found that earlier results (Foursquare NYC, Amazon CDs,
  KG expansion) were not trustworthy: no validation set (best epoch chosen on test), k-core computed on the
  whole dataset before splitting, and knowledge-graph metadata collected years after the interactions.
- Decision: archive the old repository read-only; start `t-kgat` with one pipeline, automated leakage
  checks, and MovieLens-100K as the main dataset (advisor agreed, run on Phôn's machine) with
  MovieLens-1M run in parallel on Triết's machine, same protocol and code.
  The "Confidence" part is out of scope and dropped from the title: T-KGAT — Temporal Knowledge
  Graph Attention Network for Recommendation.
- Consequences: no number from the old repository is reused. Ideas may be reused with a note of origin.
  KGAT is re-run only after the advisor approves the new protocol (review on 2026-10-31).
