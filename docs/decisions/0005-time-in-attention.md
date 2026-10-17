# 0005 — Where time enters the attention

- Status: accepted (2026-10-17); supersedes the proposal to move time onto KG triples
- Decision: keep the 2026-08-07 T-KGAT design as the main model: attention on user–item edges is
  decayed by Δt (`a_time(h,r,t) = a(h,r,t) − δ_r·γ·Δt_norm`). Add a second variant in which time
  acts on KG triples through the validity date of each fact (decision 0004), and compare the two.
- Rationale: MovieLens-1M spans almost three years and may carry a temporal signal that
  MovieLens-100K (seven months) does not. Time on KG triples matches goal 3 on Notion and is measured
  as a comparison.
- Caveat to report: ratings are entered in sessions (78% of ML-100K and 72% of ML-1M ratings fall on
  the user's first day), so Δt between one user's edges is often close to zero. The
  timestamp-permutation control is mandatory for both variants.
- Data: experiments run on MovieLens-100K (Phôn's machine) and MovieLens-1M (Triết's machine) in
  parallel, with the same protocol and code.
- Also in phase 5 (advisor, 2026-09-18): rating value as a relation or weight, timestamp as a
  relation, LLM-assisted cleaning of film metadata.
- Paper alignment (phase 4): Recall/NDCG@20 with full ranking, 2-hop KG, CKE as a KG baseline, one
  reference run with the paper's random 80/20 protocol. Intended deviations: temporal split; 5-core
  main with 10-core as a sensitivity check.
