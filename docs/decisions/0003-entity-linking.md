# 0003 — Linking MovieLens items to Freebase / YAGO through KB4Rec

- Status: accepted (2026-10-17)
- Context: KB4Rec links MovieLens-**20M** ids to Freebase and YAGO. MovieLens-1M shares the 20M id space
  (92.7% of ML-1M titles match the YAGO entity name with the same id using a crude normaliser; mismatches
  are mostly naming differences). MovieLens-100K uses its own ids.
- Decision: ML-100K → ML-1M by normalised title + year (rules in `src/tkgat/kg/linking.py`, applied in
  order: title+year, title+year±1, full title+year, a.k.a. title+year, containment+year; matches must be
  unique). ML-1M id = ML-20M id → `ml2fb.txt` / `ml2yago.txt`.
- Result: 1,647 of 1,663 cleaned ML-100K items matched (99.0%); every non-exact match is listed in
  `data/processed/links/ml100k_links.csv` and was checked by hand. 16 items are not in ML-1M.
- Known KB4Rec errors found during the check: e.g. ML-1M 2588 "Clubland (1998)" → YAGO
  `<Cloudland_Canyon_State_Park>`. Linking errors are measured on a random sample in the entity-linking
  notebook.
