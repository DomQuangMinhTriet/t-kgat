# 0014 — Last-FM: Stage 2 runs on Kaggle

- Status: accepted (2026-11-09, Triết); proposed 2026-11-08 (Triết). Follows 0013 (official Last-FM run on `lastfm-u25`; its
  item 6 drops Stage 2 first when time runs out). It was accepted on 2026-11-09, after the Stage 2 runs had finished, so the
  Outcome section below records what happened; items 1 to 6 are the rules the runs followed.
- Context: on the laptop (RTX 3050, 4 GB) KGAT on `lastfm-u25` took 23.2 h (registry, `seconds` column of the 20 validation
  runs); T-KGAT Stage 1 was on track for about 14.6 h when this was proposed (14.8 h in the registry at the end). What was
  left after Stage 1 was estimated at about 38 h: `tkgat_perm` (5 runs), `kgat_nokg` (20), `tkgat_kg` (15) and
  `tkgat_kg_perm` (5). Stage 2 alone was estimated at about 20 h.
- Decision:
  1. `tkgat_kg` and `tkgat_kg_perm` run on a Kaggle GPU, at git commit `ce55c0c`, on the same split (train, val and test
     sha256 equal to the laptop's, checked in the notebook). Everything else stays on the laptop.
  2. Gate: `scripts/measure_kgat.py` on `lastfm-u25`, KGAT kg:default, at most 14 s per epoch on Kaggle (laptop: 20.13 and
     20.52 s). Above 14 s Stage 2 stays on the laptop. History: the gate was first written as 12 s (estimate for `tkgat_kg`
     about 8.6 h, inside one 12 h session). The measurement on a Kaggle Tesla T4 (2026-11-08, commit `ce55c0c`, mean of
     epochs 2 to 3) gave 13.81 s, above 12 s. At Triết's request the gate was raised to 14 s after seeing this number, so
     this is a deliberate change, not a pre-registered pass; the estimate for `tkgat_kg` became about 9.9 h (scaled from
     8.6 h), and the margin to 14 s is 1.4%. Risks accepted: the margin is thin and Kaggle GPUs are shared, so a session may
     hit the 12 h limit; the second session (`tkgat_kg_perm`) can only start after the first.
  3. Grids, five seeds, epochs, selection on validation and test scored once are unchanged. No number is copied by hand:
     the Kaggle registry rows are appended unchanged to `results/registry.csv` in a commit of their own; the `device` column
     records the GPU.
  4. A comparison across machines (Stage 2 against Stage 1 or KGAT) is reported with the note that the GPUs differ;
     `tkgat_kg` against `tkgat_kg_perm` is on one machine.
  5. If the Kaggle run fails, Stage 2 runs on the laptop. The two are never both run (no duplicate registry keys).
  6. Advisor: on 2026-11-02 the advisor suggested in the group chat using Kaggle (30 free GPU hours per week, at most 12 hours
     per session) for heavy runs. The group ran Stage 2 on Kaggle after that message. The specific details (Stage 2 on a
     Tesla T4, a time estimate that was too low, the gate raised from 12 s to 14 s) were not yet reported to the advisor; they
     are reported in the next report (planned 2026-11-14 or 15).
- Outcome (checked on 2026-11-09 against `results/registry.csv` and the notes log `nhat-ky-thay-doi.md`):
  - The estimate was too low: about 9.9 h estimated for the 15 `tkgat_kg` runs, about 12.3 h real (the `seconds` column of
    the 15 validation rows sums to 44,361 s; mean 2,957 s per run). The wall time of sessions 1 and 2 together was
    45,494 s (12.6 h), which includes start-up and test scoring.
  - Session 1 (`tkgat_kg`, 15 runs): reached the 12 h cap at 43,211.5 s after 14 of 15 runs. Kaggle kept the output files of
    the cancelled session.
  - Session 2 (`tkgat_kg` with `--resume`; `scripts/run.py` supports it for `tkgat_kg`, config name `tkgat`): 2,282 s,
    finished the missing run (gamma 10, seed 2023, 2,195 s in the registry), run as `timeout 38000 ... --resume` on the
    output of session 1.
  - Session 3 (`tkgat_kg_perm`, 5 runs): 14,042 s (the 5 validation rows sum to 13,982 s).
  - The `.git` directory inside a Kaggle output is unusable, so the repository was rebuilt from the bundle at `ce55c0c` and
    the earlier `results/` copied over it.
  - All three sessions: Tesla T4, commit `ce55c0c`, split `f940c132a2bf`. `results/registry.csv` grew from 656 to 686 data
    rows (657 to 687 lines with the header): 30 Kaggle rows (`tkgat_kg` 15 validation and 5 test, `tkgat_kg_perm` 5
    validation and 5 test), none with a key that a laptop row already has. All 139 `lastfm-u25` rows are at commit `ce55c0c`
    and none is `-dirty`.
- Consequences:
  - The compute of Stage 2 is no longer the laptop's. GPU nondeterminism means a rerun on the laptop would give close but
    not identical numbers.
  - Results of Stage 2 against KGAT or Stage 1 compare two machines (Tesla T4 against RTX 3050) and carry that note;
    `tkgat_kg` against `tkgat_kg_perm` is one machine.
  - The extra analyses proposed to the advisor (a block-wise permutation control, a cohort from 2014-04-01, a `drop_undated`
    KG variant, gamma {30, 100} on validation only, MostPop and BPR-MF on full Last-FM) are not part of this decision. Each
    needs its own decision record before it runs, and each would be reported as added after the test scores were seen.
