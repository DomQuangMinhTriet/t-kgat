# Contributing

## Ground rules
1. **No leakage.** Model selection (epochs, hyper-parameters, variants) uses the validation set only.
   The test set is scored once per selected configuration, by `scripts/run.py`.
2. **No hand-copied numbers.** Every number in a report or notebook comes from `results/registry.csv`
   or from files written by the scripts. Read the registry with `tkgat.registry.load()`, which keeps
   only the newest run of each (dataset, split_sha256, model, config, seed, phase); never edit or delete
   old rows. The registry "dataset" is the data config's `run_name`, so give every split its own run name.
   Spread over seeds is reported as mean ± **sample** standard deviation (`ddof=1`), as
   `scripts/run.py` does; with numpy use `np.std(x, ddof=1)` (pandas `.std()` already uses `ddof=1`).
   Per-user comparisons use `tkgat.eval.stats.paired_wilcoxon`, which applies `zero_method="wilcox"`
   (users with a zero difference are dropped before ranking); report its `n_nonzero` as the test's sample
   size next to `n_pairs`, and correct several comparisons with `holm`.
3. **Data stays out of git.** Download it with the instructions in `data/README.md`; verify with
   `python scripts/01_verify_data.py`.

## Workflow
- One GitHub issue per task (templates in `.github/ISSUE_TEMPLATE/`), labelled `data`, `kg`, `model`,
  `eval`, `docs` or `theory`, and attached to the milestone of its phase.
- One branch per issue: `feat/12-temporal-split`, `fix/15-mask-bug`, `exp/20-bprmf-ml1m`.
- Open a pull request; the other person reviews it. `main` is protected: CI must pass and one approval
  is required.
- Commit messages: `feat:`, `fix:`, `docs:`, `exp:`, `test:`, `chore:` followed by a short sentence.

## Language
Code, comments, commit messages and README files are written in English.
The data report, theory notes and the presentation notebook are written in Vietnamese.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Decisions
Every decision that changes the protocol gets a short file in `docs/decisions/` (context, decision,
consequences). Never edit an accepted decision; supersede it with a new one.
