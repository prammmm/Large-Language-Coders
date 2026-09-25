# CLAUDE.md — Operating Manual for Claude Code Sessions

Project: **Amazon ML Hackathon — Business Entity Resolution Challenge** (team: LargeLanguageCoders)
Technical lead: **Ayush** (strategy, evaluation, integration). Humans own architecture and decisions.

Claude is a technical assistant working under human direction:
**inspect → reason → propose → explain → wait for approval on major decisions → implement approved work → measure → report.**

Read these before starting substantial work:
- `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md` (architecture, phases, roles)
- `docs/DECISION_LOG.md` (what has been decided and why)
- `docs/EXPERIMENT_LOG.md` (what has been measured)
- `docs/DEVELOPMENT_LOG.md` (current state)

---

## CLAUDE OPERATING RULES

### RULE 1 — NO AUTONOMOUS GIT PUSH
Claude must NEVER `git push`, push to GitHub, create remote branches, modify remote repositories,
or open pull requests without explicit human instruction. The team manually reviews, commits and pushes.
(Local `git status` / `git diff` / `git log` are fine. Do not commit unless asked.)

### RULE 2 — PLAN BEFORE IMPLEMENTATION
Before a substantial task, state: (1) objective, (2) proposed approach, (3) affected files,
(4) dependencies, (5) risks, (6) how success will be measured.
Then WAIT for approval if the task is substantial or changes architecture.

### RULE 3 — EXPLAIN AFTER EXECUTION
After every meaningful execution, report:
```text
What I did / Files created / Files modified / Files deleted / Commands executed /
Results / Metrics / Errors / Fixes / Remaining issues / Next recommended step
```

### RULE 4 — DO NOT CHANGE ARCHITECTURE SILENTLY
If a better approach is found, explain: current approach, discovered problem, proposed
alternative, benefits, drawbacks, expected impact. Wait for approval when the change is material.
Record accepted changes in `docs/DECISION_LOG.md`.

### RULE 5 — NO RANDOM EXPERIMENTS
Every ML experiment has: Experiment ID, Hypothesis, Change, Expected effect, Actual result,
Interpretation, Decision. Log it in `docs/EXPERIMENT_LOG.md`. Never fabricate metrics — use `N/A` until measured.

### RULE 6 — REPRODUCIBILITY
Fixed seeds (default `42`), explicit config files, recorded parameters and metrics, versioned code,
deterministic preprocessing. Any result that can't be regenerated from code + config doesn't count.

### RULE 7 — DATA SAFETY
Raw data (`datasets/`) is READ-ONLY. Never modify, rewrite, re-save or "clean in place".
All derived data goes to `data/interim/` or `data/processed/`.

### RULE 8 — FAIR PLAY
Never use external business lookup or external entity-resolution information: no business directories,
Google Maps/Places, OpenStreetMap business lookup, government business registries, geocoding APIs,
commercial ER APIs, external business databases, or internet data augmentation. Only the provided
train/test data and labels, plus rule-compliant libraries/models, are allowed.
This holds unless the official challenge documentation explicitly changes it.

### RULE 9 — LEADERBOARD DISCIPLINE
Five leaderboard submissions per day (team-stated limit). Never recommend a submission without:
hypothesis, local validation result, expected benefit, and why it's worth a slot.
Anything that can be measured locally is measured locally.

### RULE 10 — SOURCE OF TRUTH
The official challenge documentation (`Problem Statement.docx`, `student_resource/README.md`, `student_resource/utils/validate_submission.py`, `student_resource/Documentation_template.md`)
is the source of truth for requirements, constraints, output format, evaluation, eligibility, allowed models,
licenses and fair play. If a general ML assumption conflicts with it, follow the documentation.
If something isn't covered, say "Not specified in the provided materials." Don't invent requirements.

---

## HARD CONSTRAINTS — NEVER VIOLATE (from the official problem statement)

1. **TSV** (tab-separated) outputs. ID lists are comma-separated, **no quoting**, no spaces.
2. `output/matching_results.tsv` has header `source1_entity_id<TAB>matched_entity_ids`.
3. `output/candidate_pairs.tsv` has header `source1_entity_id<TAB>candidate_entity_ids`.
4. **Exactly one row per test Source-1 entity** in both files. Missing S1 entities → rejection. Duplicate S1 rows → rejection.
5. **Empty list** (empty field) for singletons / no candidates.
6. **No duplicate IDs** within any list → otherwise rejection.
7. Lists contain **only S2-/S3- IDs that exist in the test set**. No S1 self-matches, no train IDs.
8. **matches ⊆ candidates** for every S1 row (validator warns; a violation means a pipeline bug).
9. `candidate_pairs.tsv` = the **exact final set the model runs inference over**, not an earlier looser blocking pass.
10. **Country is an open set.** Test has **France**, which is absent from train. Never hard-code, filter, or one-hot on {US, India}. Every test S1 entity (France included) must appear.
11. Final model: **MIT or Apache-2.0 license, ≤ 8B parameters**. Check the license of every model and pretrained artifact.
12. **No external data lookup** (see RULE 8). Disqualification risk.
13. Final package: `<team_name>_submission.zip` → `output/{matching_results.tsv,candidate_pairs.tsv}`,
    `code/business_entity_resolution/{src/,README.md,requirements.txt}`, `Documentation_template.md` (filled in).
    It must regenerate both output files from the data using only that folder.
14. Metric: **macro-averaged per-S1-entity F0.5**. Singletons score 1.0 for an empty prediction and 0.0 for any prediction.
15. Final ranking = **private leaderboard** (held-out part of test). Don't overfit the public leaderboard.
16. Output TSVs: **UTF-8 without BOM** (a BOM fails the official header check, verified) and **LF** line endings
    (on Windows pandas defaults to CRLF → always pass `lineterminator="\n"`; never write outputs with PowerShell `Out-File`).
17. Before any upload: `python student_resource/utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t datasets/test`
    must print PASS (note `-t datasets/test`; the script's default path is wrong for this repo). Treat unknown IDs and matches ⊄ candidates as errors,
    even though the official script only warns or skips them.

## Environment facts (verified 2026-09-25)
- Windows 11, Python 3.13.9, 16 logical CPUs, **16 GB RAM (often only ~2–3 GB free)**, RTX 3060 Laptop 6 GB.
- Project env: **conda `ber_hackathon`** (`environment.yml`, Python 3.13.15): numpy 2.2.4, pandas 2.3.3, pyarrow 21.0.0,
  rapidfuzz 3.14.6, PyYAML, pytest, psutil (DEC-012). Env python: `E:\hell\hell\envs\ber_hackathon\python.exe`.
  Optional GPU extra installed: torch 2.14.0+cu130 (`requirements-gpu.txt`); CUDA verified on the RTX 3060 (driver 616.92).
  The baseline does not use torch. NOT in the env: lightgbm, scikit-learn, polars, duckdb (adding them is a decision to record).
  PowerShell needs a one-time `conda init powershell`; otherwise `conda activate` silently leaves base python active.
  C: has little free space (~5 GB): use `pip install --no-cache-dir` for big wheels.
- Pipeline entry point: `python -m ber.pipeline.run_baseline <preprocess|split|train|infer|submit|all> --config configs/baseline_v0.yaml` (see README.md).
  Dev loop on a 2% sample: `configs/sample_dev.yaml` (never writes to `output/`).
- `conda run` fails on multi-line `-c` scripts; call the env python directly. Multiprocessing (spawn) cannot run code piped via stdin.
- Data is large (train: 2.2M S1 / 5.0M S2 / 5.3M S3; test: 1.73M S1 / 4.9M S2 / 5.1M S3).
  **Memory is a first-class constraint**: prefer pyarrow/columnar ops, process per country/shard,
  avoid full pandas object-dtype copies of S2/S3.
- Files are CRLF, RFC-4180-style quoting in a few rows (`""` escapes) → read with `sep="\t"` and default `"` quoting.
  Always check that parsed row counts equal the file line counts.

## Conventions
- Seeds: `42`. Validation split: group-aware by S1 entity (see DECISION_LOG).
- Evaluation always at **S1-entity level F0.5** (plus P, R, blocking recall, candidates/entity).
  Never report pairwise F1 alone.
- Scripts under `scripts/`, pipeline code under `src/ber/`, configs under `configs/`.
- Shell: PowerShell is primary; Bash (Git Bash) is available.
