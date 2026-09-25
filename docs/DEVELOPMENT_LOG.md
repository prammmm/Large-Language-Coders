# Development Log

Newest entries at the bottom. One entry per working session or phase.

---

## 2026-09-25 — Phase 0: Repository & Data Audit (Claude Code, under Ayush's direction)

### Objective
Audit the repository and challenge materials, profile the datasets (read-only), and produce the planning and
governance artifacts. **No matching pipeline was implemented.**

### Work completed
- Read `Problem Statement.docx` in full (it is the only organizer document present).
- Found that **`student_resource/` is not in the repo**: `utils/validate_submission.py` and
  `Documentation_template.md`, which the problem statement references, are missing.
- Checked the raw files for parse safety: CRLF line endings; every line has the expected column count;
  a few rows use RFC-4180 quoting (`""` escapes); parsed row counts match line counts.
- Wrote and ran the read-only profiling script `scripts/profiling/profile_datasets.py` (pyarrow, one file at a time).
- Created the governance files and the architecture/strategy report.

### Files changed
- Created: `CLAUDE.md`, `.gitignore`, `docs/DECISION_LOG.md`, `docs/DEVELOPMENT_LOG.md`, `docs/EXPERIMENT_LOG.md`,
  `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md`, `scripts/profiling/profile_datasets.py`,
  `reports/profiling/profile_summary.json` (gitignored because it contains raw record examples)
- Modified: none. Deleted: none. Raw data untouched.

### Commands executed
- `wc -l`, `head`, `grep` over `datasets/` (read-only)
- `python scripts/profiling/profile_datasets.py --sample-s1 100000` (seed 42)

### Results
See `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md` §3 for all dataset statistics.

### Errors
- The first `docx` text dump failed on a Windows cp1252 console encoding error. Fixed with `PYTHONIOENCODING=utf-8`.

### Fixes
- Train/test overlap checks use int64 hashes instead of holding full string sets, so they fit in the ~2.5 GB of free RAM.

### Current state
Planning complete. Waiting for team review and approval of the report and of decisions DEC-001…DEC-010.

### Next action
Team review, then Phase 1b/2: build the validation split and the entity-level evaluator (Ayush / Role C),
and the normalization and blocking baseline (Role A), in parallel.

---

## 2026-09-25 — Phase 0b: Review of the newly added `student_resource/` (Claude Code)

### Objective
Read the newly available organizer resources, reconcile them with the Phase 0 planning documents, and smoke-test the official validator. No pipeline code.

### Work completed
- Read `student_resource/README.md` (its content matches `Problem Statement.docx`, so no requirement changes),
  `utils/validate_submission.py` and `Documentation_template.md`.
- Smoke-tested the official validator with throwaway files in the session scratchpad (outside the repo):
  - all-empty LF file with all 1,732,544 test S1 IDs → PASS
  - CRLF file → PASS
  - UTF-8 BOM file → **FAIL** (header mismatch)
- Verified that pandas `to_csv` writes CRLF on this Windows machine by default.
- Added report §4.1 and updated the stale "missing files" statements. Added hard constraints 16–17 to CLAUDE.md. Added DEC-011 (Proposed).

### Files changed
- Modified: `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md`, `CLAUDE.md`, `docs/DECISION_LOG.md`, `docs/DEVELOPMENT_LOG.md`
- Created / deleted in repo: none. Raw data untouched. Nothing committed.

### Commands executed
- `python student_resource/utils/validate_submission.py -m <scratch>.tsv -c <missing> -t datasets/test` (×3)

### Results / Errors / Fixes
See the Work completed section above. No errors.

### Current state
Planning is complete, and the organizer resources are integrated. DEC-001…DEC-011 are awaiting team approval.

### Next action
Team approves the decisions and dependencies (rapidfuzz, lightgbm), then starts the parallel Phase 1/2 tracks from report "WHAT THE TEAM SHOULD DO NEXT".

---

## 2026-09-25 — Phase 1: Submission-1 baseline implementation (EXP-B-01) (Claude Code)

### Objective
Implement a runnable, config-driven, memory-safe baseline (blocking → pair scoring → global threshold) that the team can run on the full data to produce Submission 1.

### Work completed
- Conda env `ber_hackathon` (`environment.yml`, `requirements.txt`, `pyproject.toml`; `ber` installed editable). DEC-012.
- `src/ber/`:
  - normalization: script-aware, Latin-only accent strip, Indic-safe, multi-view.
  - ingest: streaming TSV→Parquet with row-count assertions.
  - per-country normalized views: open-set countries.
  - frozen stratified split.
  - blocking V0.
  - float32 features (rapidfuzz + token sets).
  - weighted rule scorer and global threshold.
  - official-definition evaluator.
  - the single LF/no-BOM writer and a strict validator.
  - `pipeline/run_baseline.py` with stages preprocess/split/train/infer/submit/all.
- Tests: 60 (synthetic data only), all passing.
- Scripts: `scripts/verify_env.py`, `scripts/make_sample_data.py` (proportional real-data samples under `data/sample/`).
- Memory hardening after measuring the real India test partition (peak 2.92 GB → 1.85 GB blocking / 2.01 GB with features), verified bit-identical output:
  - chunked token encoding;
  - packed in-place key sort;
  - compress blocks to S1 hits;
  - lazy per-column pool reads;
  - per-chunk unique-row gathering for features;
  - Arrow system allocator.

### Files changed
- Created: `environment.yml`, `requirements.txt`, `pyproject.toml`, `README.md`, `configs/{baseline_v0,sample_dev}.yaml`,
  `src/ber/**` (config, utils/runtime, normalization/{text,views}, data/{ingest,normalize_records,split}, blocking/blocker,
  features/pair_features, models/weighted_rule, decision/threshold, evaluation/metrics, submission/{writer,strict_validator},
  pipeline/run_baseline), `tests/{synthetic,test_normalization,test_metrics,test_blocking_features,test_writer_validator,test_pipeline_e2e}.py`,
  `scripts/{verify_env,make_sample_data}.py`
- Modified: `.gitignore` (data/sample*/), `CLAUDE.md` (env facts), `docs/DECISION_LOG.md` (DEC-012), `docs/EXPERIMENT_LOG.md` (EXP-B-01), this log.
- Raw data untouched (mtimes unchanged). Nothing committed or pushed.

### Commands executed (main ones)
- `conda env create -f environment.yml`; `python -m pytest -q`
- `python scripts/make_sample_data.py --fraction 0.02` (and 0.2, since deleted); `python -m ber.pipeline.run_baseline all --config configs/sample_dev.yaml`
- Full-data **preprocess only** (train and test) plus bounded blocking/feature memory probes on the largest partitions. No full train/infer/submit run.

### Results
See EXPERIMENT_LOG EXP-B-01 (sample metrics clearly labelled; full-data metrics N/A).

### Errors → fixes
- Arrow `ArrowMemoryError` / access violation while building samples near the Windows commit limit → system allocator + streaming sample builder.
- An earlier "hang" was an interrupted session blocking on stdout; it was not reproducible.
- Per-pair gathering tripled feature memory → unique-row gathering.

### Current state
Ready for the team's full run. `data/interim` and `data/processed` for both splits already exist (built by this code). `data/splits/val_s1_ids.txt` is NOT created yet (team step).

### Next action
Team runs split → train → infer → submit → validators (README), records full-data metrics in EXP-B-01, and decides on Submission 1.

---

## 2026-09-25 — Submission 1 result + error analysis (Claude Code)

### Objective
Report the full EXP-B-01 execution, explain the public score (0.751), and propose measured improvements. No pipeline changes.

### Work completed
- Collected the team's full-run results. Validation F0.5 0.7851 (threshold 0.67); public leaderboard **0.751**.
- Wrote read-only analysis scripts:
  - `scripts/analysis/analyze_exp_b01.py`: loss decomposition, true-pair fate via uncapped re-blocking, miss/false-positive descriptors, what-if simulations, test-by-country diagnostics.
  - `scripts/analysis/matcher_headroom_probe.py`: EXP-D-01.
  - `scripts/analysis/hypothesis_probe.py`.
- Wrote `reports/EXP-B-01_SUBMISSION_1_REPORT.md`.

### Key measured findings
- Blocking ceiling (oracle) 0.910. 17.9% of true pairs share no blocking key; top-K beyond 30 adds <1%.
- 34% of non-candidate true pairs have an identical `name_core`.
- Matcher/decision loss 0.125. The LR probe reaches +0.031 on the same features.
- Decision-layer tweaks give ≤ +0.001.
- The gap to the leaderboard is explained by country mix (India+US reweighted ≈ 0.762); France is estimated at ≈ 0.68.

### Files changed
- Created: `reports/EXP-B-01_SUBMISSION_1_REPORT.md`, `scripts/analysis/{analyze_exp_b01,matcher_headroom_probe,hypothesis_probe}.py`.
- Created (git-ignored): `experiments/runs/EXP-B-01/analysis/*`.
- Modified: `docs/EXPERIMENT_LOG.md` (EXP-B-01 full-run results + public score, EXP-D-01), this log.

### Current state
Proposals EXP-B-02 / EXP-M-02 / EXP-F-01 / EXP-M-03 / EXP-N-01 are awaiting team approval. Nothing committed.

---

## 2026-09-25 — Phase 2: evidence-driven improvements (Claude Code)

### Objective
Improve on Submission 1 (validation 0.7851, LB 0.751) with controlled, measured, switchable changes. Precision is a first-class metric.

### Work completed
- **Audit vs report.**
  - Confirmed the report's claims in code.
  - Corrections: the caps are global (no India-specific caps exist, and none were added: open set); the EXP-D-01 probe was fitted on validation halves; EXP-B-01's threshold was tuned on validation.
- **Code** (all switchable, defaults keep EXP-B-01 behaviour; regression check reproduced 0.7851 exactly):
  - Blocking: `exact_core` and `sorted_core` passes, `strip_leading_zeros`, chunk transforms.
  - Features: conflict features and zero-safe number features.
  - Matcher: numpy logistic regression plus a `make_scorer` factory.
  - Pipeline stages: `block-eval`, `fit` (train-fold only, inner-split threshold), `evaluate` (identical-candidate re-scoring).
  - Pipeline plumbing: per-experiment scored tags, `--set` overrides, config `inherit`, precision-first metrics (false merges, missed matches, predictions/S1, threshold table), `threshold_from`.
- **Experiments** (full frozen validation; LR fits dev-scale on 50k train-fold S1):
  - EXP-B-02 sweep (12 variants): keep sorted_core + zero-strip; reject caps / max_nums / max_words.
  - EXP-B-02 with the rule: rejected (precision 0.8879).
  - EXP-M-02 0.8158 · M-02b 0.8200 · F-01a 0.8287 · **F-01b 0.8371 (P 0.9316)**. Every step holds on both validation halves.
  - EXP-N-01 feasibility probe.
- Documentation: `docs/MODEL_AND_PIPELINE_EXPLAINED.md` (English + simple Hindi), README run guide, DEC-013/014/015, EXPERIMENT_LOG entries.
- Tests: 78 passing (new: `tests/test_phase2.py`, `tests/test_pipeline_logistic_e2e.py`). Checked on the real-data 2% sample end to end, including the official validator.

### Files changed
- Modified:
  - `src/ber/{config.py, blocking/blocker.py, features/pair_features.py, decision/threshold.py, evaluation/metrics.py, pipeline/run_baseline.py, utils/runtime.py, models/__init__.py}`;
  - `configs/baseline_v0.yaml` (disabled options only);
  - `scripts/analysis/analyze_exp_b01.py` (any experiment tag);
  - `tests/test_pipeline_e2e.py`;
  - README, DECISION/EXPERIMENT/DEVELOPMENT logs.
- Created:
  - `src/ber/models/logistic.py`;
  - `configs/{exp_m02_lr, exp_b02_blocking_v1, exp_m02b_lr_blocking_v1, exp_f01_conflicts, exp_f01b_conflicts_zeros}.yaml`;
  - `scripts/experiments/run_blocking_sweep.py`;
  - `scripts/analysis/{translit_feasibility, compare_val_halves}.py`;
  - the two new test files;
  - `docs/MODEL_AND_PIPELINE_EXPLAINED.md`.
- No dependency changes. Nothing committed. Raw data untouched.

### Errors → fixes
- A shell heredoc mangled a regex back-reference; caught by a sanity check and fixed.
- The halves comparison first applied the LR threshold to stored rule scores; caught and fixed (re-scores with the run's model).

### Current state
EXP-F-01b is the best measured system (dev-scale fit). The team runs the full fit/train/infer/submit.
