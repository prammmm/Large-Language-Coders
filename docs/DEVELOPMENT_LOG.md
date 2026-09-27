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

---

## 2026-09-25 — Submission 2 result + deep analysis (Claude Code)

### Objective
Explain LB 0.815 and find where the score is still lost.

### Work completed
- Collected the team's full EXP-F-01b run (validation 0.8372, LB 0.815).
- Ran:
  - `analyze_exp_b01.py` on EXP-F-01b;
  - a new `scripts/analysis/error_budget.py` (oracle-fix gain per error bucket);
  - a new `scripts/analysis/nonlinear_probe.py` (EXP-D-02: MLP on the same features and protocol, 3 seeds).
- Wrote `reports/EXP-F-01b_SUBMISSION_2_REPORT.md`.

### Key findings
- LB gap explained by country mix (India+US reweighted 0.823; France estimated ≈0.77).
- Error budget: blocking misses +0.073, low-scored true candidates +0.058, false merges +0.046.
- **Non-linear scorer +0.050 (0.887–0.890), with precision up and false merges halved.**

### Files
- Created: the report and the two analysis scripts.
- Modified: `nonlinear_probe.py` (seed/out args), EXPERIMENT_LOG, this log.
- Nothing committed.

---

## 2026-09-25 — Phase 3: EXP-M-03 LightGBM matcher (Claude Code)

### Objective
Test one hypothesis: a production-safe non-linear scorer extracts the headroom found by the MLP probe. The change is scorer-only.

### Work completed
- Env: installed `lightgbm=4.7.0` (conda-forge CPU, MIT) plus its dependency `scipy` 1.18.1 into the existing `ber_hackathon` env with `--freeze-installed`. No other package changed (`pip check` clean). Pinned in `environment.yml` / `requirements.txt` (DEC-016).
- Code:
  - `src/ber/models/gbdt.py` (LightGBMModel: fit with early stopping, save/load, gain importance);
  - `models/__init__.py` (`lightgbm` type);
  - `run_fit` generalised: `fit.reuse_fit_parts` (read-only re-use of another experiment's train-fold parts, with the S1-set check), `fit.dev_subsample_s1`, a leakage guard on the actual training rows, and a LightGBM branch;
  - `all` stage includes lightgbm.
- Config: `configs/exp_m03_lgbm.yaml`. Tests: `tests/test_lightgbm.py` (5 tests). Total 83 passing.
- Scripts: `scripts/analysis/test_country_profile.py` (France guard); `compare_val_halves.py` loads LightGBM too.
- Dev-scale measurement: 50k train-fold S1 subsample; full frozen validation. LR on the same 50k: 0.8375. LightGBM seeds 42/7/123: 0.9021/0.9022/0.9021, precision 0.976–0.979. France sample guard: LightGBM predicts more in France (0.19 vs 0.13 per S1).
- Docs: EXPERIMENT_LOG (EXP-M-03), DECISION_LOG (DEC-016), `docs/MODEL_AND_PIPELINE_EXPLAINED.md` (LightGBM section), README.

### Current state
EXP-M-03 is a strong candidate. The team runs the full 200k fit/evaluate/infer, then the France guard, then decides on the submission. Nothing committed.

---

## 2026-09-25 — Submission 3 result + deep analysis (Claude Code)

### Objective
Explain LB 0.889 (EXP-M-03) and find what still blocks the score.

### Work completed
- `analyze_exp_b01.py` now re-scores with the experiment's own model when it reads another experiment's validation parts, and uses the experiment's own test parts. (Before, it would have analysed the LR score for EXP-M-03.)
- Ran on the full EXP-M-03 run: analysis, error budget, a new France pattern probe (`scripts/analysis/france_pattern_probe.py`, hypothesis rejected), a new test-conflict lower bound (`scripts/analysis/test_conflicts.py`), and the capacity probe EXP-D-03.
- Wrote `reports/EXP-M-03_SUBMISSION_3_REPORT.md`; updated EXPERIMENT_LOG.

### Key findings
- The scorer is near its ceiling (gap 0.038). Blocking is ≈57% of the remaining loss.
- On test, France has ≥ 6.2% certainly-wrong predictions from conflicting claims (LR 2.3%), invisible on validation.
- Capacity and threshold are not levers.

### Current state
Waiting for the team's decision on the next experiment (EXP-B-03 blocking recall / EXP-X-01 full-competition conflict resolution / EXP-N-01). Nothing committed.

---

## 2026-09-25 — Phase 4: EXP-B-03a name-token blocking (Claude Code)

### Objective
Test one change: enable the existing name-token blocking pass, with the Submission-3 LightGBM model kept fixed.

### Work completed
- Inspected the existing `name_token` pass (name_core tokens with pool DF ≤ max_df; max_df also caps block size), the M-03 config and the model. No code change was needed for the pass itself.
- Config `configs/exp_b03a_name_token.yaml`:
  - inherits EXP-M-03;
  - name_token enabled;
  - `matcher.model_path` = the EXP-M-03 model (fixed);
  - new tag for re-blocked validation/test parts.
- Measured on the frozen validation split: block-eval + train (re-block + fixed-model scoring). F0.5 0.9027 vs 0.9037 → **rejected**.
- New analysis script `scripts/analysis/compare_candidate_sets.py`: new-only / displaced / flipped pairs, recovered-TP and new-FP profiles, counterfactual F0.5.
- Tests: `tests/test_name_token.py` (4 tests). 87 passing.
- Logs and `docs/MODEL_AND_PIPELINE_EXPLAINED.md` updated. No dependency change. Nothing committed. No test inference run.

### Key finding
The name-token candidates are safe (3,932 TP vs 182 FP accepted). The regression comes from the fixed model reacting to changed `cheap_score` / `cheap_rank` inputs on existing candidates (+3,994 FP flips). Candidate recovery alone is worth +0.0020.

---

## 2026-09-25 — Submission 4 (EXP-B-03a check, LB 0.890) report (Claude Code)
- Test-side diagnostics for EXP-B-03a vs EXP-M-03 (country profile, conflict lower bound). `test_country_profile.py` / `test_conflicts.py` now follow a run's recorded `model_path`.
- New `scripts/analysis/blocking_gap_probe.py`: 78.1% of missing true pairs are reachable by combined key families; only 19.7% by rare name tokens.
- Wrote `reports/EXP-B-03a_SUBMISSION_4_REPORT.md` (cumulative work + analysis and priorities). No pipeline changes. Nothing committed.

---

## 2026-09-25 — Phase 5 Stage A: EXP-B-04 Blocking V2 + retrained LightGBM (Claude Code)

### Objective
Test the largest measured blocking opportunity: combined keys, with the model retrained on the same blocker. Implement → self-test → block-eval → retrain → validate → analyse → report, then STOP for human review.

### Work completed
- Code (`src/ber/blocking/blocker.py`):
  - new `cross_keys` helper (windowed per-record cross products; unordered pairs for `same=True`);
  - new passes `name_addr` and `addr_pair` (PASS_ORDER now 8 passes, which fits the uint8 flags);
  - both select only words with pool DF ≥ 1 (a bug found by a new test: DF-0 words wasted key slots);
  - existing passes are unchanged.
- `scripts/experiments/run_blocking_sweep.py`: `--prefix`; custom variants-only runs.
- Config: `configs/exp_b04_blocking_v2.yaml` (inherits EXP-M-03; new passes; own tag; no fit-part reuse; own model).
- Tests: `tests/test_blocking_v2.py` (8 new). 95 passing.
- Measurements:
  - 2% sample check;
  - blocking-only sweep A/B/C/D on the frozen validation split;
  - retrain + validation for D at K=20 (EXP-B-04) and K=30 (EXP-B-04-k30);
  - halves, pairwise attribution, recovered-by-key, error budget, 2% sample unlabeled test diagnostics.
- Docs: EXPERIMENT_LOG (EXP-B-04), DECISION_LOG (DEC-017, Proposed), `docs/MODEL_AND_PIPELINE_EXPLAINED.md` (Blocking V2 + retraining, English/Hindi).

### Result
Validation F0.5 0.9274 (+0.0237), precision 0.9774, halves 0.9271 / 0.9277. **Status: READY FOR HUMAN REVIEW.**

### Not done (by instruction)
No full test inference, no submission, `output/` untouched, nothing committed. No dependency change.

---

## 2026-09-26 — POST-0.91 forensic analysis after Submission 5 (Claude Code, analysis only)

### Objective
Explain the ≈0.91 leaderboard score of EXP-B-04 (val 0.9274), build a complete error budget, and pick exactly one next experiment. No production, model, blocker, threshold, feature or `output/` changes. No submission.

### Work completed
- New read-only scripts in `scripts/analysis/`:
  - `full_competition.py`: scores all 2.2M train S1s so they compete as on test;
  - `blocking_forensics.py`: uncapped recall@K, fate of true pairs, miss causes, new key-family coverage;
  - `scorer_test_forensics.py`: calibration, cardinality, false-merge categories, test/France label-free stats;
  - `distribution_shift.py`.
- Forensic learning-curve point `FORENSIC-B04-dev50k`: re-uses the B-04 fit parts read-only, own run dir.
- Report: `experiments/runs/POST-0.91-FORENSIC/REPORT.md`, 17 sections plus an English + Hindi team explanation. `experiments/runs/` is gitignored.

### Key measured results
- **Full competition:** with all train S1s competing, a best-claimant-per-pool-record rule (margin 0) raises F0.5 on 2.0M out-of-sample S1s from 0.9273 to **0.9313**. P 0.9775 → 0.9840, R −0.0014, India +0.0060, US +0.0027. The earlier "exclusivity ≤ 0.0005" result is obsolete, because it was measured with only 10% of S1s competing.
- **LB gap:** about −0.008 of the −0.017 gap is country mix, since India is 46.8% of test S1s vs 40.0% of validation. Implied France F0.5 is ≈0.85 [estimate].
- **Blocking:**
  - uncapped recall is 0.945 at K = 200 (India 0.910, US 0.968);
  - 5.5% of true pairs are never retrieved, and 1.8% are lost to the top-K cut;
  - India misses: mostly outside the rarest-3 word selection, plus script mismatch;
  - US misses: name-only records with empty addresses.
- **Scorer and data:** the scorer is well calibrated. 50k → 200k fit S1s gives only +0.0020. Test India/US label-free distributions match validation. The test pool is denser (5.8 vs 4.7 pool records per S1).

### Recommendation (not implemented)
EXP-D-01: best-claimant conflict resolution at inference. It needs no retraining and no blocker change. Awaiting human approval.

### Notes
- `src/ber/models/` is still untracked in git; the team must commit it.
- New derived data: `data/processed/train/forensic_alltrain_B04/` (42.5M scored pairs, can be deleted).
- Nothing committed; no dependency change.

---

## 2026-09-26 — Stage 1: EXP-D-01 full-competition best-claimant resolution (Claude Code)

### Objective
Implement, test and validate, under full competition, the decision rule from the POST-0.91 forensic analysis. B-04 stays the control; no retraining and no blocker, feature or threshold change.

### Work completed
- Compliance check:
  - the official text says S1 is *deduplicated*;
  - per-record exclusivity is not an explicit rule;
  - the train ground truth shows every S2/S3 id belongs to exactly one S1.
- Code:
  - `src/ber/decision/competition.py` (rule + statistics);
  - `run_baseline.submit` applies it when `decision.competition.enabled` (off by default); competition statistics go into `submission_summary.json`;
  - new `experiment.allowed_stages` guard;
  - config `configs/exp_d01_best_claimant.yaml`, which re-uses the B-04 scores and model read-only and writes to `experiments/runs/EXP-D-01/output`.
- Scripts: `scripts/experiments/eval_exp_d01.py` (full-competition evaluation, re-using `forensic_alltrain_B04`) and `scripts/analysis/verify_d01_outputs.py`.
- Tests: `tests/test_competition.py` (12, including a synthetic end-to-end run). **107 / 107 passing.**
- Wrote test outputs to `experiments/runs/EXP-D-01/output/`. Strict and official (`--check-ids`) validators PASS. The B-04 `output/` sha256 is unchanged.
- Docs:
  - EXPERIMENT_LOG: EXP-D-01; B-04 row now shows LB ≈0.91;
  - DECISION_LOG: DEC-018 (Proposed);
  - MODEL_AND_PIPELINE_EXPLAINED: §2.7b;
  - report: `experiments/runs/EXP-D-01/REPORT.md`.

### Result
- On E (2,006,821 out-of-sample S1s, all 2.2M train S1s compete):
  - F0.5 0.9273 → **0.9313**;
  - P 0.9775 → 0.9840, R 0.8576 → 0.8562;
  - false merges −29.8%;
  - India +0.0060, US +0.0027;
  - halves +0.0039 to +0.0040.
- Test: −49,954 predictions; conflicts 41,560 → 0.
- **KEEP · READY FOR HUMAN REVIEW.**

### Not done (by instruction)
No submission. `output/` untouched. Stage 2 (blocking/retrieval) not started. Nothing committed; no dependency change.

---

## 2026-09-26 — EXP-B-05: wider address-word window (3 → 5) + retrained LightGBM, with D-01 (Claude Code)

### Objective
Test whether a 5-word window for the combined address keys recovers missed (mainly India) true pairs, with the model retrained on the new blocker and D-01 on top. Control: B-04 + D-01.

### Work completed
- Config `configs/exp_b05_addr_window5.yaml`. The window was already a config knob, so there is no blocker code change. The output dir is the run folder, never `output/`.
- Tests: `tests/test_blocking_b05.py` (12, including a synthetic end-to-end run with retrain and D-01). **119 / 119 passing.**
- Scripts:
  - `scripts/experiments/eval_exp_b05.py` (two-arm full-competition comparison, false-merge categories, test diagnostics);
  - `scripts/analysis/compare_blocking_fates.py`;
  - `scripts/analysis/d01_threshold_attribution.py` (diagnostic);
  - `blocking_forensics.py` gained `--dump` and candidate distributions.
- Runs:
  - blocking sweep (ctrl / w5 / ap5 / na5);
  - uncapped forensics for both blockers;
  - 2% sample end-to-end (official validator with `--check-ids` PASS);
  - fit + validation (F0.5 0.9277, threshold 0.62);
  - all-train scoring (`forensic_alltrain_B05`);
  - test inference (`scored_EXP-B-05`; no output files written).
- Docs: EXPERIMENT_LOG (EXP-B-05), MODEL_AND_PIPELINE_EXPLAINED (§2.4 note), report `experiments/runs/EXP-B-05/REPORT.md`.

### Result
- On E: B-04 + D-01 0.9313 → B-05 + D-01 0.9324 (+0.0010); halves +0.0010 to +0.0012; P −0.0027; false merges +18.8%.
- Attribution: the blocker is worth +0.0005 at a matched threshold.
- Block-cap growth loses 761 true pairs; top-K displacement pushes out 3,024.
- **REJECTED · NOT READY.** B-04 + D-01 stays best.
- Proposed next: EXP-N-02, train-learned transliteration. Script mismatch is 42% of India's unreachable pairs; the DEC-009 fair-play confirmation is needed first.

### Notes
- The ID EXP-D-01 is also used by an older 2026-09-25 diagnostic in EXPERIMENT_LOG; flagged for the team, not renamed.
- New derived data can be deleted:
  - `data/processed/train/forensic_alltrain_B05/`;
  - `data/processed/{train,test}/scored_EXP-B-05*`;
  - `data/sample/output_b05/`.
- Nothing committed; no dependency change; no submission.

---

## 2026-09-26 — EXP-N-02: train-pair-learned cross-script name view + retrained LightGBM, with D-01 (Claude Code)

### Objective
Bridge India's script-mismatch pairs with a token map learned only from provided train true pairs, then use it for blocking and scoring. LightGBM is retrained; the control is B-04 + D-01.

### Work completed
- Fair-play check: the official text forbids external lookups and augmentation, not learned representations of the provided data. DEC-009 is respected: the map is a normalization view and the features are similarities.
- Code:
  - `src/ber/normalization/crossscript.py` and `src/ber/pipeline/xlit.py` (learning, validation guard, final map, cross-fit plan);
  - `run_baseline.py`: `name_xlit` in LazyPool / S1, `score_split(xmap, clear, part_prefix)`, cross-fitted `fit`, map provenance in the model meta;
  - `blocker.py`: `blocking.name_view` (default unchanged);
  - `pair_features.py`: `xlit_core_jaccard`, `xlit_token_set`.
- Config: `configs/exp_n02_transliteration.yaml` (output dir = run folder).
- Tests: `tests/test_xlit.py` (15, including leakage guards and a synthetic end-to-end run). **134 / 134 passing.**
- Scripts:
  - `scripts/analysis/xlit_diagnostics.py`;
  - `scripts/experiments/eval_exp_n02.py`;
  - `eval_exp_b05.py` generalised (`--label`, `--script-analysis`);
  - `blocking_forensics.py` uses the map.
- Runs:
  - map diagnostics; block-eval; uncapped forensics + fate comparison;
  - 2% sample end-to-end (official `--check-ids` PASS);
  - cross-fitted fit; validation; all-train scoring (`forensic_alltrain_N02`); test inference (`scored_EXP-N-02`);
  - two-arm evaluation;
  - submit to `experiments/runs/EXP-N-02/output` (strict + official `--check-ids` PASS).
- Bookkeeping:
  - ID clash resolved: the older probe is renamed **EXP-D-01P** with a note; the production EXP-D-01 is unchanged.
  - B-05 scored folders deleted as authorised (about 1.07 GB); the B-05 report and evidence JSONs are kept.
- Docs: EXPERIMENT_LOG, DECISION_LOG (DEC-019 Proposed), MODEL_AND_PIPELINE_EXPLAINED, report `experiments/runs/EXP-N-02/REPORT.md`.

### Result
- On E: F0.5 0.9313 → **0.9402 (+0.0089)**, P 0.9840 → 0.9864, India +0.0210; halves +0.0089 to +0.0091; peak ≤ 4.72 GB.
- All 12 gate criteria pass. **KEEP · READY FOR HUMAN REVIEW.**

### Not done (by instruction)
No root `output/` write, no submission, no commit or push. No dependency change.

---

## 2026-09-26 (evening) — Plan execution: N-03, B-07, B-06; research review (Claude Code)

- **Step 0:**
  - backed up the N-02 output (LB 0.928) to `submissions/EXP-N-02_output/`;
  - `eval_exp_b05.py` accepts any control (`--control-label`, control scoring tags);
  - refreshed the error budget.
- **EXP-N-03** (learned Indic legal-form words): E 0.9413 (+0.0010 vs N-02). Below the gate; carried as the base.
- **EXP-B-07** (exact name × address word, `exact_core.with_addr_words`): E 0.9464 at t 0.67 (precision-first tuning rule, `scripts/analysis/d01_threshold_select.py`). KEEP.
- **EXP-B-06** (name-token pair, `name_token.pair`): E 0.9471, t 0.67. KEEP (best local).
- **Probes:** `scripts/analysis/stage2_probe.py` (+0.0017 held-out) and `scripts/analysis/collective_probe.py` (+0.0048 held-out). Review: `reports/RESEARCH_REVIEW_2026-09-26.md`; next is EXP-S-01.
- **Tests:** 145 / 145. All outputs were validated in their run folders. `output/` holds B-07 t 0.67 (copied by the team). Nothing committed.

---

## 2026-09-27 — Final senior review: audit, EXP-S-01 collective stage 2, LOCO diagnostics, final output (Claude Code)

- **Audit:** `scripts/analysis/audit_summary.py` → `experiments/runs/FINAL/audit_summary.json`. All review claims were verified. Discrepancies documented (the B-07 summary overwrite; root `output/` = B-06). The control is B-06 + D-01 @ 0.67.
- **EXP-S-01** (`src/ber/decision/stage2.py`, `src/ber/pipeline/run_stage2.py`, `configs/exp_s01_stage2.yaml`), built test-first (`tests/test_stage2.py`, `tests/test_stage2_pipeline.py`):
  - held-out E half B 0.9471 → **0.9596**, validation 0.9470 → **0.9601**, precision up;
  - all halves agree; deterministic (byte-identical re-run);
  - strict + official `--check-ids` PASS. **KEEP — final.**
- **EXP-S-01b** (no count features): E-B 0.9591 / validation 0.9595. Not selected under the pre-declared rule (test inflation not clearly lower).
- **LOCO C-01/C-02** (`fit.only_countries`, `xlit.map_path`; `tests/test_loco.py`): unseen-country loss −0.075 / −0.039, mostly precision.
- **C-03** (`stage2.train_countries`): stage 2 improves unseen India by +0.021.
- **Label-free diagnostics:** train→test feature shift; test acceptance +9.3% vs +4.0% expected; French "st"/"saint" (≤ 1.7% of French pool records).
- **Memory fixes:** stage-2 submit 5.14 → 4.79 GB; evaluate 5.41 → 2.98 GB. Tests: **161 / 161**.
- **Final output:** `experiments/runs/EXP-S-01/output/`. Report: `reports/FINAL_REPORT_2026-09-27.md`. DEC-020 Proposed. Nothing committed; `output/` untouched (still B-06).

---

## 2026-09-27 (later) — POST-S-01 public regression forensics (Claude Code)

- **Trigger:** S-01 public ≈ 0.920 (team-reported) vs best pre-S-01 0.935; local E half B 0.9596.
- **New code** (diagnostics only, test-first):
  - `src/ber/forensics/{shadow.py, diagnostics.py}`;
  - `scripts/analysis/post_s01_forensics.py` (verify / delta / shift / shadow-build / shadow-features / shadow-eval / leakage / floor);
  - `run_stage2.build_features(drop_s1, out)` and `predict_all(fdir)`, for simulated competitions.
- **Tests:** `tests/test_shadow.py` (13) plus a drop-mask test in `tests/test_stage2_pipeline.py`. Full suite **175 / 175**.
- **Shadow benchmark** (`experiments/shadow_tests/`, seed 42):
  - SHADOW-BASE 446,321 S1 (E half B minus validation);
  - random-orphaned and chain-orphaned competitions;
  - leakage checks all 0.
- **Result:** the regression comes from S-01's context-driven additions on uncontested look-alike records. Test has 1.77× the train density of such candidates, and S-01 adds 2.14× more per S1 on test. Removals transfer 1:1. Ruled out: D-01, implementation, France, per-feature shift, threshold alone.
- **Proposed** (not implemented): EXP-S-02, stage 2 as a veto only (density-invariant, +0.0027 held-out).
- **Report:** `experiments/runs/POST-S01-FORENSIC/forensic_report.md`.
- **Memory:** recorded peaks: 3.77 GB (test side of `delta`) and **5.47 GB** (train side of `delta`). The latter is above the 5 GB working target; it is a diagnostic script, not pipeline code, but should be streamed before any re-run. Other subcommands did not record a peak. `output/` untouched (still the submitted S-01 files). No submission, nothing committed.

---

## 2026-09-27 (morning) — EXP-S-02 veto + shadow benchmark v2 (Claude Code)

- **Code:**
  - `stage2.add_floor` / `stage2.model_from` (read-only guards) and a shared streamed `control_decisions()` in `run_stage2.py`;
  - `restrict_additions` in `ber.decision.stage2`;
  - v2 shadow helpers in `ber.forensics.shadow`;
  - new script `scripts/analysis/exp_s02_shadow.py`;
  - config `configs/exp_s02_stage2_veto.yaml`.
- **Tests:** 191 / 191 (16 new: the veto intersection with forced additions and removals, the null floor reproducing bytes, the read-only model, the streamed control equal to the reference, shadow v2).
- **Memory fix:** the stage-2 submit membership test now uses `searchsorted`; peak 5.15 → 3.59 GB, output byte-identical.
- **Result:** EXP-S-02 KEEP (all pre-declared gates). Validation 0.9499 vs B-06 0.9470. Density-robust; S-01's advantage collapses on the new tiers.
- **Outputs:** `experiments/runs/EXP-S-02/`. Root `output/` untouched (still S-01). No submission, nothing committed.
- **Extra run folder:** `experiments/runs/EXP-S-02_nullcheck/` (real-data null-floor regression).

---

## 2026-09-27 (late morning) — EXP-B-08 top-K 30 (Claude Code)

- **New files:**
  - config `configs/exp_b08_topk30.yaml`;
  - tests `tests/test_topk_b08.py` (6; suite 197 / 197);
  - streamed evaluation `scripts/analysis/exp_b08.py` (score-alltrain, sanity, eval, decompose, testdiag, budget, keyforensic, leakage, gates).
- **Pipeline:** block-eval, fit, train, all-train scoring, infer, submit ×2. Every step ≤ 4.05 GB; deterministic; strict and official validators PASS.
- **Result:** REJECT. +0.0017 to +0.0019 on every held-out population, below the pre-declared +0.002 bar; clean and density-stable.
- **Incident:** the EXP-S-02 reproduction was re-run outside this session between 07:56 and 08:46. It rewrote the shadow v2 manifest (timestamp only; masks byte-identical) and the S-02 analysis JSONs, and reproduced every S-02 number.
- **Known pipeline property (pre-existing):** candidate generation depends slightly on which S1 are blocked together (~0.002% of pairs).
- **Outputs:** `experiments/runs/EXP-B-08/`. Root `output/` untouched (S-01). No submission, nothing committed.

---

## 2026-09-27 (afternoon) — EXP-B-09 name-token x house-number key (Claude Code)

- **Preflight:** env numpy made self-contained (DEC-023); the B-06 reference reproduced exactly; GPU unavailable for LightGBM (OpenCL crash, no CUDA build); CPU used.
- **Code:**
  - `name_num` pass in `ber.blocking.blocker`; uint16 flags only when enabled; block-eval bit fix;
  - `configs/exp_b09_name_num.yaml`;
  - tests `tests/test_blocking_b09.py` (10; suite 213 / 213);
  - analysis `scripts/analysis/exp_b09.py`, `exp_b09_eval.py`, `exp_b09_extra.py`.
- **Result:** kill gate passed (+0.0082 recall@20); end to end +0.0032 F0.5 but precision -0.0012 and test weak acceptance +15.8%, so **REJECT** under the pre-declared gates.
- **Incident:** an external B-08 re-run (infer, then submit) competed for resources; stopped at the team's request before it could overwrite B-08 outputs. B-08's test scored parts (`data/processed/test/scored_EXP-B-08`) are incomplete (47 parts) and need a B-08 `infer` to restore. B-08 outputs and model are intact.
- **Outputs:** `experiments/runs/EXP-B-09/` (output isolated). Root `output/` untouched. No submission, nothing committed.

---

## 2026-09-27 (evening) — EXP-S-03 veto refitted on B-09 (Claude Code)

- **Code:** config `configs/exp_s03_veto_on_b09.yaml`; tests `tests/test_stage2_s03.py` (5; suite 218 / 218); analysis `scripts/analysis/exp_s03.py`. No production code change.
- **Result:** KEEP. All gates A-P pass: +0.0032 to +0.0034 F0.5 over B-09, precision +0.0068, false merges -49%, 0 additions, density-stable.
- **Output:** `experiments/runs/EXP-S-03/output/` (candidates byte-identical to B-09). Root `output/` untouched (still S-01). Not submitted, nothing committed.
- **GPU:** unavailable for LightGBM (CPU used).

---

## 2026-09-27 (evening) — S-03 public 0.942; EXP-B-10 rejected at the kill gate (Claude Code)

- **S-03 public:** 0.942 (team-reported; previous control 0.935). Root `output/` holds S-03.
- **EXP-B-10:** `name_token.max_tokens` 3 on B-09 (config + 6 tests; suite 224 / 224).
  - Blocking recall@20 +0.0028 < the +0.003 kill gate, so it stopped before the retrain.
  - The key recovers the targeted name-only pairs (77% of those entering the top-20), but 1,441 true pairs rank beyond 20 (limited by the cheap-score ranking).
- **Nothing committed.** No submission.

## 2026-09-27 (evening) — EXP-B-11 rejected at the kill gate (Claude Code)

- **EXP-B-11:** `name_token.weight` 1.0 -> 2.0 on the B-10 blocker (config + 8 tests; suite 232 / 232).
  - Blocking recall@20 0.9584: +0.0015 vs B-09, **-0.0012 vs B-10**. Kill gate KA / KB / KD fail, so it stopped before the retrain.
  - The weight moves 742 of the 1,441 targeted pairs into the top-20, but it pushes out 2,305 address-evidence true pairs (68% share a house number, none name-only). Net -941 true vs B-10.
- **Nothing committed.** No submission. Root `output/` still holds S-03.
