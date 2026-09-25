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
