# Business Entity Resolution — LargeLanguageCoders

Pipeline: **blocking → pair features → matcher → global threshold**. How it works, in plain language: [docs/MODEL_AND_PIPELINE_EXPLAINED.md](docs/MODEL_AND_PIPELINE_EXPLAINED.md).

| Config | System | Validation F0.5 / precision |
|---|---|---|
| `configs/baseline_v0.yaml` | EXP-B-01, Submission 1 (hand-weighted rule) | 0.7851 / 0.9226 (public LB 0.751) |
| `configs/exp_f01b_conflicts_zeros.yaml` | **EXP-F-01b, current best** (Blocking v1 + conflict features + logistic regression) | 0.8371 / 0.9316 (dev-scale fit) |

Decisions: `docs/DECISION_LOG.md` · measurements: `docs/EXPERIMENT_LOG.md` · Submission-1 analysis: `reports/EXP-B-01_SUBMISSION_1_REPORT.md`.

## Run guide (repo root, PowerShell or Git Bash)

```bash
# A. environment: the existing env; no new dependencies in Phase 2
conda activate ber_hackathon          # PowerShell needs a one-time `conda init powershell` + a new terminal
python scripts/verify_env.py          # expect RESULT OK and python ...envs\ber_hackathon\python.exe

# B. tests (synthetic data only, ~10 s)
python -m pytest -q

# C. preprocessing + frozen split: ALREADY DONE on this machine (skip). Only if data/ is missing:
#    python -m ber.pipeline.run_baseline preprocess --config configs/baseline_v0.yaml
#    python -m ber.pipeline.run_baseline split      --config configs/baseline_v0.yaml   (never with --force)

# D. (optional) EXP-B-02 blocking sweep, blocking only, ~20 min
python scripts/experiments/run_blocking_sweep.py

# E. current best system, EXP-F-01b (full scale)
python -m ber.pipeline.run_baseline fit    --config configs/exp_f01b_conflicts_zeros.yaml   # LR on 200k TRAIN-fold S1 (~5 min)
python -m ber.pipeline.run_baseline train  --config configs/exp_f01b_conflicts_zeros.yaml   # validation scoring + report (~4 min)
python -m ber.pipeline.run_baseline infer  --config configs/exp_f01b_conflicts_zeros.yaml   # test scoring (~15-20 min)
python -m ber.pipeline.run_baseline submit --config configs/exp_f01b_conflicts_zeros.yaml   # writes output/*.tsv + strict check

# F. validators
python -m ber.submission.strict_validator --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir datasets/test --scored-dir data/processed/test/scored_EXP-F-01b --train-dir datasets/train
python student_resource/utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t datasets/test
```

Other stages:
- `block-eval`: blocking-only recall/cost on the validation split.
- `evaluate`: re-score stored validation features with another matcher (no re-blocking).
- Any setting can be overridden with `--set key=value`, e.g. `--set fit.max_train_s1=50000`.

Outputs per experiment go to `experiments/runs/<ID>/`:
- `metrics.json`: precision-first report and threshold table;
- `model.json`: learned weights and threshold;
- `blocking_eval.json`, `fit_report.json`, `config_snapshot.yaml`.

## Layout

```
src/ber/normalization  script-aware text normalization + record views (Indic-safe)
src/ber/data           ingest (TSV->Parquet, row-count checks), per-country normalization, split
src/ber/blocking       country-partitioned multi-pass key blocking + top-K cap
src/ber/features       compact float32 pair features (rapidfuzz + token sets + conflict features)
src/ber/models         weighted rule (not trained) and logistic regression (numpy, trained on train-fold S1)
src/ber/decision       global threshold (from model / validation sweep / config)
src/ber/evaluation     official entity-level macro F0.5 + precision/recall/false-merge breakdowns
src/ber/submission     the single TSV writer + strict validator
src/ber/pipeline       run_baseline.py (stages above)
scripts/analysis       read-only error analysis and diagnostics
scripts/experiments    controlled experiment runners (blocking sweep)
```

Memory: every stage streams or works per country; peak about 2.2 GB on the largest partitions. Close other heavy applications before full runs.
