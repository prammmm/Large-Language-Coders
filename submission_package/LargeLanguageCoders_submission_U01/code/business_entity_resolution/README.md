# Business Entity Resolution — LargeLanguageCoders

**Pipeline:** country-partitioned multi-key blocking (top-20 candidates per Source-1 entity) → pair features → **LightGBM stage-1 matcher** → global threshold → **D-01 best-claimant rule** (each Source-2/3 record goes to at most one Source-1 entity) → **stage-2 LightGBM veto** (context model; it can only *remove* stage-1 matches).

**Rules followed:**
- CPU only;
- only the provided train/test data;
- no external data, lookups, dictionaries or pretrained models;
- LightGBM (MIT licence) is the only model;
- seed 42 throughout.

## 1. Environment

Python 3.13 (tested on 3.13.15, Windows 11, 16 GB RAM).

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
```

Always run from this folder, with the source on the path:

```bash
# Git Bash / Linux
export PYTHONPATH=src PYTHONNOUSERSITE=1
# PowerShell
$env:PYTHONPATH="src"; $env:PYTHONNOUSERSITE="1"; $env:PYTHONIOENCODING="utf-8"
```

**Data.** Place the official data as:
- `datasets/train/` (source1, source2, source3, ground truth);
- `datasets/test/` (source1, source2, source3).

All derived data goes to `data/` and `experiments/runs/`. The raw files are never modified.

**Memory.** Every stage streams per country or per part. Peak memory is at most 5 GB.

## 2. Full reproduction (data → blocking → matching → output)

| # | Command | What it does | Time [measured] |
|---|---|---|---|
| 1 | `python -m ber.pipeline.run_baseline preprocess --config configs/exp_b09_name_num.yaml` | TSV → Parquet, row-count checks, per-country normalised record views | ≈ 20–30 min |
| 2 | `python -m ber.pipeline.run_baseline split --config configs/exp_b09_name_num.yaml` | deterministic group-aware validation split (seed 42) | < 5 min |
| 3 | `python -m ber.pipeline.run_baseline fit --config configs/exp_b09_name_num.yaml` | learns the cross-script name map (train fit set only); blocks the 200k-entity fit set; trains stage-1 LightGBM; picks the threshold on an inner split → `experiments/runs/EXP-B-09/model.json` | ≈ 36 min |
| 4 | `python -m ber.pipeline.run_baseline infer --config configs/exp_b09_name_num.yaml` | blocks and scores every test Source-1 entity (top-20 candidates) | ≈ 26 min |
| 5 | `python -m ber.pipeline.score_alltrain --config configs/exp_b09_name_num.yaml --tag forensic_alltrain_B09` | stage-1 scores for all train entities (stage-2 training input) | ≈ 33 min |
| 6 | `python -m ber.pipeline.run_stage2 features --config configs/exp_s03_veto_on_b09.yaml` | stage-2 context features, train and test | ≈ 27 min |
| 7 | `python -m ber.pipeline.run_stage2 fit --config configs/exp_s03_veto_on_b09.yaml` | stage-2 LightGBM on held-out train entities; veto threshold on an inner split → `experiments/runs/EXP-S-03/stage2_model.*` | ≈ 22 min |
| 8 | `python -m ber.pipeline.run_stage2 submit --config configs/exp_s03_veto_on_b09.yaml` | final decision → `experiments/runs/EXP-S-03/output/{matching_results.tsv,candidate_pairs.tsv}` + strict validation | ≈ 8 min |

Step 8's output is byte-identical to `../../output/` (two runs give identical SHA-256).

**Quick path.** The trained models ship in this folder:
- `experiments/runs/EXP-B-09/model.json` and `xlit_map.json`;
- `experiments/runs/EXP-S-03/stage2_model.*`.

With them, **steps 3, 5 and 7 can be skipped**: run 1 → 2 → 4 → `run_stage2 features --split test --config configs/exp_s03_veto_on_b09.yaml` → 8. This takes about 1 h 15 min.

**Validation of the output (official script):**

```bash
python validate_submission.py -m experiments/runs/EXP-S-03/output/matching_results.tsv -c experiments/runs/EXP-S-03/output/candidate_pairs.tsv -t datasets/test --check-ids
```

## 3. Configs

Configs inherit from each other with `inherit:`. The chain for the final system:

```
baseline_v0 → … → exp_b07_name_x_addr → exp_b06_name_pair → exp_b09_name_num (stage 1)
           → exp_s03_veto_on_b09 (stage-2 veto)   [FINAL]
(exp_d04_ef_prune: expected-F0.5 pruning, evaluated, not selected)
```

## 4. Final system and output

- **Final system:** EXP-U-01 (S-03 + S-04 acceptances on new candidates; see §6 and `FINAL_SYSTEM.txt`).
- **Output files:** `../../output/` holds the exact files this pipeline writes in step 8 (UTF-8 without BOM, LF, tab-separated, one row per test Source-1 entity; `candidate_pairs.tsv` is the exact top-20 set the model scores).

## 5. Layout

```
src/ber/normalization  script-aware text normalisation, record views, train-learned cross-script name map
src/ber/data           ingest (TSV -> Parquet, row-count checks), per-country normalisation, validation split
src/ber/blocking       country-partitioned multi-pass key blocking + cheap-score top-K
src/ber/features       float32 pair features (rapidfuzz similarities, token sets, number / legal-form conflicts)
src/ber/models         LightGBM stage-1 scorer (+ earlier rule / logistic-regression scorers)
src/ber/decision       threshold, D-01 best claimant, stage-2 context features + veto, expected-F0.5 pruning
src/ber/evaluation     official per-entity macro F0.5 + diagnostics
src/ber/submission     TSV writer + strict validator
src/ber/pipeline       run_baseline (stage 1), score_alltrain, run_stage2 (stage 2 / submit)
configs/               every experiment config (inheritance chain above)
tests/                 unit tests on synthetic data (python -m pytest -q tests)
```

## 6. Final system EXP-U-01 (S-03 + S-04 acceptances on new candidates)

After steps 1–7 above (S-03 chain), run the S-04 chain, then combine:

```bash
python -m ber.pipeline.run_baseline fit   --config configs/exp_s04_stage1_b10.yaml          # B-10 blocker stage 1 (skip: model shipped)
python -m ber.pipeline.run_baseline infer --config configs/exp_s04_stage1_b10.yaml
python -m ber.pipeline.score_alltrain --config configs/exp_s04_stage1_b10.yaml --tag forensic_alltrain_S04   # skip with shipped models
python -m ber.pipeline.run_stage2 features --config configs/exp_s04_veto_on_b10.yaml              # --split test with shipped models
python -m ber.pipeline.run_stage2 fit      --config configs/exp_s04_veto_on_b10.yaml              # skip: model shipped
python scripts/analysis/exp_u01.py submit      # -> experiments/runs/EXP-U-01/output/ (= ../../output/, byte-identical)
```

`scripts/analysis/` and `experiments/shadow_tests/` (population masks used by the shared evaluation context) are included for this step.
