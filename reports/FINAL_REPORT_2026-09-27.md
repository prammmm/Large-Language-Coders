# Final report — Business Entity Resolution: from baseline to the final submission

Team LargeLanguageCoders · 2026-09-27 · Author: Claude Code (senior ML engineering review + execution).

Evidence labels:

| Label | Meaning |
|---|---|
| **[M]** | Measured on labelled train data |
| **[D]** | Label-free diagnostic |
| **[E]** | Estimate |

All local numbers come from run artifacts (`scripts/analysis/audit_summary.py` → `experiments/runs/FINAL/audit_summary.json`), not from memory. Unknown values are marked N/A.

## Executive summary

**Final system: EXP-S-01.** It is the frozen EXP-B-06 stage 1 (normalization + cross-script view + Blocking V2 + generic-name and name-pair keys + retrained LightGBM) plus a **collective second-stage LightGBM decision model** with best-claimant resolution (D-01).

Held-out gain over the verified control EXP-B-06 + D-01 @ 0.67 [M]:

| Held-out population | EXP-B-06 + D-01 | **EXP-S-01** | Delta |
|---|---|---|---|
| E half B (1.0M S1) | 0.9471 | **0.9596** | **+0.0125** |
| Frozen validation (220,682 S1) | 0.9470 | **0.9601** | **+0.0131** |
| Precision (E half B) | 0.9863 | **0.9875** | +0.0012 |
| Recall (E half B) | 0.8832 | 0.9162 | +0.0330 |

- Over the last scored submission (N-02 + D-01, public LB **0.928**), the local gain is +0.0194 on E half B (0.9403 → 0.9596).
- A leave-one-country-out diagnostic shows the new stage **also improves an unseen country** (+0.021, precision +0.015).
- **Expected public LB [E]: about 0.94–0.95.** This is not a claim; it depends on the unlabelled test shift described in the caveats.

## 1. Audit — claims verified against artifacts

| Claim (research review) | Artifact value | Verdict |
|---|---|---|
| B-04 + D-01 E 0.9313 | `EXP-D-01/full_competition_eval.json`: 0.93134 | ✔ |
| N-02 + D-01 E 0.9402, LB 0.928 | 0.94025; LB as reported by the team | ✔ |
| N-03 E 0.9413 | 0.94125 | ✔ |
| B-07 + D-01 E 0.9464 | 0.94675 @ 0.63 (`full_competition_unrestricted.json`); **0.9464 @ 0.67** (`threshold_067_vs_063_E.json`) | ✔ (threshold-specific) |
| B-06 + D-01 E 0.9471 | 0.94706 (P 0.98630) | ✔ |
| Blocking recall 0.94–0.95; oracle ≈ 0.9835 | B-06: 0.9487 (val/E); oracle 0.9835 (val), 0.9832 (E) | ✔ |
| Remaining loss dominated by scoring/decision | B-07 error budget: low-scored true +0.023, false merges +0.016, not blocked +0.021 | ✔ |

Discrepancies found (documented, history not rewritten):
- `EXP-B-07/submission_summary.json` was overwritten by the t 0.67 output run. The t 0.63 summary is preserved as `submission_summary_t063.json`.
- Root `output/` holds the **B-06** files (sha `c8b0d4cb…` / `1e765d7a…`), promoted by the team. Their public score was not reported to me: N/A.

**Control selection (Step A).** Both candidates were verified:
- configs: `exp_b07_name_x_addr.yaml` / `exp_b06_name_pair.yaml`;
- models: `model.json` threshold 0.63 / 0.67 (B-07 run at 0.67);
- D-01 on; halves agree; strict and official validators PASS; no subset violations.

B-06 was selected: E 0.9471 vs 0.9464, precision 0.9863 vs 0.9864, singleton FP 5,567 vs 5,530 (+0.7%), halves consistent.

## 2. Experiment timeline (verified numbers)

| Experiment | Main change | Local (E F0.5 with D-01 unless noted) | Public LB | Decision |
|---|---|---|---|---|
| EXP-B-01 | Baseline: blocking V0 + weighted rule | val 0.7851 | 0.751 | banked |
| EXP-F-01b | LR + conflict features | val 0.8372 | 0.815 | banked |
| EXP-M-03 | LightGBM | val 0.9037 | 0.889 | banked |
| EXP-B-03a | Name-token key, fixed model | val 0.9027 | 0.890 | rejected |
| EXP-B-04 | Blocking V2 + refit | val 0.9274; E unrestricted 0.9273 | ≈ 0.91 | banked |
| EXP-D-01 | Best claimant per record | E 0.9313 (+0.0040) | — | kept |
| EXP-B-05 | Address window 3 → 5 | E 0.9324 (+0.0010; P −0.0027) | — | rejected |
| EXP-N-02 | Train-learned cross-script view | E 0.9402 | **0.928** | kept |
| EXP-N-03 | + learned Indic legal-form words | E 0.9413 (+0.0010) | — | below gate; kept as base |
| EXP-B-07 | + exact name × address-word key | E 0.9464 @ 0.67 | N/A | kept |
| EXP-B-06 | + name-token-pair key | E 0.9471 | N/A (promoted to `output/` by the team; score not reported) | kept; **control** |
| **EXP-S-01** | **Collective stage-2 decision model** | **E-B 0.9596 / val 0.9601** | not yet | **KEEP — FINAL** |
| EXP-S-01b | S-01 without count features | E-B 0.9591 / val 0.9595 | — | validated alternative |
| EXP-C-01 / C-02 | LOCO stage 1 (diagnostic) | unseen-country loss −0.075 / −0.039 | — | diagnostic |
| EXP-C-03 | LOCO stage 2 (diagnostic) | unseen India +0.0208 | — | diagnostic |

## 3. Error evolution (dominant failure over time)

1. **B-01 → M-03:** the matcher (hand weights) and retrieval (18% of true pairs never blocked). Fixed with LR/LightGBM and conflict features.
2. **B-04:** retrieval via combined keys (blocking 0.86 → 0.93).
3. **D-01:** false merges from multiple claimants, visible only under full competition.
4. **N-02/N-03:** **cross-script loss** in India (script-mismatch oracle +0.019 → +0.002).
5. **B-07/B-06:** generic names capped by block size, and name-only records (blocking 0.933 → 0.949).
6. **S-01:** weak **per-pair decisions without entity context** (low-scored true +0.023, false merges +0.016). This is the largest remaining lever and was fixed by the collective stage.
7. **Now:** **country robustness** (unseen France; the diagnostic shows −0.04 to −0.075, mostly precision) and the remaining never-retrieved pairs (+0.021 oracle).

## 4. EXP-S-01 in detail

- **Hypothesis:** stage 1 scores each (S1, record) pair independently, but an S1's true records resemble each other and compete with other S1s' claims. A decision model that sees the candidate set and peer records recovers underscored true pairs and rejects isolated weak ones.
- **Control:** EXP-B-06 + D-01 @ 0.67.
- **Primary change:** a second-stage LightGBM (41 features). Eligible pairs: stage-1 score ≥ 0.02. The candidate universe and `candidate_pairs.tsv` are unchanged: byte-identical to B-06's (`c8b0d4cb…`).
- **Features** (`src/ber/decision/stage2.py`; country-agnostic, label-free):
  - stage-1 score;
  - S1 context: rank, count, relative position, best, 2nd best, gap, # ≥ 0.5, # ≥ 0.8;
  - record context: claimants, # ≥ 0.5, best / 2nd best claimant, rank, gap to the best **other** claimant, is-best;
  - collective: for the top-3 **other** candidates of the same S1 (the pair itself is excluded, tested), record-to-record name Jaccard, `name_xlit` Jaccard, name token-set, address token-set, number Jaccard (zero-stripped), useful-address-word Jaccard, peer score; plus peer count, max similarity, score-weighted mean similarity, best-peer similarity.
- **Training (no leakage):**
  - stage-1 scores of all train S1s come from the frozen B-06 model;
  - E S1s are out-of-sample;
  - stage 2 trains **only on E half A minus frozen validation**: 714,270 S1s / 3,148,590 pairs;
  - the inner hash split (178,824 S1s / 789,004 pairs) is used for early stopping (best iteration 1,987 of 2,000) and to choose the decision rule;
  - `populations()` raises if validation, E half B or stage-1-fit S1s enter training (tested).
- **Decision:** stage-2 probability ≥ **0.65**, then **best claimant per record** (D-01 on the stage-2 probability). Chosen on the inner split against "threshold only".

**Measurements [M]** (control → S-01):

| Population | F0.5 | Precision | Recall | False merges | Singleton FP | Empty-on-positive | Predicted / S1 |
|---|---|---|---|---|---|---|---|
| **E half B (held-out)** | 0.9471 → **0.9596** | 0.9863 → 0.9875 | 0.8832 → 0.9162 | 40,389 → 36,955 | 2,795 → 2,285 | 11,683 → 10,171 | 3.092 → 3.211 |
| **Validation (held-out)** | 0.9470 → **0.9601** | 0.9862 → 0.9878 | 0.8836 → 0.9167 | 8,890 → 7,994 | 621 → 469 | 2,630 → 2,235 | 3.093 → 3.212 |
| Validation half A / B | 0.9468 → 0.9601 / 0.9472 → 0.9600 | | | | | | |
| E half A (contains the training rows; not held-out) | 0.9470 → 0.9617 | | | | | | |
| Stage-2 training partition (in-sample) | 0.9470 → 0.9625 | | | | | | |

**By country (held-out):**

| Country | E half B F0.5 | Validation F0.5 | Precision (E-B) | False merges (E-B) |
|---|---|---|---|---|
| India | 0.9396 → 0.9530 | 0.9395 → 0.9536 | 0.9826 → 0.9862 | −22% |
| US | 0.9522 → 0.9641 | 0.9520 → 0.9644 | 0.9887 → 0.9883 (−0.0004) | +5% |

**Match buckets (E half B):**

| True matches | 0 | 1 | 2–3 | 4–5 | 6+ |
|---|---|---|---|---|---|
| F0.5 | 0.9501 → 0.9592 | 0.8678 → 0.8682 | 0.9446 → 0.9577 | 0.9573 → 0.9714 | 0.9598 → 0.9728 |

**What changed on E half B:**
- 148,327 pairs added, **134,351 true (90.6%)**, with stage-1 scores 0.26–0.57 (interquartile);
- 28,370 pairs removed, 10,960 true (38.6%), with stage-1 scores 0.71–0.86;
- conflicts after the decision: 0.

**Other checks:**
- Blocking recall 0.9487 and oracle 0.9832 are unchanged (same candidates).
- Most important features by gain: score 83%, gap to the best other claimant 3.3%, peer number overlap, best claimant score, relative position, peer max similarity.
- Determinism: a synthetic re-fit gives a byte-identical model and output (test). On real data, a re-run of evaluate gives identical metrics, and a re-run of submit gives **byte-identical** files (`4cad314e…`).

**Robustness diagnostics:**

*Train → test shift [D].* The test pool is denser (5.8 vs 4.7 pool records per S1). Test S1s have more eligible candidates (median 6–7 vs 5), and France records have more claimants (2.17 vs 1.3–1.7).

| Predictions per S1, stage 2 vs control | Train E half B | Test (observed) | Test (expected from train within-bin behaviour) |
|---|---|---|---|
| Rise | +3.9% | +9.3% | +4.0% |

- The excess is concentrated in S1s with 1–5 candidates, and is **not** caused by count features: removing them (S-01b) reduces it by only 8%.
- Without test labels we cannot tell whether the extra acceptances are true records of the same entities (plausible, given the denser pool and high peer similarity) or false merges.

*Leave-one-country-out (C-03, labels) [M].* Stage 2 trained only on US data, on top of a US-only stage 1, raises the **unseen** India from 0.8724 to **0.8932** (E-B), precision 0.8981 → 0.9133, false merges −15%. In-domain US gains +0.0101. The stage generalizes **better** to an unseen country than the stage-1 decision it replaces.

**Gate:**

| Gate | Result |
|---|---|
| E-B ≥ +0.003 | ✔ +0.0125 |
| Validation ≥ +0.003 | ✔ +0.0131 |
| Precision drop ≤ 0.001 | ✔ it rises |
| E-B and validation same direction | ✔ |
| Singleton FP | ✔ −18% |
| Production peak ≤ 5 GB | ✔ features 4.32 / 3.95, fit 3.11, submit 4.79 after the memory fix (first submit run 5.14, fixed and re-run), evaluate 2.98 after the fix (5.41 before) |
| Deterministic | ✔ |
| No leakage | ✔ guards + tests |
| Tests | ✔ 161 / 161 |
| Official validator | ✔ |

→ **KEEP.**

## 5. EXP-S-01b (validated alternative)

- Same as S-01 minus 7 count/position features.
- E-B 0.9591, validation 0.9595 (within 0.001 of S-01); precision 0.9875 / 0.9876; singleton FP 2,205 (E-B).
- Test inflation +8.6% vs S-01's +9.3%.

Pre-declared rule: prefer S-01b only if held-out F0.5 is within 0.001 **and** the test inflation is *clearly* lower. The second condition fails (only 8% of the excess is removed), so **not selected**. Its output is validated at `experiments/runs/EXP-S-01b/output/`.

## 6. Country-robustness diagnostics (Step C)

| Diagnostic | In-domain | Unseen country | Evidence |
|---|---|---|---|
| C-01: stage 1 on US only (200k) | US 0.9527 | India 0.8609 (P 0.8830) vs 0.9358 (P 0.9773) with both countries; false merges ×5.5; at the validation-optimal threshold only 0.8794 | [M] |
| C-02: stage 1 on India only (200k) | India 0.9397 | US 0.9108 (P 0.9440) vs 0.9493 (P 0.9843) | [M] |
| C-03: stage 2 on US only, on C-01 | US +0.0101 | India +0.0208, precision +0.0152 | [M] |

**Interpretation:**
- An unseen country loses 0.04–0.075 F0.5, mostly **precision**. This explains the local → LB gap (France estimated ≈ 0.88 at N-02).
- The US-only model leans on `name_ratio × addr_token_set` (56% of gain), whose meaning changes with address length. `cheap_score` (blocker magnitude) is **not** the driver: 4% in the US-only model.
- A scale-free replacement of `cheap_score` was therefore not pursued.
- The collective stage 2 is the measured robustness improvement.

**France [D, label-free]:** French S1 addresses write "saint" (10.0%); 1.7% of French pool records write "st", which normalizes to "street". Bounded, and **not** converted into a rule (no labels, no country-specific rules).

## 7. Smaller levers (Step D) — not executed, with reasons

- **D1 (top-K 30):** measured +0.0015 on B-04. The top-K cut is now 1.5–2% of true pairs.
- **D2 (more S1s):** measured +0.002 per 4×.
- **D3:** the stage-2 threshold is already tuned inside the training partition.

All are small compared with S-01, would each need a full stage-1 retrain plus a stage-2 retrain (≈ 2.5 h), and were not needed to select the final system.

## 8. Final architecture

```
normalization (multi-view, Indic-safe) → train-learned cross-script name view (N-02/N-03)
→ Blocking V2 + exact-name×address-word (B-07) + name-token-pair (B-06), top-K 20
→ stage 1: 24 pair features → LightGBM (B-06)
→ entity/context features: S1 context + record-claimant context + top-3 peer record similarity
→ stage 2: LightGBM (S-01) → p ≥ 0.65 → best claimant per record (D-01)
→ writer (UTF-8, LF, one row per S1, matches ⊆ candidates) → strict validator → official validator
```

## 9. Final local metrics (EXP-S-01) [M]

| Metric | Value |
|---|---|
| F0.5, E half B (held-out) | **0.9596** (P 0.9875, R 0.9162) |
| F0.5, frozen validation | **0.9601** (P 0.9878, R 0.9167) |
| India F0.5 (E-B / validation) | 0.9530 / 0.9536 |
| US F0.5 (E-B / validation) | 0.9641 / 0.9644 |
| Blocking recall / oracle | 0.9487 / 0.9832 |
| False merges / singleton FP (E-B) | 36,955 / 2,285 |
| Test candidates | 33,489,321 pairs (19.33 per S1; 22 empty rows) |
| Test predictions | 5,874,532 ids (3.391 per S1; 101,917 empty rows) |

Runtime and peak (process tree):

| Step | Runtime | Peak |
|---|---|---|
| Stage-2 features, train | 454 s | 4.32 GB |
| Stage-2 features, test | 419 s | 3.95 GB |
| Fit | 992 s (incl. all-pair prediction for rule selection) | 3.11 GB |
| Evaluate | 333 s | 2.98 GB |
| Submit (incl. strict validation) | 522 s | 4.79 GB |

## 10. Data-compliance statement

- **Data:** only the provided train/test data and train labels. No external data, business directories, maps/OSM/geocoding, registries, entity APIs, internet augmentation or external labels.
- **Cross-script mapping:** learned only from provided train true pairs. No external dictionary, transliteration package or pretrained model/embedding.
- **Dependencies:** LightGBM 4.7.0 (MIT), numpy, pandas, pyarrow, rapidfuzz; no new dependency in this phase.
- **Country:** open set; no country-specific rules or features.
- **Leakage:** frozen validation and E half B contributed nothing to any map, model, threshold or stage-2 training (guards + tests).
- **Raw data:** `datasets/` untouched.

## 11. Execution log (this phase)

All runs used `E:\hell\hell\envs\ber_hackathon\python.exe` with `PYTHONIOENCODING=utf-8`. Peak = process tree. "Exit 0" = completed successfully.

| # | Command | Experiment / config | Tests before | Result | Key output | Runtime / peak |
|---|---|---|---|---|---|---|
| 1 | `python scripts/analysis/audit_summary.py` | audit | — | exit 0 | `experiments/runs/FINAL/audit_summary.json` | s |
| 2 | `python -m pytest tests/test_stage2.py -W error::RuntimeWarning` | S-01 unit | — | 1 fail (nanmean warning) → fixed → 7/7 | — | 1 s |
| 3 | `python -m pytest tests/test_stage2_pipeline.py` | S-01 synthetic end-to-end | 7/7 | 5/5 | — | 6 s |
| 4 | `python -m pytest tests` | full | — | 157/157 | — | 34 s |
| 5 | `python -m ber.pipeline.run_stage2 features --config configs/exp_s01_stage2.yaml --split train` | S-01 | 157 | exit 0 | 9,726,117 pairs → `data/processed/train/stage2_EXP-S-01` | 454 s / 4.32 GB |
| 6 | `... features --split test` | S-01 | | exit 0 | 8,737,308 pairs | 419 s / 3.95 GB |
| 7 | `python -m ber.pipeline.run_stage2 fit --config configs/exp_s01_stage2.yaml` | S-01 | | exit 0 | best iteration 1,987; threshold + D-01 @ 0.65; inner 0.9596 | 992 s / 3.11 GB |
| 8 | `... evaluate` | S-01 | | exit 0 | E-B +0.0125, validation +0.0131 | 327 s / 5.41 GB (evaluation only; fixed in #20) |
| 9 | `... submit` | S-01 | | exit 0, strict PASS | `experiments/runs/EXP-S-01/output` | 496 s / 5.14 GB (fixed in #21) |
| 10 | `python student_resource/utils/validate_submission.py -m …EXP-S-01/output/matching_results.tsv -c …/candidate_pairs.tsv -t datasets/test --check-ids` | S-01 | | PASS | — | — |
| 11 | `python -m ber.pipeline.run_baseline fit/train --config configs/exp_c01_loco_us.yaml` | C-01 | 161 | exit 0 | India 0.8609 | 1,067 s + 483 s / 2.71 GB |
| 12 | same with `configs/exp_c02_loco_in.yaml` | C-02 | | exit 0 | US 0.9108 | 878 s + 459 s / 2.79 GB |
| 13 | label-free feature-shift + France `st/saint` scripts (inline, read-only) | S-01 diagnostics | | exit 0 | `EXP-S-01/feature_shift_train_vs_test.json`, `france_st_saint_labelfree.json` | < 2 min |
| 14 | `python -m ber.pipeline.run_stage2 fit/evaluate/submit --config configs/exp_s01b_stage2_robust.yaml` + official validator | S-01b | 14 stage-2 tests | exit 0, PASS | E-B 0.9591, validation 0.9595 | 912 / 296 / 464 s |
| 15 | `python scripts/analysis/stage2_shift_check.py --configs configs/exp_s01_stage2.yaml configs/exp_s01b_stage2_robust.yaml` | shift | | exit 0 | test excess +5.3% / +4.9% | 1,066 s / 5.85 GB (diagnostic) |
| 16 | `python scripts/analysis/full_competition.py --config configs/exp_c01_loco_us.yaml --scored-tag forensic_alltrain_C01 …` | C-03 prerequisite | | exit 0, sanity reproduces | — | ≈ 35 min |
| 17 | `python -m ber.pipeline.run_stage2 features --split train / fit / evaluate --config configs/exp_c03_loco_stage2_us.yaml` | C-03 | | exit 0 | unseen India +0.0208 | ≈ 26 min |
| 18 | `python -m pytest tests/test_stage2_pipeline.py tests/test_loco.py` (after each option was added) | S-01b / C | | all pass | — | — |
| 19 | `python -m pytest -q` | final | | **161 / 161** | — | 42 s |
| 20 | `python -m ber.pipeline.run_stage2 evaluate --config configs/exp_s01_stage2.yaml` (re-run) | S-01 | | exit 0; metrics identical | — | 333 s / 2.98 GB |
| 21 | `python -m ber.pipeline.run_stage2 submit --config configs/exp_s01_stage2.yaml --set paths.output_dir=experiments/runs/EXP-S-01/output_repro` | S-01 determinism | | exit 0 | byte-identical to `output/` | 522 s / 4.79 GB |
| 22 | `python -m ber.submission.strict_validator --matching … --candidate … --test-dir datasets/test --train-dir datasets/train --scored-dir data/processed/test/scored_EXP-B-06` | S-01 | | **PASS (strict)** | candidate set == scored set (33,489,321) | — |

Logged in `docs/EXPERIMENT_LOG.md` (S-01, S-01b, C-01..C-03), `docs/DECISION_LOG.md` (DEC-020, Proposed) and `docs/DEVELOPMENT_LOG.md`.

## 12. Final output locations

| Item | Path |
|---|---|
| Final candidate file | `experiments/runs/EXP-S-01/output/candidate_pairs.tsv` (sha256 `c8b0d4cbad01ec5a…`) |
| Final matching file | `experiments/runs/EXP-S-01/output/matching_results.tsv` (sha256 `4cad314e68e139f5…`) |
| Determinism copy (identical, can be deleted) | `experiments/runs/EXP-S-01/output_repro/` |
| Final config | `configs/exp_s01_stage2.yaml` (stage 1: `configs/exp_b06_name_pair.yaml`) |
| Models | `experiments/runs/EXP-B-06/{model.json, model.lgb.txt, xlit_map.json}`; `experiments/runs/EXP-S-01/{stage2_model.lgb.txt, stage2_model.json}` |
| This report | `reports/FINAL_REPORT_2026-09-27.md` |

## 13. Caveats (supported by the experiments)

1. **Test-shift acceptance.** S-01 accepts 9.3% more pairs than the control on test, against 4% expected from train behaviour. Whether the extra acceptances are true (denser test pool) or false cannot be measured locally; the public LB is the only check. If the LB score falls below the B-06 file's, revert to B-06 (backup below).
2. **France is unlabelled.** The LOCO diagnostics show unseen-country precision loss (−0.04 to −0.075 at stage 1). Stage 2 reduces it on India (+0.021), but France itself is unmeasured.
3. Stage-2 context uses stage-1 scores of the 200k stage-1 fit S1s as competitors; those scores are in-sample (≈ 9% of train S1s). On test, all S1s are out-of-sample.
4. Remaining never-retrieved pairs (oracle +0.021) and the French "st"/"saint" abbreviation (≤ 1.7% of French pool records) are not addressed.

---

# FINAL COMMANDS TO GENERATE AND VALIDATE THE SUBMISSION

PowerShell, from the repository root. Every command below exists in this repository. Steps 2, 3a/3b, 4–6 were executed during this phase (3b as separate `--split` runs of the same stage); the 3c stage-1 commands were executed in earlier phases (B-06 artefacts) and not re-run now; step 7 (promotion) is for you to run.

```powershell
# 1. Environment
cd "D:\Large Language Coders"
conda activate ber_hackathon                       # if activation fails: $PY = "E:\hell\hell\envs\ber_hackathon\python.exe" and use & $PY instead of python
$env:PYTHONIOENCODING = "utf-8"
python -c "import sys, lightgbm; print(sys.executable, lightgbm.__version__)"   # expect ...ber_hackathon\python.exe 4.7.0

# 2. Tests (expect: 161 passed)
python -m pytest -q

# 3. Generate the final output (EXP-S-01) into its experiment folder
#    3a. FAST: re-generate from the saved stage-1 test scores + saved stage-2 model (~9 min, writes experiments\runs\EXP-S-01\output, runs strict validation)
python -m ber.pipeline.run_stage2 submit --config configs\exp_s01_stage2.yaml
#    3b. FULL stage-2 rebuild from the existing stage-1 artefacts (~40 min): features (train + test) -> fit -> evaluate -> submit
python -m ber.pipeline.run_stage2 features --config configs\exp_s01_stage2.yaml
python -m ber.pipeline.run_stage2 fit      --config configs\exp_s01_stage2.yaml
python -m ber.pipeline.run_stage2 evaluate --config configs\exp_s01_stage2.yaml
python -m ber.pipeline.run_stage2 submit   --config configs\exp_s01_stage2.yaml
#    3c. FROM SCRATCH (stage 1 as well; hours): run these first, then 3b
#        python -m ber.pipeline.run_baseline preprocess --config configs\exp_b06_name_pair.yaml
#        python -m ber.pipeline.run_baseline split      --config configs\exp_b06_name_pair.yaml
#        python -m ber.pipeline.run_baseline fit        --config configs\exp_b06_name_pair.yaml
#        python -m ber.pipeline.run_baseline train      --config configs\exp_b06_name_pair.yaml
#        python -m ber.pipeline.run_baseline infer      --config configs\exp_b06_name_pair.yaml
#        python scripts\analysis\full_competition.py --config configs\exp_b06_name_pair.yaml --scored-tag forensic_alltrain_B06 --out experiments\runs\EXP-B-06\full_competition_unrestricted.json

# 4 + 5. Strict project validation of the candidate AND matching files (expect: PASS (strict))
python -m ber.submission.strict_validator --matching experiments\runs\EXP-S-01\output\matching_results.tsv --candidate experiments\runs\EXP-S-01\output\candidate_pairs.tsv --test-dir datasets\test --train-dir datasets\train --scored-dir data\processed\test\scored_EXP-B-06

# 6. Official validator (expect: PASS — no blocking issues found)
python student_resource\utils\validate_submission.py -m experiments\runs\EXP-S-01\output\matching_results.tsv -c experiments\runs\EXP-S-01\output\candidate_pairs.tsv -t datasets\test --check-ids

# 7. Promote to output\ (back up the current files first), then re-check
New-Item -ItemType Directory -Force submissions\EXP-B-06_output | Out-Null
Copy-Item output\*.tsv submissions\EXP-B-06_output\
Copy-Item experiments\runs\EXP-S-01\output\*.tsv output\ -Force
python student_resource\utils\validate_submission.py -m output\matching_results.tsv -c output\candidate_pairs.tsv -t datasets\test --check-ids
#    (revert if needed:  Copy-Item submissions\EXP-B-06_output\*.tsv output\ -Force)

# 8. Packaging: no packaging script exists in this repository. Required structure (HARD CONSTRAINT 13):
#    <team_name>_submission.zip -> output\{matching_results.tsv,candidate_pairs.tsv},
#    code\business_entity_resolution\{src\,README.md,requirements.txt}, Documentation_template.md (filled in)
```
