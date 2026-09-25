# Experiment Log

Rules (CLAUDE.md RULE 5): every experiment needs an ID, a hypothesis, a single primary change, an expected effect,
the measured result, an interpretation and a decision. **Never fabricate metrics — use `N/A` until measured.**

All local metrics are computed on the **fixed validation split** (see DEC-004) at the **S1-entity level**:
- **Local F0.5** = mean over validation S1 entities of per-entity F0.5 (singletons: 1.0 if the prediction is empty, else 0.0)
- **Precision / Recall** = macro means of per-entity P and R, using the same conventions as F0.5 (see report §5.3)
- **Blocking Recall** = (true pairs present in candidate_pairs) / (all true pairs), micro
- **Cand/S1** = mean candidates per S1 entity (median and p99 go in the run record)

ID scheme: `EXP-<phase>-<nn>` (e.g. `EXP-B-01` blocking, `EXP-M-01` model, `EXP-T-01` threshold, `EXP-S-01` submission).
Every run writes `experiments/runs/<ID>/{config.yaml,metrics.json,notes.md}` (the directory is gitignored except notes and configs we choose to copy here).

## Summary table

| ID | Hypothesis | Change | Local F0.5 | Precision | Recall | Blocking Recall | Cand/S1 | Public Score | Decision |
|----|------------|--------|------------|-----------|--------|-----------------|---------|--------------|----------|
| EXP-0-00 | Predicting all-empty establishes the floor (= validation singleton rate) | Trivial "no matches" submission, local only | N/A | N/A | N/A | N/A | 0 | N/A (do not submit) | Pending |
| EXP-B-01 | Blocking V0 (exact name ∪ address num+word ∪ rare address word, top-20) + weighted similarity rule + tuned global threshold gives a valid, precision-leaning Submission 1 | First end-to-end baseline (`configs/baseline_v0.yaml`), threshold tuned to 0.67 | 0.7851 | 0.9226 | 0.6553 | 0.7983 | 15.42 | **0.751** (Submission 1) | Baseline banked; see `reports/EXP-B-01_SUBMISSION_1_REPORT.md` |
| EXP-B-02 (sweep) | Blocking loses true pairs because keys are too narrow, not because K is small | 10 single-change blocking variants, blocking only | — | — | — | v0 0.7983 → b 0.8485 / bc 0.8597 (@20) | 15.4 → 16.1 | — | sorted_core + zero-strip kept; caps / max_nums / max_words rejected |
| EXP-B-02 | Blocking v1 (bc) lifts F0.5 with the unchanged rule matcher | bc blocking + weighted rule | 0.7841 | **0.8879 ↓** | 0.6936 | 0.8597 | 16.1 | — | **Rejected alone**: +21.5k false merges |
| EXP-M-02 | A learned LR beats the hand-weighted rule on identical candidates | LR (train-fold fit, dev 50k S1) on EXP-B-01 candidates | 0.8158 | 0.9238 | 0.7089 | 0.7983 | 15.4 | — | Keep |
| EXP-M-02b | Blocking v1 and LR are complementary | bc blocking + LR | 0.8200 | 0.9238 | 0.7215 | 0.8597 | 16.1 | — | Keep |
| EXP-F-01a | Number and legal-form conflicts cut false merges | + num_conflict, legal_form_conflict | 0.8287 | 0.9278 | 0.7318 | 0.8597 | 16.1 | — | Keep |
| **EXP-F-01b** | Leading-zero-safe number features sharpen the conflicts | + zero-stripped number features | **0.8371** | **0.9316** | 0.7421 | 0.8597 | 16.1 | pending | **Keep: current best (dev-scale fit)** |

## Detailed entries

> **Phase-2 protocol (applies to every entry below).**
> - Frozen validation split of 220,682 S1 (sha256 `ce1da2d3…`); every S1 is evaluated.
> - Learned matchers are fitted on a seeded sample of **train-fold** S1 only. The *dev* runs use 50,000 S1 via `--set fit.max_train_s1=50000`; the config default of 200,000 is for the team's full run. Validation S1 are excluded; `fit_s1_subset` raises on any overlap.
> - LR thresholds are tuned on an inner 20% of those train-fold S1 (`decision.threshold_from: model`), never on validation. EXP-B-0x rule runs keep EXP-B-01's validation-tuned threshold for comparability (optimistic).
> - Numbers come from `experiments/runs/<ID>/{metrics.json, blocking_eval.json, model.json}`. The halves check is in `experiments/runs/val_halves_comparison.json`.
> - Seed 42. No dependency changes in Phase 2.

### Precision-first comparison (validation, full split, threshold actually used)

| Run | t | F0.5 | P | R | False merges | Singleton FP | Pred/S1 | Cand/S1 | India F0.5 / P / R | US F0.5 / P / R |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-B-01 (rule) | 0.67 | 0.7851 | 0.9226 | 0.6553 | 39,489 | 2,724 | 2.45 | 15.4 | 0.692 / 0.869 / 0.566 | 0.847 / 0.956 / 0.715 |
| EXP-B-02 (bc + rule) | 0.67 | 0.7841 | 0.8879 | 0.6936 | 61,011 | 4,421 | 2.68 | 16.1 | 0.711 / 0.864 / 0.601 | 0.833 / 0.902 / 0.755 |
| EXP-M-02 (LR) | 0.49 | 0.8158 | 0.9238 | 0.7089 | 42,044 | 3,107 | 2.63 | 15.4 | 0.753 / 0.885 / 0.655 | 0.858 / 0.949 / 0.745 |
| EXP-M-02b (bc + LR) | 0.62 | 0.8200 | 0.9238 | 0.7215 | 39,955 | 3,692 | 2.67 | 16.1 | 0.759 / 0.903 / 0.652 | 0.861 / 0.937 / 0.768 |
| EXP-F-01a | 0.62 | 0.8287 | 0.9278 | 0.7318 | 38,861 | 3,420 | 2.70 | 16.1 | 0.771 / 0.909 / 0.665 | 0.867 / 0.940 / 0.776 |
| **EXP-F-01b** | 0.60 | **0.8371** | **0.9316** | 0.7421 | 37,054 | 3,247 | 2.72 | 16.1 | 0.779 / 0.908 / 0.676 | 0.876 / 0.947 / 0.786 |

- EXP-B-01: 501,169 true matches, 262,387 missed, 16,006 empty-on-positive.
- EXP-M-02: 539,227 true matches, 224,329 missed, 11,584 empty-on-positive.
- EXP-F-01b: 564,267 true matches, 199,289 missed, 9,889 empty-on-positive.
- Validation halves A/B (F0.5): B-01 0.7853 / 0.7849 · B-02 0.7842 / 0.7839 · M-02 0.8164 / 0.8153 · M-02b 0.8203 / 0.8197 · F-01a 0.8291 / 0.8284 · **F-01b 0.8374 / 0.8367**.
- Precision on halves A/B: B-01 0.9218 / 0.9234 · F-01b 0.9317 / 0.9315.

### EXP-B-02 — Blocking v1
- Hypothesis: True pairs are lost because blocking keys are too restrictive, not because top-K is small.
- Changes (one per variant): exact `name_core` key, sorted `name_core` key, leading-zero stripping of address numbers, name-token pass, `addr_num_word` cap 50→100/150/200, max_nums 2→3, max_words 4→6.
- Config: `configs/exp_b02_blocking_v1.yaml`; sweep `scripts/experiments/run_blocking_sweep.py`.
- Output: `experiments/runs/EXP-B-02/blocking_sweep.md`.
- Blocking results at K=20:

| Variant | Recall@20 | India | US | Cand/S1 | New true pairs | Extra candidates | Runtime | Peak |
|---|---|---|---|---|---|---|---|---|
| v0 | 0.7983 | 0.7790 | 0.8112 | 15.42 | — | — | 77 s | 1.64 GB |
| a exact core | 0.8431 | 0.8250 | 0.8552 | 16.04 | +34,235 | +138k | 94 s | 1.62 GB |
| **b sorted core** | 0.8485 | 0.8273 | 0.8626 | 16.08 | +38,300 | +146k | 105 s | 1.65 GB |
| **c zero-strip** | 0.8142 | 0.7837 | 0.8346 | 15.48 | +12,171 | +14.5k | 73 s | 1.63 GB |
| d name_token | 0.8177 | 0.7979 | 0.8310 | 17.10 | +14,846 | +372k | 83 s | 1.64 GB |
| e100 / e150 / e200 caps | 0.8033 / 0.8051 / 0.8062 | 0.788 / 0.792 / 0.795 | ≈0.814 | 16.3 / 16.6 / 16.8 | +3.8k / +5.2k / +6.0k | +200k / +263k / +297k | ~75 s | ~1.6 GB |
| f3 / g6 | 0.7976 / 0.7978 | — | — | 15.5 / 15.8 | −544 / −368 | +13k / +83k | 73 s | 1.6 GB |
| **bc** | **0.8597** | 0.8305 | 0.8792 | 16.13 | +46,875 | +157k | 105 s | 1.66 GB |
| bcd | 0.8686 | 0.8392 | 0.8882 | 17.58 | +53,659 | +478k | 118 s | 1.77 GB |

- Matcher-level result with the unchanged rule (EXP-B-02 row above):
  - F0.5 0.7841 (−0.001); **precision fell 0.9226 → 0.8879**.
  - +21,522 false merges; singleton FP +1,697.
- Interpretation:
  - The sorted-core key also retrieves same-name businesses at other addresses, and the rule accepts them.
  - Cap increases buy little recall for many candidates (reject). max_nums 3 and max_words 6 slightly *lower* recall@20.
  - name_token (d) adds mostly non-matching candidates (2% of its extra candidates are true); kept off.
- Decision: bc is kept **only in combination with a learned matcher** (EXP-M-02b). Blocking v1 + rule is rejected.

### EXP-M-02 — Logistic regression matcher
- Hypothesis: A fitted linear model on the same features/candidates beats the hand weights (EXP-D-01 probe: +0.031).
- Model:
  - Newton-IRLS logistic regression in numpy; L2 = 1e-4 on weights, intercept unpenalised; standardised features; no class weights.
  - 20 inputs: 16 features, cheap score/rank and 2 interactions.
  - 10 iterations; fit 2.3 s on 614,837 pairs (110,621 positive).
- Training data: 50,000 train-fold S1 (dev). Inner-tune threshold 0.49 (inner F0.5 0.8145).
- Evaluation: the stored EXP-B-01 validation features were re-scored (`evaluate`), so the candidates are identical.
- Result: F0.5 **0.8158** (+0.0307); P 0.9238 (+0.0012); R 0.7089 (+0.0536).
  - +38,058 true matches for +2,555 false merges; singleton FP +383.
  - India P ↑ (0.869 → 0.885); US P ↓ (0.956 → 0.949).
  - At t = 0.55 the LR beats the rule on F0.5, P, false merges *and* singleton FP simultaneously (0.8141 / 0.9354 / 34,141 / 2,654).
- The train-fold threshold (0.49) equals the validation optimum.
- Runtime: fit stage 91 s (blocking 84 s), peak 1.82 GB.
- Decision: **keep**.

### EXP-M-02b — Blocking v1 + LR
- Result: F0.5 0.8200 (+0.0042 vs M-02); P 0.9238 (=); false merges −2,089 vs M-02; singleton FP +585 vs M-02.
  - India P 0.903 (+0.018); US P 0.937 (−0.012).
- Interpretation: The gains are complementary, but only with a learned matcher.
- Runtime: validation scoring 179 s, peak 1.85 GB; fit 125 s.
- Decision: **keep**.

### EXP-F-01a / EXP-F-01b — Conflict features
- F-01a (`num_conflict`, `legal_form_conflict`): F0.5 0.8287 (+0.0087); **P 0.9278 (+0.0040)**.
  - False merges −1,094; singleton FP −272.
  - Both countries improve.
- F-01b (+ number features compared after leading-zero stripping): F0.5 **0.8371** (+0.0084); **P 0.9316 (+0.0038)**.
  - False merges −1,807; singleton FP −173.
  - India P flat (0.9086 → 0.9080); US P 0.940 → 0.947.
- F-01b threshold table (the train-fold choice 0.60 equals the validation optimum):

| t | F0.5 | P | R | Singleton FP | Pred/S1 | False merges |
|---|---|---|---|---|---|---|
| 0.50 | 0.8338 | 0.9078 | 0.7664 | 4,081 | 2.89 | 54,090 |
| 0.55 | 0.8360 | 0.9202 | 0.7550 | 3,647 | 2.81 | 45,160 |
| **0.60** | **0.8371** | **0.9316** | 0.7421 | 3,247 | 2.72 | 37,054 |
| 0.65 | 0.8356 | 0.9424 | 0.7249 | 2,773 | 2.63 | 30,028 |
| 0.70 | 0.8313 | 0.9520 | 0.7064 | 2,305 | 2.54 | 24,209 |
| 0.80 | 0.8161 | 0.9657 | 0.6672 | 1,558 | 2.37 | 16,149 |

- Still-open precision regressions vs EXP-B-01: singleton FP +523 (3,247 vs 2,724); US precision 0.947 vs 0.956.
- Runtime: validation scoring about 190 s, peak 2.2 GB.
- Decision: **keep F-01b as the current best system** (`configs/exp_f01b_conflicts_zeros.yaml`).
- Remaining loss (analysis rerun on F-01b):
  - incomplete lists 0.057;
  - false merges 0.046;
  - empty list with a low-scored true pair 0.024;
  - empty list with the true pair never blocked 0.021;
  - singleton FP 0.015.

  Oracle 0.941.

### EXP-N-01 (feasibility only) — Transliteration
- A token map was learned only from aligned **train-fold** true pairs (150k S1 sample; 4,635 equal-token-count pairs; 721 tokens with support ≥ 3 and share ≥ 0.6).
- It covers 47.2% of non-Latin tokens in 55,270 validation script-mismatch true pairs.
- It raises name_core Jaccard from 0.021 to 0.486 on those pairs.
- 24% of the true pairs F-01b still scores below threshold are script mismatches.
- Decision: **the proposed next experiment** (not implemented).

### Not done (by evidence)
- LightGBM (EXP-M-03): no evidence yet that the remaining errors are non-linear. No dependency added.
- Decision-layer tweaks, re-measured on F-01b: top-1 fallback +0.0008, exclusivity +0.0006, per-source thresholds +0.0014 in-sample. Not pursued.

### EXP-D-01 — Learned-matcher headroom probe (diagnostic, not adopted)
- Date: 2026-09-25 · Owner: Claude (analysis) · Script: `scripts/analysis/matcher_headroom_probe.py` · Output: `experiments/runs/EXP-B-01/analysis/matcher_headroom_probe.json`
- Hypothesis: The hand-weighted rule, not the features, limits precision/recall. A fitted linear model on the same features and candidates scores higher.
- Change: Logistic regression (16 features + cheap score/rank + 2 interactions) instead of fixed weights.
  - Validation S1s are split into 2 halves by a seeded hash; fit and tune on one half, evaluate on the other, then swap.
  - Candidates and blocking are identical.
- Expected effect: A positive delta.
- Actual result: Rule 0.7849 / 0.7853 (mean **0.7851**, threshold 0.67); LR 0.8153 / 0.8166 (mean **0.8160**, threshold 0.48). **+0.031** held-out.
- Interpretation: Most of the matcher loss comes from the weights. The strongest learned signals are address token-set similarity and name×number interactions.
  - Caveat: the probe fits on validation-fold pairs. A production model must be fitted on train-fold S1s (EXP-M-02).
- Decision: Supports prioritising EXP-M-02. Nothing adopted. Leaderboard: not submitted.

### EXP-B-01 — Submission-1 baseline
- Date: 2026-09-25 · Owner: Claude (implementation) / team (full run) · Config: `configs/baseline_v0.yaml` · git commit: not committed yet
- Hypothesis: Country-partitioned key blocking (Block A exact `name_norm`, Block B (number, word) and rare-word address keys, union → dedupe → cheap-score top-20), then a weighted similarity rule and a global threshold tuned on entity-level F0.5, gives a valid and reasonably precise Submission 1.
- Change: First implementation (no parent).
- Expected effect: Blocking recall is bounded by the missing name-token pass (pairs with empty/non-matching addresses). Precision should be high at the tuned threshold.

**A. Implementation verification (not a benchmark)**
- 60/60 unit and integration tests pass: normalization (incl. Indic), metrics (official 0.714 example, singleton rules), blocking, features, writer, strict validator (BOM, CRLF, duplicates, invalid/train IDs, matches ⊄ candidates, candidate ≠ scored set), and a synthetic end-to-end run with the official validator (`--check-ids`).
- A blocking-memory rewrite (chunked encoding, packed sort, lazy pool columns, per-chunk unique-row gathering) was verified bit-identical to the original implementation on the 20% sample: 4,297,886 pairs, same scores and ranks.

**B. Real-data SAMPLE results (optimistic: sample pools are much smaller than the full pools → easier blocking; NOT full-data metrics)**

| Sample | Val S1 | Blocking recall | Oracle F0.5 | Tuned t | F0.5 | Macro P | Macro R | All-empty F0.5 | Cand/S1 mean |
|---|---|---|---|---|---|---|---|---|---|
| 2% (`scripts/make_sample_data.py --fraction 0.02`) | 4,394 | 0.9027 | 0.9626 | 0.47 | 0.8898 | 0.9439 | 0.8370 | 0.0569 | 10.8 |
| 20% (same script, `--fraction 0.2`) | 44,120 | 0.8475 | 0.9377 | 0.64 | 0.8429 | 0.9672 | 0.7054 | 0.0564 | N/A |

- 2% sample, blocking recall at rank k: @1 0.265, @3 0.670, @5 0.832, @10 0.889, @20 0.903.
- 2% sample by country: India F0.5 0.859 (R 0.749), US 0.910 (R 0.895). There were 30 singleton false positives and 104 empty-on-positive entities out of 4,394.
- Blocking recall falls from 0.90 to 0.85 as the pool grows from 2% to 20%, because more blocks exceed the size caps. Expect the full data to be lower still.

**C. Resource measurements on the FULL data (bounded probes, no scoring/submission run)**
- Preprocess: test 190 s, peak process-tree RSS 2.29 GB · train 193 s, peak 2.24 GB (6 workers).
- India test partition (809,986 S1 × 4,717,565 pool): blocker built in 51 s; 13.4M candidates (16.5/S1). Peak RSS 1.85 GB (was 2.92 GB before the memory rewrite). With the feature phase at `max_raw_pairs_per_chunk: 4e6`, the peak is 2.01 GB.
- US train partition with ALL 1.32M S1 (upper bound; the `train` stage uses only the validation S1s): peak 1.83 GB.
- Feature and scoring throughput: 60–67k pairs/s.

**D. Full-data benchmark (team run, 2026-09-25):**
- Frozen validation split: 220,682 S1 (sha256 `ce1da2d3…`).
- Tuned threshold: 0.67.
- Validation F0.5 **0.7851**: macro P 0.9226, macro R 0.6553, blocking recall 0.7983, oracle F0.5 0.9098. All-empty scores 0.0558.
- By country: India 0.6918, US 0.8474.
- Config threshold 0.75 would have given 0.7412.
- Test outputs: 1,732,544 rows. 26,089,341 candidate ids (15.1 per S1); 4,628,270 matched ids; 192,763 empty match rows. Strict validation passed.
- **Public leaderboard: 0.751** (Submission 1, slot 1).
- Local-to-public gap: −0.034. Full analysis: `reports/EXP-B-01_SUBMISSION_1_REPORT.md`.

- Interpretation: The pipeline is valid end-to-end. On the samples, recall rather than precision is the weak side, especially India (no transliteration, no name-token pass).
- Decision: Pending the full run.
- Leaderboard: not submitted.

<!--
### EXP-X-NN — <title>
- Date:
- Owner:
- Hypothesis:
- Change (single primary change vs. parent experiment <ID>):
- Config: experiments/runs/<ID>/config.yaml  | git commit:
- Expected effect:
- Actual result: F0.5 / P / R / Blocking recall / Cand per S1 (mean, median, p99) / runtime / peak RAM
- Per-country breakdown: US / India (and, for test predictions, the France share of predicted matches)
- Interpretation:
- Error-analysis notes (top error buckets):
- Decision: adopt / reject / needs follow-up
- Leaderboard: submitted? (Y/N) — if Y: slot #, public score, justification
-->
