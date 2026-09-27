# Final 5-hour plan — honest assessment and strategy

2026-09-27 · written for the LargeLanguageCoders team (Ayush / Param) · **plan only: nothing here has been executed**

Evidence labels:

| Label | Meaning |
|---|---|
| **[M]** | Measured |
| **[UB]** | Oracle upper bound |
| **[E]** | Estimate |
| **[I]** | Inference |
| **[U]** | Unknown |

## 1. Bottom line

1. **0.98 public cannot be reached in 5 hours with this system, or with any change that can be built and verified in that time.** This is arithmetic, not pessimism (§2).
2. **No untouched lever will raise the score "for sure".** The one cheap, principled lever we have not tried yet is the **decision rule for the macro-F0.5 metric** (EXP-D-04, §4). Its realistic gain is small: **+0.0005 to +0.002 local [E]**, which means about **0.942–0.944 public [E]**.
3. **The one large lever we have never touched is learned text representations**: a pretrained multilingual encoder under the MIT/Apache ≤ 8B rule, used for retrieval and matching. That is most likely what top-100 teams do [I]. It needs **≈ 6–7 hours of machine time before the first measurement** (§5.1), so it cannot fit in this window.
4. **The final submission zip is mandatory for every team and does not exist yet.** `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + a filled-in `Documentation_template.md` (official README, "Final Submission Package"). Missing it is a larger risk than any score change. It is Step 2 of the plan.

**Recommendation:**
- run EXP-D-04 (≈ 1 h 45 min, ready to upload at about T+2 h if it passes its gate);
- build the final package in parallel;
- keep **S-03 (0.942)** as the protected fallback and default final choice.

## 2. Why 0.98 is out of reach (numbers)

**Local → public gap is stable [M]:**

| System | Local | Public | Gap |
|---|---|---|---|
| N-02 | 0.9402 | 0.928 | −0.012 |
| B-06 | 0.9471 | 0.935 (if that file was B-06) | −0.012 |
| **S-03** | **0.9536** | **0.942** | **−0.012** |

So **public 0.98 needs about local 0.992**, i.e. the local loss must shrink from **0.046 to about 0.008**.

**Where S-03 loses its 0.046 locally** [M, UB per bucket, E half B, `experiments/runs/EXP-S-03/error_budget.json`]:

| Bucket | True / false pairs | Oracle gain if fixed alone [UB] |
|---|---|---|
| True pair never retrieved (not in top-20) | 149,151 (4.3%) | **0.0162** |
| Retrieved, stage-1 score 0.1–0.5 | 131,097 | 0.0153 |
| Retrieved, score 0.5–0.65 | 57,475 | 0.0067 |
| Retrieved, score < 0.1 | 34,274 | 0.0044 |
| True pairs removed by the S-03 veto | 15,031 | 0.0022 |
| Lost the D-01 best-claimant contest | 3,145 | 0.0005 |
| False merges on S1 with true matches | 20,738 | 0.0044 |
| False merges on singletons | 1,788 | 0.0017 |

**The ceiling argument:**
- Even a **perfect** matcher and decision rule on our current candidate lists would be capped at about **1 − 0.0162 ≈ 0.984 local [UB]**, which is about **0.972 public [E]**.
- So 0.98 requires *both* a much better retrieval stage and a near-perfect matcher.

**The track record:**
- Across the last eight measured steps, gains were +0.0007 to +0.0089 each (mostly about +0.003).
- The 2026-09-27 audit classified even public 0.97 as **"currently unsupported"** (`experiments/runs/FINAL_RESEARCH_AUDIT/target_097_assessment.json`).

**Test-side check [M, unlabeled]:**
- S-03 predicts 3.11 matches per S1 on test: France 3.10, India 3.10, US 3.12.
- That is exactly the train-consistent rate (3.46 true per S1 × recall 0.889 / precision 0.992).
- So no country is "broken" in a way a quick fix could repair.

## 3. What has and has not been touched

| Component | Status | Measured return |
|---|---|---|
| Blocking keys (B-01 → B-11, 11 experiments) | Heavily worked | Diminishing: B-08 +0.0018, B-09 +0.0032, B-10 / B-11 failed kill gates |
| Hand-crafted pair features + LightGBM stage 1 | Worked | M-03 +0.065, F-01 +0.02, N-02 +0.009 |
| Record-competition decisions (D-01) | Done | +0.004 |
| Stage-2 context model | Done | Additions fail on test (S-01, public 0.920); veto works (S-02 / S-03, +0.003) |
| **Decision rule matched to the per-S1 macro-F0.5 metric** | **Never touched** (one global threshold) | EXP-D-04, this plan |
| **Stage-1 training volume** (200k of 2.2M train S1 = 9%) | **Never run** (M-04 planned, not executed) | +0.002 measured for 4× on B-04 [M]; too slow with the veto refit (§5.3) |
| **Learned text representations** (pretrained multilingual encoder, MIT/Apache) | **Never touched** (our own rule so far: no pretrained models) | [U]; the largest structural gap vs strong ER systems [I]; ≈ 6–7 h (§5.1) |
| France / unseen-country adaptation | Not measurable (no labels) | [U] |

## 4. The executable experiment: EXP-D-04 — expected-F0.5 per-S1 pruning (removal-only)

### 4.1 Why this is the right lever for the metric

The score is a **per-S1 macro F0.5**. For one S1, the value of keeping a borderline pair depends on how many other matches that S1 already has. Worked example with a calibrated probability p for the borderline pair:

| S1 situation | Keep the pair when p ≥ |
|---|---|
| The pair is the S1's only candidate (singleton risk) | **0.50** |
| 1 other confident match | **≈ 0.73** |
| 3 other confident matches | **≈ 0.77** |

- **Why the break-even rises:** with F0.5, one false merge costs more than one missed match, and the cost grows as the S1's true set grows.
- **What S-03 does instead:** one global cut (stage 2 ≥ 0.69), which is a compromise across all S1 shapes.
- **What D-04 does:** replaces the compromise with the exact per-S1 optimum. Parameter-free, so nothing is tuned on held-out data.

### 4.2 Exact change (one)

After S-03's decision (B-09 stage 1 ≥ 0.65 + D-01, AND stage-2 veto ≥ 0.69):
1. For each S1, sort its accepted pairs by stage-2 probability.
2. Keep the prefix of size k (k = 0 … |accepted|) that maximises the **exact expected F0.5** under independence.
   - Poisson-binomial over **all** of the S1's eligible candidates: accepted pairs, and rejected pairs with stage 1 ≥ 0.02.
   - k = 0 scores P(no true match), which is the singleton rule.
   - Candidates that lost D-01, or have stage 1 < 0.02, count as p = 0.
3. **Removal-only:** final ⊆ S-03 by construction, so no addition is possible. Additions are exactly what failed on test in S-01.

Other properties:
- No retraining. The S-03 stage-1 and stage-2 models and features are reused read-only.
- No new threshold and no sweep.
- Vectorised, chunked numpy; peak memory < 3 GB [E].

### 4.3 Pre-declared gates (written before any D-04 result)

This is a no-retrain decision change. Control = S-03.

| Gate | Rule |
|---|---|
| **G1** | F0.5 ≥ S-03 + 0.001 on validation, GATE-B **and** E half B |
| **G2** | both validation halves Δ > 0 |
| **G3** | precision ≥ S-03's on all three (removal-only should raise it) |
| **G4** | India and US each Δ ≥ −0.0005 |
| **G5** | density-stable: advantage on T1W and T4W ≥ advantage on T1 − 0.0005 (shadow v2) |
| **G6** | test removals per S1 within 0.5×–1.5× of E half B's; France within ±50% of overall |
| **G7** | leakage PASS: the stage-2 model was trained on E half A minus validation; evaluation only on validation / E half B / GATE-B / shadow |
| **G8** | two submits byte-identical; strict validator PASS; official validator `--check-ids` PASS; matches ⊆ candidates; candidates byte-identical to S-03 |

Kill rule: if G1 fails on any of the three, **stop**. No slot is used, and S-03 stays.

**Diagnostic, not a gate:** a reliability table of stage-2 probabilities on E half B, per decile, to explain the result.

### 4.4 Expected result (no guarantee)

- **Local:** +0.0005 to +0.002 [E]. Public ≈ 0.942–0.944 [E].
- **Density:** removal-only decisions were density-stable in S-02 / S-03 [M], so the public transfer risk is low [I].
- **Downside:** if stage-2 probabilities are over-confident on test, D-04 removes *fewer* pairs than optimal. That is the safe direction.

### 4.5 Runtime (from measured component timings)

| Step | Time |
|---|---|
| Implement decision function + tests (unit: prefix optimality vs brute force, removal-only, k = 0 rule, determinism) | 35 min |
| Evaluate validation / halves / GATE-B / E half B, reusing S-03 features and model | 25 min |
| Shadow tiers T1 / T1W / T4W + density | 20 min |
| Test output ×2 (determinism) + strict + official validators | 25 min |
| **Total** | **≈ 1 h 45 min → an upload-ready file at about T+2 h**, only if all gates pass |

Output goes to `experiments/runs/EXP-D-04/output/` only. Root `output/` (S-03) is copied over only after your KEEP decision; the backup is made first.

## 5. Options considered and not recommended in this window

### 5.1 Pretrained multilingual text encoder (the big untouched lever)

For example `multilingual-e5-small` (MIT, 118M parameters) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0).

- **Why it matters [I]:**
  - dense retrieval attacks the 0.016 never-retrieved bucket;
  - learned similarity attacks the 0.019 low-score buckets;
  - cross-script names come for free.
- **Minimum path:** install `transformers` / `sentence-transformers` and download the model (15 min) → encode ≈ 24M train + test strings on the RTX 3060 6 GB (≈ 1–2 h [E]) → GPU dot-product retrieval per country (≈ 20–40 min [E]) → new candidates and features, stage-1 retrain (≈ 2 h 20 min [M]) → veto refit (≈ 75 min [M]).
- **Total ≈ 6–7 h before the first local number.** It also needs your explicit decision to lift our own "no pretrained models" experiment rule; the official rules allow MIT/Apache ≤ 8B.
- **Not feasible in 5 h.** It is the right direction for a future iteration.

### 5.2 B-10 blocker + stage-1 retrain + veto refit

- About 3 h 35 min [M].
- Expected +0.001 to +0.002 [E]. B-10 failed its own blocking gate (+0.0028).
- No time buffer, and memory-contended with packaging.

### 5.3 Stage-1 training on 600k S1 (M-04)

- Fit ≈ 3× longer, plus train / infer and the veto refit: ≈ 4 h+.
- Measured +0.002 at B-04 time.
- Does not fit.

### 5.4 Stage-2 additions / lower stage-1 floor

- Every addition floor has a negative density slope [M, POST-S01 §9b]; S-01 lost 0.015 public.
- Rejected on evidence.

### 5.5 Threshold or top-K sweeps, or probing the public leaderboard with variants

- Top-K 30: +0.0018 [M].
- Tuning on the public LB overfits it; the **private** LB decides the ranking.

## 6. Timeline (T = your approval)

| Time | Work | Who |
|---|---|---|
| T+0:00 | Preflight (env, hashes: S-03 `46cb8fb5…` / `e6fee476…`), back up root `output/` → `submissions/EXP-S-03_output/`, write D-04 `notes.md` with the gates in §4.3 | Claude |
| T+0:10 – 0:45 | D-04 implementation + tests; full suite | Claude |
| T+0:45 – 1:30 | D-04 evaluation and shadow (background) **∥ build the final package**: `code/business_entity_resolution/{src,configs,README.md,requirements.txt}` from the repo, pinned versions (numpy 2.2.4, pandas 2.3.3, pyarrow 21.0.0, rapidfuzz 3.14.6, lightgbm 4.7.0, PyYAML, psutil) | Claude |
| T+1:30 – 2:00 | D-04 test output ×2 + validators; **gate report → your KEEP / REJECT decision** | Claude → you |
| T+2:00 | **Upload D-04 only if KEEP** (slot 1 of 5) | You |
| T+2:00 – 3:00 | Fill in `Documentation_template.md` (methodology, blocking, features, models, licenses: LightGBM MIT; no external data); assemble `LargeLanguageCoders_submission.zip` with the final `output/`; validate the zip layout | Claude, your review |
| T+3:00 – 5:00 | **Buffer.** No new experiment. The final choice is S-03 or D-04: the better *local* result, provided public agrees. | You |

**Slot policy (5 left):**
- use at most **1–2**: D-04 if KEEP, and a re-upload of the final choice if needed;
- never spend a slot on an unmeasured idea;
- whether the portal ranks your **last** upload or a **selected** one is *not specified in the provided materials*. If it is the last upload, make sure the final chosen file is uploaded last.

## 7. Risks

| Risk | Mitigation |
|---|---|
| D-04 gain below +0.001 | Kill gate; S-03 stays, and no slot is spent |
| Package not reproducible or incomplete | Built from the exact code and configs that produced the chosen output; README lists the exact command chain; zip layout checked before upload |
| RAM (16 GB, often 2–3 GB free) | D-04 is streamed (< 3 GB); packaging is file copying only; nothing else heavy runs |
| Overwriting the best file | Root `output/` untouched until your KEEP; backup first; hashes printed |

## 8. Decisions needed from you

1. **Approve EXP-D-04** as specified in §4 (single change, gates in §4.3). Yes / no.
2. **Approve building the final submission package now** (Step 2). Strongly recommended regardless of D-04.
3. **Confirm:** no pretrained encoder in this window (§5.1). It is recorded as the recommended next direction.

No execution, no upload and no change to root `output/` happens before your approval.
