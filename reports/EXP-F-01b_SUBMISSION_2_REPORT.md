# Submission 2 (EXP-F-01b): Full Execution Report, Deep Error Analysis and Where the Score Is Still Lost

**Team:** LargeLanguageCoders · **Date:** 2026-09-25
**Public leaderboard:** **0.815** (Submission 1: 0.751)
**Local validation F0.5:** **0.8372** (Submission 1: 0.7851)
**Status:** analysis complete. Every "next step" is a **proposal awaiting team approval** (CLAUDE.md Rules 2, 4 and 9).

> **Where the numbers come from.**
> - The team's full run: `experiments/runs/EXP-F-01b/{metrics.json, model.json, fit_report.json, submission_summary.json}` and `data/processed/{train,test}/scored_EXP-F-01b_scoring_stats.json`.
> - Read-only analysis of that run: `experiments/runs/EXP-F-01b/analysis/{analysis.json, error_budget.json, nonlinear_probe*.json}` and `experiments/runs/val_halves_comparison.json`.
> - The public score was reported by the team.
> - Anything marked *estimate*, *upper bound* or *probe* is labelled as such. No raw records appear in this tracked file.

---

## 1. Executive summary

1. **The run is sound and reproducible.**
   - The full-scale fit (200k train-fold S1, 2.58M pairs) reproduced the dev-scale result exactly: validation 0.8372 vs 0.8371.
   - The two validation halves agree to within 0.0006 (0.8375 / 0.8369).
   - The threshold chosen on the train fold (0.60) equals the validation optimum.
   - Strict and official validators passed.
2. **Leaderboard 0.815 vs local 0.837 (−0.022, down from −0.034).**
   - Still explained mainly by the **country mix**: India is our weakest country (0.779) and makes up 47% of test.
   - Validation reweighted to the test mix (India+US only) gives **≈0.823**.
   - The remainder is consistent with France scoring **≈0.77** (*estimate*; was ≈0.68 at Submission 1).
3. **Where the remaining 0.163 of validation loss comes from.** Error-budget upper bounds, each bucket fixed alone:

   | Bucket | Upper bound | India |
   |---|---|---|
   | True pairs that never become candidates (blocking) | **+0.073** | +0.102 |
   | True candidates scored below threshold | **+0.058** | |
   | False merges | **+0.046** | |

4. **The single biggest measured lever is the scorer's *form*, not new features.**
   - A small non-linear model (MLP) trained on *exactly the same features, training S1s and threshold protocol* reaches **0.887–0.890** on validation, stable across 3 seeds.
   - That's **+0.050** over LR, and it is *more* precise:
     - precision **0.970 vs 0.932**;
     - false merges **16.6k vs 37.0k**;
     - singleton false positives **1,456 vs 3,223**;
     - India **0.848 vs 0.779**.
   - The linear model cannot express rules like "strong name match *but* different house number → reject".
5. **Other structural gaps (measured):**
   - India transliteration: worth up to +0.10 for India across blocking and scoring;
   - name-matching pairs that blocking never retrieves (+0.044);
   - singleton false positives (+0.015).
6. **My overall view (§9):**
   - We are no longer limited by data plumbing or thresholds. We are limited by (a) a scorer that is too simple for the interactions in this data, and (b) India-specific recall (scripts, long addresses).
   - Fixing (a) is the cheapest, largest and best-measured step. It needs a team decision on the model library (LightGBM, MIT, vs a from-scratch MLP).

---

## 2. What was executed (Submission 2 pipeline)

```mermaid
flowchart LR
    A[Raw TSVs] --> B[Ingest → Parquet]
    B --> C[Normalize<br/>multi-view, Indic-safe]
    C --> D[Blocking v1 per country<br/>exact name · sorted core name<br/>number+word (zero-safe) · rare word]
    D --> E[top-20 candidates / S1]
    E --> F[18 features<br/>+ num & legal-form conflicts]
    F --> G[Logistic regression<br/>trained on 200k train-fold S1]
    G --> H[threshold 0.60<br/>from inner train-fold split]
    H --> I[(2 TSVs, validated)]
```

| Step | Command (team) | Result |
|---|---|---|
| fit | `run_baseline fit --config configs/exp_f01b_conflicts_zeros.yaml` | 200,000 train-fold S1 (validation excluded) → 2,580,936 training pairs (477,336 positive). Newton-LR, 10 iterations, 10.5 s. Inner-tune (39,954 S1) threshold **0.60**, inner F0.5 0.8363 |
| train | `run_baseline train …` | validation 220,682 S1: **F0.5 0.8372**, 182 s, peak 1.86 GB |
| infer | `run_baseline infer …` | test candidates 27,460,419 (15.85/S1); 7,450 S1 with no candidates |
| submit | `run_baseline submit …` | 4,874,401 matched ids (2.81/S1); 150,156 empty rows; strict + official PASS |

**Learned weights** (standardised, largest first):
- `name_ratio` +2.87
- `addr_token_set` +2.55
- `name_concat_jw` −2.06
- `legal_form_conflict` −1.43
- `addr_any_empty` +1.04
- `name_core_jaccard×addr_num_jaccard` +0.92
- `addr_num_jaccard` +0.83
- `cheap_rank` −0.72
- `name_core_jaccard` +0.68
- `num_conflict` −0.48

Two things stand out in these weights:
- **Large opposite-signed weights on correlated name features** (+2.87 / −2.06) are a classic sign of a linear model "fighting" to express a non-linear shape.
- The legal-form conflict became the fourth-strongest signal.

---

## 3. Scores

### 3.1 Submission history

| | Submission 1 (EXP-B-01) | **Submission 2 (EXP-F-01b)** | Δ |
|---|---|---|---|
| **Public LB** | 0.751 | **0.815** | **+0.064** |
| Validation F0.5 | 0.7851 | **0.8372** | +0.052 |
| Macro precision | 0.9226 | **0.9319** | +0.009 |
| Macro recall | 0.6553 | **0.7418** | +0.087 |
| Blocking recall (top-20) | 0.7983 | **0.8597** | +0.061 |
| Oracle F0.5 (perfect decisions on our candidates) | 0.9098 | **0.9413** | +0.032 |
| True matches found | 501,169 | **563,994** | +62,825 |
| False merges | 39,489 | **36,952** | −2,537 |
| Singleton false positives | 2,724 | 3,223 | **+499 ⚠** |
| Non-singleton S1 with empty prediction | 16,006 | 9,910 | −6,096 |
| Predicted matches / S1 | 2.45 | 2.72 | |
| Candidates / S1 | 15.4 | 16.1 | |
| LB − validation | −0.034 | −0.022 | |

### 3.2 By country (validation)

| Country | F0.5 | Precision | Recall | Blocking recall | Oracle | False merges | Singleton FP | Empty-on-positive |
|---|---|---|---|---|---|---|---|---|
| India | **0.779** (was 0.692) | 0.909 | 0.676 | 0.831 | 0.921 | 19,055 | 1,537 | 6,571 |
| US | **0.876** (was 0.847) | 0.946 (was 0.956 ⚠) | 0.786 | 0.879 | 0.955 | 17,897 | 1,686 | 3,339 |

- India's **matcher gap** (oracle − F0.5) is 0.142, versus 0.079 for the US. India's **blocking gap** is also larger (0.921 vs 0.955 ceiling).
- India loses at both stages.

### 3.3 By number of true matches (validation)

| True matches | S1 | F0.5 | Precision | Recall | Note |
|---|---|---|---|---|---|
| 0 (singletons) | 12,325 | 0.739 | — | — | 26% of singletons get a wrong match |
| 1 | 11,916 | **0.686** | 0.859 | 0.750 | worst bucket: one miss or one false merge zeroes the entity |
| 2–3 | 90,605 | 0.830 | 0.935 | 0.748 | |
| 4–5 | 80,607 | 0.868 | 0.961 | 0.738 | |
| 6+ | 25,229 | 0.882 | 0.975 | 0.727 | |

### 3.4 Threshold table (validation)

| t | F0.5 | Precision | Recall | Singleton FP | Predicted/S1 | False merges |
|---|---|---|---|---|---|---|
| 0.50 | 0.8338 | 0.9079 | 0.7663 | 4,072 | 2.89 | 54,090 |
| 0.55 | 0.8361 | 0.9203 | 0.7550 | 3,650 | 2.81 | 45,152 |
| **0.60** | **0.8372** | **0.9319** | 0.7418 | 3,223 | 2.72 | 36,952 |
| 0.65 | 0.8358 | 0.9426 | 0.7250 | 2,778 | 2.63 | 29,915 |
| 0.70 | 0.8310 | 0.9520 | 0.7059 | 2,293 | 2.54 | 24,202 |

- The curve is flat around 0.60, so the threshold is not a lever: ±0.05 changes F0.5 by ≤ 0.0015.
- Decision-layer what-ifs re-measured on this run:
  - top-1 fallback +0.0008;
  - exclusivity +0.0006;
  - separate S2/S3 thresholds +0.0014 (in-sample).

  None is worth a slot.

---

## 4. Why the leaderboard (0.815) is below validation (0.837)

1. **Country mix (measured inputs, arithmetic).**
   - Test is India 46.8% / US 38.3% / France 15.0%.
   - India+US reweighted: (0.468·0.7791 + 0.383·0.8760) / 0.851 = **0.823**.
2. **France (estimate).**
   - (0.815 − 0.7001) / 0.150 ≈ **0.77**, up from ≈0.68 at Submission 1.
   - This assumes the public subset is representative and India/US behave on test as on validation.
3. **Test behaves like validation for India and US** (measured):

| | Candidates/S1 | Raw blocking pairs/S1 | Predicted/S1 | Empty prediction | Median top-1 score |
|---|---|---|---|---|---|
| India val / test | 16.97 / 17.03 | 41.5 / 41.5 | 2.55 / 2.67 | 11.3% / 11.1% | 0.998 / 0.998 |
| US val / test | 15.57 / 15.24 | 28.6 / 28.0 | 2.84 / 2.96 | 6.8% / 6.3% | 0.999 / 0.999 |
| France test | 13.72 | 22.1 | 2.90 | 7.0% | 1.000 |

4. **Test density.**
   - Test has 24% more S2/S3 records per S1 than train (India 5.82 vs 4.68), yet raw blocking pairs per S1 are *identical*.
   - So the extra records mostly don't resemble our S1 names or addresses, and are probably distractors or otherwise unreachable.
   - This suggests density is *not* a major part of the gap, but it's an inference; test labels are unavailable.
5. **Unknowns.** The public/private split size is *not specified in the provided materials*.

**Conclusion:** the gap is structural (India share + France), not overfitting. Any gain on India moves the leaderboard almost 1:1 with its 47% weight.

---

## 5. Loss decomposition (validation; total loss 0.1628)

| Entity outcome | S1 | Share | F0.5 points lost |
|---|---|---|---|
| Perfect (F = 1) | 85,482 | 38.7% | 0 |
| **Precise but incomplete** (missed matches) | 90,980 | 41.2% | **0.0571** |
| **At least one false merge** | 31,087 | 14.1% | **0.0462** |
| Empty list; true pair was a candidate but scored low | 5,254 | 2.4% | 0.0238 |
| Empty list; no true pair among candidates | 4,656 | 2.1% | 0.0211 |
| Singleton wrongly given a match | 3,223 | 1.5% | 0.0146 |

In the "incomplete" group, **52% of the missing matches never became candidates** (blocking) and 48% were scored too low (scorer).

---

## 6. Error budget: what each fix is worth (upper bounds, measured)

Each row gives the validation F0.5 gained if **only that bucket** were fixed perfectly: missed pairs predicted, or false merges removed. The gains overlap, so they don't add up exactly. Their purpose is **ranking**, not forecasting.

| Bucket | Pairs | Gain (all) | India | US |
|---|---|---|---|---|
| **Never a candidate (all blocking misses)** | 107,130 | **+0.0734** | **+0.102** | +0.054 |
| ↳ shares a name token (reachable by a name key) | 60,086 | +0.0442 | +0.051 | +0.040 |
| ↳ non-Latin pool name (script mismatch) | 18,344 | +0.0204 | +0.051 | 0 |
| ↳ address empty on one side | 15,660 | +0.0103 | +0.011 | +0.010 |
| ↳ other | 13,040 | +0.0105 | +0.012 | +0.009 |
| **Candidate scored below threshold (all)** | 92,432 | **+0.0583** | **+0.085** | +0.041 |
| ↳ other (mixed evidence) | 36,986 | +0.0238 | +0.022 | +0.025 |
| ↳ script mismatch | 21,980 | +0.0194 | +0.049 | 0 |
| ↳ name weak, address strong (trade names) | 20,164 | +0.0121 | +0.014 | +0.011 |
| ↳ address empty | 11,928 | +0.0068 | +0.008 | +0.006 |
| **False merges (all)** | 36,952 | **+0.0457** | +0.057 | +0.038 |
| ↳ other (similar name and address) | 22,628 | +0.0210 | +0.032 | +0.014 |
| ↳ on singletons | 3,975 | +0.0146 | +0.017 | +0.013 |
| ↳ same core name (different business) | 6,651 | +0.0062 | +0.007 | +0.006 |
| ↳ with number conflict (model still accepted) | 3,644 | +0.0037 | +0.001 | +0.006 |
| ↳ with legal-form conflict | 54 | +0.0001 | | |

What this shows:
- **The conflict features did their job:** only 54 false merges with a legal-form conflict remain.
- **Transliteration (script mismatch) appears in both blocking and scoring.** For India alone it's worth up to about +0.10 (overlapping upper bounds).
- **56% of blocking misses share a name token.** The name-token pass was rejected with the *old rule* because it adds hard negatives. With a stronger scorer, that trade-off must be re-measured.
- **The "other" buckets are the largest single scorer buckets.** Those are exactly the mixed-evidence cases a non-linear scorer handles better (§7).

---

## 7. The biggest finding: the scorer is too simple (EXP-D-02 probe, measured)

- **Setup (identical protocol to production `fit`):**
  - same 22 inputs;
  - same 200k train-fold S1s and pairs;
  - same inner train-fold threshold tuning;
  - evaluated once on the frozen validation split.
- **Model:** a small MLP (22 → 64 → 32 → 1, Adam, trained in 3 s on the GPU). Diagnostic only; not in the pipeline.

| Scorer | Seed / epochs | t (train-fold) | F0.5 | Precision | Recall | False merges | Singleton FP | India | US |
|---|---|---|---|---|---|---|---|---|---|
| LR (production) | — | 0.60 | 0.8372 | 0.9319 | 0.7418 | 36,952 | 3,223 | 0.779 | 0.876 |
| MLP | 42 / 4 | 0.71 | 0.8870 | 0.9704 | 0.7876 | 16,640 | 1,456 | 0.848 | 0.913 |
| MLP | 7 / 4 | 0.73 | 0.8885 | 0.9681 | 0.7913 | 18,241 | 1,507 | 0.854 | 0.912 |
| MLP | 123 / 8 | 0.80 | **0.8903** | **0.9729** | 0.7891 | **15,166** | **1,292** | **0.854** | **0.915** |

- **The gain is stable** across seeds (0.887–0.890).
- **It isn't overfitting:** the train-fold inner-tune F0.5 predicted validation within 0.001 each time.
- **It improves precision and recall together**, which is the ideal shape for F0.5.
- **Singleton false positives fall below Submission 1's level** (1,292–1,507 vs 2,724).
- India gains most (+0.07).

**Why:** a linear model adds up evidence, but this data needs *conditional* rules:
- "same name only counts if the house number agrees";
- "a different name is fine if the address is identical and the name is non-Latin";
- "rank-1 candidate with a moderate score is likely; rank-15 with the same score is not".

The LR weights in §2 show it straining to approximate these. Trees and neural nets represent them directly.

**Caveats (honest):**
- The probe is not a production model.
- Test behaviour, especially France, must be checked: predictions per S1 and the empty rate per country must look sane.
- The final model library must satisfy the license rule. LightGBM is MIT. PyTorch is a BSD-3-licensed *library*, and a model we train from scratch is not a third-party pretrained model. Whether that satisfies "MIT/Apache 2.0 License model" is *not specified in the provided materials*, so it's a team decision.

---

## 8. What is still weak, component by component

| Component | State | Measured weakness | Evidence |
|---|---|---|---|
| Normalization | solid, Indic-safe | no transliteration: non-Latin names can't match Latin ones | script-mismatch buckets: India up to +0.10 |
| Blocking | 0.860 recall@20, ceiling 0.884 even uncapped | 11.6% of true pairs share no key. India: 53% of those are keys dropped by caps / rarest-k; US: 69% share address words but no number | §6, analysis.json |
| Features | 18, conflicts effective | no transliterated-name similarity; no token-rarity (IDF) weighting; no "other candidates of this S1" context | error-budget "other" buckets |
| **Scorer** | linear LR | **cannot express conditional interactions: about 0.05 left on the table** | §7 probe |
| Decision | global t = 0.60 | flat curve; tweaks ≤ +0.0015 | §3.4 |
| Singletons | 26% get a wrong match | +0.015 available | §3.3, §6 |
| n = 1 entities | F0.5 0.686 | all-or-nothing entities | §3.3 |
| France | implied ≈0.77 (*estimate*) | no labels; only generic handling | §4 |
| Engineering | stable, ~2 GB peak, reproducible | — | full = dev results |

---

## 9. My overall view: why the score is still where it is

1. **We fixed the "easy" losses and the remaining ones are structural.**
   - Submission 1 → 2 removed bad blocking keys, hand-tuned weights and missing conflict signals: +0.064 LB.
   - The threshold and decision layer are exhausted (flat curve, ≤ +0.0015).
   - Further gains must come from **better separation** (the scorer) or **more reachable true pairs** (blocking, transliteration).
2. **The scorer is the bottleneck, and it's a *form* problem, not a *feature* problem.**
   - The same 22 numbers, used non-linearly, give +0.05 with higher precision.
   - This is the largest, cheapest and best-evidenced step available. It should come **before** adding more features, because every future feature will also be used better by a non-linear model.
3. **India is where the leaderboard is won or lost.**
   - It's 47% of test with the lowest score (0.779).
   - It loses on both stages: blocking (dense, repetitive addresses, where caps drop the keys) and scoring (non-Latin names).
   - Transliteration learned from train pairs is the targeted fix: already shown feasible (EXP-N-01, 47% token coverage from a small sample).
4. **Recall is still the weaker side of F0.5, but it can't be bought with a lower threshold.**
   - That just adds false merges (§3.4).
   - It has to come from blocking (the name-token pass, re-measured with a stronger scorer) and from the scorer accepting the mixed-evidence true pairs, which the MLP probe does (recall 0.742 → 0.789).
5. **Reaching ~0.85+ on the leaderboard is plausible *on evidence*, but not guaranteed.**
   - If a non-linear scorer holds its +0.05 on test with the same ~−0.02 local→LB offset, the leaderboard would be around 0.86–0.87.
   - That's an *estimate*, not a promise: France is unmeasured.

---

## 10. Recommended next steps (proposals; each needs team approval)

| Order | Experiment | Change (one) | Evidence | Measure / keep rule |
|---|---|---|---|---|
| **1** | **EXP-M-03 non-linear matcher** | Replace LR with a gradient-boosted tree (LightGBM, MIT; new dependency → DEC) **or** the MLP (license question for the team). Same features, same `fit`/threshold protocol | §7: +0.050 measured, precision ↑, stable over 3 seeds | Keep if validation F0.5 ≥ 0.85 on both halves with precision ≥ 0.9319; check test predictions/S1 and empty rate per country, France included |
| 2 | EXP-N-01 transliteration | Train-fold-learned token map → transliterated-name feature + blocking key | India up to +0.10 upper bound; feasibility 47% token coverage | India F0.5 ↑ with India precision not ↓ |
| 3 | EXP-B-03 name-token pass, re-measured | Enable `name_token` with the non-linear scorer | 56% of blocking misses share a name token (+0.044 bound) | F0.5 ↑ and precision not ↓ (it was rejected only with the old rule) |
| 4 | EXP-F-02 context features | Per-S1 context: score gap to best candidate, candidate count, name frequency (IDF) | "other" buckets are the largest remaining scorer buckets | measured per feature |
| — | Not now | threshold / fallback / exclusivity / per-source / per-country rules | ≤ +0.0015 | — |

**Leaderboard plan (Rule 9):** spend the next slot only on EXP-M-03, and only if it clears the keep rule above. Its purpose is to confirm the non-linear gain transfers to test and to France.

---

## 11. Rules compliance

| Rule | Status |
|---|---|
| Raw data read-only | ✔ |
| No external data or lookups | ✔; transliteration probe uses only train pairs |
| Output format / validators | ✔ strict + official PASS; UTF-8 no BOM, LF, 1 row per S1 |
| matches ⊆ candidates = scored set | ✔ strict validator `--scored-dir` |
| No train/validation leakage | ✔ fit on train-fold S1s only (code raises on overlap); threshold from inner train-fold split |
| Open-set countries | ✔ no country feature or rule |
| Model license | ✔ production LR is our own numpy code. PyTorch is used only in diagnostic probes; the production choice (LightGBM vs MLP) is open (§7) |
| No fabricated metrics / no git push | ✔ |

## 12. Reproducing this report

```bash
python scripts/analysis/analyze_exp_b01.py    --config configs/exp_f01b_conflicts_zeros.yaml   # decomposition, blocking fate, test by country
python scripts/analysis/error_budget.py       --config configs/exp_f01b_conflicts_zeros.yaml   # §6
python scripts/analysis/nonlinear_probe.py    --config configs/exp_f01b_conflicts_zeros.yaml   # §7 (seed 42); add --seed 7 / --seed 123 --epochs 8
python scripts/analysis/compare_val_halves.py --runs EXP-B-01-recheck:scored EXP-F-01b:scored_EXP-F-01b
```
