# Submission 3 (EXP-M-03, LightGBM): Execution Report, Deep Analysis and What Still Blocks the Score

**Team:** LargeLanguageCoders · **Date:** 2026-09-25
**Public leaderboard:** **0.889** (Submission 2: 0.815 · Submission 1: 0.751)
**Local validation F0.5:** **0.9037** (Submission 2: 0.8372)
**Status:** analysis complete. Every "next step" is a **proposal awaiting team approval** (CLAUDE.md Rules 2, 4 and 9).

> **Where the numbers come from.**
> - The team's full run: `experiments/runs/EXP-M-03/{metrics.json, model.json, fit_report.json, submission_summary.json, val_halves_full.json, test_country_profile.json}`.
> - Read-only analysis: `experiments/runs/EXP-M-03/analysis/{analysis.json, error_budget.json, test_conflicts.json, france_pattern_probe.json}` and the capacity probe `experiments/runs/EXP-D-03-dev50k-r3000/`.
> - The public score was reported by the team.
> - *Estimates* and *upper bounds* are labelled. No raw records appear here.

---

## 1. Executive summary

1. **Submission 3 worked as the evidence predicted.**
   - Replacing logistic regression with LightGBM on identical blocking and features raised validation F0.5 from 0.837 to **0.904**, and the leaderboard from 0.815 to **0.889** (+0.074).
   - The full 200k-S1 run reproduced the dev result. The two validation halves agree (0.9037 / 0.9038), and the threshold chosen on train-fold data (0.67) sits on the validation optimum's plateau.
   - Validators passed.
2. **The local→leaderboard gap shrank to −0.015.**
   - India+US reweighted to the test mix gives **0.896**.
   - The rest is consistent with **France ≈ 0.845** (*estimate*; was ≈ 0.77).
   - **France is now the weakest country by this estimate**, below India's 0.872.
3. **What still costs score (validation, total loss 0.096):**

   | Source | F0.5 points | Share |
   |---|---|---|
   | **Blocking**: true pairs never reach the candidate list | ≈ **0.055** | ~57% |
   | Scoring: missed matches + false merges | ≈ 0.041 | ~43% |

   Precision is 0.979; **recall (0.810) is now the weak side**, and it's capped by blocking. Blocking recall is 0.860 at top-20 and only 0.884 even without any cap.
4. **A new, test-only problem: conflicting claims in France (measured, no labels needed).**
   - In training, every S2/S3 record belongs to at most one S1 (verified earlier).
   - On test, **≥ 6.2% of France's predictions are certainly wrong**: the same record is claimed by several S1s. That's up from 2.3% with LR.
   - India is at 1.2% and US at 0.13%.
   - The validation split can't show this because only 10% of train S1s are scored there. On test, all S1s compete.
5. **What is *not* the problem (measured, rejected):**
   - model capacity (3× the rounds: +0.0006);
   - thresholds and decision tweaks (≤ +0.0005);
   - the France "city-in-name / same number, other street" hypothesis (not more frequent in France, not less precise).
6. **Where to work next, in evidence order** (§9):
   - (1) Blocking recall: name-token pass, top-K 20→30, India key caps.
   - (2) Conflict resolution under full competition (France/India), validated by scoring *all* train S1s.
   - (3) Transliteration for India.

---

## 2. What was executed

| Step | Command | Result |
|---|---|---|
| fit | `run_baseline fit --config configs/exp_m03_lgbm.yaml` | Re-used the EXP-F-01b train-fold candidates (200,000 S1; validation overlap = 0, checked in code). 2,580,936 training pairs (477,336 positive); 643,798 inner-tune pairs for early stopping. **1,000 rounds (early stopping did not trigger)**, 71 s. Threshold **0.67** (inner train-fold F0.5 0.9036) |
| evaluate | `run_baseline evaluate …` | Frozen validation (220,682 S1) on the EXP-F-01b candidates: **F0.5 0.9037**; halves 0.9037 / 0.9038 |
| infer | `run_baseline infer …` | 27,460,419 candidate ids (15.85/S1), same as Submission 2 (same blocking) |
| submit | `run_baseline submit …` | 4,974,452 matched ids (2.87/S1); 145,285 empty rows; strict + official validators PASS |
| France guard | `test_country_profile.py` | predicted/S1 LightGBM ÷ LR: France **1.063**, India 1.030, US 0.994 (within the agreed tolerance) |

Model: LightGBM 4.7.0 CPU, 22 inputs.
- Params: lr 0.05, 31 leaves, max depth 8, min 100 per leaf, feature fraction 0.9, bagging 0.8, λ2 1.0.
- Deterministic, seed 42.

Gain importance (usage, not causation):
- `name_ratio×addr_token_set` 0.68
- `name_core_jaccard×addr_num_jaccard` 0.05
- `addr_token_set` 0.045
- `addr_num_jaccard` 0.035
- `cheap_score` 0.035
- `addr_jaccard` 0.028

---

## 3. Scores

### 3.1 Submission history

| | Sub 1 (rule) | Sub 2 (LR) | **Sub 3 (LightGBM)** | Δ vs Sub 2 |
|---|---|---|---|---|
| **Public LB** | 0.751 | 0.815 | **0.889** | **+0.074** |
| Validation F0.5 | 0.7851 | 0.8372 | **0.9037** | +0.067 |
| Macro precision | 0.9226 | 0.9319 | **0.9790** | +0.047 |
| Macro recall | 0.6553 | 0.7418 | **0.8100** | +0.068 |
| Blocking recall (top-20) | 0.7983 | 0.8597 | 0.8597 | 0 (same blocking) |
| Oracle F0.5 (blocking ceiling) | 0.9098 | 0.9413 | 0.9413 | 0 |
| **Gap to ceiling (scorer loss)** | 0.125 | 0.104 | **0.038** | −0.066 |
| True matches found | 501,169 | 563,994 | **617,025** | +53,031 |
| False merges | 39,489 | 36,952 | **12,428** | −66% |
| Singleton false positives | 2,724 | 3,223 | **926** | −71% |
| Non-singleton S1 with empty prediction | 16,006 | 9,910 | **6,970** | −30% |
| Predicted / S1 | 2.45 | 2.72 | 2.85 | |
| LB − validation | −0.034 | −0.022 | **−0.015** | |

### 3.2 By country (validation)

| Country | F0.5 | Precision | Recall | Blocking recall | Oracle | False merges | Singleton FP | Empty-on-positive |
|---|---|---|---|---|---|---|---|---|
| India | **0.872** (was 0.779) | 0.970 | 0.769 | 0.831 | 0.921 | 7,035 | 484 | 4,247 |
| US | **0.925** (was 0.876) | 0.985 | 0.838 | 0.879 | 0.955 | 5,393 | 442 | 2,723 |

India's remaining gap to its ceiling is 0.049, versus 0.030 for the US. **India's ceiling itself is lower** (0.921 vs 0.955): India loses more in blocking.

### 3.3 By number of true matches

| True matches | S1 | F0.5 | Precision | Recall | Blocking recall |
|---|---|---|---|---|---|
| 0 (singletons) | 12,325 | 0.925 | — | — | — |
| **1** | 11,916 | **0.794** | 0.958 | 0.817 | 0.867 (= its ceiling) |
| 2–3 | 90,605 | 0.896 | 0.980 | 0.813 | 0.861 |
| 4–5 | 80,607 | 0.919 | 0.988 | 0.809 | 0.860 |
| 6+ | 25,229 | 0.925 | 0.993 | 0.799 | 0.857 |

Single-match entities are the weakest. **Their ceiling is 0.867, purely from blocking:** if their one true record isn't a candidate, the entity scores 0.

### 3.4 Threshold table (validation)

| t | F0.5 | Precision | Recall | Singleton FP | Predicted/S1 | False merges |
|---|---|---|---|---|---|---|
| 0.50 | 0.9013 | 0.9656 | 0.8256 | 1,505 | 2.95 | 21,328 |
| 0.60 | 0.9033 | 0.9740 | 0.8167 | 1,156 | 2.89 | 15,636 |
| **0.67** | **0.9037** | **0.9790** | 0.8100 | 926 | 2.85 | 12,428 |
| 0.75 | 0.9026 | 0.9841 | 0.7998 | 715 | 2.80 | 9,094 |
| 0.85 | 0.8968 | 0.9897 | 0.7798 | 457 | 2.71 | 5,607 |

- The curve is flat from 0.60 to 0.75 (±0.001).
- The scores are well separated: the median true pair scores 0.999 and the median false pair 0.00004.
- **The threshold is not a lever.**

---

## 4. Why the leaderboard (0.889) is below validation (0.904)

1. **Country mix (arithmetic on measured inputs).**
   - Test is India 46.8% / US 38.3% / France 15.0%.
   - India+US: (0.468·0.8718 + 0.383·0.9251) / 0.851 = **0.896**.
2. **France (estimate).**
   - (0.889 − 0.7623) / 0.150 ≈ **0.845**.
   - The trend is 0.68 (Sub 1) → 0.77 (Sub 2) → 0.845 (Sub 3), but France is now **below India**.
   - This assumes the public subset is representative and India/US transfer from validation.
3. **Measured France evidence (test, no labels): conflicting claims** (§6). France has **≥ 6.2% certainly-wrong predictions** vs India 1.2% and US 0.13%. That alone lowers France pair precision to at most ~0.94, far below the 0.98 seen in India/US. It fits the estimate.
4. **India/US behave on test as on validation:**
   - predicted/S1: India 2.75 test vs 2.73 validation; US 2.94 vs 2.93;
   - empty share: India 10.0% vs 9.8%; US 7.2% vs 7.3%.

---

## 5. Loss decomposition and error budget (validation)

### 5.1 Where the remaining 0.0963 goes

| Entity outcome | S1 | F0.5 points lost |
|---|---|---|
| Perfect (F = 1) | 120,856 (54.8%) | 0 |
| **Precise but incomplete** (missed matches) | 81,630 | **0.0462**, of which **72% of the missing matches never became candidates** |
| **Empty list: no true pair among candidates** (pure blocking) | 4,963 | **0.0225** |
| At least one false merge | 10,300 | 0.0143 |
| Empty list: true pair was a candidate but scored low | 2,007 | 0.0091 |
| Singleton wrongly given a match | 926 | 0.0042 |

**Blocking now causes ≈ 0.055 of the 0.096** (0.0225 + 0.72 × 0.046).

### 5.2 Error budget (upper bound if one bucket were fixed perfectly; for ranking only)

| Bucket | Pairs | All | India | US |
|---|---|---|---|---|
| **Never a candidate (all blocking misses)** | 107,130 | **+0.0646** | **+0.0865** | +0.0500 |
| ↳ shares a name token | 60,086 | **+0.0400** | +0.0447 | +0.0369 |
| ↳ non-Latin pool name | 18,344 | +0.0170 | **+0.0425** | 0 |
| ↳ other | 13,040 | +0.0095 | | |
| ↳ address empty | 15,660 | +0.0091 | | |
| Candidate scored below threshold (all) | 39,401 | +0.0229 | +0.0294 | +0.0185 |
| ↳ other (mixed evidence) | 19,418 | +0.0121 | | |
| ↳ script mismatch | 5,838 | +0.0043 | +0.0107 | 0 |
| ↳ name weak, address strong | 7,887 | +0.0040 | | |
| False merges (all) | 12,428 | +0.0147 | +0.0201 | +0.0111 |
| ↳ on singletons | 1,088 | +0.0042 | | |

### 5.3 Why blocking misses them (never-blocked true pairs)

| Cause | India (41,205 pairs) | US (47,628 pairs) |
|---|---|---|
| Share number + word, but the key was dropped (block > 50 records, or not among the 2 rarest numbers / 4 rarest words) | **53%** | 5% |
| Share address words but no house number | 20% | **69%** |
| Address empty on one side | 13% | 21% |
| Non-Latin name and weak address overlap | 12% | — |

Also: 2.4% of true pairs are found by blocking but **cut by the top-20 limit** (India 3.5%). Top-30 would recover 1.2 points of recall at +33% candidates.

---

## 6. New finding: conflicting claims on test (France)

In training, **no S2/S3 record belongs to two S1 businesses**. On test every S1 is scored, so a record predicted for k ≥ 2 S1s contains **at least k−1 wrong predictions**. That is a hard lower bound needing no labels (`analysis/test_conflicts.json`):

| Test | Predicted pairs | On conflicted records | **Certainly wrong (≥)** | Share of that country's predictions |
|---|---|---|---|---|
| France, Sub 2 LR | 753,285 | 4.3% | 17,202 | 2.3% |
| **France, Sub 3 LightGBM** | 800,800 | **9.1%** | **49,602** | **6.2%** |
| India, Sub 3 | 2,225,059 | 2.2% | 26,256 | 1.2% |
| US, Sub 3 | 1,948,593 | 0.25% | 2,536 | 0.13% |

- LightGBM **cut US conflicts 4×** but **nearly tripled France's**. On France it accepts pairs where several French S1s look equally plausible for the same record.
- **Why validation can't see it:** only 10% of train S1s are scored there, so competing S1s are rarely both present (the val-only exclusivity simulation gave just +0.0004).
- **Rejected explanation** (`analysis/france_pattern_probe.py`): "a city word in the name" and "same house number on another street" are **not** over-represented among France's accepted pairs (4.2% vs India 6.0%, US 5.2%) and are not materially less precise on validation.
- The real driver is **several French S1s competing for the same record.**
- **Implication:** resolving conflicts, i.e. each S2/S3 record goes to at most one S1 (the highest score), would remove ≥ 49.6k certainly-wrong France predictions. The rule was measured useless on validation *only because of the missing competition*. It must be validated under full competition before use.

---

## 7. Measured and rejected this round

| Idea | Result | Verdict |
|---|---|---|
| More model capacity (3,000 rounds) | Early stopping at 1,558; F0.5 0.9027 vs 0.9021 (dev, +0.0006) | not a lever |
| Threshold changes | flat ±0.001 over 0.60–0.75 | not a lever |
| Top-1 fallback / per-source thresholds (validation) | ≤ +0.0001 | not a lever |
| France city-name / same-number patterns | not over-represented, not less precise | hypothesis rejected |

---

## 8. What is still weak, component by component

| Component | State | Weakness (measured) | Evidence |
|---|---|---|---|
| **Blocking** | recall@20 0.860, uncapped ceiling 0.884 | 11.6% of true pairs share no key; 2.4% cut by top-20; India keys dropped by caps; US pairs with no house number | §5.3; budget +0.065 |
| Scorer | LightGBM, precision 0.979 | only 0.038 below its ceiling; remaining misses are mixed-evidence and non-Latin names | §5.2 |
| **Cross-S1 consistency** | none | test conflicts: France ≥ 6.2% certainly wrong | §6 |
| Normalization | Indic-safe | no transliteration (India non-Latin names) | §5.2 (+0.043 India blocking, +0.011 scoring) |
| Decision | t = 0.67 | flat; exhausted | §3.4 |
| Single-match entities | F0.5 0.794 | ceiling 0.867 from blocking alone | §3.3 |
| Engineering | reproducible, ~2 GB peak | — | full = dev |

---

## 9. My analysis: what blocks the score and what to work on

1. **The scorer problem is essentially solved.**
   - LightGBM sits 0.038 below the ceiling set by blocking, with precision 0.979.
   - More capacity or threshold work won't move the leaderboard.
2. **Blocking is now the main bottleneck.**
   - The ceiling is 0.941 at top-20 and 0.884 recall even uncapped.
   - About 57% of the remaining validation loss is true pairs that never become candidates.
   - It hurts India most (ceiling 0.921) and single-match entities most (ceiling 0.867).
   - The largest blocking bucket is **true pairs that share a name token but have no usable address key** (60k pairs, up to +0.040). The name-token pass was rejected in Phase 2 *only* because the old rule couldn't handle the extra candidates; LightGBM can.
3. **France is the other visible loss, and it's a consistency problem, not a similarity problem.**
   - The test data itself proves ≥ 6.2% of France predictions are wrong (conflicting claims).
   - This is invisible to our validation set by construction. So the next validation must **score all train S1s** (full competition) to measure a conflict-resolution rule honestly before using it on test.
4. **India transliteration remains a real but smaller lever:** +0.043 India blocking, +0.011 scoring (upper bounds, overlapping).
5. **Realistic expectation:** the remaining upper bounds (blocking +0.065, conflicts on test, transliteration) are larger than the gaps to typical top scores, but upper bounds are never fully realised. Each step has to be measured locally first. **No leaderboard slot should be spent before a change beats 0.9037 locally** (for conflicts, under full competition).

### Proposed work, in evidence order (each one experiment, each needing approval)

| # | Experiment | One change | Evidence | Keep rule |
|---|---|---|---|---|
| 1 | **EXP-B-03 blocking recall with LightGBM** | enable the `name_token` pass (then, as separate runs, top_k 30 and addr_num_word cap 100) | largest bucket +0.040; top-20 cut 2.4%; India key-drop 53% | F0.5 > 0.9037 on both halves, precision ≥ 0.975, France profile sane |
| 2 | **EXP-X-01 conflict resolution under full competition** | score all 2.2M train S1s (validation evaluated inside the full competition), then keep each S2/S3 record only for its best-scoring S1 | test: ≥ 49.6k certainly-wrong France predictions | validation F0.5 and precision ↑ under full competition; then compare test conflicts |
| 3 | EXP-N-01 transliteration (India) | train-fold-learned token map → feature + blocking key | India +0.043 / +0.011 upper bounds; 47% token coverage | India F0.5 ↑, precision not ↓ |
| — | not now | capacity / threshold / fallback / per-country rules | measured ≤ +0.0006 | — |

---

## 10. Rules compliance

| Rule | Status |
|---|---|
| Raw data read-only; no external data | ✔ |
| Validators | ✔ strict + official PASS |
| Leakage | ✔ training rows checked in code: 0 validation S1; threshold/early stopping on the train-fold inner split |
| Open-set countries | ✔ no country feature or rule; France guard passed (1.063) |
| Model license | ✔ LightGBM 4.7.0 (MIT); trees ≪ 8B params |
| No fabricated metrics / no git push | ✔; estimates labelled |

## 11. Reproducing this report

```bash
python scripts/analysis/analyze_exp_b01.py      --config configs/exp_m03_lgbm.yaml
python scripts/analysis/error_budget.py         --config configs/exp_m03_lgbm.yaml
python scripts/analysis/france_pattern_probe.py --config configs/exp_m03_lgbm.yaml
python scripts/analysis/test_country_profile.py --runs EXP-F-01b:scored_EXP-F-01b EXP-M-03:scored_EXP-M-03
python -m ber.pipeline.run_baseline fit      --config configs/exp_m03_lgbm.yaml --set fit.dev_subsample_s1=50000 --set lightgbm.num_boost_round=3000 --set experiment.id=EXP-D-03-dev50k-r3000
python -m ber.pipeline.run_baseline evaluate --config configs/exp_m03_lgbm.yaml --set experiment.id=EXP-D-03-dev50k-r3000
python scripts/analysis/test_conflicts.py      --runs EXP-F-01b:scored_EXP-F-01b EXP-M-03:scored_EXP-M-03   # §6
```
