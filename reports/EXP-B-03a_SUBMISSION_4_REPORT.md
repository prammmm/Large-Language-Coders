# Submission 4 (EXP-B-03a check): Full Report of All Work So Far, and Where the Score Must Come From Next

**Team:** LargeLanguageCoders · **Date:** 2026-09-25
**Leaderboard history:** 0.751 → 0.815 → 0.889 → **0.890** (EXP-B-03a, submitted by the team as a check)
**Best local validation:** **0.9037** (EXP-M-03). EXP-B-03a: 0.9027.
**Status:** analysis complete. Every "next step" is a **proposal awaiting team approval**. One leaderboard slot remains today.

> **Where the numbers come from:** the files under `experiments/runs/*` named in each section, plus read-only diagnostics in `scripts/analysis/`. Leaderboard scores were reported by the team. *Estimates* and *upper bounds* are labelled. No raw records appear here.

---

## 1. Executive summary

1. **EXP-B-03a scored 0.890 on the leaderboard vs 0.889 for Submission 3.** That is +0.001, within public-subset noise.
   - On validation it was **−0.0010** (0.9027 vs 0.9037), on both halves.
   - On test it made **~20.6k more certainly-wrong predictions** (conflicting claims) than Submission 3.
   - **It is not a real improvement.** For the final (private) selection, EXP-M-03 is the safer choice.
2. **Why the score has plateaued around 0.89:**
   - The **scorer** is close to its ceiling: LightGBM is 0.038 below the best possible result on its candidates, with precision 0.979.
   - The **candidate list** (blocking) is now the limit: 14% of true pairs never become candidates, which costs about 0.055 of the remaining 0.096 validation loss.
   - **France** (15% of test) carries a test-only consistency error: ≥ 6.2% of its predictions are certainly wrong.
3. **The most important new measurement** (`blocking_gap_probe`):
   - Of the 107,130 true pairs missing from the candidate list, **only 19.7% share a *rare* name word**. That's why the name-token pass recovered so little.
   - **78.1% could be reached by two combined key families** we don't have yet: *(name word + address word)* and *(pair of address words)*.
   - **0% share nothing at all.** Every missing pair has some shared evidence; our keys just don't use it.
4. **The second most important lesson (EXP-B-03a):**
   - The model takes two blocking-derived inputs (`cheap_score`, `cheap_rank`). **Any blocking change must be followed by retraining the model** on candidates from that blocker, or the model misjudges pairs it already had (+3,994 false merges in B-03a).
5. **Recommended direction (proposals):**
   - **(1)** Blocking v2 with combined keys, with the model retrained, measured locally.
   - **(2)** Conflict resolution under full competition (France/India).
   - **(3)** Transliteration for India.
   - **Don't spend today's last slot** on anything not locally validated.

---

## 2. Everything built so far (cumulative)

| Phase | What was built | Key files | Result |
|---|---|---|---|
| 0 | audit, architecture, governance | `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md`, `CLAUDE.md`, logs | — |
| 1 | end-to-end pipeline: ingest, Indic-safe normalization, country-partitioned blocking v0, 16 features, weighted rule, writer, strict validator, tests | `src/ber/**`, `configs/baseline_v0.yaml` | **Sub 1: LB 0.751** (val 0.7851) |
| 2 | blocking v1 (sorted core-name key, zero-safe numbers), logistic regression (train-fold fit, inner-split threshold), conflict features, precision-first reporting, `block-eval` / `fit` / `evaluate` stages | `models/logistic.py`, `configs/exp_f01b_*.yaml` | **Sub 2: LB 0.815** (val 0.8372) |
| 3 | LightGBM matcher (MIT), reuse of train-fold parts, leakage guards | `models/gbdt.py`, `configs/exp_m03_lgbm.yaml`, DEC-016 | **Sub 3: LB 0.889** (val 0.9037) |
| 4 | EXP-B-03a: name-token pass with the fixed LightGBM model; pairwise candidate comparison; test conflict and country checks; blocking-gap probe | `configs/exp_b03a_name_token.yaml`, `scripts/analysis/{compare_candidate_sets, test_conflicts, test_country_profile, blocking_gap_probe}.py`, `tests/test_name_token.py` | **Check: LB 0.890** (val 0.9027, rejected) |

### 2.1 Changes made in Phase 4 (this round)

- **Configs:** new `configs/exp_b03a_name_token.yaml`.
  - Inherits EXP-M-03, with `name_token.enabled: true` (max_df 200).
  - `matcher.model_path` = the EXP-M-03 model (fixed, not retrained); its own validation/test parts.
- **New scripts:**
  - `scripts/analysis/compare_candidate_sets.py`: new / displaced / flipped pairs, recovered-TP and new-FP profiles, counterfactual F0.5;
  - `scripts/analysis/blocking_gap_probe.py`: which key family could reach each missing true pair;
  - `scripts/analysis/test_conflicts.py`: certainly-wrong test predictions.
- **Updated scripts:**
  - `analyze_exp_b01.py` re-scores with the experiment's own model and uses its own test parts;
  - `test_country_profile.py` / `test_conflicts.py` follow a run's recorded `model_path`;
  - `compare_val_halves.py` loads LightGBM models.
- **Tests:** `tests/test_name_token.py` (4 new). **87 passing.**
- **Docs:** EXPERIMENT_LOG, DEVELOPMENT_LOG, `docs/MODEL_AND_PIPELINE_EXPLAINED.md` (name-token section, English + Hindi).
- **Environment:** the existing `ber_hackathon` env; **no dependency changes**.
- **Not changed:** production code paths (the pass already existed), the model, features, thresholds.
- **Nothing committed.** Reminder: `src/ber/models/` is still **untracked** in git (the old `.gitignore` bug), so add it to your next commit.

---

## 3. EXP-B-03a results

### 3.1 Validation (frozen 220,682 S1; real measurement)

| Metric | EXP-M-03 | EXP-B-03a | Δ |
|---|---|---|---|
| F0.5 | 0.9037 | 0.9027 | −0.0010 |
| Precision | 0.9790 | 0.9735 | −0.0055 |
| Recall | 0.8100 | 0.8165 | +0.0065 |
| Blocking recall | 0.8597 | 0.8686 | +0.0089 |
| Ceiling (oracle F0.5) | 0.9413 | 0.9450 | +0.0037 |
| False merges | 12,428 | 16,311 | +3,883 |
| Singleton false positives | 926 | 1,111 | +185 |
| Candidates / S1 | 16.13 | 17.58 | +9% |
| India F0.5 / P / R | 0.8718 / 0.9698 / 0.7686 | 0.8722 / 0.9658 / 0.7759 | ≈0 |
| US F0.5 / P / R | 0.9251 / 0.9849 / 0.8376 | 0.9230 / 0.9785 / 0.8437 | −0.0021 |
| Halves A / B | 0.9037 / 0.9038 | 0.9026 / 0.9027 | stable direction |

### 3.2 Why (pairwise attribution, same model)

| Source | True matches | False merges |
|---|---|---|
| 391k new candidates (98.9% rejected by the model) | **+3,932** | **+182** |
| 70k candidates pushed out of the top-20 | −287 | −10 |
| **Existing candidates whose prediction flipped** (inputs `cheap_score`/`cheap_rank` changed) | +2,267 / −777 | **+3,994** / −283 |

- The counterfactual with the new candidates only and no input shift gives F0.5 **0.9057 (+0.0020)**.
- The new candidates are safe. **The coupling between blocker and model is what hurt.**

### 3.3 Test side (no labels)

| | EXP-M-03 (LB 0.889) | EXP-B-03a (LB 0.890) |
|---|---|---|
| Candidate ids | 27.46M | 30.53M |
| Matched ids | 4,974,452 | 5,064,688 |
| Empty rows | 145,285 | 141,440 |
| Predicted/S1: France / India / US | 3.09 / 2.75 / 2.94 | 3.19 / 2.78 / 2.99 |
| **Certainly-wrong predictions:** France / India / US | 49,602 / 26,256 / 2,536 | 51,859 / 34,589 / 11,965 |

The +0.001 leaderboard change is noise-level and goes against both the validation result and the test consistency signal. **For the final selection, prefer EXP-M-03.**

---

## 4. Where the remaining score is lost (EXP-M-03, the best system)

### 4.1 Validation loss (total 0.0963)

| Outcome | S1 | F0.5 points |
|---|---|---|
| Precise but incomplete | 81,630 | 0.0462 (**72%** of the missing matches were never candidates) |
| Empty list, no true pair in candidates | 4,963 | 0.0225 |
| At least one false merge | 10,300 | 0.0143 |
| Empty list, true pair scored low | 2,007 | 0.0091 |
| Singleton wrongly matched | 926 | 0.0042 |

**Blocking ≈ 0.055, scoring ≈ 0.041.** By country, the best possible F0.5 on our candidates is India 0.921 and US 0.955; the matcher's gap below it is India 0.049 and US 0.030.

### 4.2 Upper bounds per error type (error budget; for ranking only)

| Bucket | All | India | US |
|---|---|---|---|
| **Not in the candidate list** | **+0.065** | **+0.087** | +0.050 |
| ↳ shares a name word | +0.040 | +0.045 | +0.037 |
| ↳ non-Latin name | +0.017 | +0.043 | 0 |
| Candidate scored below threshold | +0.023 | +0.029 | +0.019 |
| False merges | +0.015 | +0.020 | +0.011 |

### 4.3 What kind of key could reach the missing pairs (new: `blocking_gap_probe`)

| 107,130 missing true pairs share… | All | India | US |
|---|---|---|---|
| a **rare** name word (current name-token key) | 19.7% | 16.4% | 22.8% |
| a name word + a useful address word | 55.0% | 45.0% | 64.4% |
| ≥ 2 useful address words | 68.1% | 70.1% | 66.2% |
| **either of the two combined keys** | **78.1%** | **80.6%** | **75.7%** |
| a house number | 39.7% | 61.2% | 19.6% |
| non-Latin vs Latin name | 17.2% | 35.5% | 0% |
| nothing | 0.0% | 0% | 0% |

These are coverage upper bounds, ignoring block-size caps and the top-K limit; real keys need caps and will reach less. They show **where** to build: keys combining *common* name words with address words, and *pairs* of address words. These reach what single rare tokens and number+word keys cannot.

### 4.4 France (test-only, no labels)
- Implied France F0.5 is ≈ **0.845** (*estimate*), the lowest country.
- Measured on test: **≥ 49,602 France predictions (6.2%) are certainly wrong** (several S1s claim the same record), up from 2.3% with LR.
- Validation can't see it, because only 10% of train S1s compete there.

---

## 5. What has been measured and ruled out

| Idea | Measured result | Status |
|---|---|---|
| Hand-weighted rule | 0.785 | replaced |
| Logistic regression | 0.837 | replaced |
| More trees (3,000 rounds) | +0.0006 | not a lever |
| Threshold changes | flat ±0.001 | not a lever |
| Top-1 fallback / per-source thresholds / validation-only exclusivity | ≤ +0.0005 | not a lever |
| Cap increases in blocking v1 (50→200) | +0.5 to +0.8 pt recall for +200k–300k candidates (rule era) | weak |
| Name-token pass (rare words) with the fixed model | −0.0010 (recovery alone +0.0020) | rejected |
| France city-name / same-number pattern | not over-represented | rejected |

---

## 6. My analysis: what to work on to raise the score substantially

**Diagnosis:** we have a strong, precise scorer (0.979) sitting on a candidate generator that can't reach 14% of true pairs. There's also a test-only consistency error concentrated in France. Scorer and threshold work is exhausted. The big remaining levers are **recall at blocking** and **consistency across S1s**. All proposals below respect the rules: provided data only, country-agnostic, measured locally before any slot.

### Priority 1: Blocking v2 with combined keys, **with the model retrained** (largest measured lever)
- **Evidence:** the blocking bucket is worth +0.065 (India +0.087); 78% of missing true pairs share evidence that a *combined* key can use; only 20% share a rare name word.
- **What to build** (each key switchable and capped, measured with `block-eval` first, which is cheap and blocking-only):
  - **(a)** *name word × address word*: key = (name_core token, useful address word), block-size cap. It reaches pairs sharing common name words (e.g. a brand) at the same street or locality. Measured coverage upper bound: 55%.
  - **(b)** *address word pair*: key = two rarest useful address words of the record, block-size cap. It reaches pairs with no house number (US: 69% of US misses) and India pairs whose number+word keys were dropped. Measured upper bound: 68%.
- **Then retrain LightGBM** on train-fold candidates from the *same* blocker. This is mandatory per the EXP-B-03a lesson, and the `fit` stage already supports it (no `reuse_fit_parts`).
- **Measure:** recall and candidates/S1 per key (block-eval); then F0.5 / precision / halves vs 0.9037.
- **Cost:** more candidates per S1. Consider top-K 30 as part of the design (blocking cuts 2.4% of true pairs at 20).

### Priority 2: Conflict resolution under full competition (France/India precision)
- **Evidence:** on test, ≥ 49.6k France and ≥ 26.3k India predictions are certainly wrong; LightGBM tripled France's.
- **What:** each S2/S3 record goes to at most one S1 (the highest score).
- **How to validate honestly:** score **all 2.2M train S1s** so validation sees full competition, then compare F0.5 / precision with and without the rule. It's a longer run (~30 min) but decisive.
- **Expected:** mainly France precision; also India.

### Priority 3: India transliteration (learned from train pairs only)
- **Evidence:** 35.5% of India's missing true pairs have a non-Latin name; the error budget gives India +0.043 (blocking) and +0.011 (scoring); feasibility showed 47% token coverage.
- **What:** a train-fold-learned token map, then a transliterated-name feature and blocking key (then retrain).

### Engineering rule going forward
- **Any blocking change ⇒ refit the model on the same blocker's train-fold candidates.**
- Optionally test a model without `cheap_score`/`cheap_rank` to decouple it from the blocker. That's a small ablation, only if Priority 1 shows instability.

### Leaderboard use (Rule 9)
- **Today's last slot:** don't spend it. Nothing untested beats EXP-M-03 locally. EXP-B-03a's +0.001 is noise.
- **Final (private) selection:** EXP-M-03 unless a locally validated improvement arrives.
- **Next slot:** only for a Blocking v2 + retrained model, or a full-competition conflict rule, that beats **0.9037** on both validation halves with precision ≥ 0.975.

---

## 7. Rules compliance

| Rule | Status |
|---|---|
| Raw data read-only; no external data | ✔ |
| Validators | ✔ strict + official PASS for all submissions |
| Leakage | ✔ training on train-fold S1 only (code-checked); thresholds from the train-fold inner split |
| Open-set countries | ✔ no country rules or features |
| Licenses | ✔ LightGBM (MIT), own LR code; no pretrained models |
| No fabricated metrics, no git push | ✔; estimates labelled |

## 8. Reproduce this report's numbers

```bash
python -m ber.pipeline.run_baseline block-eval --config configs/exp_b03a_name_token.yaml --set experiment.id=EXP-B-03a-blockeval
python -m ber.pipeline.run_baseline train      --config configs/exp_b03a_name_token.yaml
python scripts/analysis/compare_candidate_sets.py --base EXP-M-03:scored_EXP-F-01b --new EXP-B-03a:scored_EXP-B-03a
python scripts/analysis/compare_val_halves.py --out experiments/runs/EXP-B-03a/val_halves.json --runs EXP-M-03:scored_EXP-F-01b EXP-B-03a:scored_EXP-B-03a
python scripts/analysis/test_country_profile.py --runs EXP-M-03:scored_EXP-M-03 EXP-B-03a:scored_EXP-B-03a --out experiments/runs/EXP-B-03a/test_country_profile.json
python scripts/analysis/test_conflicts.py       --runs EXP-M-03:scored_EXP-M-03 EXP-B-03a:scored_EXP-B-03a --out experiments/runs/EXP-B-03a/test_conflicts.json
python scripts/analysis/blocking_gap_probe.py   --config configs/exp_m03_lgbm.yaml
```
