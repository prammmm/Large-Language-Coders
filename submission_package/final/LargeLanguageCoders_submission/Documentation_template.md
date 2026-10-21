# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** LargeLanguageCoders
**Team Members:** Ayush Singh, Atharva Patil, Kshitij Kediya, Param Bapu  
**Submission Date:** 2026-10-02

---

## 1. Executive Summary

**Approach.** We combine three blocking-plus-LightGBM pipelines:
- **Primary pipeline:** multi-key blocking per country, then a LightGBM pair classifier, then a one-owner-per-record rule, then a second LightGBM that can only remove matches.
- **Two further pipelines** use different blocking, and contribute matches only on candidates the primary pipeline never scored, and only for records not already matched.

**Results.** Held-out F0.5 is 0.956 and the public leaderboard score is 0.944.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** about 24.2M records in total (train 2.2M / 5.0M / 5.3M, test 1.7M / 4.9M / 5.1M across sources 1 / 2 / 3). Every stage runs per country and in chunks, to fit in 16 GB of RAM.
- **Names:** legal-form variants ("pvt ltd" / "private limited"), word-order changes, and the same name written in Indic and Latin scripts.
- **Addresses:** often empty (name-only records), leading zeros, unit numbers.
- **Generic names:** some repeat thousands of times, and chains appear at many addresses.
- **Countries:** France appears only in test, so country is treated as an open set. No country-specific rule exists.
- **Metric:** a precision-weighted F0.5 per entity, where singletons score only when nothing is predicted. Wrong merges therefore cost more than missed matches.
- **Test pools are denser than train,** so the final decision steps were designed to only remove matches.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage LightGBM) + ensemble of pipelines.

**Core Innovation:**
1. A cross-script token map learned only from training pairs and used in name keys and features.
2. Combination blocking keys for generic and name-only records.
3. Decision layers matched to F0.5: one owner per record, and a removal-only veto.
4. Pipelines combined so that each adds matches only where the others had no candidate.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (per country; block-size cap 50, except the rare-address-word key, which uses document frequency ≤ 20):
  - exact normalised name (weight 2);
  - sorted-token core name, with legal forms removed (weight 2);
  - sorted-token core name × address word (up to 30 words);
  - rare address word;
  - house number × address word (the 2 rarest numbers × 4 words);
  - pairs of the 3 rarest address words;
  - name token × address word (the 2 rarest name tokens × 3 words);
  - pairs of the rarest name tokens (2 tokens in pipelines A and C, 3 in B);
  - name token × house number (pipelines A and B only).

  Every name key except the exact normalised name uses the learned cross-script view. All other weights are 1.
- **Ranking:** candidates are ranked by Σ weight · log(1 + pool size / block size), summed over the keys a pair shares. Pipelines A and B keep the top 20; pipeline C keeps the top 30.
- **Candidate pairs generated (test):** 51,662,238, the union of the three scored lists (29.8 per entity).
- **How we ensured true matches were not lost:** every key was accepted only after measuring recall@K, displacement and cost on a frozen validation split. Validation recall@K rose from 0.798 (baseline) to 0.957 / 0.960 / 0.956 for pipelines A / B / C; the union of the three lists reaches 0.966. Each model was retrained whenever its candidates changed.

---

## 4. Matching Model

**Features used** (24 per pair):
- **Name features:** exact and core equality, token Jaccard, shared tokens, ratio, token-set ratio, Jaro-Winkler, length difference, cross-script Jaccard and token-set ratio, legal-form conflict.
- **Address features:** exact equality, Jaccard, ratio, token-set ratio, house-number Jaccard and shared numbers, shared useful words, empty flag, number conflict.
- **Other:** blocking score and rank, and two name × address interaction terms.

**Model type:** LightGBM (gbdt, binary log loss, 31 leaves, depth 8, learning rate 0.05, seed 42), trained on 200k training entities in each pipeline (80% for fitting, 20% as an inner split for early stopping and the threshold).
- **Second stage** (pipelines A and B): a LightGBM with 41 context features (stage-1 score and rank, rival scores, record competition, top-3 peer similarities), trained on 714k held-out training entities and used only to veto matches.

**Threshold selection method:** entity-level F0.5 maximisation on an inner split of the training data. Stage 1: 0.65 / 0.69 / 0.68 (A / B / C), then the one-owner-per-record rule. Stage 2: 0.69 / 0.67 (A / B), selected with the one-owner rule applied.
- **Final combination:** all of pipeline A's matches; plus pipeline B's matches (stage 1 and veto) on candidates outside A's list; plus pipeline C's matches (stage 1 only, no veto) on candidates outside both A's and B's lists. B and C contribute only for records not already matched. The result is 5,422,914 matches.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.956 on held-out validation (precision 0.992, recall 0.895; India 0.952, US 0.959). Public leaderboard: 0.944.
- **Common false positives (wrong merges)** (from error analysis of earlier versions; not re-measured on the final ensemble): branches of one chain, generic names in the same area, and different businesses sharing a building number.
- **Common false negatives (missed matches):** records sharing only very common words, name-only records, heavy abbreviations, and cross-script spellings outside the learned map.

---

## 6. Conclusion

- A two-stage gradient-boosting pipeline with metric-aware, removal-only decision layers and an ensemble of differently blocked pipelines reached 0.944 on the leaderboard.
- Validating under test-like density was essential.
- The next step would be learned text representations for retrieval and matching.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:
- `src/ber/`: the source code;
- `configs/`;
- `artifacts/`: the trained models;
- `README.md` and `requirements.txt`.

**Entry points:**
- `ber.pipeline.run_baseline`: preprocess, split, fit, infer;
- `ber.pipeline.score_alltrain`;
- `ber.pipeline.run_stage2`: features, fit;
- `ber.pipeline.final_submit`: writes both output files.

### B. Additional Results

| System | Validation F0.5 | Public |
|---|---|---|
| Rule baseline | 0.785 | 0.751 |
| LightGBM | 0.904 | 0.889 |
| Cross-script + best claimant | 0.940 | 0.928 |
| Stage-2 veto | 0.954 | 0.942 |
| Final ensemble | 0.956 | 0.944 |
