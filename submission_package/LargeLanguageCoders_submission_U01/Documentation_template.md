# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** LargeLanguageCoders
**Team Members:** Ayush Singh, Atharva Patil, Kshitij Kediya, Param Bapu  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

- **Pipeline:** a precision-first, two-stage entity-resolution pipeline.
  - Country-partitioned multi-key blocking keeps the top-20 candidates per Source-1 entity; 95.7% of true pairs are retrieved on our held-out validation split.
  - A LightGBM pair matcher scores those candidates.
  - A *record-competition* rule assigns every Source-2/3 record to at most one Source-1 entity.
  - A second LightGBM stage, using the context of each entity and record, acts as a **veto** that can only remove matches.
- **Key innovations:**
  - a cross-script name view (Latin ↔ Indic) **learned only from the training labels**;
  - combination blocking keys for generic names;
  - decision rules designed for the per-entity macro F0.5 metric: best claimant and a removal-only veto that stays robust under the test set's denser candidate pool.
- **Results:** held-out validation F0.5 **0.9536** (precision 0.992); public leaderboard **0.942**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:**
  - train: 2.2M S1 / 5.0M S2 / 5.3M S3;
  - test: 1.73M S1 / 4.9M S2 / 5.1M S3.
  - Every stage streams per country or per file part (peak ≤ 5 GB RAM on a 16 GB laptop).
- **Countries:**
  - train has India and the US;
  - **test adds France, which is unseen in train.** Country is treated as an open set: no country-specific rule, feature or threshold exists anywhere.
- **Noise patterns found in EDA:**
  - legal-form variants ("pvt ltd", "private limited", "llc", "inc");
  - leading zeros and unit/suite numbers in addresses;
  - word-order changes;
  - empty addresses (name-only records, frequent in the US);
  - **mixed scripts**: Indian records in Devanagari or other Indic scripts vs Latin transliterations of the same business;
  - generic names ("sri balaji traders") that occur thousands of times;
  - chains, where one name appears at many addresses.
- **Metric structure:** macro F0.5 per S1 entity.
  - Singletons score 1.0 only for an empty prediction.
  - One false merge on an entity costs more than one missed match.
  - The decision rules are built around this.
- **Train/test shift (label-free diagnostic):**
  - test candidate lists contain about 1.4× (France about 2×) more mid-score look-alike candidates per S1 than train;
  - a context model that *adds* matches overfits the train candidate environment.
  - We measured this on a shadow benchmark that simulates denser pools and designed the final stage to be removal-only.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage LightGBM) + metric-aware decision layer.

**Core Innovation:**
1. A cross-script token map learned from train true pairs only, cross-fitted to avoid leakage, feeding name keys and similarity features.
2. Combination blocking keys that recover pairs sharing only common tokens:
   - name token × address word;
   - name-token pairs;
   - name token × house number.
3. Decision layers aligned with per-entity macro F0.5:
   - best claimant per record (D-01);
   - a removal-only context-model veto.

---

## 3. Candidate Generation (Blocking)

Blocking is done per country, with the country as an open set. Several passes run over normalised record views; each yields exact-key blocks with a block-size cap of 50.

**Blocking keys used:**
- exact normalised name;
- exact name core (legal forms removed) and **sorted-token** name core, i.e. word-order invariant;
- rare address word; address house number × address word;
- pairs of the rarest address words;
- rarest name token × rarest address words;
- **name-token pair**: the 2 rarest name tokens, a key for name-only records;
- exact name core × address word, for generic names whose exact-name block exceeds the cap;
- **name token × house number**;
- all name keys use the learned cross-script name view, so Devanagari and Latin spellings share keys.

**Ranking and candidate list:**
- Candidates are ranked by a cheap score: the sum over shared keys of weight × log(1 + N_pool / block size).
- The **top 20** per S1 are kept. `candidate_pairs.tsv` is exactly this scored set.
- Numbers are zero-stripped.

**Candidate pairs generated (test):** 33,733,279 (19.5 per S1). Only 13 S1 have no candidate.

**How we ensured true matches were not lost:**
- Every key family was accepted only after a blocking-only experiment on the frozen validation split measured recall@20, candidate displacement and cost.
- Several families were rejected by pre-declared kill gates (wider windows, top-K 30, a 3-token name pair).
- Validation recall@20 rose from 0.798 (first baseline) to **0.957**.

---

## 4. Matching Model

**Stage 1: LightGBM pair classifier.** 24 float32 features per candidate pair.

**Features used:**
- **Name features:**
  - exact equality of the normalised name and of the core name;
  - core-token Jaccard; shared tokens;
  - rapidfuzz ratio and token-set ratio;
  - Jaro-Winkler on the concatenated name; length difference;
  - **cross-script features**: Jaccard and token-set ratio of the learned transliteration view;
  - **legal-form conflict**.
- **Address features:**
  - exact equality; token Jaccard; ratio; token-set ratio;
  - house-number Jaccard and shared numbers (zero-stripped); shared useful words;
  - any-empty flag;
  - **number conflict** (different unit/suite numbers).
- **Other:**
  - blocking cheap score and cheap rank;
  - two interaction terms (name Jaccard × number Jaccard, name ratio × address token-set).

**Model type:** LightGBM gbdt, binary log loss.
- Settings: 31 leaves, max depth 8, learning rate 0.05, up to 1,000 rounds with early stopping, seed 42, deterministic.
- Training data: 200k train S1 → 3.1M candidate pairs.

**Threshold selection method:** F0.5-optimal global threshold (0.65) chosen on an inner split of the training fold. The frozen validation split is never used for fitting or tuning.

**Decision layer (all tuned only on training-side data):**
1. **D-01, best claimant:** each S2/S3 record is kept only for the S1 that scores it highest (full competition among all S1).
2. **Stage-2 veto:**
   - a second LightGBM with 41 features: stage-1 score and rank, S1 context, record-competition context, and the top-3 peer records' similarities;
   - trained on held-out train entities scored out-of-sample;
   - its threshold (0.69) is chosen on an inner split;
   - it can only **remove** stage-1 matches. Removals transfer to the denser test pool; additions do not (measured).
3. **Evaluated, not selected: expected-F0.5 pruning.**
   - Per S1, keep the prefix of accepted matches that maximises the exact expected per-entity F0.5.
   - It gained only +0.0001: the stage-2 probabilities are already calibrated at the veto cut.

---

4. **Final combination (EXP-U-01):**
   - keep the S-03 decisions;
   - add pairs found only by a second, independently retrained pipeline whose blocker includes a 3-token name-pair key, where its stage 1 and its own refitted veto both accept, and the record is not matched by S-03;
   - the candidate file is the union of the two scored top-20 lists.
   - Validation F0.5 0.9536 → **0.9544**; 97.6% of the added pairs are true.

## 5. Results & Error Analysis

Held-out validation: 220,682 S1 never used for fitting or tuning. "E half B" is 1.0M further held-out train S1.

| System | Held-out F0.5 (validation; † = E, 2.0M held-out S1) | Precision | Recall | Public LB |
|---|---|---|---|---|
| Baseline (rule matcher) | 0.7851 | 0.9226 | 0.6553 | 0.751 |
| + learned logistic regression + conflict features | 0.8372 | 0.9319 | 0.7418 | 0.815 |
| + LightGBM | 0.9037 | 0.9790 | 0.8100 | 0.889 |
| + combined blocking keys + retrain | 0.9274 | 0.9774 | 0.8579 | ≈ 0.91 |
| + learned cross-script view + D-01 | 0.9402 † | 0.9864 | 0.8702 | 0.928 |
| + generic-name keys, name-token pairs | 0.9470 | 0.9863 | 0.8832 | ≈ 0.935 |
| + name × number key + **stage-2 veto (S-03)** | 0.9536 | 0.9920 | 0.8889 | 0.942 |
| + pairs from a 3-token-name-pair pipeline on new candidates (**U-01, final**) | **0.9544** | 0.9919 | 0.8912 | pending |

- **F0.5 score (macro):** 0.9536 on validation; 0.9536 on E half B.
  - India 0.949, US 0.957.
  - France has no labels; test prediction rates per S1 are the same as India/US (3.10 vs 3.10 / 3.12).
- **Common false positives (wrong merges):**
  - branches of the same chain at different addresses;
  - generic names in the same locality;
  - records sharing a house number and street but with different businesses.
  - The veto removed 49% of the remaining false merges.
- **Common false negatives (missed matches):**
  - 4.3% of true pairs are never retrieved (mostly records sharing only very common name tokens, or name-only records);
  - about 6% are retrieved but scored below threshold (heavy abbreviation, different business descriptors, cross-script spellings the learned map does not cover).
- **Rejected on evidence:**
  - stage-2 *additions*: +0.012 locally but −0.015 public, because of the denser test pool;
  - top-K 30;
  - wider address windows;
  - more trees;
  - reweighting the name-token key (it displaced address-matched true pairs).

---

## 6. Conclusion

- A carefully validated two-stage LightGBM pipeline with metric-aware, removal-only decision layers reaches 0.954 held-out F0.5 at 0.992 precision.
- Every decision was gated on pre-declared held-out tests and a density shadow benchmark, which protected us from a context model that looked +0.012 better locally but transferred worse.
- **Next step:** learned text representations (a permissively licensed multilingual encoder) for retrieval and matching of the remaining hard pairs.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/ber/`: all source code;
- `configs/`: experiment configs with inheritance;
- `README.md`: exact end-to-end commands;
- `requirements.txt`: pinned versions;
- `tests/`;
- the trained models under `experiments/runs/`, for the quick path.

**Entry points:**
- `python -m ber.pipeline.run_baseline {preprocess,split,fit,infer}` (stage 1);
- `python -m ber.pipeline.score_alltrain` (stage-2 training input);
- `python -m ber.pipeline.run_stage2 {features,fit,submit}` (stage 2 and the output files).

### B. Additional Results

- **Leakage controls:**
  - stage-1 fit set ∩ validation = ∅;
  - the stage-2 training partition is disjoint from validation and E half B, enforced in code;
  - the cross-script map is learned from the fit set only (cross-fitted inside the fit);
  - no external data or pretrained model.
- **Reproducibility:**
  - seed 42;
  - two submits give byte-identical files;
  - both outputs pass the official validator with `--check-ids`.
