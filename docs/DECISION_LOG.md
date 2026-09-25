# Decision Log

Status values: **Proposed** (awaiting team approval) · **Accepted** · **Rejected** · **Superseded by DEC-xxx**.
Evidence numbers come from `scripts/profiling/profile_datasets.py` (seed 42, 100k-S1 sample for pair probes),
summarised in `reports/INITIAL_ARCHITECTURE_AND_STRATEGY.md` §3.

---

## DEC-001 — Use a multi-stage pipeline (blocking → pair scoring → entity-level decision)

Date: 2026-09-25

Context: The challenge requires `candidate_pairs.tsv` (the exact set the model scores) plus `matching_results.tsv`.
The candidate pool is ~10M S2/S3 records for 1.73M test S1 entities, so a full cross-product (~1.7×10¹³ pairs) is impossible.

Problem: Choose the overall architecture.

Options considered:
- Option A: Pure deterministic rules (exact/normalized key joins).
- Option B: Blocking → pairwise ML classifier → entity-level decision layer.
- Option C: End-to-end embedding retrieval + cross-encoder.

Decision: **Option B**, with rules as Baseline 1 inside the same framework and embeddings as a *possible later* blocking pass or feature.

Reason: Only 25.9% of true pairs have an exactly equal normalized name and 8.3% an equal normalized address, so exact rules miss most matches.
Option C is heavy for 10M records on a 6 GB GPU and ~2–3 GB of free RAM, and it's unproven here.
The output format itself describes Option B (blocking plus a model).

Expected impact: A working end-to-end baseline quickly, with room for measured improvements.

Trade-offs: More stages means more integration work. Each stage needs its own metric.

Validation plan: Entity-level F0.5 on the validation fold (DEC-004), plus stage metrics (blocking recall, candidates per S1).

Status: Proposed

---

## DEC-002 — Country is a hard blocking partition (open-set string, never one-hot)

Date: 2026-09-25

Context: In 346,678 sampled true pairs, `country` is equal in **100%** of them. Test adds **France** (15.0% of test S1), which never appears in train.
The problem statement forbids hard-coding, filtering or one-hotting countries.

Options considered:
- Option A: Block within each country value, whatever the country strings are (generic group-by).
- Option B: Ignore country entirely.
- Option C: Country-specific pipelines for US/India.

Decision: **Option A.** Group by the raw `country` string, iterate over whatever values exist, and never enumerate specific countries in code.
Country is NOT a model feature (no one-hot). Country-*agnostic* features only.

Reason: No recall loss on train, a large reduction in comparisons, and it naturally handles France.

Trade-offs: If a test pair ever had mismatched country labels, it would be unreachable. Train evidence says this doesn't happen, but it's unverified for France.
Mitigation: log test S2/S3 country values, and fail loudly on unexpected empty or unknown labels.

Validation plan: Blocking recall with and without the country partition on the validation fold (expected identical).

Status: Proposed

---

## DEC-003 — Normalization keeps multiple representations; accent stripping is Latin-only

Date: 2026-09-25

Context: During profiling, a generic NFKD plus "strip combining marks" normalization **destroyed Devanagari and Tamil words**
(`प्राइवेट` → `पर इवट`), because Indic vowel signs are combining marks.
On the India side, 27.6% of S2 names and 18.5% of S3 names are non-ASCII (Devanagari, Tamil, Kannada, Bengali, Odia, …),
while S1 names are always Latin script. Latin-script sources also contain injected accents (`Àmicale`, `Léarning`) and domain-style names (`truefactory.com`, 4% of S2/S3).

Options considered:
- Option A: One aggressive canonical string per field.
- Option B: Several parallel representations per field (raw, normalized, tokens, no-legal-suffix, alnum-concatenated, transliterated).

Decision: **Option B.** Strip accents only for Latin-script characters. Never strip combining marks from Indic scripts.
Keep `name_raw`, `name_norm`, `name_tokens`, `name_core` (legal and generic tokens removed), `name_concat` (alphanumeric only, no spaces),
and `name_translit` (for non-Latin input), plus the address equivalents (`addr_norm`, `addr_tokens`, `addr_numbers`, `addr_words`, `addr_state_canon`).

Reason: Different noise types need different views. Destroying information early can't be undone downstream.

Trade-offs: More memory per record. Mitigate by computing views lazily per country shard and storing them as Arrow/Parquet.

Validation plan: Unit tests on hand-picked examples (Indic, French, domain names, junk prefixes), and blocking-recall deltas per representation.

Status: Proposed

---

## DEC-004 — Validation: group split by S1 entity, with blocking run on the full train universe

Date: 2026-09-25

Context: Ground truth is a strict partition (7,638,365 matched S2/S3 IDs, each in exactly one S1 list). 26–27% of S2/S3 train records are unmatched distractors.
There is no train/test ID overlap and no row-order or ID-number leakage (Spearman ≈ 0.000–0.003).
Test has more S2+S3 records per S1 (5.75) than train (4.68).

Options considered:
- Option A: Random split of *pairs*. This leaks: the same S1 ends up in train and validation.
- Option B: Split S1 entities into disjoint "mini-universes", each with its own S2/S3 pool. This makes blocking artificially easy because the pools are smaller and sparser.
- Option C: Run blocking once on the **full** train universe (all S1 × all S2/S3, unsupervised).
  Then split **S1 entities** into folds (stratified by country and match-count bucket). Train the matcher on fold-train S1 pairs and evaluate on fold-validation S1 pairs.

Decision: **Option C.** A fixed validation fold of about 10% of S1 (≈220k entities), seed 42, stored as an ID list under `data/splits/`.
Blocking hyper-parameters (DF caps, top-K) are tuned on validation-fold recall only.

Reason: Keeps realistic candidate density and hard negatives, and keeps S2/S3 exclusivity (DEC-008) usable. No label leakage, because blocking is unsupervised.

Trade-offs: The full-universe blocking run is expensive (~12.5M records). It has to be sharded by country.

Validation plan: Also report **test-mix-reweighted F0.5** (India 46.7% / US 38.3% / France 15.0% → use India and US scores, and report France as unknown).
Plus a **leave-one-country-out** robustness check as the proxy for France (DEC-009).

Status: Proposed

---

## DEC-005 — Primary metric: macro per-S1 F0.5 (official definition), never pairwise F1 alone

Date: 2026-09-25

Decision: A single evaluator module, `src/ber/evaluation/metrics.py`, implements the official definition.
Per-entity F0.5 = 1.25·P·R/(0.25·P+R). For an empty true set: score 1.0 if the prediction is empty, else 0.0.
For a non-empty true set with an empty prediction: score 0.0. The evaluator is unit-tested against the worked example in the problem statement (0.714).
It also reports macro P and R, blocking recall, candidates per S1, and per-country and per-match-count breakdowns.

Reason: The leaderboard metric is entity-level and macro-averaged. Pairwise metrics over-weight large clusters and ignore singletons.

Status: Proposed

---

## DEC-006 — Model progression: rules → logistic regression → GBDT; embeddings only if error analysis demands them

Date: 2026-09-25

Decision: Baseline 1 is a weighted similarity rule. Baseline 2 is logistic regression. Baseline 3 is LightGBM (MIT license), pending approval to add the dependency.
The same features and the same evaluator are used for all three.

Reason: Establish a strong, cheap, interpretable baseline before adding complexity. GBDT handles non-linear interactions (for example, strong name with weak address).

Status: Proposed

---

## DEC-007 — Decision layer optimized for entity-level F0.5, not a 0.5 probability threshold

Date: 2026-09-25

Decision: Start with a global threshold tuned on validation F0.5.
Then test (a) top-k per entity chosen to maximize expected F0.5 from calibrated probabilities, and (b) per-entity relative margins.
Source-specific or country-specific thresholds are allowed **only** if validation shows a gain and they generalize to unseen countries.

Status: Proposed

---

## DEC-008 — Exploit S2/S3 exclusivity (each S2/S3 record matches at most one S1)

Date: 2026-09-25

Context: In train, `other_ids_in_multiple_s1_lists = 0` across 7.64M matched IDs. The problem statement does NOT state this rule for test.

Decision: Test as an experiment. When an S2/S3 record is predicted for several S1s, keep it only for the highest-scoring S1 (optionally requiring a margin).

Reason: A cheap precision lever that directly removes false merges.

Risk: Relies on a structural property that was observed in train but isn't stated for test. It is only adopted if validation shows a gain, and it will be confirmed on the public LB with a paired submission if it matters.

Status: Proposed

---

## DEC-009 — France (unseen country) strategy: country-agnostic features and conservative thresholds

Date: 2026-09-25

Decision: Features are similarity-based and language-agnostic (no token-identity or country features).
Legal-suffix and street-abbreviation handling is derived from the data, plus small generic linguistic lists (for example `SARL`, `SAS`, `R.`→`rue`, `AV`→`avenue`).
The lists must be documented; approval for them is pending in the report's Questions section.
Robustness is measured with leave-one-country-out training (train on US → evaluate on India, and vice versa).

Status: Proposed

---

## DEC-010 — Memory-aware data engineering (pyarrow/Parquet, per-country shards, float32 features)

Date: 2026-09-25

Context: The dev machine has 16 GB RAM, and only ~2–3 GB was free during the audit. Profiling a single S2 file peaked at ~2 GB RSS.

Decision: Read the TSVs once and convert them to typed Parquet in `data/interim/`. Process per country and per S1 chunk. Store features as float32.
Never materialize a full pandas object-dtype S2/S3 frame. Sample S1 entities for model training (the pair budget is set in config).

Status: Proposed
