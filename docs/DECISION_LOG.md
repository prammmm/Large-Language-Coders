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

## DEC-011 — Submission gate = official validator + stricter local checks; fixed output encoding

Date: 2026-09-25

Context: `student_resource/` arrived. The official validator is lenient in three places. ID existence is off by default, matches ⊄ candidates only produces a warning, and CRLF is tolerated.
The README says unknown IDs are rejected, while the validator says they only lower the score. A UTF-8 BOM fails the header check (verified by smoke test). On Windows, pandas writes CRLF by default (verified).

Options considered:
- Option A: Rely on the official validator only.
- Option B: Write our own validator from scratch.
- Option C: Run the official validator unmodified, **plus** a thin wrapper adding stricter checks: memory-light ID existence as an error, matches ⊆ candidates as an error, no BOM, LF only, no whitespace in fields.

Decision: **Option C.** A single shared writer emits UTF-8 (no BOM) with `lineterminator="\n"`, no quoting, and exactly one row per test S1.

Reason: The organizer's own checks are the authority. The extra checks cover the ambiguous and lenient cases conservatively, at almost no cost.

Trade-offs: Two tools must both pass. The ID-existence check must be implemented memory-efficiently (~10M IDs).

Validation plan: A trivial all-empty test file (LF, no BOM) already passes the official validator (smoke test, 2026-09-25). The wrapper gets unit tests on crafted bad files (BOM, CRLF, duplicate row, duplicate ID, S1 self-match, unknown ID, match ∉ candidates).

Status: Proposed

---

## DEC-021 — Stage 2 restricted to a veto (EXP-S-02); shadow benchmark v2 for public-transfer checks

Date: 2026-09-27

Context: S-01 scored about 0.920 publicly, vs 0.935 before it. POST-S01-FORENSIC traced the drop to context-driven additions on weak, uncontested, look-alike candidates, which are 1.77× denser on test.

Decision (proposed):
- Final matches = (B-06 + D-01 @ 0.67) AND S-01, via `stage2.add_floor: .inf`. `null` keeps the S-01 behaviour.
- Future decision-layer experiments must also pass the v2 density-tier gate (`experiments/shadow_tests/v2/`): no material decay of the advantage over T1 → T4 / T1W / T4W on DEV and on both LOCKED sets.
- LOCKED-A/B are evaluated once, after an experiment is frozen.

Evidence: EXP-S-02 local +0.0028 to +0.0029 (validation / GATE-B / E half B). On every density tier including test density, the advantage is +0.0023 to +0.0027, while S-01's falls to −0.0002. All gates pass (`experiments/runs/EXP-S-02/final_report.md`).

Trade-offs: gives up S-01's ≈ +0.010 local recall gain from additions, which does not transfer. The tiers reach test density only through reweighting, and are calibrated on one public point.

Status: Proposed (not submitted; the team decides)

---

## DEC-020 — Collective second-stage decision model (EXP-S-01)

Date: 2026-09-27

Context: After retrieval work (B-07/B-06; blocking recall 0.9487, oracle 0.9832), most remaining loss is in the per-pair decision. Probes showed context-only features give +0.002 held-out, and collective features give +0.005 (on a 20% sample).

Decision (proposed):
- Final matches are decided by a second LightGBM on stage-1 pairs (score ≥ 0.02) using S1 context, record-claimant context and record-to-record collective similarity to the S1's top-3 other candidates.
- Threshold 0.65 + best claimant (D-01) — chosen on an inner split of the training partition.
- Stage 1, candidates and `candidate_pairs.tsv` are unchanged.
- Stage 2 is trained only on E half A minus frozen validation, from out-of-sample stage-1 scores. Stage-2 training requires the stage-1 all-train scores (`scripts/analysis/full_competition.py`).

Evidence: E half B +0.0125, validation +0.0131, precision up, all halves agree; an unseen country (LOCO C-03) improves +0.021.

Trade-offs:
- Two models and an extra feature pass (≈ 8 min train, 7 min test).
- Test acceptance rises more than on train (+5.3% unexplained; direction unknown without labels).

Status: Proposed. Awaiting human review.

---

## DEC-019 — Train-pair-learned cross-script name view (EXP-N-02)

Date: 2026-09-26

Context:
- Script mismatch (Latin S1 vs Indic-script pool names) was India's largest remaining retrieval and scoring loss.
- The official materials forbid external lookups and data, but not representations learned from the provided train data.

Decision (proposed):
- Learn a non-Latin → Latin token map **only from provided train true-match pairs**: co-occurrence over distinct S1 entities, support ≥ 3, confidence ≥ 0.5, ties rejected.
- Use it for:
  - a `name_xlit` view feeding the name-token blocking keys (`blocking.name_view`);
  - two similarity features.
- Leakage rules for any learned representation:
  - the evaluation/test map is learned from the fit S1s only;
  - training rows use cross-fitted out-of-fold maps;
  - the inner-tune fold contributes to no map;
  - validation S1s raise `LeakageError`.
- LightGBM is retrained (DEC-017).

Evidence (EXP-N-02, E = 2,006,821 out-of-sample S1s, full competition + D-01):
- F0.5 0.9313 → 0.9402; P 0.9840 → 0.9864;
- India +0.0210;
- halves +0.0089 to +0.0091.

Trade-offs:
- `fit` about 3× slower (cross-fit: ≈ 20 min).
- 518 validation true pairs are lost to block-cap growth.
- Indic legal-form tokens stay unmapped.
- No new dependency.

Status: Proposed. Awaiting human review.

---

## DEC-018 — Full-competition best-claimant resolution in the decision layer (EXP-D-01)

Date: 2026-09-26

Context:
- DEC-007 kept the decision layer to one global threshold. Exclusivity was measured at ≤ +0.001, but only on the validation fold, where about 90% of a record's competing S1s are absent.
- On test, all S1s are scored together. The forensic analysis (POST-0.91) scored all 2.2M train S1s to reproduce this.
- Official text: "Source 1 is the deduplicated reference source."
- Train ground truth: every matched S2/S3 id belongs to exactly one S1.
- Per-record exclusivity is **not** an explicit challenge rule.

Decision (proposed):
- After the threshold, each S2/S3 record predicted by several S1s is kept only for the highest-scoring S1. Margin 0; ties go to the lowest S1 row.
- Implemented in `src/ber/decision/competition.py`. `submit` applies it when `decision.competition.enabled`; the default is off.
- Rules that depend on competition must be validated with all S1s competing (all-train scoring, out-of-sample evaluation), never on the 10% validation fold alone.

Evidence (EXP-D-01, E = 2,006,821 out-of-sample S1s):
- F0.5 0.9273 → 0.9313, P 0.9775 → 0.9840, R 0.8576 → 0.8562.
- False merges −29.8%.
- Halves +0.0039 to +0.0040.
- Out-of-sample competitors only: +0.0037.

Trade-offs:
- About 18% of removed claims are true matches, mostly in close contests between near-duplicate S1s.
- If test did not share the train exclusivity structure, the only effect is lost recall on 0.95% of predictions.
- Runtime is negligible (about 5 s).

Status: Proposed. Awaiting human review (supersedes the "no exclusivity" part of DEC-007 if accepted).

---

## DEC-017 — Blocking V2 (combined keys) and the rule "blocking change ⇒ retrain"

Date: 2026-09-25

Context:
- After EXP-M-03, blocking caused about 57% of the remaining validation loss.
- 78% of the missed true pairs share a (name word + address word) or ≥ 2 address words; only 20% share a rare name word.
- EXP-B-03a showed that the blocker-derived inputs `cheap_score` / `cheap_rank` couple the model to the blocker.

Decision (proposed):
- Add the `name_addr` and `addr_pair` passes (config-switchable, block cap 50, pool DF ≥ 1 selection), top-K 20.
- **Any blocking change must be followed by re-generating train-fold candidates with that blocker and refitting the model.**

Evidence (EXP-B-04):
- Validation F0.5 0.9037 → 0.9274, precision 0.9790 → 0.9774; halves 0.9271 / 0.9277.
- K=30 adds +0.0015 for +45% candidates (not adopted).

Trade-offs:
- +19% candidates.
- Validation scoring 182 s → 378 s; peak memory ≈ 2.5 GB.
- Test inference will take longer (not yet measured).

Status: Proposed. Awaiting human review before any full test run.

---

## DEC-016 — Add LightGBM (MIT) as the non-linear matcher library for EXP-M-03

Date: 2026-09-25

Context:
- EXP-D-02 (MLP probe) showed about +0.05 F0.5 from a non-linear scorer on the same features and protocol.
- DEC-013's LR cannot express conditional interactions.
- The final model must be MIT/Apache-2.0 and ≤ 8B parameters.

Options considered:
- Option A: MLP (PyTorch, BSD-3 library; the license rule's wording is ambiguous for from-scratch models).
- **Option B: LightGBM gradient-boosted trees (MIT)**, CPU build.
- Option C: scikit-learn HistGradientBoosting (BSD-3).

Decision: **Option B.**
- Package: `lightgbm=4.7.0` (conda-forge CPU build) in the existing `ber_hackathon` env; its required dependency `scipy` 1.18.1 comes with it. No other package changed (verified).
- Model: a few thousand small trees (≪ 8B parameters).
- Used only by `matcher.type: lightgbm`; LR stays available.

Reason: MIT license (clearly compliant), strong on tabular features, fast on CPU (about 14 s to fit 516k pairs), deterministic with a seed.

Trade-offs: New dependency, and a text model file (`model.lgb.txt`) next to `model.json`.

Validation plan: EXP-M-03 (scorer-only change vs EXP-F-01b). Keep only if F0.5 and precision improve and the France test profile stays sane.

Status: Proposed (dependency installed and pinned; adoption pending the team's full-scale run)

---

## DEC-013 — Learned matcher: numpy logistic regression, trained on train-fold S1, threshold from an inner train-fold split

Date: 2026-09-25

Context: The weighted rule (DEC-006 Baseline 1) left 0.125 F0.5 between the realised score and the oracle. EXP-D-01P (headroom probe; renamed from EXP-D-01) and EXP-M-02 measured about +0.031 from fitting the weights. scikit-learn and LightGBM are not in the env.

Options considered:
- Option A: Keep the rule.
- Option B: scikit-learn LogisticRegression (new dependency).
- Option C: A small Newton-IRLS logistic regression in numpy (no dependency, deterministic, unit-tested).
- Option D: LightGBM now.

Decision: **Option C.**
- The model is fitted on a seeded sample of **train-fold** S1s; validation S1s are excluded by construction and checked.
- The threshold is tuned on an inner 20% of those train-fold S1s (`decision.threshold_from: model`).
- The model is stored as readable JSON (`experiments/runs/<ID>/model.json`).

Reason:
- The measured gain holds on both validation halves.
- The train-fold threshold equalled the validation optimum in every run.
- No dependency or license risk.

Trade-offs: Linear in its inputs (interactions must be listed explicitly). LightGBM stays a separate, evidence-gated experiment.

Status: Proposed (evidence: EXP-M-02, EXP-M-02b, EXP-F-01a/b)

---

## DEC-014 — Blocking v1 = sorted name_core key + leading-zero-safe address numbers (only with a learned matcher)

Date: 2026-09-25

Context: The EXP-B-02 sweep changed one thing at a time on the frozen validation split.

Decision:
- Enable `sorted_core` (recall@20 +0.050 for +4.3% candidates) and `strip_leading_zeros` (+0.016 for +0.4%). Together: 0.7983 → 0.8597.
- **Reject:** addr_num_word cap increases (+0.005 to +0.008 recall for +200k to +297k candidates), max_nums 3 and max_words 6 (lower recall@20), and the exact_core key (subsumed by sorted_core).
- The name_token pass (+0.9 points on top of bc but +321k candidates, 2% of them true) stays off.
- Caps remain global (no country-specific values: open-set rule).

Constraint: With the hand-weighted rule, Blocking v1 **lowers** precision (0.9226 → 0.8879; EXP-B-02), so it is only adopted together with DEC-013.

Status: Proposed

---

## DEC-015 — Conflict features: number conflict and legal-form conflict

Date: 2026-09-25

Decision:
- Add `num_conflict` (both addresses have numbers, none shared) and `legal_form_conflict` (canonical legal forms present on both sides and disjoint).
- Number comparisons use leading-zero stripping (`features.strip_leading_zeros`).
- The legal-form list is the existing generic list in `normalization/views.py` with 3 synonym merges (limited→ltd, incorporated→inc, corporation→corp). No new external resource.

Evidence:
- F0.5 +0.0087 then +0.0084.
- Precision +0.0040 then +0.0038.
- False merges −2,901 in total.

Status: Proposed

---

## DEC-012 — One conda env `ber_hackathon`; add rapidfuzz, PyYAML, psutil; Arrow system allocator

Date: 2026-09-25

Context: The Phase-1 brief asked for a single reproducible environment. Report §10 flagged `rapidfuzz` as needing approval.
On this machine, Arrow's default (mimalloc) pool kept freed memory. In one run this led to an access-violation crash near the Windows commit limit.

Decision:
- `environment.yml` (conda-forge, Python 3.13) plus pinned `requirements.txt`.
- Packages: numpy 2.2.4, pandas 2.3.3, pyarrow 21.0.0, **rapidfuzz 3.14.6 (MIT)**, PyYAML 6.0.3, pytest 8.4.2, psutil 7.2.2. The `ber` package is installed editable.
- No torch, LightGBM or scikit-learn is needed for Submission 1.
- Every entry point calls `configure_arrow()`, which switches Arrow to the system allocator.

Reason: rapidfuzz gives vectorised, multithreaded string similarity (`process.cpdist`), measured at about 60k pairs/s for the full feature set.
The system allocator cut the CSV-streaming peak from 0.99 GB to 0.61 GB (measured) and returns memory to the OS.

Trade-offs: The system allocator is slightly slower (+0.5 s per 500 MB file).

Addendum (2026-09-25, at the user's request): **optional GPU extra.** `torch 2.14.0+cu130` was installed into the same env via `requirements-gpu.txt`.
CUDA is verified on the RTX 3060 Laptop (driver 616.92) with a GPU matmul. The Submission-1 code does not use torch.
Any future GPU model must still be MIT/Apache-2.0 and ≤8B parameters, and only the models actually used go into the submission `requirements.txt`.

Status: Proposed (already applied in the Phase-1 code; reversible via config/code)

---

## DEC-010 — Memory-aware data engineering (pyarrow/Parquet, per-country shards, float32 features)

Date: 2026-09-25

Context: The dev machine has 16 GB RAM, and only ~2–3 GB was free during the audit. Profiling a single S2 file peaked at ~2 GB RSS.

Decision: Read the TSVs once and convert them to typed Parquet in `data/interim/`. Process per country and per S1 chunk. Store features as float32.
Never materialize a full pandas object-dtype S2/S3 frame. Sample S1 entities for model training (the pair budget is set in config).

Status: Proposed
