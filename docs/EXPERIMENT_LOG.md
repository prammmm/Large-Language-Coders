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
| EXP-F-01b (full) | Same system, full 200k train-fold fit | team full run | **0.8372** | **0.9319** | 0.7418 | 0.8597 | 16.1 | **0.815** (Submission 2) | Banked; see `reports/EXP-F-01b_SUBMISSION_2_REPORT.md` |
| **EXP-M-03** | A non-linear tree scorer extracts the interaction headroom the MLP probe found | LR → LightGBM (same blocking, same 22 inputs, same train-fold/inner protocol); **dev fit 50k S1** | **0.9021** (3 seeds 0.9021–0.9022) | **0.9779** | 0.8080 | 0.8597 | 16.1 | pending | **Strong candidate**; France guard required on full test |
| EXP-M-03 (full) | Same, full 200k train-fold fit | team full run | **0.9037** | **0.9790** | 0.8100 | 0.8597 | 16.1 | **0.889** (Submission 3) | Banked; see `reports/EXP-M-03_SUBMISSION_3_REPORT.md` |
| EXP-B-03a | Name-token blocking recovers true pairs and the strong LightGBM keeps precision | enable `name_token` (max_df 200); scorer = fixed EXP-M-03 model | 0.9027 (−0.0010) | 0.9735 ↓ | 0.8165 | 0.8686 | 17.6 | **0.890** (team check) | **Rejected** (coupling via cheap_* inputs; recovery alone +0.002) |
| **EXP-B-04** | Combined blocking keys + a model retrained on that blocker recover missed true pairs without losing precision | + `name_addr` + `addr_pair` passes; LightGBM refit on the new train-fold candidates (top-K 20) | **0.9274** | 0.9774 | **0.8579** | **0.9267** | 19.3 | **≈0.91** (Submission 5) | Banked; forensic analysis in `experiments/runs/POST-0.91-FORENSIC/REPORT.md` |
| **EXP-D-01** | Under full competition, keeping only the best claimant of each S2/S3 record removes more false merges than true matches | Decision layer only: best claimant per record after the threshold (margin 0); B-04 scores, model and threshold re-used; no retraining | **0.9313** (E, 2.0M out-of-sample S1s, all S1s compete; control 0.9273) | **0.9840** | 0.8562 | 0.9267 | 19.3 | pending | **Keep; ready for human review** |
| EXP-B-05 | A wider address-word window (3 → 5) for name_addr / addr_pair recovers missed India pairs; model retrained; D-01 on top | `max_words` 3 → 5 (config only) + LightGBM refit (threshold 0.62) | 0.9324 (E, full competition + D-01; control 0.9313) | 0.9813 ↓ | 0.8636 | 0.9285 | 19.6 | — | **Rejected** (+0.0010, of which the blocker is ~+0.0005; P −0.0027) |
| **EXP-N-02** | A cross-script name view learned only from train true pairs recovers and correctly scores India script-mismatch pairs | Train-learned token map → `name_xlit` view for the name keys + 2 similarity features; LightGBM refit (cross-fitted maps); D-01 on top | **0.9402** (E, full competition + D-01; control 0.9313) | **0.9864** | 0.8702 | 0.9328 | 19.3 | **0.928** (Submission 6) | **KEEP**: current best |
| EXP-N-03 | Indic legal-form words (learned) dropped from the cross-script view make generic cross-script names exact | legal-form stage of the learned map + refit; D-01 | 0.9413 (E; control N-02 + D-01 0.9402) | 0.9859 | 0.8735 | 0.9348 | 19.3 | — | Below gate (+0.0010); kept only as the base for B-06/B-07 |
| **EXP-B-07** | Exact name × address word finds generic names whose exact-name block exceeds the cap | `exact_core.with_addr_words` key + refit; threshold 0.67 (precision-first rule on the tuning set E half A minus validation) | **0.9464** (E; N-02 + D-01 0.9402) | 0.9864 | 0.8815 | 0.9448 | 19.3 | pending | **KEEP** |
| **EXP-B-06** | A name-token-pair key finds name-only records | `name_token.pair` key on B-07 + refit (t 0.67) | **0.9471** (E) | 0.9863 | 0.8832 | 0.9487 | 19.4 | pending | **KEEP**: best local (+0.0007 over B-07) |
| **EXP-S-01** | Pairs are decided in isolation; a second-stage model with S1/record context and COLLECTIVE record-to-record peer similarity decides better | 2nd-stage LightGBM on B-06 stage-1 pairs (score ≥ 0.02), trained on E half A minus validation only; decision threshold+D-01 @ 0.65 chosen on an inner split | **0.9596 (E half B)**, 0.9601 (validation); control B-06 0.9471 / 0.9470 | 0.9875 (E-B) | 0.9162 | 0.9487 (unchanged) | 19.4 | **≈0.920** (team-reported; best pre-S-01 file 0.935, B-07 t0.67 or B-06, unattributed) | **Public regression**: see POST-S01-FORENSIC |
| EXP-S-01b | Count/position context features may not transfer to the denser test pool | S-01 minus 7 count-dependent features | 0.9591 (E-B), 0.9595 (val) | 0.9875 | 0.9148 | 0.9487 | 19.4 | — | Validated alternative; not selected (pre-declared rule: test inflation not clearly lower) |
| EXP-C-01 / C-02 (diagnostic) | How much does an unseen country lose? | Stage 1 trained on one labelled country only (200k, size-controlled) | India (unseen) 0.8609 vs 0.9358; US (unseen) 0.9108 vs 0.9493 | India 0.883 / US 0.944 | — | same | — | — | Diagnostic: transfer loss is mostly precision |
| EXP-C-03 (diagnostic) | Does stage 2 help or hurt an unseen country? | Stage 2 trained on US only on top of C-01 | India (unseen) E-B 0.8724 → 0.8932 (+0.0208) | 0.8981 → 0.9133 | — | same | — | — | Diagnostic: stage 2 improves unseen-country precision |
| EXP-B-10 | No-key pairs share common name tokens; pairing the 3 rarest tokens (not 2) recovers name-only records | `name_token.max_tokens` 2 -> 3 on B-09 (config only) | blocking only: recall@20 0.9569 -> 0.9596 (+0.0028); uncapped +0.0043 | - | - | 0.9596 | 19.6 | not submitted | **REJECT at kill gate** (< +0.003; stopped before retrain; 1,441 newly generated true pairs rank beyond 20; `experiments/runs/EXP-B-10/final_report.md`) |
| EXP-B-11 | B-10's beyond-20 true pairs are held back by too little cheap-score weight; name_token.pair weight 2.0 lifts them into the top-20 | `name_token.weight` 1.0 -> 2.0 on the B-10 blocker (config only) | blocking only: recall@20 B-09 0.9569 / B-10 0.9596 / B-11 0.9584 (+0.0015 vs B-09, -0.0012 vs B-10); 742 / 1,441 targeted pairs enter top-20 but 2,305 address-evidence true pairs pushed out | - | - | 0.9584 | 19.6 | not submitted | **REJECT at kill gate** (KA +0.0015 < +0.004; KB displaced/new true 0.70 > 0.25; KD US +0.0003 with 0.94 displacement; stopped before retrain; `experiments/runs/EXP-B-11/final_report.md`) |
| EXP-D-04 | Macro F0.5 per S1 has an S1-dependent optimal cut; exact expected-F0.5 prefix pruning beats one global veto threshold | `stage2.ef_prune: true` on S-03 (removal-only, parameter-free; S-03 model/features read-only) | 0.9537 (val; S-03 0.9536), E half B 0.9537, GATE-B 0.9537 | +0.0005 | -0.0014 | 0.9569 (unchanged) | 19.5 | not submitted | **REJECT** (G1: +0.00006..+0.00009 << +0.001; removed pairs 70% true = stage 2 already calibrated at the cut; `experiments/runs/EXP-D-04/final_report.md`) |
| EXP-S-04 | B-10's extra candidates + a retrained stage 1 + a refitted S-03-style veto beat S-03 | B-10 blocker (3-token pair, w 1.0, K20) + fresh stage-1 fit (inner t 0.69) + fresh veto (inner t 0.67); no additions | 0.9528 (val; S-03 0.9536), GATE-B 0.9529 (0.9536), E half B 0.9529 (0.9536) | 0.9921 | 0.8868 | 0.9596 | 19.6 | not submitted | **REJECT** (G1-G3: -0.0007..-0.0008 vs +0.0005 gate; G4-G8 pass, 0 additions; +10k retrieved true pairs lost to a stricter inner-split stage-1 threshold 0.69; `experiments/runs/EXP-S-04/final_report.md`) |
| **EXP-U-01** | S-04's losses come from its stricter threshold on S-03's own candidates; adding only S-04 acceptances on candidates new to the B-10 blocker keeps S-03 and gains the retrieval | S-03 + S-04 final pairs outside the B-09 top-20 list (record not matched by S-03); no fit, no tuning | **0.9544** (val; S-03 0.9536), GATE-B 0.9544, E half B 0.9545 | 0.9919 | 0.8912 | union | 20.2 (test) | pending | **KEEP, ready for human review** (all pre-declared gates; +0.0008 incl. test-density weighted; added pairs 97.6% true; test additions 0.72x train; `experiments/runs/EXP-U-01/final_report.md`) |
| **EXP-U-02** | A third independent pipeline (B-08, top-30 retrained stage 1) sees candidates neither B-09 nor B-10 lists contain | U-01 + B-08 stage-1 + D-01 @ 0.68 acceptances on candidates in neither list (record unmatched); V1 (all remaining S-04 pairs) rejected on precision | +0.0013 over U-01 on validation / GATE-B / E half B (added pairs 97.5% true) | -0.00007 | - | union | - | **0.944 (final)** | **FINAL** (offline gates pass; pre-declared test gate G failed: France additions 2.5x overall; team chose a public check; public 0.943 -> 0.944) |
| **EXP-S-03** | B-09's precision loss is a decision-layer effect that a refitted veto removes | S-01/S-02 stage-2 recipe refitted on B-09 stage-1 outputs, veto only (final = B-09 + D-01 AND stage 2 @ 0.69 + D-01) | **0.9536** (val; B-09 0.9502, B-06 0.9470), E half B 0.9536, GATE-B 0.9536 | 0.9920 | 0.8889 | 0.9569 | 19.5 | **0.942** (team-reported; control 0.935) | **KEEP** (all gates A-P; false merges -49%; 0 additions; density-stable; `experiments/runs/EXP-S-03/final_report.md`) |
| EXP-B-09 | Pairs that share a COMMON name token plus a house number are missed by rarest-token keys | new key `name_num` (2 rarest name-view tokens x 2 rarest zero-stripped numbers, cap 50) + stage-1 refit (t 0.65) + D-01; control B-06 + D-01 | 0.9502 (val; 0.9470), E half B 0.9504 (0.9471), GATE-B 0.9503 | 0.9851 (-0.0012) | 0.8931 | 0.9569 | 19.5 | not submitted | **REJECT** (gates C precision -0.0011..-0.0012 and J weak acceptance +15.8% fail; +0.0032 F0.5, recall@20 +0.0082; `experiments/runs/EXP-B-09/final_report.md`) |
| EXP-B-08 | ~35% of never-retrieved true pairs sit at ranks > 20; a retrained stage 1 recovers them | `blocking.top_k` 20 -> 30 + stage-1 refit (inner threshold 0.68) + D-01; control B-06 + D-01 | 0.9489 (E half B; control 0.9471), 0.9489 (val; 0.9470), GATE-B 0.9489 | 0.9867 | 0.8865 | 0.9562 | 28.25 | not submitted | **REJECT** (pre-declared +0.002 gate: +0.0017..+0.0019; density-stable, precision up; `experiments/runs/EXP-B-08/final_report.md`) |
| **EXP-S-02** | S-01's removals transfer, its additions do not: a veto-only stage 2 gives a density-robust gain | final = (B-06 + D-01) AND S-01 (`stage2.add_floor: .inf`); no retraining; S-01 model/features read-only | **0.9499** (E half B; control 0.9471), 0.9499 (val), GATE-B 0.9499 | 0.9917 | 0.8800 | 0.9487 (unchanged) | 19.4 | not submitted | **KEEP** (all pre-declared gates; advantage +0.0023 to +0.0027 on every density tier incl. test density; `experiments/runs/EXP-S-02/final_report.md`) |
| POST-S01-FORENSIC (diagnostic) | Why did S-01 drop from local ≈ 0.96 to public ≈ 0.920? | No production change: submission check, B-06 → S-01 test flips, PSI shift, D-01 variants, a shadow benchmark (Base / Test-Matched / Chain-orphaned), a stage-1 floor sweep | Shadow-Base S-01 0.9596 vs B-06 0.9472; the S-01 advantage shrinks +0.0124 → +0.0079 as look-alike density rises 0.399 → 0.433 (test 0.706) | S-01 P 0.9875 → 0.9792 (Chain members) | — | same | — | — | Cause: context-driven additions on uncontested look-alike records (test 2.14× train rate). Proposed EXP-S-02 (veto only), not implemented |
| EXP-B-04-k30 | Same blocker, top-K 30 | K 20 → 30 (own refit) | 0.9289 (+0.0015) | 0.9770 | 0.8618 | 0.9339 | 27.8 | — | Not selected (+45% candidates for +0.0015) |
| EXP-D-03 (probe) | More trees help | 3,000 rounds (dev 50k) | 0.9027 (+0.0006) | 0.9778 | 0.8091 | 0.8597 | 16.1 | — | Rejected (capacity not a lever) |
| EXP-D-02 (probe) | A non-linear scorer beats LR on the same features/protocol | MLP 22→64→32→1 (diagnostic, torch) | 0.8870–0.8903 (3 seeds) | 0.968–0.973 | 0.788–0.791 | 0.8597 | 16.1 | — | Evidence for EXP-M-03 (not adopted) |

## Detailed entries

### EXP-S-03 — S-02-style veto refitted on EXP-B-09 stage 1 (2026-09-27)
- **Change (one):** stage 2 refitted on B-09 out-of-sample stage-1 scores (S-01 recipe unchanged; inner-split threshold 0.69 + D-01), used as a veto on B-09 + D-01. No code change (config only).
- **Result [M]:**
  - vs B-09: validation +0.0034, GATE-B +0.0033, E half B +0.0032;
  - precision +0.0068; false merges -49%; singleton FP about -47%; 0 additions;
  - shadow +0.0031 to +0.0035; tiers +0.0031 (T1) to +0.0026 / +0.0027 (T1W / T4W); locked +0.0029 / +0.0032;
  - India +0.0047, US +0.0024.
- **Veto [M]:** 1.17% of B-09 acceptances; 59% of vetoes remove false pairs; concentrated in the 0.65-0.8 stage-1 band (24% vetoed; precision 0.80 -> 0.93).
- **Test [D]:** -1.11% predictions; 0 additions; test/train removals 0.95; France 0.98x overall; candidates byte-identical to B-09.
- **Decision: KEEP** (not submitted). Cumulative: validation 0.9470 (B-06) -> 0.9536.
- **Open risk:** no decision-layer system has transferred successfully to the public board yet.

### EXP-B-09 — Name-view token x house-number blocking key (2026-09-27)
- **Change (one):** a `name_num` pass; stage-1 refit (inner threshold 0.65); D-01. Control: B-06 + D-01 @ 0.67.
- **Blocking [M]:** validation recall@20 +0.0082 (India +0.0138, US +0.0044); 20% of the no-key pairs recovered; candidates +0.7%; 480 true pairs displaced vs 6,746 entering.
- **End to end [M]:**
  - F0.5 +0.0032 on validation / GATE-B, +0.0033 on E half B;
  - shadow sets +0.0032 to +0.0043; density-stable;
  - LOCKED +0.0032 / +0.0034;
  - India +0.0050.
- **Cost [M/D]:** precision -0.0011 to -0.0012; false merges +8% to +9%; test weak acceptance (score < 0.8) +15.8% (France +18.6%).
- **Decomposition [M]:** the list change alone gives +0.0029 (its new acceptances are 97% precise); re-scoring at threshold 0.65 adds a net 2,906 false pairs.
- **Decision: REJECT** (pre-declared gates C and J). Not submitted.
- **Environment:** the env's numpy was replaced by the byte-identical 2.2.4 wheel; the reference reproduced exactly; `PYTHONNOUSERSITE=1` from now on.

### EXP-B-08 — Candidate list top-K 20 -> 30, stage 1 retrained (2026-09-27)
- **Change (one):** `blocking.top_k` 30; LightGBM refit on B-08 candidates (inner threshold 0.68) + D-01. Control: B-06 + D-01 @ 0.67.
- **Result [M]:**
  - validation +0.0019, GATE-B +0.0018, E half B +0.0018, halves +0.0020 / +0.0018;
  - SHADOW-BASE +0.0017, TEST-MATCHED +0.0018, CHAIN-ORPHANED +0.0017, CHAIN-MEMBERS +0.0031;
  - locked sets +0.0015 / +0.0016;
  - density tiers +0.0015 to +0.0018 with no decay;
  - precision up, false merges down, India and US up;
  - blocking recall 0.9487 -> 0.9562.
- **Mechanism [M]** (E half B): ranks 21-30 add 8.87M candidates, 26,307 of them true. The model accepts 17,663 of them (17,246 true, 417 false), worth +0.0019; re-scoring of ranks 1-20 is -0.0002.
- **Test behaviour [D]:** +0.48% predictions (France +1.45%, carrying 32% of the new-rank acceptances; India +0.51%; US +0.06%); candidates per S1 19.3 -> 28.0; the B-06 list is the exact prefix in every row.
- **Decision: REJECT.** The pre-declared +0.002 gate fails on 6 of the 7 validation / GATE-B / E half B / shadow gate populations; everything else passes. Not submitted.
- **Next:** never-retrieved is still the largest bucket (4.38% of true pairs, oracle +0.0168); about 76% of it is generated by no key. A key-level forensic is proposed (not run: the brief made it conditional on a pass).

### EXP-S-02 — Stage 2 as a veto only + shadow benchmark v2 (2026-09-27)
- **Hypothesis:** S-01's removals are pair-level corrections that transfer; its additions are what break under the test's look-alike regime.
- **Change (one):** final = control (B-06 + D-01 @ 0.67) AND S-01. No retraining; S-01 model, features and threshold reused read-only.
- **Local result [M]:**
  - validation 0.9470 → **0.9499**, GATE-B 0.9471 → 0.9499, E half B 0.9471 → 0.9499;
  - precision 0.986 → 0.992; false merges −43%; singleton FP −37%;
  - India +0.004, US +0.002; 0 additions.
- **Shadow benchmark v2** (DEV / LOCKED-A / LOCKED-B + density tiers):
  - Orphaning reaches look-alike density only 0.462 (test 0.706). The test excess is in the candidate lists (1.5× weak candidates, 1.2× peer-similar), not in claimants (contested share 15.9% vs 17.1%).
  - Weighted tiers T1W / T4W reach 0.706.
  - S-02's advantage: +0.0027 (T1) → +0.0025 (T4) → +0.0024 / +0.0023 (T1W / T4W), with no material decay.
  - S-01's advantage: +0.0125 → −0.0002 at T4 (the tiers reproduce the public inversion), with false merges 36.8 → 120.4 per 1k.
- **Test behaviour [D]:** −0.86% predictions (France −0.84%, India −1.13%, US −0.53%); 0 additions; test/train removals 0.94.
- **Engineering:**
  - Submit memory breach (5.15 GB) found by a trace and fixed (3.59 GB), output byte-identical.
  - `add_floor=null` reproduces the S-01 bytes on real data.
  - 191 tests.
- **Decision: KEEP** (not submitted).
- **Next (proposed, not implemented):** EXP-B-08, top-K 20 → 30. Largest post-S-02 bucket: never-retrieved true pairs, oracle +0.0198; about 35% of them sit at ranks > 20.

### POST-S01-FORENSIC — Public regression of EXP-S-01 (diagnostic, 2026-09-27)
- **Question:** why S-01 (E half B 0.9596, validation 0.9601) scored ≈ 0.920 publicly, vs 0.935 for the best pre-S-01 file.
- **Full report:** `experiments/runs/POST-S01-FORENSIC/forensic_report.md`. Shadow benchmark: `experiments/shadow_tests/`.
- **Measured:**
  - The submitted files are byte-identical to the validated S-01 output.
  - Test predictions per S1 rose +9.3% (train E-B +3.9%). Test additions were 0.316 per S1 vs 0.148 on train (2.14×); removals were 0.027 vs 0.028 (1:1).
  - Test additions have stage-1 median 0.300 (train 0.427) and are on uncontested records (mean claimants 1.04).
  - Only `s1_n_cand` shifts moderately (PSI 0.118).
  - D-01 variants A–D are identical in count.
  - France has the smallest rise.
  - Test has 1.77× more weak, uncontested, peer-similar candidates per S1 (0.706 vs 0.399; train truth rate in that group 16.6%).
  - Shadow sets: S-01's advantage over B-06 falls monotonically with look-alike density (+0.0124 → +0.0079), and S-01's false merges rise faster (36.9 → 65.8 vs 39.8 → 44.8 per 1k S1). No simulation reaches the test's density.
  - A stage-1 floor sweep: every addition floor keeps a negative density slope. **Removals only (veto)** is density-invariant (+0.0027 / +0.0026 / +0.0027).
- **Interpretation:** the regression is a joint concept shift (outcome G, with parts of A and C). S-01 turns weak pair evidence into matches using "uncontested + looks like the S1's other records". Test contains far more look-alike distractor records, so that evidence is much less reliable there.
- **Decision:** no production change and no submission. Proposed EXP-S-02 (S-01 as a veto on B-06 + D-01, `stage2.add_floor: .inf`), not implemented and awaiting approval.

### EXP-S-01 — Collective second-stage decision model (FINAL candidate)
- **Hypothesis:** After B-06, most remaining loss is in the decision (retrieved true pairs scored low: +0.023 oracle; false merges +0.016), because the stage-1 matcher scores each (S1, record) pair in isolation. A second stage that sees the S1's other candidates, the record's other claimants, and whether the record resembles the S1's other strong records decides better.
- **Control:** EXP-B-06 + D-01 @ 0.67 (verified: E 0.94706, P 0.98630).
- **Change (one):** a second LightGBM re-decides every retrieved pair with stage-1 score ≥ 0.02 using 41 label-free, country-agnostic features:
  - stage-1 score;
  - S1 context: rank, count, relative position, best, 2nd best, gap, # ≥ 0.5, # ≥ 0.8;
  - record context: claimants, best/2nd best claimant, rank, gap to the best other claimant, is-best;
  - collective: for the top-3 OTHER candidates of the same S1 (never itself), record-to-record name Jaccard, cross-script name Jaccard, name token-set, address token-set, number Jaccard (zero-stripped), useful-word Jaccard, and the peer's score; plus peer count, max, score-weighted mean and best-peer similarity.
- **Unchanged:** candidates and stage 1. Final decision = stage-2 probability ≥ 0.65, then best claimant per record (D-01) — both chosen on an inner split of the training partition.
- **Training (no leakage, guards raise):** E half A minus frozen validation only (714,270 train S1 + 178,824 inner S1). Stage-1 scores out-of-sample. All train S1 compete when features are built.
- **Result (held-out):**
  - E half B 0.9471 → **0.9596** (+0.0125), P 0.9863 → 0.9875, R 0.8832 → 0.9162;
  - validation 0.9470 → **0.9601** (+0.0131); validation halves +0.0133 / +0.0128;
  - India +0.0134, US +0.0119 (E-B);
  - false merges −8.5%, singleton FP −18% (E-B);
  - 148k pairs added (90.6% true), 28k removed (38.6% true).
- **Robustness:**
  - the test feature distribution shifts (denser pool);
  - test acceptance rises 9.3% vs control (4.0% expected from train within-bin behaviour) — unexplained excess +5.3%, not caused by count features (S-01b removes only 8% of it);
  - the leave-one-country-out stage-2 diagnostic (C-03) shows stage 2 improves an UNSEEN country by +0.021 (precision +0.015).
- **Decision:** **KEEP — final system.** Output `experiments/runs/EXP-S-01/output/` (strict + official `--check-ids` PASS; deterministic re-run byte-identical).
- **Public (team-reported): ≈ 0.920**, below the best pre-S-01 file (0.935). Diagnosed in POST-S01-FORENSIC (above). The "+5.3% unexplained excess" noted under Robustness was the warning sign.

### EXP-C-01 / C-02 / C-03 — Leave-one-country-out diagnostics
- **C-01 (US-only stage 1, 200k):** India val 0.8609 (P 0.8830) vs B-06 0.9358 (P 0.9773); false merges ×5.5; at the validation-optimal threshold only 0.8794.
- **C-02 (India-only):** US val 0.9108 (P 0.9440) vs 0.9493 (P 0.9843).
- **C-03 (stage 2 on top of C-01, trained on US only):** India E-B 0.8724 → 0.8932, P 0.8981 → 0.9133; US +0.0101.
- **Interpretation:** an unseen country loses 0.04–0.075 F0.5, mostly precision. This is consistent with the France estimate. The US-only model over-relies on `name_ratio × addr_token_set` (56% gain), whose meaning changes with address length; `cheap_score` is not the cause. Stage 2's relative/collective features reduce the loss.
- **Label-free France check:** French S1 write "saint" (10.0%) where 1.7% of French pool records write "st", which normalizes to "street". Bounded; no production change (no French labels, no country rules).

### EXP-B-07 / EXP-B-06 — Name-driven retrieval keys (on N-03, with D-01)
- **B-07 hypothesis:** N-03's unreachable cross-script pairs have identical but generic names (the exact-name block exceeds cap 50) and share the city. A key on (exact sorted name view × each useful address word) makes the block small again.
- **B-07 blocking:** recall@20 0.9348 → 0.9448 (India +0.0237); unreachable cross-script pairs 7,390 → 516; candidates +0.3%.
- **B-07 threshold:** with D-01 the curve is flat (0.60–0.67 → 0.9464–0.9468). A precision-first rule on the tuning set (E half A minus validation; the lowest threshold with tuning precision ≥ the control's) picks 0.67. Held-out: E half B 0.9465 / P 0.9863; validation 0.9465 / P 0.9864.
- **B-07 result (E, D-01, t 0.67):** 0.9402 → **0.9464**; P 0.9864 → 0.9864; India +0.0149; singleton FP −49; halves agree.
- **B-06 hypothesis:** name-only records (empty or useless addresses) need a name-pair key.
- **B-06 blocking:** recall@20 0.9448 → 0.9487 (US +0.0029).
- **B-06 result (E, D-01, t 0.67 from the inner split):** **0.9471** (P 0.9863); +0.0068 vs N-02 (halves +0.0066 to +0.0068); +0.0007 vs B-07.
- **Decision:** both KEEP (gate vs N-02). B-06 is the best local system. Outputs:
  - `experiments/runs/EXP-B-07/output_t067/`;
  - `experiments/runs/EXP-B-06/output/`.
  Both pass strict and official `--check-ids` validation.
- **Probes (diagnostic, held-out):** a second-stage decision model on B-07 scores gives +0.0017 with context features only, and **+0.0048 with collective (record-to-record) features** (20% sample). This leads to the proposed EXP-S-01 (`reports/RESEARCH_REVIEW_2026-09-26.md`).

### EXP-N-03 — Learned Indic legal-form words dropped from the cross-script view
- **Hypothesis:** Unmapped Indic legal-form words ("private limited") keep cross-script names from matching the S1 core exactly (only 12% did).
- **Change vs N-02:** a legal-form stage in the train-learned map. Tokens co-occurring with S1 legal forms in ≥ 20 distinct businesses at ≥ 0.8 confidence, and not explained by a content word, are dropped from `name_xlit`. LightGBM refit.
- **Map:** 31 legal tokens in 9 scripts. Exact-core equality 12% → 95.5%; Jaccard 0.586 → 0.973 (validation, measurement only).
- **Result (E, full competition + D-01):**
  - F0.5 0.9402 → 0.9413 (+0.0010); halves +0.0007 to +0.0010;
  - P −0.0005, R +0.0033;
  - India +0.0023, US +0.0001;
  - blocking recall@20 +0.0020.
- **Diagnosis:** the 7,390 still-unreachable cross-script pairs have identical names but generic ones (the name block exceeds cap 50); the shared address word is the city. This leads to EXP-B-07.
- **Decision:** Below the +0.003 gate, so no standalone submission. Kept as the base for B-06/B-07; the combined candidate is gated against N-02.
- **Artefacts:** `experiments/runs/EXP-N-03/` (REPORT.md, comparison JSON, `output/` validated); `configs/exp_n03_legal_tokens.yaml`; `tests/test_xlit.py` (+4 tests).

### EXP-N-02 — Train-pair-learned cross-script name view (transliteration)
- **Hypothesis:** Script mismatch (Latin S1 vs Indic-script pool names) is India's largest remaining loss. A token correspondence learned from provided train true pairs bridges it for retrieval and scoring, without external resources.
- **Change vs control (B-04 + D-01):**
  - map learned by co-occurrence over distinct S1 entities (support ≥ 3, confidence ≥ 0.5, strict winner, ties rejected);
  - `name_xlit` view (= `name_core` when nothing is mapped) used by the name-token blocking keys (`blocking.name_view`);
  - 2 features (`xlit_core_jaccard`, `xlit_token_set`);
  - LightGBM retrained from scratch.
- **Leakage design:**
  - final map from the 200k fit S1s only (validation raises);
  - training rows use cross-fitted out-of-fold maps;
  - fold 0 = the inner-tune split, which contributes to no map.
- **Fair play:** provided train pairs only. No external dictionary, package, pretrained model or hard-coded script rule. No new dependency.
- **Map:** 1,262 mappings (42 low support, 31 low confidence, 6 ambiguous rejected). Validation name Jaccard on cross-script pairs 0.021 → 0.586 (measurement only).
- **Blocking (validation):**
  - recall@20 0.9267 → 0.9328 (India 0.8816 → 0.8968, US identical); candidates unchanged;
  - +5,150 true pairs enter top-20 / 508 pushed out; 518 lost to block-cap growth.
- **Model:** inner-split threshold 0.67 (unchanged procedure); inner-tune F0.5 0.9384. Validation without competition 0.9376 (B-04 0.9274).
- **Result (E = 2,006,821 out-of-sample S1s, all train S1s compete):**
  - F0.5 0.9313 → **0.9402 (+0.0089)**; P 0.9840 → 0.9864; R 0.8562 → 0.8702;
  - false merges −13%, singleton FP −15%;
  - India 0.9025 → 0.9235, US 0.9506 → 0.9514;
  - halves +0.0089 to +0.0091.
- **Cross-script (E):**
  - true cross-script pairs predicted 296,580 → 394,719;
  - cross-script false merges 11,315 → 3,244.
- **D-01** still helps (+0.0028); conflicts −33%.
- **Test (unlabelled):** India predictions/S1 2.88 → 2.98; India conflicts 21,809 → 10,974; France and US unchanged. Output validated (official `--check-ids` PASS).
- **Interpretation:** The largest single gain since LightGBM. It improves both recall and precision, because the model now sees name agreement for cross-script pairs instead of relying on address evidence alone.
- **Decision:** **KEEP** (all 12 gate criteria pass). Ready for human review; not submitted.
- **Artefacts:** `experiments/runs/EXP-N-02/` (REPORT.md, `xlit_map.json`, comparison JSONs, `output/`); `configs/exp_n02_transliteration.yaml`; `tests/test_xlit.py`; DEC-019.

### EXP-B-05 — Wider address-word window (3 → 5) + retrained LightGBM, with D-01
- **Hypothesis:** Many India never-retrieved pairs share address words outside the rarest-3 selection. A 5-word window recovers them without uncontrolled candidate growth.
- **Change vs control (EXP-B-04 + D-01):**
  - `addr_pair.max_words` and `name_addr.max_words` 3 → 5 (config only);
  - LightGBM retrained on B-05 train-fold candidates (same 200k S1s, same parameters, inner-split threshold → 0.62).
- **Expected effect:** Recall up, especially India; precision protected by D-01.
- **Blocking (validation):**
  - recall@20 0.9267 → 0.9285 (India 0.8816 → 0.8863, US 0.9568 → 0.9566);
  - uncapped 0.9450 → 0.9488;
  - candidates +1.9%.
- **Pair fates:**
  - 3,664 recovered, 761 lost (block cap 50 now drops keys the wider window made too common);
  - top-20: 4,391 in / 3,024 out.
- **Result (E = 2,006,821 out-of-sample S1s, all train S1s compete):**
  - F0.5 0.9313 → 0.9324 (+0.0010); halves +0.0010 to +0.0012;
  - P 0.9840 → 0.9813, R 0.8562 → 0.8636;
  - false merges +18.8%, singleton FP +1,153;
  - India +0.0021, US +0.0003.
- **Attribution (diagnostic):** at a matched threshold the blocker effect is +0.0005. The rest comes from the lower retrained threshold, which lifts B-04 + D-01 equally.
- **Interpretation:** The effect is real but small. Block-cap growth and top-K displacement cancel most of the recovery. Only 146 new-candidate false merges appear, so there is no new hard-negative pattern. Script mismatch remains 42% of India's unreachable pairs.
- **Decision:** **Rejected** (not meaningful: +0.0010 < +0.003; precision −0.0027). B-04 + D-01 stays best.
- **Next (proposed):** EXP-N-02, train-learned transliteration bridge. Needs the DEC-009 fair-play confirmation.
- **Artefacts:** `experiments/runs/EXP-B-05/` (REPORT.md, sweep, forensics, comparison JSONs); `configs/exp_b05_addr_window5.yaml`; `tests/test_blocking_b05.py`.

### EXP-D-01 — Full-competition best-claimant resolution (decision layer only)
- **Hypothesis:** When several S1s predict the same S2/S3 record, keeping only the highest-scoring claimant removes more false merges than true matches, provided all S1s compete as on test.
- **Basis:**
  - [O] official: S1 is the *deduplicated* reference source.
  - [M] train observation: every matched S2/S3 id belongs to exactly one S1.
  - Per-record exclusivity is **not** an explicit challenge rule.
- **Change vs EXP-B-04:** after the 0.67 threshold, group the predicted pairs by pool record over all S1s, keep the top score (tie → lowest S1 row) and drop the rest. Margin 0, nothing fitted.
- **Unchanged:** candidates, 22 features, LightGBM model, threshold, blocker, normalization. Everything is re-used read-only; no retraining.
- **Expected effect:** precision up, recall slightly down, F0.5 up ≈ +0.004 (forensic benchmark).
- **Validation design:**
  - all 2,206,821 train S1s scored with B-04 compete;
  - evaluated on E = 2,006,821 non-fit S1s (the 200k fit S1s are competitors only);
  - also the frozen validation fold, its halves, and E halves;
  - control reproduces production validation to 9e-7.
- **Actual result on E:**
  - F0.5 0.9273 → **0.9313** (+0.0040);
  - P 0.9775 → 0.9840;
  - R 0.8576 → 0.8562;
  - false merges 129,023 → 90,576;
  - singleton FP 9,102 → 6,565;
  - India 0.8965 → 0.9025, US 0.9479 → 0.9506.
- **Stability:** validation halves +0.0040 / +0.0039; E halves +0.0040 / +0.0040.
- **Sensitivity:**
  - out-of-sample competitors only: +0.0037;
  - margins 0.05 / 0.10 / 0.20 / 0.30 give 0.9301 / 0.9294 / 0.9284 / 0.9275 (monotonically worse; not tuned).
- **Removed claims (E):** 47,098, of which 81.6% false.
  - Singleton losers: 100% false.
  - True removals concentrate in close contests (78% have margin < 0.05).
- **Test (unlabelled diagnostics):**
  - −49,954 predictions (France −2.62%, India −1.07%, US −0.15%);
  - conflicted records 41,560 → 0;
  - candidates byte-identical to B-04;
  - strict and official (`--check-ids`) validators PASS.
- **Interpretation:** the gain is real, test-like, stable, and comes almost entirely from precision. The validation-only fold hid it because about 90% of competitors were missing.
- **Decision:** **Keep** (all criteria met). Ready for human review; not submitted.
- **Artefacts:** `experiments/runs/EXP-D-01/` (REPORT.md, `full_competition_eval.json`, `output/`, `output_invariants.json`); `configs/exp_d01_best_claimant.yaml`; DEC-018.

### EXP-B-04 — Blocking V2 (combined keys) + LightGBM retrained on the same blocker
- Hypothesis: Combined name/address and address/address keys recover the missed true pairs (78% of misses share such evidence), and a model retrained on that blocker keeps precision.
- Changes vs EXP-M-03:
  - two new blocking passes: `name_addr` (2 rarest name_core tokens × 3 rarest useful address words) and `addr_pair` (unordered pairs of the 3 rarest useful address words);
  - both use pool DF ≥ 1 words only and are capped at max_block_size 50;
  - the train-fold candidates are re-generated with this blocker and LightGBM is **retrained** (identical hyper-parameters, the same seeded 200k train-fold S1, the same inner split).
  - Config `configs/exp_b04_blocking_v2.yaml`.
- Blocking-only sweep (frozen validation, recall@20; `experiments/runs/EXP-B-04/blocking_sweep.md`):

| Variant | Recall@20 | India / US | Cand/S1 | New true pairs | Extra candidates | Runtime / peak |
|---|---|---|---|---|---|---|
| A (current) | 0.8597 | 0.831 / 0.879 | 16.13 | — | — | 126 s / 1.58 GB |
| B (+ name×address) | 0.9152 | 0.866 / 0.948 | 18.73 | +42,380 | +575k | 228 s / 2.49 GB |
| C (+ address pair) | 0.9017 | 0.863 / 0.928 | 18.10 | +32,104 | +434k | 193 s / 1.92 GB |
| **D (A+B+C)** | **0.9267** | **0.882 / 0.957** | 19.25 | **+51,144** | +689k | 225 s / 2.53 GB |
| D @ K=30 | 0.9339 | 0.892 / 0.962 | 27.84 | | | |

- Retrain: 3,083,070 training pairs (514,475 positive); early stopping on 769,674 inner pairs (did not trigger, 1,000 rounds); 103 s. Threshold 0.67 (inner train-fold F0.5 0.9284).
  - Leakage: 0 validation S1 in the training rows (code-checked); the model metadata records the exact blocker.
- **Validation (frozen, 220,682 S1):**
  - F0.5 **0.9274** (+0.0237 vs 0.9037); P 0.9774 (−0.0016); R 0.8579 (+0.0479).
  - Oracle 0.9726 (+0.031).
  - False merges 14,134 (+1,706); singleton FP 1,012 (+86); empty-on-positive 3,986 (−2,984); pred/S1 3.03.
  - India 0.8963 / 0.9676 / 0.8075 (blocking recall 0.882); US 0.9481 / 0.9839 / 0.8916 (0.957).
  - **Halves 0.9271 / 0.9277.**
  - Validation run: 378 s, peak 2.47 GB. Fit: 452 s, peak 2.53 GB.
- Match-count buckets (F0.5; blocking recall): n=1 0.794 → **0.830** (0.867 → 0.928; S1s whose only true record is not a candidate 1,587 → 854); 2–3 0.896 → 0.923; 4–5 0.919 → 0.942; 6+ 0.925 → 0.948; singletons 0.925 → 0.918.
- Threshold table: flat 0.9259–0.9274 over t 0.55–0.75. The threshold is not the lever.
- Attribution (`candidate_comparison.json`):
  - 1.62M new candidates (97.5% rejected); accepted: **38,413 true vs 2,008 false**;
  - 932,804 old candidates displaced by top-20 (1,560 accepted TPs lost);
  - common-pair flips roughly cancel (+4,206 / −3,793 true, +1,950 / −2,122 false). Retraining removed the EXP-B-03a coupling problem.
  - Recovered by `name_addr` only 10,306; `addr_pair` only 6,010; both 22,097. India 12,792 / US 25,621.
- K=30 variant (own refit): F0.5 0.9289; halves 0.9287 / 0.9292. +0.0015 for +45% candidates and a 3.18 GB fit peak → not selected.
- Unlabeled test diagnostics (2% sample only, same threshold): predicted/S1 France 0.182 → 0.156, India 0.071 → 0.073, US 0.060 → 0.062; conflicting claims 186 → 99. Not an accuracy measurement.
- Remaining error budget: not blocked +0.032 (script mismatch: India +0.030); low-scored true +0.029; false merges +0.017.
- Decision: **implemented and validated; READY FOR HUMAN REVIEW.** No test inference, no submission.

### EXP-B-03a — Name-token blocking with the fixed LightGBM (rejected)
- Hypothesis: The name-token pass recovers blocking misses that share a name word, and LightGBM (unlike the old rule) rejects the extra hard negatives.
- Change (one): `blocking.passes.name_token.enabled: true`, max_df 200 (token must occur in ≤ 200 pool records of the country; the same value caps block size).
  - Config `configs/exp_b03a_name_token.yaml`.
  - Scorer: the exact EXP-M-03 model file and threshold (0.67); no refit.
- Data: frozen validation (220,682 S1), re-blocked. Real validation measurement, not a sample.
- Results (EXP-M-03 → EXP-B-03a):
  - F0.5 0.9037 → **0.9027**; P 0.9790 → 0.9735; R 0.8100 → 0.8165.
  - Blocking recall 0.8597 → 0.8686; oracle 0.9413 → 0.9450.
  - False merges 12,428 → 16,311; singleton FP 926 → 1,111; empty-on-positive 6,970 → 6,659; pred/S1 2.85 → 2.89.
  - Candidates/S1 16.13 → 17.58 (3.56M → 3.88M, +9%; median, p95 and p99 all 20; max 20).
  - India F0.5 / P / R 0.8718 / 0.9698 / 0.7686 → 0.8722 / 0.9658 / 0.7759 (blocking recall 0.831 → 0.839).
  - US 0.9251 / 0.9849 / 0.8376 → 0.9230 / 0.9785 / 0.8437 (0.879 → 0.888).
  - Halves 0.9026 / 0.9027 (base 0.9037 / 0.9038).
  - Validation run: 234 s, peak 1.79 GB; block-eval 122 s, 1.69 GB.
- Where the change comes from (`experiments/runs/EXP-B-03a/candidate_comparison.json`):
  - New-only candidates: 391,352; the model rejects 98.9% of them.
    - Accepted: **3,932 true vs 182 false** (95.6% correct).
    - 3,232 new true candidates are still rejected.
    - Recovered true pairs: 76% share ≥ 2 name tokens, 84% have strong address overlap, 63% were found only by the name-token key; 45% India / 55% US.
  - New-only false candidates: 84% share a single token, 89% weak address, 81% number conflict. These are the hard negatives the model rejects.
  - Displaced by the top-20 cap: 70,443 candidates (287 accepted TPs lost).
  - **Common pairs whose prediction flipped:** +3,994 false and +2,267 true accepted; 777 true and 283 false rejected.
    - The new keys raise `cheap_score` and change `cheap_rank`. Both are model inputs, and the fixed model was trained on the old distribution.
  - Counterfactual (only new candidates, common pairs keep the EXP-M-03 prediction): **F0.5 0.9057 (+0.0020)**, P 0.9788, false merges +172. India +0.0025, US +0.0016.
- Interpretation:
  - (1) The name-token candidates themselves are safe with LightGBM.
  - (2) The blocker and model are coupled through the blocking-derived inputs, so a fixed model cannot absorb a blocking change.
  - (3) The pass reaches only about 7k of the ~60k "shares a name token" misses: most share only common words (DF > 200). Its ceiling is therefore about +0.002, far below the +0.040 bucket upper bound.
- Decision: **reject** as configured. The retrained variant is recorded as a possible follow-up; its expected gain is small (≈ +0.002).
- Leaderboard (team check submission): **0.890** vs 0.889 for EXP-M-03. That's noise-level and contradicts the validation result (−0.0010 on both halves) and the test consistency signal: certainly-wrong test predictions France 51,859 / India 34,589 / US 11,965 vs 49,602 / 26,256 / 2,536 (`experiments/runs/EXP-B-03a/test_conflicts.json`). The final selection should prefer EXP-M-03.
- Blocking-gap probe (`scripts/analysis/blocking_gap_probe.py`, EXP-M-03 candidates): of 107,130 missing true pairs, 19.7% share a rare name token, 55.0% share a name token + address word, 68.1% share ≥ 2 address words, 78.1% are reachable by either combined family (upper bound, no caps), and 0% share nothing.

### EXP-M-03 (full run) — Submission 3
- Fit: 200,000 train-fold S1 (EXP-F-01b parts re-used; 0 validation overlap) → 2,580,936 pairs. 1,000 rounds (early stopping not triggered); 71 s; threshold 0.67 (inner F0.5 0.9036).
- Validation: **F0.5 0.9037**, P 0.9790, R 0.8100; false merges 12,428; singleton FP 926; empty-on-positive 6,970; pred/S1 2.85.
  - India 0.8718 (P 0.970, R 0.769); US 0.9251 (P 0.985, R 0.838).
  - Halves 0.9037 / 0.9038.
- Test: 4,974,452 matched ids; validators PASS. France guard ratio 1.063 (India 1.030, US 0.994).
- **Public LB 0.889** (Submission 3). LB − validation = −0.015. India+US reweighted 0.896; France implied ≈0.845 (*estimate*).
- Remaining validation loss 0.096, ≈0.055 of it from blocking. Error budget: not blocked +0.065 (name-token +0.040; script mismatch, India +0.043); low-scored +0.023; false merges +0.015.
- **Test conflict lower bound** (`scripts/analysis/test_conflicts.py`): certainly-wrong predictions France 49,602 (6.2%; LR 2.3%), India 26,256 (1.2%), US 2,536 (0.13%). Invisible on validation (only 10% of train S1s compete).
- France pattern probe (city-in-name, same number on another street): rejected as the France driver.

### EXP-D-03 — Capacity probe (diagnostic)
- Change: `lightgbm.num_boost_round` 1,000 → 3,000 (dev 50k).
- Result: early stopping at 1,558; F0.5 0.9027 vs 0.9021. Not a lever.

### EXP-M-03 — LightGBM matcher (scorer-only change vs EXP-F-01b)
- Hypothesis: A non-linear tree scorer models conditional interactions (name × address × number × legal form) that the linear LR cannot.
- Change: `matcher.type: lightgbm`. Everything else is identical to EXP-F-01b:
  - blocking, top-20, and the same 22 inputs;
  - the same seeded 200k train-fold S1 parts (re-used read-only; the S1 set and zero validation overlap are verified in code);
  - the same inner train-fold split for early stopping and the threshold.
- Model: LightGBM 4.7.0 CPU.
  - Params: lr 0.05, num_leaves 31, max_depth 8, min_data_in_leaf 100, feature_fraction 0.9, bagging 0.8/1, λ2 1.0, deterministic, 8 threads.
  - Up to 1,000 rounds with early stopping 50; **early stopping never triggered**.
  - Config: `configs/exp_m03_lgbm.yaml`.
- Data (**dev scale**): a seeded 50,000-S1 subsample of the train-fold parts (516k inner-train pairs); fit 14–15 s, peak RSS 1.06 GB.
  - Evaluated on the FULL frozen validation split with the EXP-F-01b validation candidates/features.
  - Same-data LR baseline: LR refit on the identical 50k subsample.

| Run | t (train-fold) | F0.5 | P | R | False merges | Singleton FP | Pred/S1 | India F0.5 / P / R | US F0.5 / P / R |
|---|---|---|---|---|---|---|---|---|---|
| EXP-F-01b LR (full 200k, Submission 2) | 0.60 | 0.8372 | 0.9319 | 0.7418 | 36,952 | 3,223 | 2.72 | 0.779 / 0.909 / 0.676 | 0.876 / 0.946 / 0.786 |
| LR, same 50k subsample | 0.61 | 0.8375 | 0.9336 | 0.7398 | 35,762 | 3,173 | 2.71 | 0.779 / 0.910 / 0.675 | 0.876 / 0.948 / 0.783 |
| **LightGBM seed 42** | 0.68 | **0.9021** | **0.9779** | **0.8080** | **12,990** | **1,007** | 2.85 | **0.869 / 0.968 / 0.765** | **0.924 / 0.984 / 0.837** |
| LightGBM seed 7 | 0.69 | 0.9022 | 0.9787 | 0.8068 | 12,426 | 989 | 2.84 | 0.869 / 0.969 / 0.764 | 0.924 / 0.985 / 0.835 |
| LightGBM seed 123 | 0.66 | 0.9021 | 0.9764 | 0.8099 | 13,947 | 1,066 | 2.86 | 0.869 / 0.966 / 0.768 | 0.924 / 0.983 / 0.838 |

- Validation halves A/B (F0.5): LR 0.8375 / 0.8369; LightGBM 0.9020 / 0.9023 (all three seeds within 0.0003).
- The inner train-fold F0.5 (0.9037) predicted validation (0.9021).
- Blocking recall 0.8597 and oracle 0.9413 are unchanged (scorer-only).
- Empty-on-positive: 7,014 vs 9,910.
- True matches found: 615,457 vs 563,994 (+51,463); missed: 148,099 vs 199,562.
- LightGBM threshold table (seed 42; the train-fold choice is 0.68):

| t | F0.5 | P | R | Singleton FP | Pred/S1 | False merges |
|---|---|---|---|---|---|---|
| 0.50 | 0.8995 | 0.9635 | 0.8247 | 1,604 | 2.95 | 22,655 |
| 0.60 | 0.9018 | 0.9721 | 0.8161 | 1,256 | 2.89 | 16,848 |
| **0.68** | **0.9021** | **0.9779** | 0.8080 | 1,007 | 2.85 | 12,990 |
| 0.75 | 0.9012 | 0.9826 | 0.7989 | 786 | 2.80 | 10,044 |
| 0.85 | 0.8960 | 0.9887 | 0.7793 | 493 | 2.72 | 6,210 |

- Feature importance (gain share; association, not causation):
  - `name_ratio×addr_token_set` 0.67
  - `name_core_jaccard×addr_num_jaccard` 0.05
  - `addr_token_set` 0.05
  - `cheap_score` 0.04
  - `addr_num_jaccard` 0.035
  - `addr_jaccard` 0.03
- **France guard (test sample, no labels):**
  - On the 2% real-data sample, the expected true-match rate per S1 is ≈0.07–0.08. India and US predictions match it for both models (LightGBM is slightly lower: India 0.071 vs 0.074, US 0.061 vs 0.070).
  - **France is over-predicted by both models, and more by LightGBM (0.19 vs LR 0.13 per S1).**
  - Accepted France pairs often share a city word in the *name* (`bordeaux`, `lille`, `nantes` …) and the *same house number on a different street*.
  - Output: `experiments/runs/EXP-M-03/test_sample_sanity.json`.
- Remaining error budget (LightGBM): not blocked +0.065 (shares a name token +0.040; script mismatch +0.017, India +0.043); low-scored +0.024; false merges +0.016.
- Decision: **strong candidate, not yet adopted.** Before a leaderboard slot, the team runs the full 200k fit plus `evaluate`, `infer`, and `scripts/analysis/test_country_profile.py`. France predicted/S1 relative to LR must not grow much more than India/US do.
- Leaderboard: not submitted.

### EXP-F-01b (full run) — Submission 2
- Team full run with `configs/exp_f01b_conflicts_zeros.yaml`.
- Fit: 200,000 train-fold S1 → 2,580,936 pairs (477,336 positive); 10.5 s; inner-tune threshold 0.60 (inner F0.5 0.8363).
- Validation: F0.5 **0.8372**, P 0.9319, R 0.7418; false merges 36,952; singleton FP 3,223; pred/S1 2.72.
  - India 0.7791 (P 0.909); US 0.8760 (P 0.946).
  - Halves A/B: 0.8375 / 0.8369.
- Test: 27,460,419 candidate ids; 4,874,401 matched; 150,156 empty rows; validators PASS.
- **Public LB 0.815** (Submission 2). LB − validation = −0.022 (was −0.034).
- Error budget (upper bounds, fixed alone): blocking misses +0.073 (India +0.102), low-scored true candidates +0.058, false merges +0.046.

### EXP-D-02 — Non-linear scorer probe (diagnostic, not adopted)
- Hypothesis: The remaining scorer loss comes from interactions a linear model cannot express.
- Change: A small MLP (torch) replaces LR, with identical inputs, training S1/pairs and inner train-fold threshold protocol.
- Result: validation F0.5 0.8870 / 0.8885 / 0.8903 (seeds 42, 7, 123; 4/4/8 epochs) vs LR 0.8372.
  - Precision 0.968–0.973; false merges 15.2k–18.2k vs 37.0k; singleton FP 1,292–1,507 vs 3,223.
  - India 0.848–0.854; US 0.912–0.915.
- The train-fold inner-tune F0.5 matched validation within 0.001 in all runs.
- Decision: evidence for **EXP-M-03** (non-linear production matcher). The library choice (LightGBM, MIT, vs MLP) is a team decision. Not adopted yet.

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
- Hypothesis: A fitted linear model on the same features/candidates beats the hand weights (EXP-D-01P probe: +0.031).
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

### EXP-D-01P — Learned-matcher headroom probe (diagnostic, not adopted)
- **ID note (2026-09-26):** originally logged on 2026-09-25 as `EXP-D-01`. Renamed to `EXP-D-01P` to resolve the clash with the production decision experiment **EXP-D-01** (full-competition best claimant, DEC-018). Content unchanged. Older dated entries in DEVELOPMENT_LOG and the Submission 1 report still say "EXP-D-01 probe"; read that as EXP-D-01P.
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
