"""FINAL RESEARCH AUDIT: 0.97 feasibility + proposed experiments (reads measured audit files; judgements labelled)."""

from __future__ import annotations

import json
from pathlib import Path

O = Path("experiments/runs/FINAL_RESEARCH_AUDIT")


def j(n):
    return json.loads((O / n).read_text(encoding="utf-8"))


def w(n, obj):
    (O / n).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


b = j("current_error_budget.json")["systems"]["S02(offline KEEP)"]["E_half_B"]
k = j("key_hypothesis_upper_bounds.json")
bf = j("blocking_forensics_B06_k1000.json")["recall_at_k"]
pt = j("public_transfer.json")
buckets = {n: v["oracle_gain[UB]"] for n, v in b["buckets"].items() if "oracle_gain[UB]" in v}
V = k["variants"]
tokxnum, pair3 = V["new: name token x house number (2 x 2)"]["total"], V["widen: name_token.pair 3 rarest tokens"]["total"]
val_true = 763556
B08_YIELD = {"recall_gain": 0.9562 - 0.9487, "f05_gain": 0.0019, "acceptance_of_new_true": 0.66, "precision_of_new_acceptances": 0.976}

w("target_097_assessment.json", {
    "labels": "[M] measured, [UB] oracle upper bound, [E] estimate, [I] inference",
    "current": {"public_control[reported]": 0.935, "S02_local_E_half_B[M]": b["f05"], "B06_local_E[M]": 0.9471},
    "gap_to_0.97": {"public[M-reported]": 0.97 - 0.935,
                    "local_needed_for_public_0.97[E]": {"using stage-1 gap -0.011..-0.017": [0.97 + 0.011, 0.97 + 0.017]},
                    "local_gap_from_S02[E]": [0.981 - b["f05"], 0.987 - b["f05"]]},
    "oracle_headroom_S02_E_half_B[UB]": buckets, "total_loss_S02[M]": 1 - b["f05"],
    "plausible_by_bucket[E]": {
        "A never retrieved (0.0198 UB)": "+0.002..0.004 from 1-2 new/widened key families (UB no-key recall ~20% each; B-08 realised ~10% of the rank-21-30 oracle)",
        "B2 weak retrieved 0.02-0.5 (0.0183 UB)": "+0.000..0.003 from pair-level features only; this is the region of the test candidate-environment shift (mid-score candidates +40% on test) where S-01 failed",
        "B3 0.5-0.67 (0.0076 UB)": "+0.000..0.001 (threshold region; D-01 and inner-split threshold already tuned)",
        "C/D false merges + singleton FP (0.0063 UB after S-02)": "+0.000..0.002",
        "G veto cost (0.0015 UB)": "~0 (trade-off already chosen)"},
    "historical_realised_gains[M]": {"EXP-N-02": 0.0089, "EXP-N-03": 0.0011, "EXP-B-07": 0.0062, "EXP-B-06": 0.0007, "EXP-S-02": 0.0028, "EXP-B-08": 0.0018,
                                     "note": "the last four accepted/measured steps average ~+0.0029 local each"},
    "expected_cumulative_from_strongest_ideas[E]": {"local": "+0.005..0.010 over S-02", "public_if_stage1-like_transfer": "0.940..0.948"},
    "classification": "C. currently unsupported",
    "reason[I]": ("Oracle headroom (~0.05 local) exists, so 0.97 is not contradicted outright (not D). But public 0.97 needs local ~0.981-0.987, i.e. +0.031..0.037 over S-02, "
                  "while the strongest measured ideas give +0.005..0.010 [E] and recent steps realised ~+0.003 each. Nearly half the headroom (B2) sits in the weak-evidence region "
                  "that failed to transfer publicly. No evidence supports reaching 0.97."),
    "no_guarantee": True})

P = [
 {"rank": 1, "id": "EXP-B-09", "hypothesis": "A (name-view token x house number) key recovers no-key true pairs that share common name tokens and a house number, which rarest-k keys miss.",
  "primary_change": "add ONE blocking key family: 2 rarest name-view tokens x up to 2 house numbers (zero-stripped), pool block cap 50; stage-1 refit (DEC-017); D-01",
  "control": "EXP-B-06 + D-01 @ 0.67 (top-K 20)", "population": "train fit set (200k) for fit; validation / GATE-B / E half B / shadow v2 for evaluation",
  "blocking_recall_impact[UB]": {"no_key_recall": tokxnum["no_key_recall(capped)[UB]"], "never_retrieved_recall": tokxnum["never_retrieved_recall(capped)[UB]"],
                                 "new_true_candidates_validation": tokxnum["new_true"], "recall_gain_UB": tokxnum["new_true"] / val_true},
  "end_to_end_headroom[E]": "+0.002..0.003 (B-08 yield: 66% of new true candidates accepted at 97.6% precision; minus top-20 displacement, which is NOT measured)",
  "india[UB]": "26% of India no-key pairs", "us[UB]": "13% of US no-key pairs", "france": "[U] France S1 have numbers in 99.6% of records; effect unknown, watch test prediction change",
  "candidate_growth": {"gross_new_per_s1_before_topK[UB]": tokxnum["new_pairs_per_validation_s1"], "after_topK": "list length unchanged at 20; displacement of existing tail candidates must be measured"},
  "memory_risk": "LOW-MEDIUM: +1 pass in the blocker; key count ~ tokens x numbers per record", "runtime_estimate": "~3.5 h pipeline (block-eval 7, fit 35, train 13, all-train 45, infer 35, submit 5, eval 15 min) on CPU",
  "leakage_risks": "none new (keys use record fields only); DF from pool only as today",
  "test_first": ["key construction equals a brute-force reference on small tables", "zero-strip consistent with addr_num_word", "cap drops blocks > 50", "superset/prefix behaviour at fixed K is NOT expected: test displacement accounting",
                 "retrained model records the new pass in model.meta.blocking", "deterministic re-submit"],
  "shadow_v2": "all v1/v2 populations incl. T1W/T4W; no material decay", "locked": "LOCKED-A/B once after freeze",
  "gate[pre-declared]": {"local": "validation, GATE-B, E half B each >= B-06 + 0.002; precision drop <= 0.001; singleton FP +<=0.002; India/US >= -0.001",
                         "shadow": "Base/TM/Chain/Members >= +0.002; locked within 0.0015 of DEV", "density": "no material decay (T4, T1W, T4W)",
                         "test": "prediction change within +-2%; France change within +-3%", "engineering": "<= 5 GB, deterministic, validators"},
  "why_rank_1": "attacks the largest measured bucket (A, 0.0198 UB) with the best measured coverage/candidate ratio; India-leaning where the gap is largest",
  "kill_criteria": ["blocking-only validation recall gain < +0.004 (half the UB) -> stop before refit", "end-to-end < +0.001", "any material density decay", "precision drop > 0.001"]},
 {"rank": 2, "id": "EXP-B-10", "hypothesis": "No-key pairs share >= 2 COMMON name tokens that are not the 2 rarest; pairing the 3 rarest tokens recovers them at low candidate cost.",
  "primary_change": "name_token.pair max_tokens 2 -> 3 (config only); stage-1 refit; D-01",
  "control": "the kept stage 1 at that time (B-06, or B-09 if kept) + D-01", "population": "as EXP-B-09",
  "blocking_recall_impact[UB]": {"no_key_recall": pair3["no_key_recall(capped)[UB]"], "never_retrieved_recall": pair3["never_retrieved_recall(capped)[UB]"],
                                 "new_true_candidates_validation": pair3["new_true"], "recall_gain_UB": pair3["new_true"] / val_true},
  "end_to_end_headroom[E]": "+0.0015..0.0025", "india[UB]": "15% of India no-key pairs", "us[UB]": "22% of US no-key pairs (name-only records with empty addresses)",
  "france": "[U]", "candidate_growth": {"gross_new_per_s1_before_topK[UB]": pair3["new_pairs_per_validation_s1"]}, "memory_risk": "LOW", "runtime_estimate": "~3.5 h",
  "leakage_risks": "none new", "test_first": ["pair keys for k=3 equal brute force", "config differs from control only in max_tokens", "refit on own candidates"],
  "shadow_v2": "as EXP-B-09", "locked": "once after freeze", "gate[pre-declared]": "same as EXP-B-09",
  "why_rank_2": "cheapest candidate growth per recovered true pair (5.9 new/S1), US-leaning complement to B-09; overlap with B-09 unmeasured [U]",
  "kill_criteria": ["blocking-only recall gain < +0.003", "end-to-end < +0.001", "material density decay"]},
 {"rank": 3, "id": "EXP-S-03", "hypothesis": "The S-02 veto is density-invariant on B-06; refitting the same veto on the final stage 1 adds a similar +0.0025..0.003 on top of the retrieval gains.",
  "primary_change": "refit stage 2 (S-01 recipe, same features) on the kept stage-1 system and apply it as a veto only (add_floor .inf)",
  "control": "kept stage 1 + D-01 (after B-09/B-10)", "population": "stage-2 training = E half A minus validation (as S-01); evaluation as S-02",
  "blocking_recall_impact": "none (decision layer)", "end_to_end_headroom[E]": "+0.0025..0.003 (S-02 measured +0.0027..0.0030)",
  "india[M-analog]": "+0.004 (S-02)", "us[M-analog]": "+0.002 (S-02)", "france": "[U] S-02 test removal rate France 0.98x overall",
  "candidate_growth": "none", "memory_risk": "LOW (S-02 submit 3.59 GB)", "runtime_estimate": "~2.5 h (stage-2 features train+test 45 min, fit 5, evaluate 6, submit 8, shadow eval 25)",
  "leakage_risks": "stage-2 partition guards (existing)", "test_first": ["existing S-02 tests + model_from on the new stage 1"],
  "shadow_v2": "full v2 incl. tiers", "locked": "once after freeze",
  "gate[pre-declared]": "as S-02 (>= +0.002 over its own stage-1 control; no additions; test removals within +-25% of train)",
  "why_rank_3": "consolidation: the only path to a cumulative, submission-grade (>= +0.004) gain with measured density robustness; must follow the stage-1 decision because stage 2 depends on stage-1 scores",
  "kill_criteria": ["veto advantage < +0.002", "any additions", "test removal ratio outside +-25%"]},
 {"rank": 4, "id": "EXP-M-05 (diagnostic first)", "hypothesis": "Weak retrieved true pairs (stage-1 0.02-0.5; 0.0183 UB) differ from weak false pairs in pair-level token-rarity agreement (shared rare vs shared common tokens), which the stage-1 features do not encode.",
  "primary_change": "STEP 0 (diagnostic, no model): profile B2 true vs false pairs; only if a separating pair-level signal exists (AUC gain >= 0.02 on the inner split), add ONE pair-level feature (IDF-weighted name-token agreement from pool DF) and refit",
  "control": "kept stage 1 + D-01", "population": "fit set / validation / E half B / shadow v2",
  "blocking_recall_impact": "none", "end_to_end_headroom[E]": "+0.000..0.003 [U until step 0]", "india": "[U]", "us": "[U]",
  "france": "HIGH risk: test mid-score candidates per S1 are 1.4x (India/US) and 2x (France) of train [D]",
  "candidate_growth": "none", "memory_risk": "LOW", "runtime_estimate": "step 0 ~30 min; refit path ~3.5 h",
  "leakage_risks": "DF must come from the pool (no labels); no validation in feature design",
  "test_first": ["feature equals brute force", "feature is pair-level (no other-S1 or other-record context)"],
  "shadow_v2": "decisive: must show no decay on T4/T1W/T4W because it acts in the shifted weak-evidence region", "locked": "once after freeze",
  "gate[pre-declared]": "as EXP-B-09 plus test prediction change <= +1% and France change <= +2%",
  "why_rank_4": "large UB but it sits in the region that failed publicly; evidence of a separating signal is not yet measured",
  "kill_criteria": ["step 0 shows no separating signal", "test acceptance rises > 2x the train rise", "density decay"]}]
w("proposed_experiments.json", {"labels": "[UB] upper bounds measured in key_hypothesis_upper_bounds.json; [E] estimates; [U] unknown",
                                "order_rationale": "stage-1 retrieval changes first (largest bucket), then the stage-2 veto refit on the final stage 1 (it depends on stage-1 scores), then the riskier weak-evidence feature",
                                "not_proposed": {"top-K > 30": f"recall@50 {bf['50']['recall']:.4f} vs @30 {bf['30']['recall']:.4f} (+{bf['50']['recall'] - bf['30']['recall']:.4f}) for +{bf['50']['candidates_per_s1'] - bf['30']['candidates_per_s1']:.1f} candidates/S1; recall@1000 {bf['1000']['recall']:.4f}: top-K is exhausted; key families give more recall per candidate",
                                                 "context-driven stage-2 additions": "failed publicly (S-01); excluded by the S-01/S-02 safety criteria",
                                                 "transliteration extensions": "cross-script pairs already exceed Latin-Latin recall; 3.7% of India no-key pairs",
                                                 "GPU training": "no working GPU path in the env; LightGBM is not the bottleneck"},
                                "experiments": P})
print("written")
