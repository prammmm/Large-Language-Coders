# Research review: where we lose points, and the path forward (2026-09-26, evening)

Author: Claude Code (as reviewing ML researcher). Evidence labels:

| Label | Meaning |
|---|---|
| **[M]** | Measured |
| **[P]** | Probe (measured, diagnostic, held-out) |
| **[E]** | Estimate |
| **[U]** | Unknown |

All local numbers use E = 2,006,821 out-of-sample train S1s with all 2,206,821 train S1s competing (test-like), with frozen-validation and E-half checks. Rules followed throughout:
- only the provided data and labels;
- no external lookup, dictionary or pretrained resource;
- MIT LightGBM;
- country open set;
- no leakage;
- official validator with `--check-ids`.

---

## 1. Where we are

| System | Local E F0.5 (P) | Public LB |
|---|---|---|
| EXP-B-04 + D-01 | 0.9313 (0.9840) | not submitted (B-04 alone: ≈ 0.91) |
| **EXP-N-02 + D-01** (cross-script view) | 0.9402 (0.9864) | **0.928** |
| EXP-N-03 (+ Indic legal words) | 0.9413 (0.9859) | — (below gate alone) |
| **EXP-B-07 + D-01** (+ exact name × address word), t 0.67 | **0.9464 (0.9864)** | *to submit* |
| **EXP-B-06 + D-01** (+ name-token pair), t 0.67 | **0.9471 (0.9863)** | *to submit* |

The local → LB gap was about 0.012 at N-02:
- about −0.004 is country mix (India is 46.8% of test and is our weakest measured country);
- about −0.008 is consistent with **France ≈ 0.88** [E; unlabelled].

## 2. Main failures (what still costs points)

Oracle-fix gains on validation, **after B-07** [M] (`experiments/runs/EXP-B-07/analysis/error_budget.json`):

| Failure | Loss (F0.5 pts) | Nature |
|---|---|---|
| **1. True pairs retrieved but scored too low** | **+0.023** (mostly "other": weak name, OK address, no simple pattern) | the scorer decides each pair in isolation |
| **2. False merges** | **+0.016** (address-driven, same-name chains, singletons) | same cause: no view of the entity's other records |
| 3. Never retrieved | +0.021 (name-only / empty address, generic names) | B-06/B-07 closed a large part; the remaining keys have diminishing returns |
| **4. France (unlabelled test country)** | about 25% of LB loss [E] | we can only improve it through country-agnostic changes |
| Solved | cross-script mismatch: +0.019 → +0.002 (N-02/N-03/B-07) | — |

**Main diagnosis.** Retrieval has been pushed hard (blocking recall 0.927 → 0.949; oracle 0.9835). Most remaining loss is now in the **decision**. The model scores every (S1, record) pair **independently**, then applies one global threshold and a hand-made conflict rule (D-01). Entity resolution is a *set* problem:
- An S1's true records (≈ 3.5 on average) usually resemble **each other**.
- A record claimed by several S1s should be resolved by comparing claimants.
- An S1 with one strong match plus a weak candidate that looks like that strong match is probably a larger entity, not a false merge.

None of this is visible to the current model.

## 3. Evidence for the next lever [P]

`scripts/analysis/stage2_probe.py`, `scripts/analysis/collective_probe.py`:
- trained on E half A minus validation;
- evaluated on **held-out** E half B and validation;
- base = B-07 at t 0.67 with D-01 (0.9465).

| Second-stage decision model | Held-out E half B | Held-out validation |
|---|---|---|
| Context only: rank/gap among the S1's candidates; claimants/gap on the record (all data) | +0.0017 | +0.0019 |
| **Context + collective**: similarity of the candidate record to the S1's best *other* record (name Jaccard / token-set, address token-set, number overlap, that record's score); **20% sample only** | **+0.0048** | **+0.0048** |

Collective evidence triples the gain. It was learned from only a 20% sample, and the train gap (+0.0070) suggests more data and richer peers will add more.

## 4. Recommended approach, one failure at a time (each ends in a gated, validated output file)

### Step A — Submit what is already validated (tonight)

- **A1. EXP-B-07 at t 0.67:** E +0.0062 over the submitted N-02, precision unchanged. Expected LB **≈ 0.934–0.936** [E].
- **A2. EXP-B-06 at t 0.67:** a further +0.0007 locally, precision-neutral. Expected LB **≈ 0.935–0.937** [E]. This also tells us whether small local gains transfer.
- Third slot: keep in reserve. There is nothing else validated tonight, and spending slots on unmeasured ideas breaks our leaderboard discipline.

### Step B — EXP-S-01: collective second-stage decision model (failures 1 and 2; biggest measured lever)

- **What.** A second LightGBM that re-decides each retrieved pair (stage-1 score ≥ 0.02) using three feature groups:
  1. stage-1 score;
  2. S1 context (rank, gap to best, 2nd best, number of strong candidates);
  3. record context (claimant count, best claimant score, this S1's rank and gap on the record; this subsumes D-01);
  4. **collective features:** similarity of the record to the S1's top-k other candidate records (k = 3, score-weighted), across name/address/number views and including cross-script views, plus the peers' scores.
- **Training data (no leakage).** Stage-1 scores of all non-fit train S1s are out-of-sample.
  - Train stage 2 on **E half A minus validation** (~0.9M S1s).
  - Early-stop and threshold on a hash-split of that set.
  - Evaluate on held-out **E half B** and **validation**; the gate compares against the current best.
- **Test.** Stage-1 test scores (already produced by `infer`) → the same feature builder → stage 2 → write. `candidate_pairs.tsv` is unchanged (matches ⊆ candidates by construction).
- **Rules.** Train labels only. Country-agnostic features. MIT LightGBM. Deterministic (seed 42).
- **Expected [E]:** E +0.006 to +0.010 over B-06, so LB **≈ 0.941–0.947**.
- **Cost:** about 3–4 h to implement and test, plus about 1 h of runs. No stage-1 retrain.

### Step C — Cross-country robustness (failure 4, France)

- **Diagnostic first:** leave-one-country-out. Train stage 1 on US only and evaluate on India, and vice versa. This measures how much an unseen country loses.
- If the loss is large, prefer **scale-free features** over blocker-magnitude features (`cheap_score` depends on pool sizes), and re-check stage-2 transfer.
- **Label-free France checks:**
  - France conflict rate (currently the highest);
  - address abbreviations: `st` → "street" and `ste` → "suite" are wrong for French *saint/sainte*. They are symmetric on both sides, so risk only arises when one side spells "saint"; this is not measurable, so no slot.
- **Expected [E]:** unknown until the LOCO diagnostic; it could be the largest remaining lever for the LB because France is 15% of test at about 0.88.

### Step D — Smaller, measured levers (only if slots remain)

- More training S1s: +0.001 to +0.002.
- Top-K 30: about +0.001.
- Threshold with the stage-2 model in the loop: already inside Step B.

## 5. Honest expectation vs the 0.97 target

- Local oracle (perfect decisions on today's candidates) = **0.9835** on E. Even a perfect decision layer would land around 0.98 locally before France.
- A public LB of 0.97 would need about India ≥ 0.965, US ≥ 0.98 **and** France ≥ 0.95, when France is ≈ 0.88 today [E].
- With Steps A–C, a realistic target is **≈ 0.94–0.95 public** [E], with upside if the France robustness work pays off.
- I do not see an evidence-based route to 0.97 within 24 h, and I won't promise one.

## 6. Prompt to give me after you approve (Step B)

> Claude Code — EXP-S-01: collective second-stage decision model. Control = the best submitted system (EXP-B-06 + D-01 at t 0.67, or EXP-B-07 if B-06 scores lower on the LB). Implement a second-stage LightGBM that re-decides retrieved pairs (stage-1 score ≥ 0.02) from: stage-1 score; S1 context (rank, gap, 2nd best, #strong); record context (claimants, best/2nd best, rank, gap; replaces D-01 inside the model); and collective features (similarity of the candidate record to the S1's top-3 other candidate records on name/name_xlit/address/number views, weighted by their scores). Train only on out-of-sample stage-1 scores of E half A minus frozen validation (all train S1 compete); early stopping + threshold on a hash split of that set; evaluate on held-out E half B and validation vs the control; gate: E-B and validation F0.5 ≥ +0.003, precision drop ≤ 0.001, halves agree, singleton FP not up, peak ≤ 5 GB. Apply to test from the existing stage-1 test scores; candidates unchanged; strict + official `--check-ids` validation; write to `experiments/runs/EXP-S-01/output` (not `output/`). Add unit and leakage tests (no validation or E-B S1 in stage-2 training; deterministic). No external data, no new dependency. Report with a KEEP/REJECT decision and submission commands, then stop.
