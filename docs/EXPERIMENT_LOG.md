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

## Detailed entries

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
