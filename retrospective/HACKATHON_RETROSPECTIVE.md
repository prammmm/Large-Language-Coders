# From 0.751 to 0.944: what we built for the Amazon ML Hackathon (Business Entity Resolution), what worked, and what kept us from 0.98

**Team:** LargeLanguageCoders
**Final public score:** **0.944** (macro F0.5), about rank 2,700 of about 9,000 teams. The top teams scored about 0.99.
**Written:** 2026-09-30, as an honest retrospective for learning and for sharing.

Every number here comes from our own logs (`docs/EXPERIMENT_LOG.md`, `experiments/runs/*/final_report.md`). Where something is a guess rather than a measurement, it says so.

---

## Part 1 — The problem in plain words

We were given business records from three sources: **Source 1**, and **Source 2 / Source 3**. The same shop or company can appear in several sources, written differently:

| Source | How it looked |
|---|---|
| Source 1 | `SRI BALAJI TRADERS PVT LTD, 12 MG Road` |
| Source 2 | `Sri Balaji Traders Private Limited, No.12, M.G. Rd` |
| Source 3 | `श्री बालाजी ट्रेडर्स, 12 एमजी रोड` (the same business, written in Hindi script) |

**The task:** for every Source-1 business, list every Source-2 / Source-3 record that is the *same real business*.

**Size:**

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| Train | 2.2 million | 5.0 million | 5.3 million |
| Test | 1.7 million | 4.9 million | 5.1 million |

- Train had **India and the US**.
- Test added **France**, a country we had never seen a single labelled example of.

**The score: F0.5, computed per business and then averaged.**
- F0.5 punishes a **wrong match** (merging two different businesses) twice as hard as a **missed match**.
- A business with **no** true match scores 1.0 only if we predict nothing for it. Predicting even one wrong match gives it 0.

**What that means:** being careful beats being greedy.

**The machine:** a Windows laptop with 16 GB RAM (often only 2–3 GB free), a 6 GB laptop GPU, and CPU-only LightGBM. One full "retrain and re-evaluate" cycle took **about 2–3.5 hours**. That single fact shaped the whole hackathon.

---

## Part 2 — The pipeline we ended up with (simple picture)

```
 All records
     │
     ▼
 1. NORMALISE      clean names/addresses: lowercase, strip "pvt ltd"/"llc", remove leading zeros,
                   learn a Hindi↔English token map from the training labels
     │
     ▼
 2. BLOCKING       we can't compare 1.7M × 10M records, so for each business we build "keys"
    (find          (same name, same sorted name words, same house number + street word, ...)
    candidates)    and keep only the 20 most promising candidates per business
     │
     ▼
 3. STAGE-1        LightGBM scores each (business, candidate) pair with 24 similarity features
    MATCHER        → probability "same business"
     │
     ▼
 4. D-01           each record may belong to only ONE business: keep it for the highest scorer
     │
     ▼
 5. STAGE-2 VETO   a second LightGBM looks at context (how many rivals, how close the
                   runner-up is, what similar records did) and may only REMOVE matches
     │
     ▼
 6. COMBINE        add matches that a second, independently trained pipeline found
    PIPELINES      on candidates the first pipeline could never see
     │
     ▼
 output/matching_results.tsv
```

---

## Part 3 — Every step we took, in order, and what it gave

"Local" is our own held-out validation score (Source-1 businesses never used for training). "Public" is the leaderboard score.

| # | What we did | Why | Local | Public | Kept? |
|---|---|---|---|---|---|
| 1 | **Baseline:** 3 simple blocking keys + a hand-weighted similarity rule + one threshold | get a valid submission fast | 0.785 | **0.751** | banked |
| 2 | Wider blocking keys alone | more true pairs found | 0.784 | – | ✗ found pairs but added 21k wrong merges |
| 3 | Learned **logistic regression** instead of hand weights | let data choose weights | 0.816 | – | ✓ |
| 4 | **Conflict features:** different house numbers, different legal forms, leading-zero-safe numbers | wrong merges often have *different* numbers | 0.837 | **0.815** | ✓ |
| 5 | **LightGBM** instead of logistic regression | non-linear interactions (name AND address) | 0.904 | **0.889** | ✓ biggest single jump (+0.067) |
| 6 | Rare name-token key, same model | find name-only matches | 0.903 | 0.890 | ✗ (the model was not retrained for the new candidates) |
| 7 | **Combined keys + retrain the model on the new candidates** | lesson from 6: the blocker and model must be trained together | 0.927 | **≈ 0.91** | ✓ |
| 8 | **D-01: one owner per record** | a record claimed by 2 businesses is at least half wrong | 0.931 | – | ✓ |
| 9 | Wider address window | India misses | +0.001 | – | ✗ below gate |
| 10 | **Learned cross-script name map** (Hindi ↔ Latin), learned *only* from training labels | many Indian pairs were the same name in two scripts | 0.940 | **0.928** | ✓ |
| 11 | Learned Indic legal-form words | – | +0.001 | – | ✗ small |
| 12 | Exact name × address key for generic names, **name-token-pair** key for name-only records | generic names overflowed blocks | 0.947 | **0.935** | ✓ |
| 13 | **Stage-2 context model** (additions + removals) | decide pairs with context, not in isolation | **0.960** 😍 | **0.920** 😱 | ✗ the big lesson (Part 4) |
| 14 | **Stage-2 as veto only** (removals only) | removals transferred, additions did not | 0.950 | – | ✓ |
| 15 | Top-20 → top-30 candidates | 35% of misses sat just past rank 20 | +0.0018 | – | ✗ below gate |
| 16 | Name × house-number key | shared common name + number | +0.0032 | – | ✗ precision fell |
| 17 | **Veto refitted on step 16 (S-03)** | the veto repairs precision | **0.9536** | **0.942** | ✓ |
| 18 | 3-token name pairs (B-10) | more name-only matches | recall +0.0028 | – | ✗ kill gate |
| 19 | Double weight on name pairs (B-11) | lift those into the top 20 | recall −0.0012 vs 18 | – | ✗ pushed out address matches |
| 20 | Per-business "expected F0.5" decision rule (D-04) | match the metric exactly | +0.0001 | – | ✗ nothing left in decisions |
| 21 | Full retrain on the B-10 blocker + new veto (S-04) | – | −0.0008 | – | ✗ stricter threshold cost recall |
| 22 | **U-01:** S-03 + S-04's matches on candidates S-03 never saw | two pipelines see different candidates | 0.9544 | **0.943** | ✓ |
| 23 | **U-02:** + B-08 (top-30 model) matches on candidates neither saw | same idea | +0.0013 more | **0.944** | ✓ **final** |

We also ran diagnostic experiments that were never submitted:
- a small neural network probe, which proved LightGBM was the right choice;
- more trees (+0.0006, not worth it);
- **"train on one country, test on the other"**, which showed that an unseen country mainly loses *precision*.

**Summary of where the points came from:**

| Source of gain | Points |
|---|---|
| Learned matcher (logistic regression → LightGBM + conflict features) | ≈ +0.12 local |
| Better blocking, retrained with the model | ≈ +0.03 |
| Cross-script name learning (India) | ≈ +0.009 |
| Decision rules (one owner per record, veto) | ≈ +0.007 |
| Combining independent pipelines | ≈ +0.002 (local and public) |

---

## Part 4 — The biggest lesson: 0.960 locally, 0.920 on the leaderboard

**What happened.** Step 13 was a second-stage model that looked at *context*: how many candidates a business had, how close the runner-up was, what the neighbouring records looked like. Locally it was our best system by far (+0.012). On the leaderboard it **dropped** from 0.935 to 0.920.

**Why** (found with a forensic analysis afterwards):
- The **test set was "denser"** than training. There were more look-alike candidates per business: 1.4× more mid-score candidates in India/US and **2× in France**.
- Context features learned in the training environment ("few rivals means probably a match") became wrong in the crowded test environment. The model *added* matches that looked safe in train but were not safe in test.
- Our validation split came from the same environment as training, so it **could not see this coming**.

**What we changed:**
1. We built a **"shadow benchmark"**: we removed businesses from training to make the pool artificially crowded, re-weighted to look like test, and re-checked every idea at several crowding levels.
2. We **turned stage 2 into a veto** (it can only remove matches). Removals held up in crowded pools; additions did not.
3. We **wrote pass/fail gates down before looking at results**, so we could not talk ourselves into a result.

From then on, **every local gain showed up on the leaderboard**: S-03 0.942, U-01 0.943, U-02 0.944, each about 0.012 below local, exactly as predicted.

> **Takeaway:** a validation set that looks like the *training* data is not enough. Build one that looks like the *test* data (same density, unseen countries) from day one.

---

## Part 5 — Why we could not reach 0.98 (the honest analysis)

### 5.1 The arithmetic ceiling

Our final system still lost about **0.046** of local F0.5. The table shows where, each bucket measured as "the gain if only this were fixed":

| Where the loss is | True / false pairs | Max gain if fixed |
|---|---|---|
| True match **never retrieved** (not in the top-20 candidates) | 4.3% of true pairs | **0.016** |
| Retrieved but scored **very low** (< 0.5) | about 165k pairs | **0.019** |
| Retrieved, scored just under the threshold | about 57k pairs | 0.007 |
| Wrong merges still predicted | about 22k pairs | 0.006 |

**What that means:**
- Even a **perfect** matcher on our candidate lists could reach at most about **0.984 locally, ≈ 0.972 public**.
- **0.98 was mathematically out of reach for our architecture.** To get there you must fix *retrieval* and *hard-pair scoring* at the same time.

### 5.2 What we did not touch (or not properly), ranked by likely impact

1. **Learned text representations: the biggest miss.** *[inference, but strongly supported]*
   - **What we did:** everything was hand-made string similarity (edit distance, token overlap, Jaro-Winkler) plus LightGBM. We set ourselves a rule "no pretrained models", even though the competition **allowed** MIT/Apache models up to 8B parameters.
   - **Why it matters:** a multilingual sentence-embedding model or a small cross-encoder, fine-tuned on the training pairs, understands that "Balaji Traders" ≈ "बालाजी ट्रेडर्स" ≈ "Sri Balaji Trading Co." without us hand-coding every rule.
   - **What it fixes:** it attacks **both** big buckets at once:
     - dense-vector search finds the "never retrieved" pairs;
     - learned similarity fixes the "scored very low" pairs.
   - **Our estimate:** when we finally costed it, it needed **6–7 hours of machine time before the first number**. We only looked at it in the last 5 hours.
   - **Lesson:** decide on the *model family* in the first 24 hours, not the last.

2. **Retrieval was keyword-exact only.** *[measured]*
   - Every blocking key was an *exact* match of some token or combination. We never tried:
     - character n-gram / TF-IDF retrieval (catches typos and spelling variants);
     - MinHash / LSH (fast fuzzy "roughly the same words");
     - phonetic keys (sounds-alike names);
     - vector nearest-neighbour search.
   - 11 blocking experiments squeezed exact keys to 95.7% recall with **diminishing returns**: the last three gave +0.003, +0.003 and −0.001.
   - **Lesson:** when a family of ideas gives smaller and smaller gains, switch families; don't polish.

3. **We trained the matcher on only 9% of the training data.**
   - Stage 1 used 200k of 2.2M Source-1 businesses, because of memory and time.
   - A measured test on an earlier system showed 4× more data gave **+0.002**, and we never ran it on the final system.
   - With a bigger machine (or cloud), this is a free win.

4. **No cluster / graph reasoning.**
   - Businesses form groups across sources (S1 ↔ S2 ↔ S3). If record A matches the business, and record B is almost identical to A, B probably matches too.
   - Our context model tried this via "peer" features, but it was the part that failed on the crowded test set. We never tried a **proper clustering step** (connected components, correlation clustering) with test-like validation.

5. **Error analysis on the "scored very low" bucket came too late.**
   - About 165k true pairs were retrieved but scored below 0.5. We never studied a sample of 200 of them by eye to learn *why* (abbreviations? different descriptors? branch names?).
   - That list would have told us which features were missing.

6. **France (15% of the test, zero labels).**
   - We could only protect France (no country-specific rules, and checks that France did not behave strangely). We could never *improve* it.
   - Options we did not try:
     - self-training on confident test matches (allowed: it uses only the provided test data);
     - language-agnostic text models (see point 1).

7. **The validation mismatch cost us about a day** (Part 4). The shadow benchmark should have existed from the start.

8. **The idea that finally worked came last.**
   - "Combine independent pipelines, add only what the other one sees" (steps 22–23) was cheap, safe and gave **+0.002 local / +0.002 public**.
   - It was also basic ensembling, a standard technique we could have used from day two.
   - **Lesson:** try ensembling early; it is often the cheapest reliable gain.

### 5.3 What the top teams probably did differently

We cannot know. From the numbers, reaching 0.99 means near-perfect retrieval **and** near-perfect hard-pair scoring. The most likely ingredients:
- learned multilingual text models for retrieval and matching (point 1);
- much more training data used;
- strong validation that mimicked the test set.

---

## Part 6 — What we did well (keep doing this)

- **Discipline:**
  - every experiment had one change, a hypothesis and pass/fail gates written **before** seeing results;
  - 40+ experiments are logged, including all the failures.
- **No leaderboard overfitting:**
  - we never tuned on the public leaderboard;
  - after the S-01 lesson, local and public scores moved together every time.
- **Fair play:**
  - no external data, lookups or dictionaries;
  - the cross-script map was *learned from the training labels*;
  - country was treated as an open set, and France was never special-cased.
- **Engineering under constraints:**
  - the whole pipeline streamed data per country and per file part, so 17M+ records ran on a 16 GB laptop with peaks ≤ 5 GB;
  - outputs were deterministic and byte-identical on re-run;
  - there were 246 automated tests.
- **Knowing when to stop:** kill gates stopped four experiments before hours of retraining (B-10, B-11, D-04, S-04 blocking).

---

## Part 7 — Lessons (the short list)

1. **Pick the right model family early.** Hand-crafted features plus gradient boosting took us from 0.75 to 0.94, but the last 0.04 needed learned text representations.
2. **Measure your ceiling on day one.** "What would a perfect matcher on my candidates score?" tells you whether to fix retrieval or scoring.
3. **Make validation look like the test set:** density, unseen countries, class balance.
4. **Retrain the model whenever the candidates change.** The model learns *which* candidates it will see.
5. **Diminishing returns mean switch families,** not tune harder.
6. **Ensemble early.** Different pipelines make different mistakes.
7. **Removals are safer than additions** when the test environment differs from training.
8. **Budget compute honestly.** A 3-hour loop allows about 8 experiments a day. Use cloud or subsample to iterate faster.
9. **Write gates before results.** It saved us from shipping several "local wins" that would have lost points.
10. **Look at the errors by eye.** 200 examples teach more than 20 metrics.

---

## Part 8 — Ready-to-use LinkedIn post (edit freely)

> **From 0.751 to 0.944: what I learned at the Amazon ML Hackathon (Business Entity Resolution)**
>
> Our team LargeLanguageCoders spent the hackathon on one problem: find every record that refers to the same real business across 3 data sources, about 17 million records, written in English and Indian scripts, with France appearing only in the test set.
>
> We finished at **0.944 F0.5** (about rank 2,700 of 9,000 teams). The top teams reached 0.99, and the gap taught me more than the score did.
>
> What moved the score:
> ✅ Learned matcher: logistic regression → LightGBM with "conflict" features (different house numbers, different legal forms): +0.12
> ✅ Smarter candidate search (blocking), retrained together with the model: +0.03
> ✅ A Hindi↔English name map learned only from training labels: +0.009
> ✅ Decision rules: one owner per record, and a second model that can only *veto* matches: +0.007
> ✅ Combining two independent pipelines: +0.002
>
> The biggest lesson: a context model scored **0.960 locally and 0.920 on the leaderboard**. The test data was "more crowded" than training, and our validation couldn't see it. We rebuilt validation to simulate the test environment. After that, every local gain showed up on the leaderboard.
>
> What we'd do differently:
> 🔹 Use learned text models (multilingual embeddings / cross-encoders) from day one; we relied on hand-crafted string similarity
> 🔹 Measure the "perfect matcher" ceiling early: ours was about 0.97, so 0.98 needed a different architecture, not more tuning
> 🔹 Make validation look like the test set from the start
> 🔹 Try ensembling early; it was our cheapest reliable gain, and we found it last
>
> 40+ logged experiments, pass/fail gates written before every result, no external data, all on a 16 GB laptop.
> On to the next one! 🚀
>
> #MachineLearning #EntityResolution #Hackathon #LightGBM #DataScience #LessonsLearned

---

## Appendix — Where to find the details

| Topic | File |
|---|---|
| Every experiment, hypothesis, result, decision | `docs/EXPERIMENT_LOG.md` |
| Every architectural decision and why | `docs/DECISION_LOG.md` |
| The public drop investigation (0.960 → 0.920) | `experiments/runs/POST-S01-FORENSIC/forensic_report.md` (local only) |
| The final audit (ceilings, 0.97 assessment) | `experiments/runs/FINAL_RESEARCH_AUDIT/final_report.md` (local only) |
| Final 5-hour plan and the "0.98 is out of reach" math | `reports/FINAL_5H_PLAN.md` |
| Final systems | `experiments/runs/EXP-S-03`, `EXP-U-01`, `EXP-U-02` (local only) |

`experiments/runs/` is git-ignored (large artefacts). Copy the reports you want to publish into `retrospective/` before committing.
