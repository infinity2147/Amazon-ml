# Project context: Amazon ML Challenge 2026, Business Entity Resolution

_Last updated: 2026-09-26, 00:40 IST. Everything below is measured from files and logs in this folder unless marked "estimate"._

---

## 000. Update 2026-09-26 06:10: candidate `sub7` with a cross-encoder (read this first)

**UPLOADED — leaderboard 0.984005 (sub5 was 0.971807).** File: `/home/24b4518/ml-projects/submissions/sub7_crossenc/matching_results.tsv`, 97,156,351 bytes, MD5 `4e9cb7318016402f076a1ad30c29de5a`. Official validator with `--check-ids`: PASS. 0 records with two owners; all test S1s present; matches ⊆ candidates.

**What changed vs sub5 (leaderboard 0.971807):**
- **Cross-encoder feature:** `microsoft/mdeberta-v3-base` (MIT) reads the raw "name | address" of the S1 and of the record as one text pair. It is fine-tuned on uncertain pairs of the 551,789 validation S1s (two models on disjoint halves) and its logit is added as a LightGBM feature for uncertain-band pairs (stage-1 p in (0.005, 0.995); 1.83 pairs/S1 on test). On held-out hard pairs its AUC is 0.9887–0.9917 against 0.974 for LightGBM.
- **Out-of-fold shortlist ranker**, so all 1,986,139 non-audit train S1s are used for training (on its own this gave no gain: audit 0.97391 = sub5).

**Evidence:**

| | Test-like validation (551,789 S1s) | Untouched audit (220,682 S1s) | Leaderboard |
|---|---|---|---|
| sub2 | – | 0.96664 (sub3) | 0.965514 |
| sub5 | 0.97210 (X1) | 0.97391 | 0.971807 |
| sub7 | 0.98637 (X4u, leakage-controlled) | **0.98643** | not yet uploaded |

The audit set has read 0.001–0.002 above the leaderboard so far, so the expected leaderboard is about 0.984–0.985. France is not measured.

**Leakage checks done:**
- every S1 is scored by a cross-encoder that never saw its pairs;
- scores on records a model saw in training are blanked (the X4u control costs only 0.0001);
- the audit S1s were never used for any training or choice.

**Rejected this round:**
- training on clones with ties broken (X5, 0.96990 vs 0.97210: the T0/T1 gain was the clone-tie shortcut);
- bigger trees (X6, +0.00006);
- more data alone (S6, +0.0000).

**Not yet done:**
- `src/run.py` runs the final and write stages with cross-encoder score files, but the cross-encoder training and scoring steps still live in `work/ce_*.py`;
- reproduction of sub7 has not been rerun;
- the France probe.

---

## 00. Update 2026-09-26 00:40: new submission candidate `sub5`

**UPLOADED — leaderboard 0.971807 (was 0.965514, +0.0063).** File: `/home/24b4518/ml-projects/submissions/sub5_density_honest/matching_results.tsv`, 96,711,229 bytes, MD5 `7c551e54fc623130d2dd019b30f8bda3`. The official validator passes with `--check-ids`; all 1,732,544 test S1s are present; every match is among the candidate pairs; 0 records have two owners.

**What it is:** stage-1 LightGBM only (stage 2 dropped), 77 features including new **density** features, trained on the original pairs of **all 1,324,327 usable train S1s** with decoy pairs weighted 1+q (honest test-density training, never on clones), weighted per-fold isotonic calibration, and one owner per record with ties broken by the ranker score. It is produced by `code/business_entity_resolution/src/run.py`.

**Evidence** (untouched audit set: 220,682 train S1s never used for any training or choice, scored exactly like test on their test-like rows):

| Model | Audit F0.5 |
|---|---|
| submission 3, two-stage (≈ submission 2, which scored **0.965514** on the leaderboard) | 0.96664 |
| sub4_honest (stage 1, honest weighting, 551k S1s) | 0.97128 |
| **sub5** | **0.97391** (+0.0073 vs submission 3, +0.0026 vs sub4_honest) |

This is not a guaranteed leaderboard gain: the audit set is US/India only, and France is only covered by the country-swap stand-in.

**Kept:** density features (+0.0016 test-like validation, and better in both country-swap directions); exactly-one-owner tie-break (correctness); honest decoy weighting; training on all usable S1s (+0.0026 on the audit set).
**Rejected, with measurements** (details in `experiments.md` rows E1…FINAL): stage 2 (−0.0009 and −0.0015 at test density); monotone constraints (−0.0035); dropping absolute IDF features (−0.0012 in-domain, mixed country swap); `p_outside` (slightly negative); rare-name rescue (≤ +0.0002 ceiling).
**Deferred:** French street-type mapping (about 1% of French records, needs a full rebuild); address n-gram rescue for India; synthetic near-twin injection.
**Open risks:** France "same address, different name" acceptance rose from 0.243 to 0.257 per S1 (not verifiable without labels); the 534 transliteration rules were mined from all train labels, audit S1s included.

---

## 0. Where we were on 2026-09-25 18:30 (history)

**One-line status then:** we have a working pipeline and a leaderboard score of **0.9655** (best so far, the two-stage submission). Our own validation had said 0.978, so we spent the day finding out *why the two numbers differ*, and we now have a validation that reproduces the leaderboard.

---

## 1. The problem in short

We get business records from three sources. Each record has a name, an address and a country.

- **Source 1 (S1)** is clean and deduplicated: every business appears once.
- **Sources 2 and 3 (S2/S3)** are messy copies: typos, abbreviations, reordered words, missing parts, names in Hindi/Telugu script, fake extra house numbers, and so on. Some S2/S3 records are **decoys** that belong to no S1 business.

**Task:** for every S1 business, list the S2/S3 record IDs that describe the same business. The list may be empty.

**Scoring:** F0.5 is computed per S1 business and then averaged over all S1 businesses.
- F0.5 counts precision twice as much as recall, so a wrong ID hurts more than a missing one.
- If an S1 has no true matches, an empty list scores 1.0 and anything else scores 0.

**Data size:**

| | S1 | S2 + S3 | countries |
|---|---|---|---|
| Train (with answers) | 2.2 M | 10.3 M | US, India |
| Test (no answers) | 1.73 M | 9.97 M | US, India, **France (never seen in train)** |

**Rules:** only the provided data may be used (no external databases, APIs or geocoders), and any model must be MIT/Apache-2.0 licensed with at most 8 B parameters. We submit `matching_results.tsv` to the leaderboard, plus a final zip with code, both output files and a methodology document. **Only 5 leaderboard submissions are allowed per day.**

**Purpose of this project:** build a pipeline that takes the raw files and produces the submission end to end, reproducibly, and scores as high as possible on the **private** leaderboard, which includes France.

---

## 2. What the data taught us (measured; details in `NOTES.md`)

- **Match sets are large.** Only 5.6% of S1s have no match; the rest have 3.5 matches on average (up to 11).
- **One owner per record.** Every S2/S3 record belongs to at most one S1, and no match crosses countries.
- **Chains share names.** 39% of S1 names are shared by several S1 businesses at different addresses (`primary care group` ×253), so the **address, especially the house number, decides**.
- **Native scripts.** 12–24% of India S2/S3 names are in native script. We romanise them and apply 534 word rules learned from training pairs (`praivet → private`).
- **Decoys are not random.** 26% of train S2/S3 records belong to no S1. A large part are **near-twins**: a copy of a real S1 with the same name and street but a changed house number. In US train, 155,548 unowned records (9.7% of all unowned) have exactly the same cleaned name and street words as some S1, and 89% of those differ only in the house number.
- **How the generator changes a house number** (measured):

  | | exact | digit dropped (`421→42`) | one digit swapped | ±1–2 | arbitrary replacement | missing |
  |---|---|---|---|---|---|---|
  | Near-twin decoys | 0% | 1% | 20% | 17% | 50% | 9% |
  | True matches | 88% | 4% | (rare) | <1% | (rare) | (rest) |

  So a candidate whose house number differs is usually a decoy, but about 12% of true matches also carry a noisy house number. That overlap is an unavoidable source of errors.
- **Test has about twice as many decoys as train, with the same number of true matches per S1:**

  | | records per S1, train | records per S1, test | decoys per S1, train | decoys per S1, test |
  |---|---|---|---|---|
  | US | 4.67 | 5.76 | 1.21 | 2.30 |
  | India | 4.68 | 5.82 | 1.22 | 2.36 |

  The extra test decoys follow the same distribution as train's decoys, in the same similarity buckets, so **test looks like train plus a second batch of decoys**. This is the single most important fact about this project.
- **France** comes from the same noise generator, with departments swapped for regions. Its names are built from a **small generic vocabulary** (Club, École, Amicale, Primaire, Sport plus a city name), and several different businesses share one street address. In France, 0.90 records per S1 sit at an S1's exact street and house number with a name that is *not* any S1's name (US 0.50, India 0.20).
- **No ID leak.** Correlation between S1 and matched ID numbers is 0.008.

---

## 3. The approach, step by step

Think of it as a hiring funnel: cheap steps shrink 10 M records to a short list per business, careful models judge the short list, and a decision rule writes the final answer.

### Step 1: Clean the text (code, no model) — `er/normalize.py`, `er/prep.py`
- Lowercase, strip accents, romanise native scripts (anyascii) and apply the 534 learned word rules.
- Remove `NULL`/`N/A`/PO boxes; take the real name out of `X DBA: Real Name`, `formerly` and `name.com`; split legal forms (`Pvt Ltd`, `LLC`, `SARL`) into their own field; shorten street words.
- Detect state and department spellings per country **without labels** (`texas`, `mharastr`, `gironde`).
- Pull out house number, number set, units and landmarks.
- **Bug found and fixed today:** a leading 5-digit number was read as a postal code, not a house number. 9.3% of US addresses start with a 5-digit number, and for those the house field was empty 71% of the time. The fix is in `normalize.py` (`house_postal`), the data was rebuilt in `work/cache/v2/`, and its effect is measured in section 5.

### Step 2: Shortlist candidates (search, no labels) — `er/blocking.py`
- Per country, TF-IDF search over three views: whole words of name+address, 5-letter chunks of the name, 5-letter chunks of the address. Both directions: each S1's most similar records and each record's most similar S1s. Very common words are dropped to bound the compute.
- **Added today:** a fourth candidate source, the **exact key** "same cleaned name and same house number" (keys shared by at most 30 S1s).
- **Why:** comparing everything with everything is 17 trillion pairs, and a true match that is not shortlisted can never be found later.

### Step 3: Cut the shortlist to 20 per S1 (small model, the "ranker")
- **Model:** LightGBM (gradient-boosted decision trees). **Row:** one (S1, candidate) pair described only by its search scores and ranks. **Label:** 1 if the candidate is in that S1's true list, else 0.
- **Trained on** a random 30% of train S1s, which are then excluded from all later training and scoring.
- **Keeps** each S1's top 20 and each record's top 1 S1, plus the exact-key pairs.
- **Share of true pairs still in the candidate set:** US 99.05%, India 97.2% (was 96.7% before the key view).

### Step 4: Describe each pair with numbers (code) — `er/features.py`
About 70 country-neutral features: name and address similarity, rare shared words, house-number comparison (exact, digit dropped, distance), state agreement, pseudo-word detector, empty address, and "competition" features (is this the best candidate for the S1, and is this S1 the best owner of the record).

### Step 5: Stage-1 judge (model)
LightGBM on one (S1, candidate) row of about 70 features, label 1/0 from the answer key. Output: probability the pair is the same business.

### Step 6: Stage-2 judge (model) — `er/stage2.py`
A second LightGBM that also sees the stage-1 probabilities of all 20 candidates of the S1 and **sibling features** (how much this candidate agrees with the S1's other confident candidates). Idea: true matches of one business agree with each other; decoys don't. Only stage-1 probabilities from models that did not train on that S1 are used.

### Step 7: Turn probabilities into lists (rule, no training) — `er/decode.py`
1. **Calibrate** with isotonic regression so "0.8" means right 80% of the time.
2. **One owner:** each S2/S3 record goes to at most one S1, the one with the highest probability.
3. **List size:** for each S1, try keeping the top 0, 1, 2… candidates and pick the size with the highest *expected* F0.5 (example: probabilities 0.9, 0.5, 0.3 → keep only the first).

### Step 8: Write and check
Write every test S1 (empty lists included) and the exact candidate set; run our checker and the official `utils/validate_submission.py`.

---

## 4. How we measure

### 4a. The original validation (train density)
- **Rows:** train S1s not used by the ranker. A random sample of **551,789 S1s** with all their candidates: about 11.0 M pairs, 17% true.
- **Cross-validation grouped by S1:** an S1 and all its candidates always sit in the same fold. Train on the other folds, predict this fold, so every S1 is scored by a model that never saw it. The experiments used 5 folds (3 folds, faster learning rate, for quick experiments).
- **Score:** the exact competition F0.5 over all 551,789 S1s, including S1s with no true matches.
- **Its flaw:** these lists contain about half as many decoys as the real test.

### 4b. The test-like validation, "TLV" (built today)
- **Idea:** since test = train + a second batch of decoys, **clone every unowned train record with probability q** (US 0.891, India 0.941, chosen so decoys per S1 match test). A clone gets a new ID, the same candidate links and the same features; then the 20-per-S1 cut is re-applied and all competition features are recomputed.
- **Check:** candidates per S1 (US 20.06 vs 20.02 on test), and per-S1 counts of dangerous candidate types match test within a few percent.
- **Result:** scoring our existing models on it drops the score by 0.013, **the same as the real leaderboard drop** (0.978 → 0.9655). So decoy density alone explains the gap.
- **Its flaw:** a clone is an **exact duplicate** of another record. A model *trained* on TLV could learn "two identical near-twins in the list means decoy", which cannot work on the real test, where second-batch decoys are separate records. See problem 2.

---

## 5. Results so far

All F0.5 numbers are macro-averaged over the 551,789 validation S1s, computed out-of-fold.

| Name | What it is | Train-density validation | Test-like validation | Leaderboard |
|---|---|---|---|---|
| First model | stage-1 LightGBM, 61 features | 0.9707 | – | – |
| Second model | + house-number detail, state agreement, name-twin features (70) | 0.9751 | – | – |
| **Submission 2** (two-stage) | second model + stage 2 | 0.9785 | 0.9663 (models trained at train density) | **0.965514** |
| Submission 3 | submission 2 + name+house exact-key candidate source | 0.9795 (stage 1 alone 0.9759) | stage 1 0.9686, two-stage 0.9663 | not uploaded |
| T0 (trained on TLV, 3 folds) | submission-3 setup, but trained *and* scored on TLV | – | stage 1 **0.9738**, two-stage 0.9795 | – |
| T1 (T0 + house-number fix) | same as T0 with rebuilt data | – | stage 1 **0.9743**, two-stage **0.9797** | – |
| H1 variant A (honest) | train on originals only (no clones), score on TLV; no correction, i.e. what submissions 2/3 did | 0.9759 | stage 1 0.9688 | – |
| H1 variant B (honest) | as A, but calibration weights decoy pairs by (1+q) | – | stage 1 0.9701 | – |
| H1 variant C (honest) | decoy pairs also weighted (1+q) in training | – | stage 1 **0.9705** | – |

**Reading T1 and H1 together (stage 1, same 551,789 S1s and the same TLV rows):** honest correction gives 0.9688 → 0.9705 (+0.0017). Training on TLV itself gives 0.9743 (+0.0055). The extra ≈ +0.004 is either the exact-duplicate clone shortcut or a real benefit of learning from candidate lists that contain more decoys. These runs cannot tell which. The house-number fix adds only ≈ +0.0005. S1s with no true match stay at 0.89–0.90 under honest training, against 0.96–0.98 when trained on TLV.

**Where the test-like drop comes from** (submission-3 models on TLV): matched S1s barely move (US 0.976, India 0.971), but **S1s with no true match collapse: US 0.816, India 0.851** (from about 0.97). They are 28% of lost points there, against 9% on train density. A no-match S1 scores zero as soon as we accept one wrong ID, and twice the decoys means twice the chances to accept one.

**Where points are lost at train density** (two-stage, 11,883 points): true IDs shortlisted but rejected ≈ +0.010 if all fixed (40% have an empty address, 20% are claimed by another same-name S1, 16% have a different house number, 8% have a pseudo-word name); true IDs never shortlisted ≈ +0.007 (India +0.005); wrong IDs accepted ≈ +0.005 (median probability 0.89, mostly near-twins).

---

## 6. Problems we are facing, and why

### Problem 1: our validation said 0.978, the leaderboard said 0.9655
- **Why:** train has half the decoys of test. Our models learned how many wrong candidates to expect from train's density, so on test they accept too many. The model accepts (probability above 0.5) 3.41 to 3.55 IDs per S1 on test (US 3.49, India 3.41, France 3.55) against 3.37 on train, where the true average is 3.40. The true count per S1 did not change on test, so the extra accepted IDs are mostly wrong.
- **Not the cause:** a global prior-shift correction (Saerens EM) found no shift; the extra decoys are specific, confusable near-twins in particular slices, not a uniform increase. France is not needed to explain the gap.
- **Status:** explained and reproduced by TLV. Not yet fixed.

### Problem 2: we cannot safely train at test density
- **Why:** the obvious fix is to train on test-like data, but the only way we have to make it (cloning) creates exact duplicates. In T0, S1s with no match recovered from 0.816 to 0.977, which is suspiciously complete.
- **What I tested:** stage 1 shows no shortcut (near-twin decoys get probability 0.0285 with a clone in the list and 0.0289 without). Stage 2 accepts near-twin decoys with a clone in the list 2.2× less often (0.62% vs 1.39%). But the two groups differ in other ways (records that keep their clone are the uncontested ones), so this is **inconclusive**: it neither proves nor rules out the shortcut.
- **Status:** stage-1 gain at test density (+0.005) looks real; stage 2's extra +0.006 is unproven. **H1** answers this cleanly by never letting a model see clones, and representing the extra decoys by a weight instead.

### Problem 3: stage 2 may not help at test density
- **Why:** the sibling features were learned on lists with fewer decoys. On TLV, the submission-3 stage 2 scored 0.9663 against 0.9686 for stage 1 alone. So the "agree with confident siblings" signal looks weaker when more decoys are around.
- **Status:** unresolved; H1 and T1 will show whether stage 2 helps once trained on honest data.

### Problem 4: near-twin decoys versus noisy true matches cannot be fully separated
- **Why:** the generator changes house numbers in decoys and in some true matches in overlapping ways (section 2). A candidate with a different house number is mostly a decoy but sometimes a true match; a candidate with the same house number is almost always real. No feature can remove that overlap, only weigh it well.

### Problem 5: India's shortlist ceiling
- **Why:** 2.8% of India's true pairs are never shortlisted (worth about 0.005 on India). They are mostly common names with an empty or very short address (nothing to search on) and pseudo-word names.
- **Status:** the exact-key source recovered 0.55 points of pair recall (96.7% → 97.2%); more channels (address-only rescue, rare-name-only) are planned.

### Problem 6: France is unmeasured
- **Why:** France has no labels, so all France results are indirect. The model learned what "same address, half the name" means from US and India names; French names are generic (Club, École), so that evidence is weaker there. The country swap was only run for the first model (US→India 0.917, India→US 0.957), never for the two-stage model. Several features have country-size scales (raw counts, IDF sums). The diagnostic upload that blanks France (to measure the France score from the leaderboard) reportedly showed "failed" on the portal, and the exact error was never captured.
- **Estimated impact** (rough estimate): about 0.0075 of the gap could come from France.

### Problem 7: consistency issues in our own pipeline (from a code audit)
- Stage-2 competition features and the one-owner rule see only the 25% validation sample in training but every S1 on test.
- Early stopping and the probability calibration use the same held-out folds that get scored; a calibration map fit on single models is applied to a 5-model average.
- **Ties:** calibration output is a step function, so two S1s often tie exactly on one record and both kept it. `decode.py` now has a `tie_col` option that keeps exactly one owner; it still needs to be measured.
- `p_outside` (the chance a true match was never shortlisted) is always 0, although India misses 2.8%.
- The 534 word rules were learned from all labels, including validation S1s (a mild leak).
- No monotone constraints yet.

### Problem 8: the code package cannot reproduce any submission
- **Why:** the real pipeline (ranker, key view, per-country features, two stages, fold averaging) lives in `work/` scripts with hard-coded paths. `src/run.py` is the old single-stage runner. Top packages are audited, so this is a hard requirement, not housekeeping.

### Problem 9: compute and working conditions
- The machine is shared: another user's job sometimes takes 60–95 GB of the 125 GB of RAM, which killed several of our jobs and background wait-shells. Blocking takes about 70 minutes per split and five-fold training about an hour. Hence the move to AWS.
- Only 5 submissions per day, so we only upload files that beat the current best on a validation we trust.

### Mistakes made along the way (so we do not repeat them)
- Character-3-gram blocking was far too expensive at 10 M scale (trillions of operations); word + 5-gram views with a compute budget replaced it.
- `np.isin` on string arrays is quadratic; forked worker pools copied memory and caused an out-of-memory kill (fixed with forkserver and bounded batches).
- `pkill -f <pattern>` kills its own shell; always kill by process ID.
- The "remove 19% of S1s" simulation of extra decoys was unfaithful (it creates groups of copies; real decoys are loners, often near-twins), so it was abandoned. It never produced a submission: only its first 3 of 5 stage-1 models were trained.
- I first claimed "stage 2 exploits the clone duplicates", then found my comparison was confounded and retracted it.

---

## 7. What is running and what comes next

**Finished (18:50 IST):** T1 and H1, logged in `experiments.md` and section 5. Result: honest correction recovers only +0.0017 of the +0.0055 that training on the test-like set gives, so the source of the other ≈ +0.004 is undecided.

**Next, in order** (from `PLAN_REVIEW_2026-09-25.md`, one change at a time, each measured on the test-like validation; only the first is current, the rest are parked until it is finished):
1. **Decide whether the test-density gain is real.** Proposed: one labelled diagnostic upload of a model trained on the test-like set (success = leaderboard clearly above 0.9655, e.g. ≥ 0.969; failure = about 0.966 or lower). Needs your decision because it spends one of the day's uploads.
2. Tie-break in the one-owner step; `p_outside` per country; near-twin group features (competing house-number groups, majority house of the confident siblings).
3. France: country swap for the full two-stage model; country-neutral rescaling of size-dependent features; name-distinctiveness features; monotone constraints; a label-free French check.
4. India shortlist: address-only rescue channel, rare-name channel.
5. Final model on all S1s (not only 551k), 3-seed bagging; optional multilingual cross-encoder on uncertain pairs if a GPU slot exists.
6. Make `src/run.py` reproduce the submission, write the README and methodology document, do a fresh-machine rerun.
7. **AWS** (pending): the work is CPU and memory heavy, so a 64-vCPU, 128–256 GB server would replace the shared machine. It needs an AWS account login (an AWS Builder ID alone cannot launch servers), a region, S3 and EC2 permissions, a 64-vCPU quota, and a spending cap. Credentials must be entered in your own terminal, never pasted in chat. A fresh AWS machine doubles as the fresh-clone reproduction test.

**Realistic target** (from the plan review): 0.975–0.978 on the leaderboard. 0.985 (the current top) is unlikely from tuning alone.

---

## 8. Submissions

| Folder | What it is | State |
|---|---|---|
| `submissions/sub1_M1/` | first model | never uploaded |
| `submissions/sub2_two_stage/` | two-stage model | **uploaded, leaderboard 0.965514** |
| `submissions/sub3_keyview/` | submission 2 + name+house key source | validated locally, not uploaded (expected gain about +0.001, inside noise) |

There is **no "sub4"**: `sub4sim` was the name of an abandoned simulation experiment, not a file. Any future upload will go in its own clearly named folder, after passing the official validator with `--check-ids`.

---

## 9. Where things are

| Path | What |
|---|---|
| `code/business_entity_resolution/src/er/` | pipeline modules: normalize, prep, blocking, features, stage2, model, decode, io_metric, mine |
| `code/business_entity_resolution/src/run.py` | old end-to-end runner (**not yet updated**) |
| `work/run_*.py`, `work/make_tlv.py`, `work/run_honest.py` | experiment drivers actually used |
| `work/cache/v1/` | first-generation data (old house-number parsing), features, predictions |
| `work/cache/v2/` | rebuilt data with the house-number fix, key-view candidates (`r3`), test-like validation (`tlv3`) |
| `NOTES.md` | data findings |
| `experiments.md` | one row per experiment |
| `context.md`, `PLAN_REVIEW_2026-09-25.md` | handoff notes and the critical review/plan written by the other working session |
| `reports/Business entity resolution state of art.md` | research report |
| `submissions/` | validated submission files |
