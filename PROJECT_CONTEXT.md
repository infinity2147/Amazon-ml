# Project context: Amazon ML Challenge 2026, Business Entity Resolution

_Last updated: 2026-09-25_

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

**Rules:** only the provided data may be used (no external databases, APIs or geocoders), and any model must be MIT/Apache-2.0 licensed with at most 8 B parameters. We submit `matching_results.tsv` to the leaderboard, plus a final zip with code, both output files and a methodology document.

## 2. Purpose of this project

Build a pipeline that takes the raw files and produces the submission end to end, reproducibly. It has to score as high as possible on the **private** leaderboard, which includes France, a country with no training labels.

## 3. What the data taught us (measured, details in `NOTES.md`)

- **Match sets are large.** Only 5.6% of S1s have no match; the rest have 3.5 matches on average (up to 11).
- **One owner per record.** Every S2/S3 record belongs to at most one S1, and no match crosses countries.
- **Chains share names.** 39% of S1 names are shared by several S1 businesses at different addresses (`primary care group` ×253), so the **address, especially the house number, decides**.
- **Near-twin decoys.** The generator makes decoys that copy a real business but change the house number slightly (`1043 → 1047`). True matches also get house-number noise (`421 → 42`). This is our main source of wrong matches.
- **Native scripts.** 12–24% of India S2/S3 names are in native script. We romanise them and apply 534 word rules learned from the training pairs (`praivet → private`, `phuds → food`).
- **France** is produced by the same noise generator: same tricks, with departments swapped for regions.
- **Test is harder than train.** Test has the same number of true matches per S1 but about **2× the decoys** (≈40% of S2/S3 against 26% in train). Expect precision on test to be lower than in our validation.

## 4. The approach, step by step

Think of it as a hiring funnel: cheap steps shrink 10 M records to a short list per business, careful models judge the short list, and a decision rule writes the final answer.

### Step 1: Clean the text (code, no model), `er/normalize.py`, `er/prep.py`
- **What it does:**
  - Lowercase, strip accents, and romanise native scripts (anyascii), then apply the 534 learned word rules.
  - Remove `NULL`/`N/A`/PO boxes.
  - Take the real name out of `X DBA: Real Name`, `X formerly Real Name` and `name.com`.
  - Split legal forms (`Pvt Ltd`, `LLC`, `SARL`) into their own field, and shorten street words (`Road → rd`).
  - Detect state and department spellings per country **without labels** (`texas`, `mharastr`, `gironde`).
  - Pull out house number, number set, units and landmarks.
- **Why:** so that two records of the same business look alike before any comparison.
- **Output:** one cleaned row per record (20 M rows), cached as parquet.

### Step 2: Shortlist candidates (search, no labels), `er/blocking.py`
- **What it does:**
  - For each country separately, find similar records with TF-IDF search over three "views":
    1. whole words of name + address,
    2. 5-letter chunks of the name (catches typos),
    3. 5-letter chunks of the address.
  - Search in both directions: each S1's most similar records, and each record's most similar S1s.
  - Drop very common words to keep the compute bounded.
- **Why:** comparing everything with everything would be 17 trillion pairs. A true match that isn't shortlisted can never be found later, so this step sets the ceiling.
- **Output:** about 91 candidates per S1.

### Step 3: Cut the shortlist to 20 per S1 (small model), the "ranker"
- **Model:** LightGBM (gradient-boosted trees).
- **Input row:** one (S1, candidate) pair described only by its search scores and ranks.
- **Label:** 1 if the candidate is in that S1's true list (`train_ground_truth.tsv`), else 0.
- **Training data:** a random 30% of train S1s. These S1s are then **excluded from all later training and scoring**, so they can't leak into the scores.
- **Keeps:** each S1's top 20, plus each record's top 1 S1.
- **Why:** the later models are expensive per pair. Cutting by raw similarity kept only 90% of India's true pairs at 20 per S1; the learned ranker keeps 99.8% of what the search found.
- **Result, share of true pairs still in the final candidate set:**
  - US 99.0%;
  - India 96.7%, which is our biggest ceiling.

### Step 4: Describe each pair with numbers (code), `er/features.py`
About 70 features per (S1, candidate) pair:
- name and address similarity (several fuzzy measures, and a no-space version for `name.com`);
- shared rare words ("zydus" counts, "traders" doesn't);
- house-number comparison: exact, digit dropped, or how far apart;
- state agreement;
- pseudo-word name detector, empty address, native script, source S2 or S3;
- "competition" features: is this the best candidate for this S1, and is this S1 the best owner for this record, and by how much.

Everything is country-neutral, with no "is India" input, so it can transfer to France.

### Step 5: Stage-1 judge (model)
- **Model:** LightGBM.
- **Input row:** one (S1, candidate) pair with the ~70 features.
- **Label:** 1/0 from the answer key.
- **Output:** the probability that the pair is the same business.
- **Why:** it learns how to weigh all the signals together, which hand-written rules can't do.

### Step 6: Stage-2 judge (model), `er/stage2.py`
- **Model:** a second LightGBM.
- **Input:** the stage-1 features, plus the stage-1 probabilities of all 20 candidates of the same S1, plus **sibling features**: how much this candidate agrees with the S1's *other* confident candidates (name, address, house number).
- **Why:** true matches of one business agree with each other, and decoys don't. For example, `Dovafaye | 007986 FLORES…` has a meaningless name, but its address equals the S1's confident matches, so stage 2 accepts it.
- **Leakage guard:** stage 2 only ever sees stage-1 probabilities made by models that didn't train on that S1.

### Step 7: Turn probabilities into the final lists (decision rule, no training), `er/decode.py`
1. **Calibrate:** correct the probabilities so that "0.8" really means right 80% of the time (isotonic regression fit on held-out predictions).
2. **One owner:** each S2/S3 record goes to at most one S1, the one with the highest probability.
3. **Choose the list size:** for each S1, try keeping the top 0, 1, 2… candidates and pick the count with the highest *expected* F0.5. Worked example: probabilities 0.9, 0.5, 0.3 → keep only the first, because a wrong ID costs more than a missing one.

### Step 8: Write and check the output
- Write every test S1, including those with empty lists, and the candidate file (exactly the set the model scored).
- Run our own check and the official `utils/validate_submission.py`.

## 5. How we measure (validation)

- **Rows:** train S1s not used by the ranker. From them, a random sample of **551,789 S1s** with all their candidates: 11.0 M pairs, 17% true.
- **5-fold cross-validation, grouped by S1:**
  - An S1 and all its candidates always sit in the same fold.
  - Train on 4 folds (~441k S1s), predict the 5th (~110k S1s), and repeat 5 times.
  - Every sampled S1 gets a prediction from a model that never saw it.
- **Score:** the exact competition F0.5, averaged over all 551,789 S1s, including S1s with no true matches and S1s whose matches the shortlist missed.
- **France stand-in:** train on US only and score India, and the reverse ("country swap").
- **Known optimism:** early stopping and calibration use the same held-out folds, and test has more decoys than train.

## 6. Results so far

All scores use the validation from section 5.

| Model | What it is | Validation F0.5 | Training time (5 folds) |
|---|---|---|---|
| First model | stage-1 LightGBM, 61 features | 0.9707 | ~27 min |
| Second model | + house-number detail, state agreement, name-twin features (70) | 0.9751 | ~52 min |
| **Two-stage** | second model + stage-2 sibling model | **0.9785** | +11 min sibling features, +17 min stage 2 |

- **Country swap** (first model): train US → score India 0.917; train India → score US 0.957.
- **Where points are still lost** (two-stage): about 91% on S1s that have matches (missed or wrong IDs) and about 10% on no-match S1s that we wrongly gave an ID.

**Submissions** (both pass the official validator):
- `submissions/sub1_M1/`: first model only.
- `submissions/sub2_two_stage/`: two-stage, the current best. **Upload this one.**

## 7. Next plan (in order)

1. **Get a leaderboard score for submission 2** (you upload it).
   - It tells us how far test is from our validation, most likely lower because of the extra decoys. Every later decision uses that gap.
   - Optionally also submit an all-empty file once: its score equals the share of no-match S1s in the public test, which checks our 5.6% assumption.
2. **Raise India's shortlist ceiling** (96.7% → ~98%).
   - The missed true pairs are mostly common names with empty or short addresses, and pseudo-word names.
   - Add cheap exact-key searches: same cleaned name (small groups only), same name + house number, same house number + rare street word.
   - Measure recall gained against extra pairs added, and keep only what helps.
3. **Protect precision against test's extra decoys.**
   - Look at the decoys we accept: near-twins with a different house number.
   - Try stricter house-number handling and a slightly more cautious decision rule, judged by validation *and* the leaderboard.
4. **France hardening.**
   - Rerun the country swap for the two-stage model.
   - Try "monotone" constraints, so higher similarity can never lower the probability.
   - Eyeball 30 French matches and 30 French rejections by hand.
   - Check that the French department ↔ region mapping was learned.
5. **Use all the training data for the final model.**
   - Experiments use 551k S1s; the final model can use all ~1.5 M non-ranker S1s.
   - Use a faster learning rate (0.1) during experiments and the careful one (0.05) for the final.
6. **Optional, only if time and GPU allow:** a small multilingual cross-encoder (`mdeberta-v3-base`, MIT) scoring only the uncertain pairs, used as one more feature.
7. **Package:**
   - Update `run.py` to the two-stage pipeline.
   - Finish `README.md` and the methodology document (`Documentation_template.md`).
   - Do a clean rerun from scratch to prove reproducibility, then zip.

**Open question for you:** what is the deadline? It decides whether step 6 fits.

## 8. Where things are

| Path | What |
|---|---|
| `code/business_entity_resolution/src/er/` | pipeline modules (normalize, prep, blocking, features, stage2, model, decode, io_metric, mine) |
| `code/business_entity_resolution/src/run.py` | end-to-end runner (to be updated to two stages) |
| `work/run_*.py` | experiment drivers used so far |
| `work/cache/v1/` | cached cleaned records, candidates, features, predictions |
| `NOTES.md` | data findings |
| `experiments.md` | log of every experiment |
| `submissions/` | validated submission files |
