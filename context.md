# Handoff context: Amazon ML Challenge 2026, Business Entity Resolution

_Written 2026-09-25 15:05 IST, to continue in a new Claude Code session. Project root: `/home/24b4518/ml-projects`._

---

## 0. Read first: how to work with this user

- **Always reply in English**, even if the user writes in Hindi.
- **Explain concretely:** name the model, its input rows, its labels and where they come from, and how each score was computed (which rows were held out, which folds). Never use unexplained shorthand such as "M1" or bare numbers.
- **Study the raw data yourself** (real rows) before and while building; don't just relay notes.
- **Submissions are scarce: 5 per day.** 3 of today's (2026-09-25) were used, 2 remain. Only hand over a file to upload when:
  1. it has **beaten the current best on offline validation**, or it is an explicitly labelled diagnostic probe; and
  2. it passes `student_resource/utils/validate_submission.py --check-ids`; and
  3. you give the **absolute path, byte size and MD5**, with a **distinct folder name** (several files are called `matching_results.tsv`, and the user once uploaded a wrong or duplicate one).
- **The machine is shared** (64 cores, 125 GB RAM; another user's job sometimes takes 60–95 GB). Watch `free -g`. Claude Code kills idle background wait-shells under memory pressure. The Bash tool's max timeout is 600 s: use `timeout 590 bash -c 'until …; do sleep 10; done'`.
- **Never use `pkill -f <pattern>`.** It matches and kills its own shell. Kill by PID.

---

## 1. Problem in short

- **Sources:**
  - S1 is a clean, deduplicated reference (train 2.21M, test 1.73M businesses).
  - S2 and S3 are noisy copies (train 10.3M, test 9.97M records).
  - Fields: `entity_id, business_name, business_address, country`.
- **Output:** for every test S1, the list of S2/S3 IDs describing the same business (the list may be empty).
- **Metric:** F0.5 per S1, macro-averaged over all S1s. An S1 with no true matches scores 1 for an empty list and 0 for anything else. Precision matters more: a wrong ID costs about 0.19 on an S1 with 3–4 matches; a missed ID costs about 0.08.
- **Countries:** train has US and India; test adds **France (no labels)**.
- **Rules:**
  - Only the provided data may be used (no geocoders, registries or external lookups).
  - Models must be MIT or Apache-2.0 and ≤8B parameters.
  - The final zip contains `output/matching_results.tsv`, `output/candidate_pairs.tsv` (the exact candidate set the model scored), `code/business_entity_resolution/{src,README.md,requirements.txt}` and a filled `Documentation_template.md`.
- **Official files:** `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`, `student_resource/README.md`, `student_resource/utils/validate_submission.py`.

---

## 2. Key data facts (measured; details in `NOTES.md` "Deep dive 2")

- **Match counts:** 5.6% of S1s have no match; the mean is 3.46 matches per S1 (max 11; at most 5 from S2 and 6 from S3).
- **One owner:** each S2/S3 record belongs to at most one S1, and there are no cross-country links.
- **Decoys:**
  - 26% of train S2/S3 records have no owner. They are copies of businesses *absent from S1*.
  - Test: about 40% decoys (5.8 records per S1 vs 4.7 in train), with the **same number of true matches per S1** (checked with a label-free proxy).
- **Chains:** 39% of S1 names are shared (`primary care group` ×253), so the address and house number decide.
- **Noise operators seen:**
  - native-script names (India S2 23%, S3 12%);
  - `X DBA:/formerly/t/a Real Name`;
  - glued website names (`name.com`), pseudo-word names (`Dovafaye`), acronyms;
  - `NULL`/`N/A`/`<NULL>`, PO box / PMB;
  - junk city (`INCORPORATED`);
  - comma-inverted names, bracketed legal forms, legal words moved to the front;
  - house numbers: zero-padding, `#`/`No`/`HN`, fake extra numbers, digit drop, ±1;
  - empty addresses in about 3% of records;
  - state spellings (abbreviation / full / native script); France uses departments vs regions.
- **Near-twin decoys:** copies of a business with a slightly different house number. True matches also get house-number noise.
- **France** comes from the same generator (same operators), but its names are built from a **small generic vocabulary** (Club, École, Amicale, Primaire, Sport, a city name) and several different businesses share one street address.
- **Leak check:** no ID leak (correlation 0.008). Do not exploit IDs or row order.

---

## 3. What is built (code in `code/business_entity_resolution/src/er/`)

| Module | What it does |
|---|---|
| `io_metric.py` | Safe TSV loader, exact metric, `score_breakdown`, writer that emits every S1, `check_outputs` |
| `normalize.py` | anyascii romanisation (**licence ISC: replace with uroman (MIT)**); NFKC; `N°` handling; DBA/formerly/web extraction; honorific and legal-form edge stripping; ADDR short-form map; unsupervised per-country admin vocabulary (`fit_admin_vocab`: state components, frequent never-in-S1 noise tokens, alias components such as `texas`/`gironde`); number, house, unit and landmark parse; skeleton; translit map hook |
| `mine.py` | `mine_translit`: 534 romanised→English word rules mined from train pairs (**leak note:** mined from ALL train labels, including validation S1s; should be mined only from non-validation S1s) |
| `prep.py` | Multiprocess normalisation → `{split}_s1p.parquet`, `{split}_cp.parquet` |
| `blocking.py` | Hashed TF-IDF (word tokens, char-5 name, char-5 address) per country, sparse_dot_topn top-k in both directions, **cost-budget pruning** of common keys (1e11 multiply-adds), per-view caching; numpy `group_rank`; `rank_features` and `cap_by_ranker` (stage-1 ranker) |
| `features.py` | ~72 pair features: rapidfuzz name/address similarities, glued/no-space, acronym, alt-name, skeleton, IDF overlap, pseudo-word (`r_known_frac`), number-set features (cover/exact/extra, house_eq/diff/drop/first), 3-state postal/units/legal/name-nums, state agreement via unsupervised `build_state_map`, empty address, native, landmark, name×address joints, context ranks/gaps/margins (na_prod, v_word, rk), name-twin house context. Runs per chunk in a **forkserver** pool (fork copied memory through refcounts, which OOM'd) |
| `stage2.py` | Sibling features from stage-1 probabilities: p ranks/gaps/sums; agreement (name/address token-set, house equality, number Jaccard) with the S1's other confident candidates (p ≥ 0.5); chunked |
| `model.py` | LightGBM params (lr 0.05, 127 leaves, early stop, cap 4000 rounds), `oof_predict` (GroupKFold by S1), isotonic `calibrate`, `evaluate`, `loco` |
| `decode.py` | One-owner (each record goes to its highest-probability S1), then per-S1 exact expected-F0.5 top-k (Poisson-binomial), including k=0 |
| `run.py` | Old end-to-end runner: **not yet updated** to ranker / key view / two stages |

Experiment drivers are in `work/`:
- `run_prep.py`, `run_block_views.py` (views only), `run_rank.py` (union → ranker → cap 20/S1 + top-1 per record), `run_keyview.py` (name + house key view → `pairs_r2`), `run_feats.py` (per country → `{sp}_feats_{tag}_f2/part_<ctry>.parquet`);
- `run_train.py` (stage-1 OOF), `run_stage2.py`, `run_full.py` (two stages, 5 folds each, test fold-average, writes `output/`), `run_submit.py`;
- `run_sim.py` (decoy simulation), `bucket_prior.py` (slice prior correction);
- analysis scripts: `dd1–7.py`, `err_ana.py`, `err_fn.py`, `ana_keys.py`, `ana_block_ctry.py`, `test_ranker.py`.

Caches are in `work/cache/` (raw parquet copies of the TSVs, `gt.parquet`) and `work/cache/v1/`:
- cleaned records, per-view blocking `blk_*`, `ranker.txt`, `ranker_s1.parquet` (the 30% of train S1s used only for the ranker);
- pairs `*_pairs_r1/r2/sim2`, features `*_feats_r1_f2/`, `*_feats_r2_f2/`;
- OOF `oof_*.parquet`, test predictions `test_pred_*.parquet`, `removed_s1.parquet`.

---

## 4. Pipeline and validation

1. **Normalise** all records.
2. **Block** per country: 3 TF-IDF views, top-k in both directions (union ≈ 91 candidates per S1).
3. **Stage-1 ranker** (LightGBM on blocking scores and ranks, trained on the 30% ranker S1s): keep the top 20 per S1 plus each record's top 1.
4. **Key view:** same cleaned name + same house number (keys shared by ≤ 30 S1s).
5. **Features** (~72 per pair).
6. **Stage-1 LightGBM:** 5-fold GroupKFold by S1 → out-of-fold probabilities; test gets the fold average.
7. **Stage-2 LightGBM** on stage-1 features + sibling features.
8. **Isotonic calibration**, one-owner rule, expected-F0.5 decoding.

**Validation:**
- 551,789 train S1s: a random sample from non-ranker S1s, `frac 0.25/0.7`, `random_state=0`, about 11.0M pairs, 17% positive.
- 5-fold GroupKFold by S1; the exact competition metric over all sampled S1s.
- **Known optimism:**
  - early stopping and calibration use the same OOF;
  - stage 2 reuses the stage-1 folds;
  - translit rules were mined from all labels;
  - above all, **train has far fewer decoys than test**.

---

## 5. Results

| Name | What | Validation F0.5 | Notes |
|---|---|---|---|
| M1 | stage 1, 61 features | 0.97068 | Country swap: US→India 0.917, India→US 0.957 |
| M2 | + house-number detail, state agreement, name-twin features (70); cap 4000 rounds | 0.97513 | |
| **Two-stage (sub2)** | M2 + stage-2 siblings | **0.97846** | **Leaderboard 0.965514** |
| sub3 (running) | + name+house key view (India pair recall 0.9672→0.9727, US 0.9904→0.9905) | pending | same sample, comparable with 0.97846 |
| sub4sim (queued) | trained and validated at test-like decoy density | pending | see §7 |

- **Shortlist recall** (candidate set after cap, non-ranker S1s): US 0.990, India 0.967 → 0.973 with the key view.
- **Top of the leaderboard:** 0.985.
- **Timing (shared CPU):**
  - Stage-1 5-fold training: 52–59 min. Stage 2: about 17 min. Sibling features: about 11 min.
  - Train features: about 8–15 min. Blocking views: about 70 min (train) and 72 min (test).

**Submissions:**
- `submissions/sub1_M1/`: never uploaded.
- `submissions/sub2_two_stage/` = `UPLOAD_THIS/matching_results.tsv` (MD5 `eb7c3e26c074654657bdc5cc5b19c052`): **scored 0.965514**. Its first upload attempt failed; the same file was then accepted.
- `UPLOAD_THIS/probe_blank_france/matching_results.tsv` (MD5 `203e0b78d3fe3beb5cac439a7b9740e6`, 86,459,332 bytes): submission 2 with every French row empty. **The portal reported "failed".**
  - The file is byte-identical in format to the accepted one and passes `--check-ids`.
  - The cause is unknown: the user may have uploaded the wrong or duplicate file, or the transfer went wrong.
  - **Ask for the exact portal error before any retry.**
  - Purpose: France score ≈ (0.9655 − probe score)/0.15 + 0.05.
- `UPLOAD_THIS/sub2_slice_prior/`: slice-level prior correction, US/India near-twins. Bounded effect −0.0014 to +0.0024. **Do not spend a submission on it.**
- `UPLOAD_THIS/matching_results.zip`: zipped submission 2 (fallback in case of a size limit).

---

## 6. Diagnosis: why the leaderboard (0.9655) is below validation (0.978)

- **The model accepts more IDs on test than on train.** It accepts (p > 0.5) 3.49 (US), 3.41 (India) and 3.55 (France) IDs per S1 on test, against 3.37 on train (true 3.40). The true count per S1 is unchanged, so the extras are mostly wrong.
- **A global Saerens EM prior estimate shows NO drop** (odds ×1.03–1.08). The extra decoys are confusable and slice-specific.
- **Where the extras are** (per S1, test vs train):
  - **US/India near-twins** (same name and street, different house number):
    - candidates: US 1.44 vs 0.81, India 0.83 vs 0.50;
    - accepted: US 0.199 vs 0.146.
  - **France, exact house but different name:** 0.61 accepted vs 0.36 in train. Examples: `Roubaix Sport SARL` ↔ `Roubaix Primaire SARL`, `Global Ecole` ↔ `Global Amicale`, each at the same address. French names are generic, so "same address + half the name" is weak evidence there, but the model learned the US/India meaning.
- **Estimate:** France about −0.0075 overall and US/India near-twins about −0.005, which together explain the gap. Not yet confirmed; the France probe is pending.
- **Train OOF error analysis** (two-stage; 11,883 points lost):
  - shortlisted true IDs rejected: ceiling +0.010 (58.7k IDs; 40% empty address, 20% taken by another same-name S1, 16% house differs, 8% pseudo-word);
  - never shortlisted: +0.007 (India +0.0048);
  - wrong IDs: +0.005 (median p 0.89, near-twins).
- **Hypotheses tested and rejected** (the model is already calibrated per slice):
  - name uniqueness for empty-address records (a unique-name empty-address record is only 18% true);
  - house difference 1–2 vs 3+ (already learned);
  - the phantom-twin count.
  - Only a small miscalibration found: empty-address records whose S1 confidently owns other same-name records (true 0.92–0.97 vs p 0.85–0.89).

---

## 7. Currently running (check with `ps aux | grep -E "run_|chain_"`)

1. **`chain_r2.sh` → `run_full.py r2_f2 0.25 sub3`** (key-view candidates; 4 of 5 stage-1 folds done at 15:05). It writes:
   - `logs_full_sub3.txt`, `cache/v1/oof_sub3.parquet`, `cache/v1/test_pred_sub3.parquet`;
   - `output/*.tsv`, **which the next job overwrites**. Regenerate sub3's files from `test_pred_sub3.parquet` with `decode_all` + `write_idlist_tsv` if needed.
   - Compare its validation F0.5 with 0.97846.
2. **`chain_sim.sh`:** simulation of test-like decoy density on train.
   - `run_sim.py` is **done**: it removed 419,526 random train S1s (19%, `removed_s1.parquet`), so their records become decoys. The owned share of S2/S3 is now **0.599**, matching test's ~0.60.
   - Pairs: `train_pairs_sim2.parquet` (36.7M; reused cached views minus removed S1s, then ranker cap and key view).
   - Next steps in the chain:
     - `run_feats.py train sim2` (running) → `train_feats_sim2_f2/`;
     - then, after sub3 finishes: `SIM=1 TEST_TAG=r2_f2 run_full.py sim2_f2 0.25 sub4sim` → `logs_full_sub4sim.txt`.
   - **Expectations:** the validation score should drop toward the real leaderboard (~0.965), and the model trained at the right decoy density should transfer better.
   - **First checks when done:**
     1. Accepted-per-S1 on test by bucket, compared with the table in §6.
     2. Near-twin candidates per S1 in the simulation vs test (1.44 US / 0.83 India).
     3. Whether the simulated validation score is close to 0.9655. That confirms the simulation is a faithful proxy.

---

## 8. Research done (full report: `reports/Business entity resolution state of art.md`; notes: `research_notes/Business entity resolution state of art/*.md`)

Six strands: SOTA matchers, collective ER, blocking at scale, industry/Kaggle, cross-lingual and transfer, prior shift.

- **Our architecture already matches what winning systems use.** The Foursquare Location Matching Kaggle winners used several candidate searches, a cheap LightGBM filter, gradient boosting with 120–200 similarity features, an XLM-R/mDeBERTa cross-encoder in the ensemble (1st place CV: LightGBM 0.875 → mDeBERTa 0.907 → ensemble 0.911), and graph post-processing (+0.002 to +0.022).
  - The Foursquare leaderboard had a leak: the leak-free top 5 scored 0.920–0.933.
  - Shopee winners: decision/post-processing tricks were worth more than better encoders.
- **Blocking:**
  - Sparkly (BM25 on char 3-grams, top-k) beats unsupervised dense blocking.
  - A supervised contrastive bi-encoder (SC-Block) wins at scale.
  - Sparse + dense union helps.
  - Allowed encoders: multilingual-e5 and BGE-M3 (MIT), LaBSE and gte-multilingual (Apache).
- **Matchers:**
  - Fine-tuned RoBERTa-class matchers beat generic Magellan features on messy text; the gap to a tuned GBDT is unknown.
  - Open LLMs ≤8B are weak zero-shot; fine-tuned they reach about 70–80 F1 on benchmarks.
  - Allowed: DeBERTa-v3, XLM-R, mDeBERTa (MIT); Qwen2.5/3 ≤8B, Mistral-7B, Flan-T5, ByT5 (Apache).
  - Not allowed: Llama 3.x, Jellyfish (CC-BY-NC).
- **Collective ER:**
  - Plain connected components is worst.
  - Global greedy / mutual-best assignment beat per-record best (Gemmell et al.: 90/90 vs 83/81 precision/recall).
  - Small groups mean the clustering algorithm matters little.
  - Suggested near-twin *group* features: competing house-number groups, and the group margin.
- **Prior shift:**
  - Saerens EM with calibrated probabilities is the standard fix, but only for pure label shift, which ours is not.
  - Validate by simulating shift; use a few leaderboard probes.
  - The expected-F decoder is sound.
  - Candidates of one S1 are negatively correlated under decoys; an exact dependent decoder exists (Dembczyński/Waegeman).
- **Cross-lingual:**
  - uroman (MIT), IndicXlit (MIT, 11M params).
  - DADER domain adaptation: +11 to +44 F1 across domains; smaller gains when domains are close.
  - A multilingual encoder beats an English-only one for a French target (mBERT 87.7 vs BERT 65.3).
- **Report ranking by expected gain for us** (estimates, overlapping):
  1. per-slice decoy correction (+0.003 to +0.008);
  2. train on test-like decoy density (+0.003 to +0.010), **in progress** as the simulation;
  3. near-twin group + density/uniqueness features (+0.001 to +0.004);
  4. French rules / a per-country fix (0 to +0.005, depends on the probe);
  5. mDeBERTa/XLM-R cross-encoder as an OOF LightGBM feature on uncertain pairs (+0.002 to +0.005, 1–2 GPU-days);
  6. global-greedy ownership for contested records (up to +0.002);
  7. supervised dense retriever + name-only channel (up to +0.002);
  8. dependent-label decoder (about +0.001);
  9. a fine-tuned 7–8B LLM (unknown).
- **The user shared an external review** (it had no data access). Adopted from it:
  - fix the translit leakage and keep an untouched audit set of S1s;
  - use the "oracle macro ceiling" (per S1: 1.25m/(m + 0.25n)) instead of pair recall to value shortlist changes;
  - a joint ownership experiment on small conflict components;
  - shorter n-grams / an address-only rescue channel;
  - per-slice and transfer reporting.
- **Rejected from it:** injecting extra training decoys from unowned records is impossible, because all are already in the pool. The *S1-removal simulation* is the realistic way.

---

## 9. Next steps (in order)

1. **When sub3 finishes:** record its validation score in `experiments.md` and snapshot `submissions/sub3_keyview/` from `test_pred_sub3.parquet`.
2. **When sub4sim finishes:** check that the simulation is faithful (§7). If it is, **adopt simulated validation as the main yardstick** and compare the sub4sim test predictions' accepted-per-S1 by bucket against sub2's.
3. **France:** get the probe's error or result. Then build **name-distinctiveness features** (per-country IDF share of the *shared* name tokens; is the shared part only generic words or a city name?) and check that France's "exact house, different name" acceptance moves toward about 0.36 per S1 without hurting the country-swap score.
4. **Near-twin group features and global-greedy ownership**, validated on the simulated set.
5. **Housekeeping:**
   - swap anyascii → uroman (licence);
   - mine translit rules only from non-validation S1s;
   - keep an untouched audit S1 set;
   - update `code/business_entity_resolution/src/run.py` to the full pipeline (ranker, key view, per-country features, two stages, forkserver, per-view caches);
   - fill in `Documentation_template.md`;
   - do a fresh-clone rerun before zipping.
6. **Optional, if GPU is free** (the two RTX A5000s are shared with other users): an mDeBERTa-v3 cross-encoder as an OOF feature on the uncertain band.

---

## 10. Gotchas learned

- `np.isin` on string arrays is quadratic. Use `pd.Index(...).isin(set)`.
- Forked Pool workers copy Python string objects (refcount writes), which caused an OOM. Use `forkserver` plus bounded batches. Driver scripts need `if __name__ == "__main__":`.
- Pandas groupby-rank on 1e8 rows is slow. Use `blocking.group_rank` (numpy lexsort).
- Char-3-gram TF-IDF at 10M scale costs trillions of operations. Use word + char-5 with cost-budget pruning.
- Load parquet per country with pyarrow filters to bound memory.
- `run_full.py` always writes to `/home/24b4518/ml-projects/output/`. Snapshot before the next run overwrites it.
- Log every experiment in `experiments.md` (it has the full history of rows B1–B3, M1–M2, S2a–S2b and A1–A2).
- Other docs: `PROJECT_CONTEXT.md` (plain-language project overview), `NOTES.md` (data findings), `Entity Resolution Solution Spec.md`, `RESEARCH_BRIEF.md`.

---

## 11. Critical review and execution plan (added 2026-09-25 15:40 IST)

**Read `PLAN_REVIEW_2026-09-25.md` before doing anything else.** It replaces §9 as the ordered plan. New facts measured during the review (scripts were run ad hoc; re-derive with `work/cache/v1/{train,test}_cp.parquet` and `{train,test}_s1p.parquet`, filtering by `ctry`):

- **Near-twin decoys are synthetic hard negatives, one per targeted S1.** US train: 155,548 unowned records (9.7% of unowned) have exactly the same `name_n` + `addr_words` as an S1; 89% differ only in house number; 99% of unowned records are unique under that key (decoys do not cluster); S1–S1 pairs with the same name+street are 0.03%. House change types in near-twin decoys: arbitrary replacement 50%, one digit substituted 20%, ±1–2 17%, missing 9%, digit dropped 1%. In true matches: exact 88%, digit dropped 4%, ±1–2 <1%.
- **Test has more of them:** same-name-same-street records with a different house number per S1: US 0.175 (train) → 0.271 (test), India 0.087 → 0.101. France: 0.90 records per S1 at an S1's exact street+house with a name that is not any S1 name (US 0.50, India 0.20).
- **Consequence:** the S1-removal simulation (`sub4sim`) cannot reproduce near-twins; the plan injects synthetic near-twins mined from the generator's own recipe instead.
- **Bug:** `normalize.py:198-199` parses a leading 5-digit number as postal. 9.3% of US S1 addresses start with a 5-digit house number; their `house` is empty 71% of the time. Real ZIPs occur in only 1.6% of US addresses.
- **Code audit findings:** stage-2 candidate-side features (`p1_rank_c`, `p1_gap_c`, `n_conf_c`) and one-owner see only the 25% sample in training but all S1s on test; isotonic and early stopping are fit on the scored folds; isotonic fit on single-model scores is applied to the 5-model average; one-owner `>=` keeps tied owners; `p_outside` is always 0; no monotone constraints; no LOCO for the two-stage model; `src/run.py` cannot reproduce any submission.
- **User preference:** run Python with `source /home/24b4518/ml-projects/.venv/bin/activate && python ...`, not `.venv/bin/python`.
