# Amazon ML Challenge 2026 entity resolution: critical review and execution plan

_Written 2026-09-25 for execution by Opus 5.5 in `/home/24b4518/ml-projects`. Read `context.md` §0 first (how to work with this user: English only, explain every model, score and number concretely, 5 submissions per day, never `pkill -f`, activate the venv with `source /home/24b4518/ml-projects/.venv/bin/activate` before any Python)._

## 1. Context

Task: for each of 1.73M test businesses in Source 1 (S1), list the Source 2/3 (S2/S3) records that describe the same business. Score: F0.5 per S1 (precision counts double), averaged over all S1s. Train has US and India with labels; test adds France without labels.

Current state: a full pipeline exists (normalise → 3 TF-IDF blocking views → LightGBM ranker keeps 20 candidates per S1 → 71 pair features → stage-1 LightGBM → stage-2 LightGBM with "sibling" features → isotonic calibration → one-owner rule → expected-F0.5 decoding). It scores **0.9785 on our own validation but 0.9655 on the leaderboard**; the leaderboard top is 0.985. Two jobs are running: `sub3` (adds a name+house exact-key blocking view; stage-1 OOF 0.97592 vs 0.97513 before) and `sub4sim` (the "remove 19% of S1s so their records become decoys" simulation).

## 2. Verdict: right architecture, wrong yardstick, and a few real bugs

The architecture is the one winning systems use (multi-view blocking, GBDT on similarity features, stacked second stage, calibrated decision rule). The team's data study is thorough. Keep all of it.

What is off, in order of importance. Each point below was verified on the real data or in the code during this review.

### 2.1 The validation does not measure what the leaderboard measures (the 0.013 gap)

1. **The extra test decoys are per-S1 synthetic near-twins, and the running simulation cannot create them.** Measured on US train: 155,548 unowned S2/S3 records (9.7% of all unowned) have exactly the same cleaned name and street words as an S1; 89% of them differ only in the house number; almost every one targets a different S1 (one twin per S1). S1 itself contains only 0.03% such pairs, so these are not copies of hidden chain siblings; the generator makes one hard negative per chosen S1 by replacing the house number (type of change: arbitrary replacement 50%, one digit substituted 20%, ±1–2 17%, number removed 9%, digit dropped 1%) and swapping the legal form. True matches change the house number very differently (dropped digit 4%, ±1–2 under 1%, exact 88%). In test the rate is higher: same-name-same-street records with a different house number per S1 are **US 0.175 → 0.271, India 0.087 → 0.101**. Removing S1s (what `run_sim.py` does) creates unrelated decoys, which the model already rejects easily, and leaves IDF, vocab and blocking slots fitted with the removed S1s present. Expect `sub4sim` to show only a small drop and to say nothing about near-twins.
2. **France is 15% of test S1s and is never measured.** France has **0.90 records per S1 sitting at an S1's exact street+house with a name that is not any S1 name** (US 0.50, India 0.20). With France's generic vocabulary (`Club`, `École`, `Sport`, city names), "same address, half the name" is weak evidence, but the model learned the US/India meaning. No leave-one-country-out (LOCO) number exists for the two-stage model, and several features carry absolute, country-sized scales (`n_idfw`, `n_idfmax`, `aw_idfw`, `aw_idfmax`, `house_diff`, raw counts) that shift for a country with 259k S1s instead of 663k–810k.
3. **Train/test skew inside the pipeline** (`work/run_full.py`, `code/.../er/stage2.py`): the stage-2 features `p1_rank_c`, `p1_gap_c`, `n_conf_c` and the one-owner rule see only the 25% sampled S1s during training but all S1s at test time. Isotonic calibration and early stopping are fit on the same out-of-fold rows they are scored on, and the isotonic map fit on single-model scores is applied to a 5-model average. There is no untouched holdout; every design decision so far was made on the same 551,789 S1s. The transliteration rules were mined from all labels.
4. **Data use:** 30% of train S1s are burnt on a ranker that needs far less, 25% train the matcher, 45% are never used.

### 2.2 Bugs and gaps found in the code

- **5-digit house numbers are parsed as postal codes** (`normalize.py:198-199`): 9.3% of US S1 addresses start with a 5-digit house number; for those the house field is empty 71% of the time and `postal` is set 100% of the time. Real ZIPs appear in only 1.6% of US addresses. All house-number features (`house_eq/diff/drop/first`, `sib_h_eq_mean`) are blind on exactly the suburban US addresses where near-twin decoys live.
- **One-owner ties** (`decode.py:52-56`): isotonic output is a step function, ties are common, and `>=` keeps every tied S1, so a record can be given to two S1s.
- **`p_outside` is always 0** although India's shortlist misses 2.7% of true pairs.
- **No monotone constraints**, no LOCO run for the two-stage model.
- **French normalisation gaps:** `cedex` unhandled, `ter` collides with `terrace`, `sainte→ste` collides with `suite→ste`, missing `quai/cours/faubourg/résidence/ZA/ZI`.
- **The package cannot reproduce any submission:** `src/run.py` is the old single-stage pipeline; the ranker, key view, state map, per-country features, stage 2, fold averaging and simulation live only in `work/` with hard-coded paths. Top packages are audited; this is a hard requirement, not housekeeping.
- Licence note: the rule restricts *models* to MIT/Apache; `anyascii` (ISC) is a library, so the swap to uroman is optional, but document it.

### 2.3 Realistic target

Fixing the yardstick does not add score by itself; it makes the next ten decisions right. Honest expected gains on the leaderboard: test-density training and per-slice priors +0.003 to +0.008; France fixes +0.003 to +0.005; house parse and near-twin features +0.002 to +0.004; consistent stage 2, all data and cross-fitted calibration +0.001 to +0.003; India shortlist +0.001 to +0.002. These overlap. A realistic landing zone is **0.975–0.978**; 0.985 is not reachable by tuning alone and should not be chased with submissions.

## 3. The plan

Every experiment: one change at a time, logged in `experiments.md` with its number defined in the row. Every run reports: pair recall of the shortlist, OOF F0.5 on the test-like validation (§3.1), per country × {singleton, matched}, accepted IDs per S1 by bucket on test, and min(LOCO) when features change. Fast loop for feature work: 300k S1s, 2 folds, learning rate 0.1, fixed rounds; full 5-fold run only for a candidate submission.

### Phase 0 (today): land what is running, stop spending submissions

1. When `sub3` finishes: record stage-2 OOF in `experiments.md`, snapshot `output/*.tsv` to `submissions/sub3_keyview/`, but **do not upload**: it was validated on the old yardstick and the expected gain (+0.001) is inside the leaderboard noise.
2. Let `sub4sim` finish (it is the unrelated-decoy half of the simulation and stays useful as a component). Record its number, but do not adopt it as the yardstick.
3. Ask the user for the exact portal error of the failed France probe before any resubmission. If it is retried, it is still the single most informative submission available (France score ≈ (0.9655 − probe)/0.15 + 0.05).

### Phase 1: build a test-like yardstick (the "Test-Like Validation", TLV)

Goal: an offline score that lands within ±0.003 of the leaderboard, so that a change that helps offline helps online.

1. **Use all training S1s.** Train the ranker out-of-fold (5-fold GroupKFold by S1, each S1 ranked by a model that never saw it; test uses the average) so no S1 has to be held out for it. Then the matcher trains on all 2.2M S1s minus the audit set. Cost: stage-1 training goes from ~55 min to ~2.5 h at lr 0.05; use lr 0.1 with a fixed round count for experiments. Files: `work/run_rank.py` (ranker OOF), `work/run_full.py` (drop the `frac` sample).
2. **Untouched audit set:** 10% of S1s chosen once (`random_state` fixed), stored in `work/cache/v1/audit_s1.parquet`, never used for training, early stopping, calibration, feature selection or translit mining. Reported once per candidate submission only.
3. **Consistent stage 2 and one-owner:** compute candidate-side stage-2 features and one-owner over *all* S1s' stage-1 predictions (OOF for training S1s, fold-averaged for the audit set, exactly as on test). Fix in `er/stage2.py` and `run_full.py`.
4. **Cross-fitted calibration and fixed rounds:** early stopping on an inner 10% of the training folds (or a fixed round count chosen once), isotonic fit on folds ≠ k and applied to fold k; fit the test isotonic map on averaged fold scores of the audit set so it matches what test receives. `er/model.py`.
5. **Mine the translit rules only from non-audit S1s** (`er/mine.py`).
6. **Inject synthetic near-twin decoys that follow the generator's own recipe** (new script `work/make_twins.py`). Mine the recipe from the 155k US and the India train near-twin decoys by comparing each decoy with its S1: house-number change type and magnitude, legal-form swap frequencies, casing (S2 uppercase, S3 title case), state spelling, abbreviation and typo rates; then apply it to a random subset of S1s to raise the same-name-same-street-different-house rate per S1 from train to test level (US 0.175 → 0.271, India 0.087 → 0.101). Give the synthetic records IDs like `S2-SYN-…`, add them to the S2/S3 pool, re-run train blocking once (~70 min, cached), ranker cap, key view, features. Combine with the existing S1-removal (19%) for the unrelated-decoy share.
7. **Faithfulness check, before anything else is built on it:** on the TLV set and on test, compare (a) records per S1, (b) near-twin candidates per S1, (c) accepted IDs per S1 by bucket (near-twin, exact house different name, empty address), (d) the OOF F0.5 against 0.9655. If (d) is within ±0.003 and (a)–(c) match, adopt TLV as the only yardstick. If not, adjust the injection rate, not the model.

Deliverable: one full run on TLV with the current features, giving the new baseline number. Expect it near 0.965–0.970.

### Phase 2: bug fixes and cheap wins, each measured on TLV

1. **House number parse** (`er/normalize.py`): a number at the start of the address (after `#`, `No`, `HN`, `N°`) is the house number whatever its length; postal = a 5–6 digit number that is not the leading number (India 6-digit PIN, French 5-digit before the city, US 5-digit after the state). Keep `bis/ter` attached to the house (`15 bis`), map `ter` only when it follows a number, separate `sainte` from `suite`, drop `cedex NN`. Re-run prep + features (~1.5 h). Expected: fewer near-twin false positives and fewer digit-drop false negatives in the US.
2. **One-owner tie-break** on the raw stage-2 score, then on `rk`; try `slack` 0 vs 0.02 on TLV.
3. **`p_outside` per country** = 1 − shortlist pair recall (US 0.010, India 0.027, France: use India's), tuned by a factor {0.5, 1, 1.5} on TLV.
4. **Near-twin group features** in `er/features.py`/`er/stage2.py`, all country-neutral: number of same-name-same-street candidates of this S1 with distinct house numbers; house-relation type as a categorical (exact, dropped digit, ±1–2, one digit substituted, other, missing); whether this candidate's house equals the majority house of the S1's confident siblings; rank of this candidate's house-relation quality among the twins; margin of this candidate's stage-1 p over the best rival twin.
5. **Per-slice prior correction** (already prototyped in `work/bucket_prior.py`): keep only if TLV improves; with TLV in place, retraining at test density (Phase 1.6) should make most of it unnecessary.

### Phase 3: France

1. **Measure first:** LOCO for the full two-stage pipeline (train US → score India and the reverse) as the France stand-in; per-country prediction stats on test (empty-list rate, mean list size, p histogram, acceptance in the "exact house, different name" bucket).
2. **Country-neutral rescaling of absolute features:** replace `*_idfw`, `*_idfmax`, `house_diff` and raw counts by per-country percentiles or ratios (e.g. IDF mass shared ÷ IDF mass of the S1 name). Keep a feature only if TLV holds and min(LOCO) rises.
3. **Name-distinctiveness features:** share of the S1 name's IDF mass that the candidate reproduces; whether the shared tokens are only generic (top-quantile frequency in that country) or include a city name; number of S1s in the same country at the same street+house (density of the address). These target France's 0.90 same-address different-name records per S1.
4. **Monotone constraints** on all similarity and agreement features in LightGBM (`er/model.py`); accept if TLV loses < 0.0005 and min(LOCO) rises.
5. **Label-free French check:** apply the mined noise operators (Phase 1.6 recipe plus the S2/S3 name/address operators from `NOTES.md`) to French test S1s to create known-positive pairs, and create known-negative near-twins the same way; report the model's separation on French text. Optional and documented: self-training on mutually-best French pairs with p > 0.98 as positives and rejected near-twins as negatives.
6. **France decoding prior:** if the probe result arrives, set a France-only logit offset so the "exact house, different name" acceptance moves to about 0.36 per S1, and verify on the label-free check.

### Phase 4: shortlist recall for India (0.973 → 0.98+)

Value each change by the oracle macro ceiling (per S1: 1.25m/(m + 0.25n) where m = true matches shortlisted, n = true matches) rather than pair recall. Try, in order: an address-only rescue channel with char 3–4-grams on `addr` at small k; a name-only channel restricted to rare names (document frequency ≤ 5); skeleton view for native-script names. A supervised dense retriever (multilingual-e5-small, MIT) only if a GPU slot is free and the above plateau.

### Phase 5: model upgrades (only with time left and on TLV)

Full-data final model at lr 0.05 with 3-seed bagging; a stacked mDeBERTa-v3-base (MIT) cross-encoder scored on the uncertain band (p 0.2–0.8) as one extra stage-2 feature, trained with the same folds; 1–2 GPU-days on a shared A5000, expected +0.002 to +0.005.

### Phase 6 (runs in parallel from Phase 1): make the package reproduce the submission

1. Move the `work/` drivers into `code/business_entity_resolution/src/` as one `run.py` with a config (data dir, out dir, cache dir, sample fraction, seeds) and stages `prep | block | rank | keyview | feats | train | predict | decode | write`; no absolute paths.
2. `README.md`: exact commands, hardware, runtime per stage, seeds, every pretrained model with licence; `requirements.txt` stays pinned (it is today).
3. Fill `Documentation_template.md` from the experiment log: blocking table (recall, full coverage, candidates per S1, reduction ratio), feature groups, the two-stage diagram, calibration and decoding, France strategy with LOCO numbers, ablation table, compliance statement.
4. Fresh-clone rerun (new venv, `pip install -r requirements.txt`, `python src/run.py`) and diff against the submitted files before zipping.

### Submission policy (5 per day)

Upload only a file that beat the current best **on TLV**, passes `student_resource/utils/validate_submission.py --check-ids`, and is handed over with absolute path, byte size, MD5 and a distinct folder name. Reserve one submission for the France probe. Never tune against the leaderboard.

### Time tiers

- **≤ 3 days left:** Phase 0, Phase 1 (items 1–4, 6–7), Phase 2 (1, 2, 4), Phase 6. Skip the rest.
- **4–7 days:** add Phase 3 (1–4, 6) and Phase 4.
- **> 7 days:** add Phase 3.5 and Phase 5.

## 4. Critical files

- `work/run_full.py`, `work/run_rank.py`, `work/run_sim.py`: validation, ranker, simulation (Phase 1).
- `code/business_entity_resolution/src/er/normalize.py:178-202`: house/postal parse (Phase 2.1).
- `code/business_entity_resolution/src/er/decode.py:52-64`: one-owner ties, `p_outside` (Phase 2.2–2.3).
- `code/business_entity_resolution/src/er/stage2.py`, `features.py:249-271`: sibling and context features (Phase 1.3, 2.4, 3.3).
- `code/business_entity_resolution/src/er/model.py`: calibration, monotone constraints, LOCO (Phase 1.4, 3.1, 3.4).
- `code/business_entity_resolution/src/run.py`, `README.md`, `student_resource/Documentation_template.md` (Phase 6).
- Reuse: `er/io_metric.py` (`macro_fbeta`, `score_breakdown`, `write_idlist_tsv`, `check_outputs`), `er/blocking.py` (`group_rank`, `cap_by_ranker`, per-view caches in `work/cache/v1/blk_*`), `work/bucket_prior.py`, `work/err_ana.py`, `work/err_fn.py`.

## 5. Verification

- Phase 1 is done when: TLV OOF F0.5 is within ±0.003 of the last leaderboard score for the same model; near-twin candidates per S1 and accepted IDs per S1 by bucket match test within 10%; the audit set score is reported once and agrees with OOF within 0.002.
- Every later change: TLV up, per-country breakdown, min(LOCO) not down (for feature changes), 25 false positives and 25 false negatives eyeballed per country and tagged.
- Before any upload: `check_outputs` PASS, official validator PASS, matches ⊆ candidates, every test S1 present, MD5 recorded.
- Before the final zip: fresh-clone rerun reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` byte-for-byte or with a documented seed-level difference.
