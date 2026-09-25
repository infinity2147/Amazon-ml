# RESEARCH_BRIEF.md: Amazon ML Challenge 2026, Business Entity Resolution

> **Read this first (for the coding agent).** This file hands over everything established so far: the task specification, facts measured on the real data, the research method and results, the chosen architecture with fallbacks, licence rules, the state of the existing code, and an ordered task list with acceptance criteria.
>
> **How to read it:**
> - Anything marked **[MEASURED]** was computed on the real data.
> - **[EVIDENCE]** is backed by a published source (Section 14).
> - **[INFERENCE]** is a reasoned design choice to be confirmed on our own validation.
> - Never treat an [INFERENCE] as settled. If a measurement contradicts this file, trust the measurement and update this file.
>
> Prepared 2026-09-25 (IST).

---

## 0. TL;DR

- **Task:** for each Source 1 (S1) business, output the set of matching Source 2/Source 3 (S2/S3) records. The set can be empty.
- **Metric:** macro F0.5 per S1 entity. Precision counts 2×. An empty prediction on an entity with no matches scores 1.0.
- **Scale:**
  - Test S1: 1.73 M rows [MEASURED].
  - Test S2 and S3: about 509 MB and 506 MB, likely 3–5 M rows each [INFERENCE from file size].
  - Train is similar in size.
- **Languages:**
  - S1 is 100% Latin script [MEASURED].
  - S2/S3 contain Telugu, Hindi (Devanagari) and other scripts, per the user. **Not yet measured.**
  - Cross-script matching is a first-class requirement.
- **Architecture (primary):**
  1. Normalise and create **two views per record**: original text and romanised text.
  2. Blocking = union of per-S1 top-k from **char n-gram TF-IDF** (on the romanised view) and a **fine-tuned multilingual embedding model** (on the original view), plus reverse top-k.
  3. Pair scoring = **LightGBM stacker** over script-free features plus out-of-fold scores from a **fine-tuned multilingual cross-encoder**, with a **small LLM** only on uncertain pairs.
  4. **Calibrate** the probabilities.
  5. Give each S2/S3 record to at most one S1 owner.
  6. Per S1, choose **top-k or empty by maximum expected F0.5**.
- **Models (all Apache-2.0, about 5.2 B parameters in total):**
  - `Qwen/Qwen3-Embedding-0.6B` for retrieval.
  - `Qwen/Qwen3-Reranker-0.6B` for pair scoring.
  - `Qwen/Qwen3-4B` (LoRA) for hard cases.
  - Fallbacks are in Section 6.
- **France** (test only, no labels): generalise by design, with no per-country models. Proxy validation = leave-one-country-out (US↔India).

---

## 1. Task specification

### 1.1 Inputs

Every file is a TSV with a UTF-8 header.

- **Source files:** columns `entity_id, business_name, business_address, country`.
  - The ID prefix gives the source: `S1-`, `S2-`, `S3-`.
  - There is no source column.
- **`train_ground_truth.tsv`:** columns `source1_entity_id, matched_entity_ids`.
  - `matched_entity_ids` is a comma-separated list, empty for singletons.
- **Countries:**
  - Train: US and India.
  - Test adds France.
  - Treat country as an **open set**: no hard-coding, no filtering, no one-hot encoding of {US, India}.
- **Read with:** `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)`.
  - `keep_default_na=False` keeps names like "NA" as strings.
  - `QUOTE_NONE` stops stray quotes in addresses from merging rows.

### 1.2 Outputs

Put both files in `output/`.

- **`matching_results.tsv`:** header `source1_entity_id\tmatched_entity_ids`. This is the only file that is scored.
- **`candidate_pairs.tsv`:** header `source1_entity_id\tcandidate_entity_ids`. This must be exactly the set the final model runs inference on (the last filtering stage).

Rules for both files:
- Exactly one row per test S1 entity, including singletons, which get an empty list.
- No duplicate rows and no duplicate IDs within a list.
- Only S2/S3 IDs that exist in the test set.
- ID lists are comma-separated and unquoted.
- Every matched ID must also appear in the candidates.
- Validate with `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.

### 1.3 Metric

For each entity: `F0.5 = 1.25·P·R / (0.25·P + R)`, then averaged over all S1 entities.
- A singleton scores 1.0 for an empty prediction and 0.0 for any prediction.
- Public leaderboard = a subset of test; private leaderboard = the rest, and it decides the ranking.

### 1.4 Hard rules

- **No external data**, APIs, geocoders, business registries or internet augmentation. Packages are audited, and violations mean disqualification.
- **Final model:** MIT or Apache-2.0 licence and ≤ 8 B parameters. Whether the cap is per model or total is unclear (see Section 10).

### 1.5 Final package

```
<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt(pinned)}
└── Documentation_template.md (filled)
```

The template's sections are:
1. Executive Summary
2. Methodology: problem analysis and solution strategy
3. Candidate generation: keys, number of pairs, how recall was protected
4. Matching model: features, model type, threshold method
5. Results and error analysis: F0.5, common FPs and FNs
6. Conclusion
7. Appendix: code artefacts and extra results

---

## 2. Data inventory (Google Drive `student_resource`)

- **Folder:** `1Ct8trZUcDx1QZF1lBO0nLPgEhhs49tVt`, shared "anyone with link: reader".
- **Owner:** kanakamurthyhavale@gmail.com

| Path | Drive file ID | Size | Status |
|---|---|---|---|
| README.md | 1VlxERdN6SBipUPmSjm49t96-qNol7S_0 | 14 KB | Read; identical to the PDF problem statement |
| Documentation_template.md | 1HRkw0HYhe2W8soYusDLwnCvFuhjxyZnO | 2 KB | Read; sections listed in 1.5 |
| dataset/train/train_source1.tsv | 1CvW9nbvIpR_Yfs-d_yttElG2-YvXpNNg | 210.1 MB | Not yet analysed |
| dataset/train/train_source2.tsv | 1Fx1Nwkc0jPpd_TDD5tTITj8xPoPg3An6 | 489.3 MB | Not yet analysed |
| dataset/train/train_source3.tsv | 1NeLynftVEBz8esjKxKB0fGAkK1tr1D2D | 503.7 MB | Not yet analysed |
| dataset/train/train_ground_truth.tsv | 1RAnMoi8r9R9DF9uJ6gnR4ucKZ_CfC29k | 127.0 MB | Not yet analysed |
| dataset/test/test_source1.tsv | 1J20DEu9iahDq0ZhUhwVuZ3ijd2IYzztI | 175.0 MB | **Profiled (Section 3)** |
| dataset/test/test_source2.tsv | 1Gt5lAWw57jAbelYfszL0GHRaO9jNZ3l7 | 509.5 MB | Not yet analysed |
| dataset/test/test_source3.tsv | 1w7NF6uR7spBprP6aYI8nvfK8wuxKiUBU | 506.0 MB | Not yet analysed |
| utils/validate_submission.py | (folder 1-AnBGslUi-2rTFxiYnWBPwtDJtIrPH_Z) | small | Not yet opened |

**Download on a normal machine:** `pip install gdown && gdown --folder https://drive.google.com/drive/folders/1Ct8trZUcDx1QZF1lBO0nLPgEhhs49tVt`. Alternatively, `gdown <file_id>` per file.

The research environment could not download the large files: the Drive connector passes content through the chat, and scripted fetches were blocked. That is why only `test_source1.tsv` (uploaded directly) has been profiled.

---

## 3. Data facts

### 3.1 Measured on `test_source1.tsv` [MEASURED]

**Size and IDs**
- 1,732,544 rows × 4 columns; no empty names or addresses.
- IDs look random (e.g. `S1-714132312`) and are all unique, so there is no ordering leak to exploit (and we must not).
- Every name+address combination is unique, consistent with S1 being deduplicated.

**Countries**
- India 809,986 (46.8%), US 663,106 (38.3%), France 259,452 (15.0%).
- Last address component is mostly a state or region. Top values:
  - US: tx 57k, ny 44k, nc 41k.
  - India: Maharashtra 148k, Delhi 95k.
  - France: Hauts-de-France 88k, Nouvelle-Aquitaine 74k, Pays de la Loire 63k.
- France is concentrated in a few cities: Bordeaux 37k, Nantes 32k, Lille 30k, Tourcoing 16k, Dunkerque 15k.

**Script and characters**
- 100% Latin script: zero letters outside Latin/Latin-Extended across all 1.73 M rows.
- Accents: é 83k, è 29k, É 13k, ç 1.8k, and others.
- Mojibake from Windows-1252: `\x80`, `\x93`, `\x99`, `\x92`, `Â` in about 460 fields, e.g. `D\x92aide & Frères SARL`.
  - Fix: map C1 control characters via cp1252, or use `ftfy` (MIT; verify the licence).
- Template leakage: 108 names contain placeholders such as `<CITY_NAME> Buildsquare Private Limited`.
- Other odd characters in names: `!`, `<<`, `@`.

**Lengths**
- Name: median 24 characters, max 92.
- Address: median 50 characters, max 268.

**Postcodes are nearly absent**
- 5-digit numbers: US 10.8%, India 0.3%, France 0.4%.
- 6-digit or 3+3 numbers: India 0.3%.
- So postal features have little value. House numbers, street tokens, city and state are the anchors.
- Share of addresses with any digit: US 100%, France 99.6%, India 91.3%.

**Names repeat heavily, so a name alone is not an identifier**
- Share of unique names: US 79.0%, France 74.4%, India 64.4%.
- Most common names:
  - "bordeaux club sarl" 205 times
  - "nantes club sarl" 157
  - "urgent care group" 123
  - "shree & co" 97
  - "new delhi india private limited" 74

**Addresses repeat, so an address alone is not an identifier**
- Share of unique addresses: US 97.6%, India 97.1%, France 93.1%.
- "12 rue lyderic, lille, hauts-de-france" hosts 101 different businesses.

**Other variation**
- Component order: about 6.4% of US addresses start with the state ("IA, Iowa City, 1064 Newton Rd, Unit 11"); about 6.6% of French addresses start with the region.
- Legal forms:
  - India: 48.9% of names contain "Private Limited" and 13.7% contain "Pvt".
  - US: LLC, Inc, Corp, PC, PLLC, LP, "L.L.C.", "P.C.".
  - France: SARL, SAS, SASU, EURL, SA, SCI, EI, Cie, Ets.
- Indian landmarks: 11.2% of addresses contain near/opp/opposite/behind/beside, e.g. "Prahaladpur Near Hero Showroom".
- Hindi or Indian words written in Latin script (nagar, marg, shree, chowk, vihar, …) appear in about 226k India rows.
- The same French word appears with and without accents: école 4,674 vs ecole 10,995; comité 3,875 vs comite 9,957; santé 2,505 vs sante 3,118; lycée 3,311 vs lycee 1,786. **Strip accents.**
- Mixed languages within one record: "Refuge Services SARL" (English + French), "Westend Marg, New Delhi" (English + Hindi). **Never route by language.**

### 3.2 Reported by the user, not yet verified

- S2/S3 contain **Telugu, Hindi and other native scripts**, while S1 is Latin-only. So many true matches are **cross-script**, e.g. S1 "Shree Ganesh Traders" ↔ S2 "श्री गणेश ट्रेडर्स".

### 3.3 Must-measure first (unknowns that change the design)

1. Row counts of every file; script distribution per source (Unicode block of each letter) per country.
2. From ground truth:
   - singleton rate per country;
   - match-set size distribution;
   - share of matches from S2 vs S3;
   - **whether any S2/S3 ID appears under more than one S1** (if never, the one-owner rule is valid);
   - cross-country links (expected 0);
   - **share of matched pairs that cross scripts**.
3. Share of S2/S3 records that match nothing (distractor density).
4. Noise census on matched pairs:
   - per field: abbreviation, legal suffix swap, typo, reorder, dropped component, landmark added, transliteration or translation;
   - broken down by country × source × script.
5. Romanisation quality: for cross-script matched pairs, char-3-gram cosine between S1 and the romanised S2/S3 text.

---

## 4. How the research was done

**Method.** Two rounds of parallel literature and web research were run on seven tracks. Each track wrote structured notes: findings with citations, then evidence vs inference, then gaps. A synthesis step then merged the tracks, resolving conflicts in favour of primary sources.

1. **Blocking / candidate generation:** Sparkly, DeepBlocker, UniBlocker, pyJedAI, Papadakis benchmarks, sentence-transformers docs, FAISS.
2. **Pairwise matching models:** WDC Products, Ditto, AnyMatch, Peeters/Steiner/Bizer LLM matching papers, ComEM, Hugging Face model cards.
3. **Name and address handling:** libpostal, deepparse, usaddress, cleanco, jellyfish, Delhivery/Flipkart/Myntra engineering blogs, UPU/La Poste address guides, legal-form research.
4. **Winning competition solutions:** Kaggle Foursquare Location Matching 2022 (multilingual POI name+address matching), Kaggle Shopee 2021 (per-item match sets), past Amazon ML Challenges.
5. **Set decoding, calibration, generalisation:**
   - F-measure optimisation theory (Ye et al. 2012; Waegeman et al. 2014);
   - multi-source ER clustering (FAMER);
   - NIL linking;
   - calibration under shift (Ovadia 2019);
   - domain adaptation for ER (DADER).
6. **Language-agnostic / cross-script matching:** transliteration-robust retrieval (Chari 2025), ByT5 vs mT5 noise robustness, PAWS-X zero-shot transfer, romanisers (uroman, anyascii, ICU), IndicXlit, GlotLID.
7. **Model shortlist and licences:** verified on Hugging Face model cards and API plus PyPI on 2026-09-25, with MMTEB and reranking benchmarks and throughput estimates.

In addition, `test_source1.tsv` was profiled with pandas (Section 3.1), and the romanisers were run on sample strings (Section 5.6). A code toolkit was also built and tested on synthetic data (Section 11).

**Limitations**
- No published benchmark covers multilingual business name + address ER with zero-shot transfer to a new country. The *combination* of components is [INFERENCE].
- Some Kaggle write-ups could not be fetched. The Foursquare 0.878 → 0.911 figure comes from a secondary summary.
- Throughput figures are FLOP-based estimates, not measurements.

---

## 5. Key research results

### 5.1 Blocking

- **[EVIDENCE] Sparse per-record top-k TF-IDF on character 3-grams (Sparkly, PVLDB 2023) beat every DeepBlocker variant on all 15 benchmark datasets.**
  - On Amazon-Google it reached 98% recall with 2.5% of candidate pairs; deep blockers needed about 10%.
  - Recall was 99–100% at k=50 on structured name/address data.
  - The key factors are per-record top-k rather than a global threshold, and IDF weighting. Term frequency barely matters for short strings.
- **[EVIDENCE] Cardinality/top-k methods dominate threshold methods, and LSH underperforms** (Papadakis et al.).
- **[EVIDENCE] Sparse + dense union raised pair completeness by up to 5%** (UniBlocker 2024). Sparse alone was about 2% mAP above the best universal dense blocker.
- **[EVIDENCE] Fine-tuning a bi-encoder with hard negatives can add a lot:** Abt-Buy F1 went 0.34 → 0.64 (KBS 2026). About 1:1 hard negatives worked best; fine-tuning can hurt where the base model already fits.
- **[EVIDENCE] Foursquare top teams unioned several candidate generators.** The 7th place used 5 generators and 32 candidates per record, with a recall ceiling IoU of 0.978 (their pipeline included coordinates).
- **[EVIDENCE] FAISS IVF-PQ cut memory by up to 96% with negligible recall loss.** Exact search is preferred when the GPU allows.

### 5.2 Pair matching

- **[EVIDENCE] Feature-only matchers look good only on easy data.** Magellan reached about 42 F1 on WDC Products, while deep matchers score 20–50 points higher. The classic benchmarks are nearly linearly separable.
- **[EVIDENCE] Contrastive (bi-encoder) matchers overfit to seen entities:** R-SupCon fell from 82.2 to 53.3 F1 on unseen entities. Cross-encoders fell less (78.2 → 69.8). So **bi-encoders belong in blocking, not in the final decision.**
- **[EVIDENCE] Ditto trained on one dataset scored only 29–36 F1 on WDC.** LLMs are the most robust to unseen entities: GPT-4 scored 89.6 on WDC.
- **[EVIDENCE] LoRA fine-tuning an 8B LLM (Steiner et al. 2024) took it from 53.4 to 69.2 F1**, and to 74.1 with structured-explanation targets.
  - Recipe: r=64, α=16, dropout 0.1, lr 2e-4, ≤10 epochs, keep the best epoch.
- **[EVIDENCE] AnyMatch (small model): about 82 F1 zero-shot on unseen datasets**, against 86.4 for GPT-4. Training used 2 negatives per positive plus hard examples mined from a weaker model.
- **[EVIDENCE] ComEM (COLING 2025): choosing among a candidate list beat independent pairwise matching by about 17.6% F1**, but position bias means the candidate order must be permuted.
- **[EVIDENCE] Cross-script Indian person-name matching (SGER, 2026) reached 0.994 F1** with a LoRA-tuned 8B LLM using a parse-then-match recipe. Fine-tuned BERT scored 0.806 and Levenshtein 0.726. The study used Llama 3, which is not allowed here; replicate the recipe on Qwen3.
- **[EVIDENCE] Stacking beats either part alone.** Foursquare re:waiwai: CatBoost over 120+ string features scored 0.878 CV IoU alone. Blending with fine-tuned XLM-R-large and mDeBERTa-v3-base raised it to 0.911 (secondary source).

### 5.3 Set decoding and post-processing

- **[EVIDENCE] Under label independence, the expected-F-optimal prediction is top-k by marginal probability, or empty.** The best k for Fβ with rational β² can be computed exactly in O(n²) (Ye et al., ICML 2012).
  - The General F-measure Maximizer (Waegeman et al. 2014) handles dependence, but its authors found "no consistent winner" in practice.
  - A tuned global threshold is more robust when probabilities are misspecified (Ye et al.).
- **[EVIDENCE] FAMER (multi-source ER with duplicate-free sources): CLIP (strong links that are best from both sides) had the best precision.** Connected components had the lowest F-measure, so **never use union-find over the pair graph.**
- **[EVIDENCE] NIL ("no match") handling materially affects accuracy, and NIL examples must be in training** (Learn-to-Not-Link, ACL Findings 2023).
- **[EVIDENCE] Post-hoc calibration fitted in-distribution degrades under dataset shift; ensembles are more robust** (Ovadia 2019).
- **[EVIDENCE] Shopee, on the effect of set rules:**

  | Rule | F1 |
  |---|---|
  | Global threshold | 0.8211 |
  | + min-2 rule | 0.8285 |
  | + neighbour blending | 0.8345 |

  **Do NOT copy forced-minimum-set rules here**, because empty predictions are rewarded.

### 5.4 Generalising to an unseen country and domain

- **[EVIDENCE] DADER (SIGMOD 2022), domain adaptation for ER:**
  - MMD alignment is the most stable method; InvGAN+KD is best but hyperparameter-sensitive; GRL adversarial training is often unstable.
  - Gains were 0–27 F1 between similar domains and 11–44 between different ones.
  - Lower source–target MMD predicts better transfer.
  - An unadapted model can predict *everything* as a non-match on the target, so monitoring is required.
- **[EVIDENCE] Ditto augmentation (span deletion, shuffle, attribute operators) added up to +18.8 F1 in low-resource settings.**
- **[EVIDENCE] Cross-language training helps EM:** adding English pairs raised a German product matcher from 65.3 to 89.8 F1.
- **[EVIDENCE] PAWS-X zero-shot from English:** XLM-R averages about 85% across languages, and 92.8% on French with code-switching augmentation.

### 5.5 Cross-script / language-agnostic matching

- **[EVIDENCE] Off-the-shelf multilingual retrievers collapse across scripts** (Chari et al. 2025):
  - BGE-M3/mT5 with romanised Chinese queries against native-script documents: MRR@10 fell from 0.2342 to 0.0078 (−97%).
  - Russian fell 49%.
  - **Training on 50% native + 50% transliterated inputs** raised the Chinese transliterated score about 17× (to 0.1382) without hurting native-script accuracy.
- **[EVIDENCE] Byte-level ByT5 is far more robust to character noise:** under random-case noise it lost 1.5 XNLI points, against 25.7 for mT5. Lowercasing first removes most of the gap for subword models.
- **[EVIDENCE] Script detection needs no model:** Unicode block ranges are deterministic.
  - If language ID is needed, use GlotLID (Apache-2.0). fastText lid.176 is CC-BY-SA and not allowed.

### 5.6 Romaniser test (run in this session) [MEASURED]

| Input | anyascii (ISC) | uroman |
|---|---|---|
| श्री गणेश ट्रेडर्स (Shree Ganesh Traders) | `sri gnes tredrs` | `shrii gannesh ttreddarsa` |
| శ్రీ లక్ష్మీ ట్రేడర్స్ (Sri Lakshmi Traders) | `sri lksmi tredrs` | `shrii lakssmii ttreeddars` |
| महासमुंद | `mhasmumd` | — |
| Native digits `१२ ౧౨` | `12 12` | — |

- `unicodedata.digit()` also gives `12 12` for the native digits.
- Romanised output **never equals** the English spelling, so it must only be compared with fuzzy measures (char n-grams, edit distance, skeleton keys), never exact equality.

---

## 6. Chosen architecture (specification)

### 6.1 Stage 1: Normalisation (deterministic code, no model)

Order of operations:
1. Fix mojibake: C1 control characters → cp1252 re-decode, or ftfy.
2. NFKC.
3. Casefold.
4. Unify whitespace and punctuation. Collapse dotted acronyms (`S.A.S.` → `sas`). Map `&` → `and`.
5. NFKD and drop combining marks (é → e).
6. Map every Unicode digit to ASCII: `str(unicodedata.digit(c))` for category `Nd`.
7. Strip template placeholders such as `<CITY_NAME>`.
8. Detect the script per record from Unicode blocks.
9. If any letters are non-Latin, add a **romanised view**: primary uroman, fallback anyascii, fallback PyICU `Any-Latin; Latin-ASCII`. **Keep the original too.**
10. Map long forms to short canonical forms: street → st, road → rd, private → pvt, limited → ltd, and so on. Mapping *to the short form* avoids wrongly expanding ambiguous abbreviations (st = street / saint).
    - Extend the map by **mining rewrite pairs from the train ground truth**: align the unexplained tokens in matched pairs and count them.
11. Extract the legal form into its own field. Strip it only at the name's edges: trailing forms (Pvt Ltd, LLC), and leading French forms (SARL …).
12. Split the address into parts:
    - digit tokens and house number;
    - unit (suite/ste/apt/unit/flat/shop/#);
    - landmark clause (near/opp/behind/beside/next to, and French près de/face à/en face de/à côté de) up to the next comma;
    - remaining street, city and state words.
    - Treat the address as a **bag of tokens**, because component order varies.
13. Skeleton key for transliteration: consonant skeleton on the romanised view, e.g. Agarwal/Aggarwal/Agrawal → `agrvl`.

**Libraries**

| Status | Library | Licence |
|---|---|---|
| Allowed | stdlib `unicodedata` | stdlib |
| Allowed | anyascii | ISC |
| Allowed | PyICU | MIT |
| Allowed | rapidfuzz | MIT |
| Allowed | jellyfish | MIT |
| Check licence text | uroman | Apache classifier on PyPI, but the LICENSE text reads as custom |
| Banned | Unidecode | GPL |
| Banned | abydos | GPL-3 |
| Banned | libpostal and deepparse | Trained on external address data (deepparse is also LGPL) |

### 6.2 Stage 2: Blocking

For each S1 record, build the union of:

| View | Text | Method | Suggested k (tune) |
|---|---|---|---|
| name | romanised | char_wb 2–4-gram TF-IDF, sublinear tf, IDF fit on S1+S2+S3 (transductive) | 20–30 |
| address | romanised | same | 20–30 |
| name+address | romanised | same | 30–40 |
| skeleton | skeleton key | char 2–3-gram TF-IDF | 10–15 |
| dense | **original** text | fine-tuned Qwen3-Embedding-0.6B, cosine, exact FAISS flat on GPU | 20–50 |
| reverse | all of the above | each S2/S3 record's top-k S1 records, added to the owners' lists | 3–5 |
| key | digits | shared house number + shared rare token, drop buckets > 50 | — |

**Other rules**
- Block within country if the ground truth shows 0 cross-country links. Records with an empty country go to every bucket.
- Cap candidates per S1 after fusing scores, e.g. by reciprocal rank fusion or a light LightGBM over channel scores.

**Metrics to log for every run**
- pair recall (pair completeness);
- **entity full-coverage** (share of S1 entities whose *entire* gold set is inside the candidates);
- candidates per S1;
- reduction ratio `1 − pairs/(|S1|·|S2∪S3|)`.
- All of these per country and per script-pair.
- **Target:** pair recall ≥ 0.98, entity full-coverage ≥ 0.97.

**Fine-tuning the dense retriever** (sentence-transformers)
- Loss `CachedMultipleNegativesRankingLoss` on (S1, matched S2/S3) pairs, large batch, `NoDuplicatesBatchSampler`.
- Round 2 with mined hard negatives at about 1:1.
- **Transliterate-train:** feed every record 50% in its original script and 50% romanised.
- Augment with synthetic noise matching the mined noise census: suffix swaps, abbreviations, typos, dropped components, "Near X" landmarks, accent stripping.

**Memory warning at this scale**
- The existing toolkit's `_topk_sparse` makes a dense `chunk × n_candidates` float64 block. At 512 × 3 M that is about 12 GB.
- Use chunk ≤ 32, float32, or a sparse top-k.
- Also shard by country and by state or region token where the ground truth allows.

**Fallbacks, in order**
1. Dense model → BGE-M3 (MIT) → granite-embedding-311m-multilingual-r2 (Apache-2.0) → LaBSE (Apache-2.0, strong on cross-script).
2. Off-the-shelf rather than fine-tuned embeddings.
3. TF-IDF only on the romanised view: near state of the art, and runs on CPU.
4. FAISS IVF-PQ or HNSW if memory-bound. Never LSH.

### 6.3 Stage 3: Pair features (script-free by construction)

Features for each (S1, candidate) pair:

**Similarities**
- Name and address separately, on the romanised view: `ratio`, `token_set`, `token_sort`, `partial`, Jaro-Winkler, Levenshtein.
- Char-3-gram TF-IDF cosine on both the native and the romanised views.

**Rarity**
- IDF-weighted token Jaccard, and the IDF mass of shared tokens (sharing a rare token counts, "traders" doesn't).
- IDF computed over the union of all sources, and per country as an extra feature.

**Three-state structured agreement** (+1 agree / −1 conflict / 0 missing)
- house number, unit, digit set, legal form, numbers inside the name.
- **Missing ≠ conflicting.**

**Joint agreement**
- `name_sim × addr_sim` and `min(name_sim, addr_sim)`. This catches chains (same name, different address) and shared buildings (same address, different business).

**Landmarks**
- Presence, and landmark similarity, kept separate.

**Context / competition**
- The candidate's rank and gap-to-best within the S1's list.
- The S1's rank within the candidate's list of S1s, and its margin over the second-best S1.
- Number of candidates.
- These were top features by gain in the synthetic test (Section 11).

**Other**
- Blocking view membership and scores.
- Dense cosine.
- Script-pair indicator (same script / cross-script).
- Source indicator (S2 vs S3).

**Not used as a feature:** the raw country categorical, because France would be an unseen level. Use country only for CV stratification [INFERENCE].

### 6.4 Stage 3: Models

| Role | Primary | Fallbacks (in order) | Licence | Notes |
|---|---|---|---|---|
| Cross-encoder (runs on every pair) | `Qwen/Qwen3-Reranker-0.6B`, fine-tuned | `jhu-clsp/mmBERT-base` (307M) · `FacebookAI/xlm-roberta-large` · `microsoft/mdeberta-v3-base` · `BAAI/bge-reranker-v2-m3` · `Alibaba-NLP/gte-multilingual-reranker-base` | Apache-2.0 / MIT / MIT / MIT / Apache-2.0 / Apache-2.0 | See benchmark notes below |
| LLM (uncertain band only, p ∈ [0.3, 0.7], about 5–10% of pairs) | `Qwen/Qwen3-4B` + LoRA yes/no classifier (`Qwen/Qwen3-8B` if the cap is per model) | `Qwen/Qwen3.5-4B` (check tool support for its hybrid architecture) · `Qwen/Qwen3-1.7B` · `microsoft/Phi-4-mini-instruct` (no Hindi tag) | Apache-2.0 / Apache-2.0 / MIT | Output is a stacker **feature**, never an override |
| Character-robust extra member (optional) | `google/byt5-base` encoder | `google/canine-s` | Apache-2.0 | Robust to typos and transliteration |
| Stacker | LightGBM with `monotone_constraints` (+1 on similarities, −1 on distances and ranks; method `intermediate` or `advanced`) | CatBoost · XGBoost | MIT (LightGBM's PyPI field is blank; verify) / Apache-2.0 / Apache-2.0 | Monotonicity limits country-specific quirks under shift |

**Benchmark notes on the cross-encoder choices**
- Qwen3-Reranker-0.6B: MMTEB-R 66.36; reads Hindi, Telugu and French. It scores with softmax over the "yes"/"no" logits, which gives a calibratable probability.
- mmBERT-base: PAWS-X 87.7, against 85.9 for XLM-R-base.
- bge-reranker-v2-m3: MMTEB-R 58.36. gte-multilingual-reranker-base: 59.44.

**Training recipes**
- **Cross-encoder input:** `name | address` [SEP] `name | address`. Include both views when the scripts differ. Train on both pair orders.
  - Negatives: hard negatives from the blocker at about 2:1 against positives.
  - Augmentation: Ditto-style (span deletion, shuffle) plus transliteration views.
  - Use the same GroupKFold folds as the stacker and emit **out-of-fold** scores.
- **LLM LoRA** (Steiner et al.): r=64, α=16, dropout 0.1, lr 2e-4, up to 10 epochs, keep the best epoch.
  - Prompt: both records, then "Same business? yes/no".
  - Probability = softmax over the yes/no logits.
  - Optional: a parse-then-match target (SGER-style), where the model first normalises each name.

### 6.5 Stage 4: Calibration and decoding

1. **Calibrate:** Platt or temperature scaling fit on pooled **leave-one-country-out out-of-fold** predictions, after averaging seeds. Use isotonic only with a large OOF set.
2. **One owner:** each S2/S3 record keeps only its argmax-probability S1, plus a CLIP-style bonus when the link is best from both sides.
   - Valid only if the ground truth shows no S2/S3 ID under more than one S1.
   - Tune a small slack on OOF.
3. **Per-S1 set:** sort candidates by probability; for k = 0..n compute the exact expected F0.5; output the argmax. k = 0 is the empty set.
   - E[F | empty] = P(no true match), including a `p_outside` mass for true matches missed by blocking.
   - Implemented and verified against brute force in `decode.py` (Section 11).
4. **Optional:** an entity-level "has any match" model (features: max p, second p, sum p, candidate count, name rarity), used as P(empty).
5. **Fallback decoder:** one global threshold plus a top-1 rule gated by a confidence floor, tuned directly on OOF macro F0.5. Switch to it if the France drift monitors fire.
6. **Optional precision booster:** a ComEM-style listwise "select" pass by the Qwen3 LLM over the top candidates of ambiguous entities, with permuted candidate order.
7. **Never:** connected components or union-find merging, forced minimum set sizes, or per-country thresholds (France has no labels).

### 6.6 Handling France and other unseen countries

- One shared model. No per-language or per-country routing.
- Signals that are the same in any language: characters of the romanised view, digits, corpus IDF, ranks.
- Fit IDF on the test records themselves (transductive; confirm with the organisers).
- **Synthetic French positives:** apply the noise rates measured on US/India pairs to French test-S1 records to make (S1, noisy copy) pairs, with hard negatives from the same blocks. No labels are used.
- Optional: conservative self-training on mutual-best, high-margin French pairs, and an MMD penalty on the cross-encoder.
- **Drift monitors, comparing France with US and India:**
  - share of S1 predicted empty;
  - max-probability histogram;
  - mean set size;
  - share of records claimed by more than one S1;
  - KS/MMD on the feature distributions.
  - If France falls outside the US/India range, switch to the fallback decoder.

---

## 7. Validation protocol

- **Main CV:** GroupKFold (5) grouped by S1 entity. Keep the **full** S2/S3 pool for blocking in every fold, so distractor density matches test.
- **France proxy:** leave-one-country-out (train US → score India, train India → score US).
  - Choose features, calibrators and thresholds by the **worse** of the two directions.
  - Prefer settings whose optimal threshold barely moves between folds.
- **Report for every experiment** (keep an `experiments.md` log, one change per run):
  - blocking metrics (6.2);
  - OOF macro F0.5 overall, per country, singleton vs matched, same-script vs cross-script;
  - LOCO minimum.
- **Error analysis after every run:** 25 false positives and 25 false negatives from the worst bucket, tagged as:
  - FN not in candidates → blocking;
  - FN in candidates with low p → missing feature or normaliser;
  - FP chain (same name) → address-conflict features;
  - FP shared building → name weighting;
  - FP on a singleton → decoding or calibration.
- **Public leaderboard:** a subset of test and noisy. Trust CV.
  - An all-empty probe submission scores exactly the public subset's singleton fraction; use it to sanity-check the prior.
- **Before every upload:** `check_outputs` (toolkit) + the official `utils/validate_submission.py`. Also assert matches ⊆ candidates.

---

## 8. Compute estimates [INFERENCE: FLOP-based, not measured]

- **Assumed volume:** about 1.73 M S1 × ~20 candidates ≈ 35 M pairs on test, similar on train.
- **Speed on one H100** (A100 about 3× slower, L4 about 8× slower):

  | Model | Time per 10 M short pairs |
  |---|---|
  | ~0.1B encoder | ~6–10 min |
  | Qwen3-Reranker-0.6B | ~35–45 min, so ~2–3 h for 35 M |
  | 4B LLM | ~4–6 h, so it runs only on the ~5% uncertain band |
  | 8B LLM | ~8–11 h |

- Run tree models on GPU: 35–100× faster in Kaggle reports.
- Cache every stage (prepared frames, candidates, features, OOF scores) as parquet.

---

## 9. Licence table (verified 2026-09-25 unless noted)

**Allowed**
- **Qwen3** (0.6–8B), Qwen3-Embedding, Qwen3-Reranker, Qwen3.5-4B: Apache-2.0.
- **MIT:** BGE-M3, multilingual-e5, XLM-R, mDeBERTa-v3, mmBERT, Phi-3.5/Phi-4-mini, IndicXlit.
- **Apache-2.0:** bge-reranker-v2-m3, gte-multilingual-reranker-base, granite-embedding-r2, LaBSE, ByT5, CANINE, GlotLID.
- **Tooling:** sentence-transformers, transformers, PEFT, vLLM, XGBoost and CatBoost are Apache-2.0. FAISS and rapidfuzz are MIT. scikit-learn is BSD-3.
- **Unconfirmed:** LightGBM's licence was not confirmed from PyPI.

**Not allowed**

| Item | Reason |
|---|---|
| Llama, Gemma, EmbeddingGemma | Custom licences |
| Qwen2.5-3B | Qwen research licence (other Qwen2.5 sizes are Apache) |
| jina embeddings and rerankers v2/v3/v3.5 | CC-BY-NC |
| Jellyfish EM models | CC-BY-NC |
| Aya / Command-R | CC-BY-NC |
| Sarvam-1 | Non-commercial |
| Ministral-3-8B | 8.4–8.8 B |
| Qwen3.5-9B | Over 8 B |
| Unidecode | GPL |
| abydos | GPL-3 |
| deepparse | LGPL, external weights |
| Zingg | AGPL |
| fastText lid.176 | CC-BY-SA |
| libpostal parser | Trained on OpenStreetMap/OpenAddresses, so external data |

- Unicorn and AnyMatch repos have no licence file; don't use them.

---

## 10. Open compliance questions (ask the organisers; safe defaults in brackets)

1. **Is the 8 B cap per model or the total across the pipeline?** Does a multimodal checkpoint's vision encoder count? [Safe default: total ≤ 8 B, text-only models → 0.6B + 0.6B + 4B.]
2. **Do auxiliary pretrained tools trained on outside corpora count as "external data"?** Examples: IndicXlit, GlotLID, uroman tables. [Use only for training-time augmentation; the pipeline must still run without them.]
3. **Does the MIT/Apache rule cover libraries** (ISC anyascii, BSD scikit-learn, uroman's custom text)? [Prefer MIT/Apache; keep the others swappable.]
4. **Are small hand-written dictionaries allowed** (abbreviations, legal forms, city renames)? [Learn them from train labels; keep any hand list minimal and documented.]
5. **May unlabelled test records be used transductively** (IDF, self-training, domain alignment)? [Measure the gain, but ship a version that works without it.]
6. **Is synthetic data made from training records by an allowed LLM "external data"?** [Use rule-based corruption only.]

---

## 11. Existing code: status and required fixes

A tested toolkit exists as the Claude skill **`amazon-ml-entity-resolution`** (six modules). Copy them into `code/business_entity_resolution/src/er/`.

| Module | Contents |
|---|---|
| `io_metric.py` | Safe TSV loader, exact metric, `score_breakdown`, writer that emits every S1, `check_outputs` validator |
| `normalize.py` | Accent strip, dotted acronyms, short-form maps, legal-form extraction at name edges, landmark/unit/house/postal parse, skeleton, acronym |
| `blocking.py` | Multi-view TF-IDF top-k (forward and reverse), key view, `blocking_report` |
| `features.py` | rapidfuzz similarities, IDF overlap, three-state features, joint and context features |
| `decode.py` | Exact expected-Fβ top-k decoder (Poisson-binomial DP), `enforce_one_owner`, `decode_all` |
| `recon.py` | Ground-truth statistics, noise census, rewrite-rule miner |

**Tests passed on synthetic US/India → US/India/France data**
- The decoder equals brute force on 300 random cases.
- The metric reproduces the PDF example (0.714).
- The writer round-trips.
- Full pipeline: OOF 0.995, unseen-France test 0.997.
- Records with an empty country were handled.
- Synthetic scores do **not** predict real performance.

**Fixes needed before running on the real data**
1. **Memory:** in `_topk_sparse`, use a small chunk and float32, or a sparse top-k (Section 6.2).
2. **Scripts:**
   - add the romanised view and native-digit mapping (Section 6.1);
   - `base_clean` currently deletes non-Latin letters (regex `[^a-z0-9#]`), which **would erase Telugu and Hindi text**. Romanise *before* `base_clean`, and keep the original for the dense and cross-encoder models.
3. **Postcode features:** keep them, but expect them to be weak (Section 3.1).
4. **Cleaning:** add the mojibake fix and placeholder stripping.
5. **Dense channel:** add it with FAISS, plus the cross-encoder, LLM stage and stacker (Sections 6.2 and 6.4).
6. **Calibration:** add pooled LOCO calibration before `decode_all`.

---

## 12. Ordered task list for the coding agent

| # | Task | Done when |
|---|---|---|
| 1 | Download the data (Section 2) and set up the package layout (1.5) with pinned `requirements.txt` | All 8 TSVs load with the safe loader; row counts logged |
| 2 | Recon (3.3): script distribution per source, ground-truth stats, one-owner check, cross-script share, noise census, mined rewrites | `NOTES.md` has every number from 3.3 |
| 3 | Baseline end to end: normalise → TF-IDF blocking → about 10 features → LightGBM → threshold → write both files → validator | Validator PASS; OOF macro F0.5 logged; first submission |
| 4 | Blocking to target: add the romanised view, reverse top-k, skeleton, off-the-shelf dense channel; tune k | Pair recall ≥ 0.98 and entity full-coverage ≥ 0.97 overall *and* on cross-script pairs, at the fewest candidates per S1 |
| 5 | Fine-tune the dense retriever (MNRL + hard negatives + transliterate-train) | Blocking recall at a fixed k improves on the LOCO folds |
| 6 | Full feature bank (6.3) + monotone LightGBM + calibration + one-owner + expected-F decoder | OOF and LOCO beat step 3; decoder ≥ tuned threshold |
| 7 | Cross-encoder (Qwen3-Reranker-0.6B), OOF scores into the stacker | LOCO minimum improves |
| 8 | LLM on the uncertain band (Qwen3-4B LoRA) as a feature | LOCO minimum improves within the time budget |
| 9 | France hardening: synthetic French positives, drift monitors, fallback decoder switch | Monitors in range, or fallback triggered |
| 10 | Freeze: seed bagging, fresh-clone reproduction, filled `Documentation_template.md` with blocking and ablation tables | Zip reproduces the submitted files byte for byte |

---

## 13. Evidence vs inference summary

**Backed by evidence**
- Sparse char-gram top-k is near state of the art for blocking.
- A sparse + dense union adds recall.
- Bi-encoders belong in blocking only.
- LLMs and cross-encoders transfer better than feature-only models.
- GBDT + transformer stacking wins on multilingual POI matching.
- Top-k-or-empty is F-optimal under independence.
- CLIP-style one-owner linking beats connected components on precision.
- Calibration degrades under shift.
- Multilingual retrievers collapse across scripts unless trained on transliterated pairs.
- Byte-level models resist character noise.

**Inference, to confirm on the data**
- The exact five-stage combination and the model ranking for *this* dataset.
- That LOCO predicts France performance.
- That mined rewrites and IDF replace hand lexicons for France.
- Excluding the country feature.
- The LLM-on-uncertain-band cascade, S2↔S3 corroboration, and the synthetic French positives.
- All throughput numbers.
- That S2/S3 contain Telugu and Hindi (user report, not yet measured).

---

## 14. Key sources

- Sparkly (PVLDB 2023): https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf
- Papadakis et al., filtering benchmark: https://arxiv.org/pdf/2202.12521 ; ER benchmark difficulty: https://arxiv.org/abs/2307.01231
- UniBlocker (2024): https://arxiv.org/html/2404.14831
- Bi-encoder fine-tuning for ER (KBS 2026): https://www.sciencedirect.com/science/article/pii/S095070512600170X
- FAISS IVF-PQ for ER (Information Systems 2026): https://www.sciencedirect.com/science/article/pii/S0306437926000992
- WDC Products benchmark: https://webdatacommons.org/largescaleproductcorpus/wdc-products/
- Peeters, Steiner & Bizer, LLMs for EM (EDBT 2025): https://arxiv.org/pdf/2310.11244v4
- Steiner et al., fine-tuning LLMs for EM (2024): https://arxiv.org/html/2409.08185v1
- AnyMatch: https://arxiv.org/html/2409.04073v1
- ComEM (COLING 2025): https://arxiv.org/html/2405.16884v2
- SGER, Indian cross-script name matching (2026): https://arxiv.org/html/2605.23597v1
- Ditto: https://ar5iv.labs.arxiv.org/html/2004.00584
- Cross-language EM (Peeters & Bizer line): https://www.arxiv.org/pdf/2110.03338v1
- Foursquare Location Matching: https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ ; 7th place: https://future-architect.github.io/articles/20220720a/ ; stacking summary: https://www.jingsailian.com/news/135346.html
- Shopee solution: https://github.com/jingxuanyang/Shopee-Product-Matching
- Ye et al., Optimizing F-measure (ICML 2012): https://www.comp.nus.edu.sg/~leews/publications/fscore.pdf
- Waegeman et al., Bayes-optimal F-measure (JMLR 2014): https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
- FAMER clustering (Saeedi, Peukert, Rahm): https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf
- Learn to Not Link (ACL Findings 2023): https://ar5iv.labs.arxiv.org/html/2305.15725
- Ovadia et al., calibration under shift (NeurIPS 2019): https://arxiv.org/abs/1906.02530
- DADER (SIGMOD 2022): https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf
- Chari et al., transliteration-robust retrieval (2025): https://arxiv.org/pdf/2505.08411
- ByT5 (TACL 2022): https://arxiv.org/html/2105.13626v3
- Code-switching PAWS-X (IJCAI 2024): https://www.ijcai.org/proceedings/2024/0706.pdf
- mmBERT: https://arxiv.org/html/2509.06888
- Model cards:
  - https://huggingface.co/Qwen/Qwen3-Embedding-8B
  - https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
  - https://qwenlm.github.io/blog/qwen3/
  - https://huggingface.co/BAAI/bge-m3
  - https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2
  - https://huggingface.co/jhu-clsp/mmBERT-base
- Tools:
  - uroman: https://github.com/isi-nlp/uroman
  - anyascii: https://github.com/anyascii/anyascii
  - IndicXlit: https://github.com/AI4Bharat/IndicXlit
  - GlotLID: https://huggingface.co/cis-lmu/glotlid
  - LightGBM parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
  - sentence-transformers losses: https://sbert.net/docs/sentence_transformer/loss_overview.html
