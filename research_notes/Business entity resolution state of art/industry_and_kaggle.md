# Industry and Kaggle: how winners and production systems do business / place (POI) entity resolution

Scope note for the report writer: this file covers (1) the Kaggle "Foursquare - Location Matching" competition (2022), (2) other competition entity-resolution (ER) solutions (Shopee 2021, SIGMOD contests), (3) production/industry systems (Yelp, Facebook/Meta, Microsoft, Overture Maps, D&B, Senzing, Splink, Zingg, Tamr, Quantexa, Amazon), and (4) the cross-cutting tricks. Kaggle write-ups were read in full from the raw forum posts (Kaggle's discussion API), not from aggregators. Our target setting: business name + address + country only, no coordinates, clean reference source S1 and noisy sources S2/S3, macro F0.5 averaged per reference entity.

Key definitions used below:
- **Foursquare metric**: for each record id, IoU (Jaccard) between the predicted set of matching ids and the true set, averaged over all ids. Like our macro-per-entity F0.5, it is a per-entity average, so small groups count as much as large ones.
- **Max IoU**: the IoU you would get if a perfect classifier kept exactly the true pairs among the generated candidates. It is the recall ceiling of the candidate-generation (blocking) step.
- **CV**: local cross-validation score; **LB**: Kaggle leaderboard score (public/private test split).
- **The leak**: a Kaggle platform bug put training rows into the test set (~67% of test rows overlapped), so the official leaderboard is inflated for teams that exploited it. Kaggle re-scored the top teams on the intended non-overlapping test set (see below). All "LB" numbers from Foursquare write-ups are on the overlapping test set unless stated otherwise.

---

## Q1. Kaggle "Foursquare - Location Matching" (2022): top solutions in detail

### Takeaway
Every top solution used the same skeleton: (a) generate ~25–130 candidates per record (up to 400 in 9th place) from two independent retrievers (geographic nearest neighbours + name-text nearest neighbours via TF-IDF or a fine-tuned embedding), (b) a cheap LightGBM filter to cut candidates by ~10x while keeping max IoU ≈0.98, (c) a heavy pairwise classifier (LightGBM/CatBoost/XGBoost with ~100–200 string-similarity features, ensembled with an XLM-RoBERTa / mDeBERTa cross-encoder that reads both records' text), and (d) graph post-processing over the match graph (merge overlapping neighbourhoods, n-hop paths, GNN, union-find), which alone was worth +0.005 to +0.022 IoU. On the leak-free re-score the top five were within 0.920–0.933 IoU, so architecture choice mattered less than candidate recall, text-similarity features, and graph post-processing.

### Cited Findings

**Competition facts and the leak**
- 1,079 teams, 1,290 submitting participants, 22,050 submissions; winners announced 21 July 2022 — [Kaggle recap (Addison Howard, 2022-07-21)](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338720)
- Kaggle confirmed "a mashup of dataset versions, which meant that rows in the training and test set overlapped" and re-ran top submissions on the intended non-overlapping data. Non-overlap private IoU: "Ri" 0.932782 (1st; was 10th on the leaked LB), "team merge master" 0.922937, "Ethan & qyxs & hyd" 0.922847, "re:waiwai" 0.920594 (was 1st at 0.977683 on the leaked LB), "Psi" 0.919698 (was 3rd at 0.967847) — [Kaggle investigation update (2022-07-18)](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338035)
- Train ≈ 1.1M records, test ≈ 600k records — [4th place](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810); [3rd place](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338112)

**1st place (team re:waiwai, posted by Takoi, 2022-07-09)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)
- Validation: 2-fold CV (the later-cited "GroupKFold by point_of_interest" is the standard among top teams); final models trained on both folds.
- Stage 1 (candidates): for each id, 100 candidates by Euclidean lat/lon distance and 100 by cosine similarity of `bert-base-multilingual-uncased` name embeddings (kNN with cuML). A small LightGBM with only a few features (Jaro distance of names, Jaro of categories) kept top 20 from each list → ~40 candidates/id. Max IoU 0.979. GPU ForestInference used for speed.
- Stage 2: ~120 features — Levenshtein and Jaro-Winkler similarity, **per-id statistics of those similarities (max/min/mean over all candidates of the same id) and ratios of a pair's similarity to those statistics**, Euclidean distance, SVD-reduced BERT name embedding. LightGBM CV 0.875. Threshold 0.01 cut candidates to ~10%.
- Stage 3: CatBoost on same features CV 0.878; `xlm-roberta-large` (text = name + categories + address + city + state, plus ~70 stage-2 numeric features, 3 epochs); `mdeberta-v3-base` (same text, ~90 numeric features incl. Manhattan/haversine distances, trained with FGM adversarial training + EMA, 4 epochs) CV 0.907. Weighted ensemble (LGB 0.01, CatBoost 0.32, XLM-R-large 0.29, mDeBERTa 0.38) CV 0.911, threshold 0.5.
- Stage 4 (post-processing): "Compare the matches of two ids and merge the matches of two ids if the common id exceeds 50% from either side", then re-score the newly created pairs with xlm-roberta-large (threshold 0.02). CV 0.911 → 0.9166 (+0.0056).
- Leak exploitation (joining test to train on name + lat + lon, adding all train-POI true positives, removing false positives) moved a stage-2 submission from LB 0.900 → 0.943 → 0.971. Final private: 0.941 without leak merge, 0.977 with it.
- Component gains on CV (their numbers): GBDT 0.875 → transformer 0.907 (+0.032) → ensemble 0.911 (+0.004) → graph merge + re-score 0.9166 (+0.0056).

**2nd place (T0m, tkm2261, yukia18, ria, colum2131; 2022-07-09)** — [summary](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336062); [candidate/blocking part](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072); [GBDT+BERT part](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090)
- Candidates: 128 nearest by lat/lon and 128 nearest by name TF-IDF cosine, then a total of 128 chosen "with country-specific optimized ratios". Max IoU 0.9895.
- "Transformer candidate blocking": a neural model over absolute features (lat, lon, country embedding, category embedding, xlm-roberta-base name embedding, TF-IDF-SVD vectors) and relative features (edit distances on name/address/zip/phone/url/categories, TF-IDF-SVD cosine). Single-model LB 0.918. With threshold 0.005 + union-find, average candidates/id fell to 4.1 while max IoU stayed 0.986 (from 0.9895).
- Stack: XGBoost + LightGBM + xlm-roberta-base cross-encoder. XGBoost features: raw lat/lon, ordinal country and category; for each of name/categories/address/city/state/zip/url/phone: gestalt, Levenshtein, Jaro, RapidFuzz simple/partial/token_set/token_sort/token/partial_token/WRatio/QRatio; Jaccard/Dice/Simpson on token sets; TF-IDF cosine (word 1-gram and char 1–3-gram) on name/categories/address/all; plus the transformer's probability as a feature (stacking).
- BERT input: `country [SEP] name1 [SEP] category1 [SEP][SEP] name2 [SEP] category2`, with log haversine distance concatenated to the CLS vector before a 2-layer head; dropout set to 0; 5 epochs.
- Scores: LB 0.949 without leak post-process, 0.971/0.972 with it. They deliberately folded by `src_id` (not by POI) to exploit the overlap.

**3rd place (Psi = Philipp Singer, 2022-07-19)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338112)
- Deep-learning only. Stage 1: `xlm-roberta-large` trained with **ArcFace** (classes = point_of_interest) on a serialized record `name </s> category </s> lon digits split every 3 </s> lat </s> zip </s> city </s> address`; all-pairs similarity above a threshold gives candidates; ArcFace "clearly superior" to triplet losses but hard to stabilise.
- Stage 2: pairwise model over both records' columns interleaved in one sequence (he calls it "bi-encoder", but the input concatenates both records, i.e. a cross-encoder); trained on stage-1 in-sample candidates, all true positives plus progressively more false positives.
- Validation insight: retrieval scores depend on the size of the candidate pool, so he held out 600k records with unique POIs (same size as test) as validation; he later relied on public LB.
- Final = blend of stage-1 and stage-2 probabilities; "minor QE [query expansion] post processing". Non-overlap private 0.9197 (5th).

**4th place (Vincent Schuler, ymatioun, theoviel; 2022-07-08)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810)
- Cleaning: remove special characters/spaces, `unidecode`, lowercase; grouped synonymous categories ("gyms", "gyms or fitness centers"...) and computed per-category-group distance radii containing 50/75/90/95% of true matches; grouped equivalent names/cities/states.
- Candidates from many rules: near neighbours, name similarity, shared name words, grouped-category equality, cleaned phone equality, same address, TF-IDF. **TF-IDF added "less than 3% of true matches" and hard false positives.**
- 5-fold LightGBM with few features removed pairs below 0.007; then 200+ features into a 20-fold LightGBM. XGBoost/CatBoost stacking gave "very poor" gain.
- Post-processing key idea: **"adapt the thresholds to the sizes of the groups we were merging"** — merging two singletons differs from merging two groups of 10.
- 0.939 (15th) without leak → 0.957 with leak (key: cleaned_name + round(lat,5) + round(lon,5)).
- Machine translation was "not a game-changer compared to unidecode" (except pykakasi for Japanese); offline reverse geocoding not useful.

**6th place (tomo20180402, 2022-08-28)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/348399)
- 28 candidates/id (IoU ceiling 0.978): KD-tree 6,500 spatial neighbours re-ranked by a weighted sum of sqrt distance, 1−Jaro-Winkler, normalised Levenshtein, 1−Simpson on name and category tokens.
- Features include count encodings of name/address/url/phone, language-ID of name, and **rank of each feature among the 28 candidates of the same id**. Single CatBoost.
- Post-processing "graph probability convolution": new P(A–B) = weighted average over A's confident neighbours C of P(C–B), weighted by P(A–C) (edges ≥0.5), then add 2-hop nodes with prob ≥0.5; repeated twice.

**7th place (nadare, 2022-07-08)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335800); [summary blog in Japanese](https://future-architect.github.io/articles/20220720a/)
- 32 candidates from 5 retrievers (4 haversine, 12 regression-weighted haversine+embedding, 4 word-level name, 8 char-level name, 4 name-embedding). Retrieval max IoU 0.9778; **with post-processing max IoU 0.9935** (graph expansion recovers matches missed by retrieval).
- Features: gestalt, Levenshtein, Jaro-Winkler, ROUGE-N/L, per-language normalisation, address NaNs filled from 3 nearest neighbours; custom embeddings trained jointly with skip-gram, category metric learning and SimCSE with hard negatives (mix-embedding precision@16 0.8997).
- LightGBM with num_leaves 4095, lr 0.1, 2,000 iterations; FP-weighted samples; GPU ForestInference ~100x faster.
- Post-processing: union-find components, prune by betweenness centrality, predict pairs within graph distance 2.

**9th place (taksai, 2022-07-11)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415)
- 400 candidates/id (350 lat/lon kNN + 50 lat/lon+TF-IDF), max IoU 0.994, 420M pairs → light CatBoost (40 features, threshold 0.005) → 3M pairs, max IoU 0.987.
- Main CatBoost/LightGBM with ~140 features incl. 12 string-similarity methods and **2-fold target encoding of (name, name_match) and (categories, categories_match) value pairs** (CatBoost text encoding +0.010).
- Post-process: node score = average of neighbour predictions, threshold 0.4.
- LB timeline: CatBoost 0.925 → +post-process 0.930 → +LightGBM 0.933 → +train on all data 0.943 → +5-model ensemble 0.945 → +more iterations 0.948 (the last two steps are leak-inflated).

**11th place (sakusaku team, 2022-07-08)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335924)
- Metric-learning BERT (ArcMargin, margin ramped 0.2→0.8, 40 epochs) with normalised lat/lon concatenated; ensemble of xlm-roberta-large, LaBSE, paraphrase-multilingual-mpnet, RemBERT embeddings; weighted DBA/QE; 50 FAISS candidates + BallTree spatial candidates.
- One XGBoost per candidate rank (50 models) — memory-efficient.
- **Fill NaN text fields with text from the 5 nearest points: 0.931 → 0.940 (+0.009)**. Dijkstra post-processing (edge weight 1 − p, undirected by averaging both directions, collect nodes within a distance threshold): 0.945 → 0.947.

**12th place (Matt Motoki team)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336051)
- 25 candidates/id, max IoU 0.97675, from lat/lon top-k, (name embedding ⊕ lat/lon) top-k, (category embedding ⊕ lat/lon) top-k, using Universal Sentence Encoder (multilingual).
- Post-processing: match if a path exists with 1 hop at p>0.5, 2 hops at p>0.9, 3 hops >0.95, 4 hops >0.998, 5 hops >0.999 (stricter thresholds for longer paths).

**13th place (213tubo, GNN)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124)
- 60 candidates (30 TF-IDF text over name+categories+address+city+state, 30 haversine) → LightGBM filter to 3.7M pairs, max IoU 0.971.
- DITTO-style cross-encoder (xlm-roberta-base, mdeberta-v3-base) with input `name_1 [SEP] name_2 [COL] categories_1 [SEP] categories_2 [COL] address_1 ...`, plus haversine features; multi-task with country (+0.003); swap/delete/split word augmentation; flip-pair TTA.
- **GNN post-processing (PNAConv) on the 2-hop subgraph around each id, trained with a differentiable IoU loss + 0.1·BCE: public LB 0.924 → 0.946 (+0.022)**; 2-hop subgraph raised max IoU to 0.993. Earlier: 20→60 candidates and adding XLM-R took LB 0.907 → 0.924.

**8th place (kkanayj)** — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335928)
- Fine-tuned `all-MiniLM-L12-v2` (English model on unidecoded/pykakasi text beat multilingual models, attributed to limited data) with contrastive loss; **two-step fine-tuning where the second round adds hard negatives mined by the first-round model: candidate max IoU 0.97 → 0.986**. 10 distance-kNN + 20 embedding-kNN candidates; embedding cosine from 3 SBERT variants as LightGBM features.

**Aggregator caveat**
- Foursquare's own blog lists "Philipp Singer (1st Place)" and describes re:waiwai's pipeline as three-stage with Levenshtein/Jaro features, attributions that conflict with the Kaggle write-ups above (Singer = 3rd; the Levenshtein/Jaro pipeline is re:waiwai's = 1st; the GNN 0.907→0.946 story is the 13th-place solution) — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/). Prefer the Kaggle primary posts.

### Inferences
- **Transferable skeleton for our setting (no coordinates)**: the geographic retriever in Foursquare solutions must be replaced by an address-based retriever (TF-IDF char n-grams on normalised address, postcode/city exact blocks) plus a name retriever. The 4th-place finding that TF-IDF name retrieval added <3% true matches was *with* coordinates available; without coordinates, text retrievers become the primary source of recall, so the candidate union (name-TF-IDF ∪ address-TF-IDF ∪ fine-tuned embedding kNN) should be evaluated by max achievable per-entity F0.5, the analogue of "max IoU".
- **Cheap-filter-then-heavy-model** is universal: a few-feature LightGBM at a very low threshold (0.005–0.01) removes ~90% of pairs with ~0.3–0.7 pt ceiling loss (2nd: 0.9895→0.986; 9th: 0.994→0.987), which makes a cross-encoder affordable.
- **The GBDT + cross-encoder ensemble** was worth about +0.03 CV over GBDT alone in the 1st place solution (0.875/0.878 → 0.907/0.911). The cross-encoder input is simply both records' fields serialised with separators, plus a few numeric similarity features concatenated to the CLS vector.
- **Per-id context features** (similarity rank among the id's candidates, ratio to the id's max similarity, count encodings) recur in 1st, 6th and 9th place and cost almost nothing; they let the model express "this is the best candidate for this reference", which matches our one-reference-entity-at-a-time metric.
- **Graph post-processing is the highest-return late step** (+0.005 1st, +0.005 9th, +0.002 11th, +0.022 13th). With a clean reference S1, the natural analogue is: S2/S3 records that match each other strongly should inherit each other's S1 match (transitivity through the noisy side), with stricter thresholds for longer paths (12th place) and group-size-aware thresholds (4th place).
- Because the leak rewarded overfitting, "train longer / more iterations / train on all data" gains reported on the leaderboard (7th: 0.933→0.951; 9th: 0.945→0.948) should not be trusted as generalisable.

### Gaps
- The non-overlap winner "Ri" (0.9328, the best leak-free system) did not appear to publish a write-up among the forum's top-voted ~200 topics; its method is unknown.
- 5th and 10th place write-ups were not found. No team published a clean leak-free ablation on the non-overlap set.
- Winners' exact feature importances (e.g., how much address similarity contributed vs name vs distance) were not reported.

---

## Q2. Other competition ER solutions (Shopee 2021, SIGMOD contests, company-name matching)

### Takeaway
Shopee Product Matching (2021) showed that turning embeddings into matches (thresholding with a minimum of 2 matches per item, taking the union of per-modality match sets, iterative neighbourhood blending/query expansion) gave larger gains than better encoders, and that per-group-size sample weighting, mutual-edge filtering and graph pruning help a per-item-averaged F1 metric. SIGMOD 2022 was won by a small distilled transformer + FAISS + Jaccard re-rank under a CPU budget.

### Cited Findings
- **Shopee 1st place (harangdev & limerobot, 2021-05-11)**, metric = mean per-item F1 over matched sets. Public LB history: baseline 0.70 image-only / 0.64 text-only → concat image+text embeddings 0.724 → "min2" (always keep at least the nearest neighbour) 0.743 → normalise-then-concat 0.753 → full-data training 0.757 → **union of combined, image and text match sets + tuned thresholds 0.776** → **Iterative Neighbourhood Blending (INB)** + diverse text models 0.784 → using all three embeddings in INB stage 1 with jointly tuned thresholds 0.793. Authors: "individual model improvement helped less… Even when CV for each image/text model increased quite a lot, CV for ensemble and LB didn't increase much." Encoders trained with ArcFace; text models xlm-roberta-large/base, IndoBERT, mBERT; FAISS kNN k=51 — [Shopee 1st place write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)
- **Shopee 2nd place (lyakaap & tkm2261)**: stage 1 metric-learning similarities; stage 2 LightGBM + Graph Attention Network pair classifier. Timeline: 0.777 (TF-IDF + image emb + simple LGB) → +Indonesian-BERT 0.780 → +multimodal 0.783 → +mBERT/Paraphrase-XLM 0.788 → +PageRank 0.789 → +query expansion 0.790 → +GAT 0.792. Graph features: mean/std of top-K cosine similarity per item (K=5,10,15,30), normalised to handle train/test distribution shift. **Sample weight = 1/(label group size)^0.4** "important to predict smaller label groups". Ensembling: subtract each model's own best threshold, sum, accept if >0; **drop pairs without a mutual edge (A→B kept only if B→A kept)**. Betweenness-centrality edge pruning +0.001. cuDF/cuGraph/ForestInference: "40 min inference on CPU became 2 min" — [Shopee 2nd place write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)
- Shopee 18th place: replacing each item's predictions with the average of its nearest neighbours' predictions; **forcing the predicted group-size distribution to match train's was "the largest single trick"** — [summary of Shopee tricks (Foursquare forum, 2022-06-06)](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/329472)
- **ACM SIGMOD Programming Contest 2022 (blocking for ER)**: won by team WBSG (Brinkmann & Peeters, Univ. Mannheim) out of 55 teams; pipeline = domain-specific normalisation and grouping of identical descriptions → embeddings from a fine-tuned, very small distilled transformer indexed in FAISS → ANN search → re-rank by average of neural similarity and Jaccard similarity; constraint 16-core CPU, 32 GB RAM, 35 minutes — [Univ. Mannheim news](https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/)
- A retrospective of the SIGMOD ER contests exists — [SIGMOD Record paper](https://dl.acm.org/doi/10.1145/3615952.3615965) (not read in full).
- NetOwl claims to have won the MITRE Multicultural Name Matching Challenge (vendor claim) — [NetOwl](https://www.netowl.com/name-matching-software/)

### Inferences
- For our macro-per-entity F0.5, Shopee's lessons map directly: (i) the decision rule on top of scores (per-entity thresholds, min-1 match per reference when the evidence is strong, union of rules from different signals) can matter more than a better encoder; (ii) group-size-aware weighting/thresholds protect small entities, which dominate a macro average; (iii) mutual-edge filtering is a precision tool, useful because F0.5 weights precision twice as much as recall.
- The SIGMOD 2022 result supports using a small distilled sentence-embedding model plus a lexical (Jaccard/TF-IDF) score for candidate generation under compute limits.

### Gaps
- No Amazon KDD Cup product-matching competition was found: the Amazon KDD Cup 2022 was a shopping-query relevance task, not record matching (not verified in this session; I found no source for a product-matching KDD Cup).
- No public company-name matching competition with a published winning solution was found beyond the vendor claim above.

---

## Q3. Industry systems: architectures, reported accuracy, signals

### Takeaway
Production place/business matchers share a retrieve → pairwise-score → cluster design with humans in the loop for uncertain pairs. Yelp and D&B are tuned for precision (Yelp targets F0.1). Facebook and Overture moved to learned embeddings for blocking plus a GBDT or metric-learning scorer. Rule/probabilistic tools (D&B MatchGrade, Splink, Senzing) weight evidence by how rare a value is. Published accuracy numbers are few and mostly self-reported.

### Cited Findings
**Yelp**
- 2014 "Business Match": input is partial business info (name, location, phone, address text); Elasticsearch retrieval with sub-queries per field (name TF-IDF, geo-distance, phone); a pointwise learning-to-rank regression model over the per-field ES component scores, trained on manually labelled past match requests; **F1 improved from 91% to 95%** vs hand-tuned scoring — [Yelp Engineering, 2014-12-10](https://engineeringblog.yelp.com/2014/12/learning-to-rank-for-business-matching.html)
- 2015 duplicate detection: Kafka-triggered Business Match for new/changed businesses → candidate pairs → scikit-learn Random Forest using name, location, phone, distance, synonym-aware field matching, edit distance and Jaccard, plus two intermediate classifiers (an NER model that detects professionals such as lawyers/doctors/agents, and a logistic-regression "word alignment" classifier measuring how discriminative the differing name words are). Labels crowdsourced (CrowdFlower + internal ops). Held-out: **F0.1 = 0.966, precision 99.1%, recall 27.7%**, vs exact-match baseline F0.1 = 0.915; precision prioritised "since merges are hard to undo"; >500,000 duplicates merged — [Yelp Engineering, "Seeing Double", 2015-05-27](https://engineeringblog.yelp.com/amp/2015/05/seeing-double-on-yelp.html)

**Facebook / Meta**
- "Place Deduplication with Embeddings" (Yang, Hoang, Mikolov, Han; WWW 2019): signals = name, address, coordinates, category. Training data: ~4.6M labelled pairs from 19 sources (curation, crowdsourcing, user feedback) covering ~4M places, of noisy and varying quality; only ~10⁻⁵ percent of pairs labelled. Test: a 47K-page golden set curated for production quality evaluation. Two steps: embedding-based candidate fetch (kNN), then pairwise duplicate prediction. Name/address encoded by unsupervised embeddings. **Concatenating raw coordinates and category dummies to the embedding "not helpful and even lead to worse performance"**; instead used graph smoothing over same-grid-bin and same-category edges. Supervised metric learning with contrastive loss + batch-wise hard-negative sampling + source-oriented attentive weighting + soft-clustering label denoising. Pairwise accuracy: LR 0.7514, SVM 0.7699, RF 0.7786, **GBDT 0.7828 vs full model 0.9279**; ablation of candidate-fetch recall@K: unsupervised features 0.6476 → contrastive 0.7534 → +hard sampling 0.8434 → +attentive 0.8554 → +denoising 0.9119 — [arXiv 1910.04861](https://arxiv.org/abs/1910.04861)
- An earlier unsupervised Facebook places-dedup method is reported to beat simple TF-IDF models — ["Deduplicating a places database", WWW 2014](https://dl.acm.org/doi/10.1145/2566486.2568034) (abstract only seen).

**Overture Maps Foundation (AWS, Meta, Microsoft, TomTom), places conflation (docs, 2026)**
- Pipeline per provider feed: (1) history matches against the existing corpus ("roughly 90% of feed records are history matches; only the remaining ~10% of new records continue"); (2) high-recall blocking using "embedding representations of each place's name, category (taxonomy), and address, together with distance thresholds"; (3) pairwise **XGBoost** over name (string + embedding), address, website, phone, category-embedding and spatial features; (4) clustering, with the record from the source having the most matches in the cluster promoted. Training labels: "a large set of human- and LLM-generated match labels sampled across providers, countries, and categories"; LLM labelling uses an iteratively optimised prompt with an evidence hierarchy (functional identity, phone, name uniqueness and distance, then supporting signals such as website, category, brand); two independent LLMs label each pair, agreements accepted automatically, disagreements sent to humans. Analysis found **undermatching** for records >100 m apart or with only moderately similar names; the July 2026 release upgraded blocking to address it — [Overture Places guide](https://docs.overturemaps.org/guides/places/)

**Microsoft (Bing Maps)**
- Zheng et al., ACM SIGSPATIAL GIS 2010 (Best Paper): candidate selection → features (name similarity, address similarity, category similarity) → trained classifier; "both the precision and recall of our method exceeded 90%" on large real datasets — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/detecting-nearly-duplicated-records-in-location-datasets/). Search snippet also reports manual detection at 85% accuracy vs 98% for the ML method on tourism attractions — [ResearchGate](https://www.researchgate.net/publication/221589504_Detecting_nearly_duplicated_records_in_location_datasets) (not verified against full text).

**Dun & Bradstreet (reference-data matching, closest analogue to S1-as-reference)**
- Matching = connecting customer records to the D&B reference (D-U-N-S) data; key inputs are business name, street number, street name, city, state, mailing address, telephone. Output: a Confidence Code (1–10) mapped from a MatchGrade pattern, plus a Match Data Profile saying which data was used — [D&B "The Basics on Data Matching" PDF](https://dnb.com/content/dam/english/dnb-data-insight/DNB_The_Basics_On_Data_Matching.pdf)
- MatchGrade = 11 component grades in order: Name, Street Number, Street Name, City, State, PO Box, Phone, Postal Code, Density, Uniqueness, SIC; Match Data Profile covers 14 elements (adds DUNS, National ID, URL). Confidence code "from 4 (low) up to 10 (high)" in API responses — [D&B Direct 2.0 Match REST docs](https://docs.dnb.com/direct/2.0/en-US/company/latest/match/rest-API)
- Grade letters: A = same, B = similar, F = different, Z = one or both null; process phases: cleansing, parsing, standardising, candidate retrieval, evaluation; confidence 4–10 is the standard range for batch matching — [D&B/Salesforce entity matching doc](https://appexchange.salesforce.com/partners/servlet/servlet.FileDownload?file=00P3A00000eR5L2UAK) (via search snippet; not opened in full)

**Senzing (principle-based ER)**
- Each attribute gets three behaviours: frequency (F1 = one entity, FF = few, FM = many, FVM = very many; NAME special), exclusivity, stability. Preconfigured examples: ADDRESS FF, PHONE FF, WEBSITE FF, TAX_ID F1. ~35 general-purpose principles; no training or tuning required; values seen on too many entities are auto-labelled "generic" and prior decisions re-evaluated — [Senzing principle-based ER PDF](https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf)
- Entity-centric learning: matches a new record against the accumulated entity (all name/address/phone variants seen), not record-to-record — [Senzing ML page](https://senzing.com/how-senzing-uses-machine-learning-for-entity-resolution/)

**Splink (UK Ministry of Justice)**
- Fellegi-Sunter model with parameters estimated by EM, unsupervised ("training data is not required"); links 1M records on a laptop in under two minutes with DuckDB, 100M+ records on Spark/Athena — [Robin Linacre, 2020, updated 2022](https://www.robinlinacre.com/introducing_splink/)
- Term-frequency adjustment: a value-specific u-probability estimated from the value's frequency replaces the average one, so matching on a rare value earns more weight than matching on a common one; `tf_adjustment_weight` (0–1) scales it for fuzzy levels; `tf_minimum_u_value` floors it — [Splink TF docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)

**Zingg, Tamr, Quantexa, AWS**
- Zingg learns a blocking model; "Typical Zingg comparisons are 0.05-1% of the possible problem space"; active learning surfaces the pairs the model is least sure about; vendor claims 30–50 labelled pairs suffice — [Zingg GitHub](https://github.com/zinggAI/zingg); [Zingg blog part 4](https://www.zingg.ai/post/entity-resolution-at-scale-part-4-thresholds-active-learning)
- Tamr: labelled pairs → ML model → clusters → human curation that merges/splits clusters; verified clusters are the unit of expert feedback and are fed back as training signal; confidence thresholds gate auto-merges — [Tamr docs](https://docs.tamr.com/new/docs/overall-workflow-mastering); [Tamr patents](https://www.tamr.com/patents)
- Quantexa claims "99% data matching accuracy in an independent test with Dun & Bradstreet" (vendor claim, no method given) — [Quantexa ER page](https://www.quantexa.com/platform/entity-resolution-software/)
- AWS Entity Resolution offers rule-based, ML-based and data-provider matching — [AWS](https://aws.amazon.com/entity-resolution/)

**Amazon research**
- AutoKnow (KDD 2020): product-knowledge-graph system (taxonomy, extraction, cleaning, synonyms); +~200% facts in consumables graph, product-type identification 87.7% accuracy, >11K product types. It is not a business-record matcher and no ER-specific numbers were found — [Amazon Science blog](https://www.amazon.science/blog/building-product-graphs-automatically); [paper](https://www.amazon.science/publications/autoknow-self-driving-knowledge-collection-for-products-of-thousands-of-types)
- Ditto (Megagon Labs, VLDB 2021; not Amazon): pre-trained transformer cross-encoder for entity matching, up to 29% F1 over prior SOTA, +9.8% from domain-knowledge injection, summarisation and augmentation; **matched two company datasets of 789K and 412K records at F1 96.5%** — [arXiv 2004.00584](https://arxiv.org/abs/2004.00584)

### Inferences
- **The D&B design is the closest industrial analogue** to our task (clean reference file + noisy inquiry records, name + address, no coordinates): per-field categorical grades (same/similar/different/missing) for name, street number, street name, city, state, postcode, phone, plus density and uniqueness grades, mapped to a confidence level with a tunable auto-accept cut-off. Encoding our pairwise features as per-field A/B/F/Z grades (including an explicit "missing" state) and adding a "density/uniqueness" feature (how many S1 entities share the name or the street) is directly transferable.
- **Frequency/uniqueness weighting** appears independently in D&B (Uniqueness grade), Senzing (F1/FF/FM, generic detection), Splink (TF adjustment), Yelp (name word discriminativeness classifier) and Overture (name uniqueness in LLM evidence hierarchy). For chains (many S1 entities with the same brand name), name similarity must be discounted by name frequency and address evidence must carry the decision.
- **Precision-first operating points** (Yelp F0.1, D&B confidence cut-offs, Overture/Tamr human review of uncertain pairs) match F0.5's precision emphasis; the Yelp result (P 99.1% at R 27.7%) shows how far a precision-optimised system trades recall, and F0.5 will want a less extreme point.
- Facebook found raw coordinate features unhelpful when concatenated to embeddings; this weakly suggests that losing coordinates costs less than one might fear when name/address embeddings are good, though Overture's undermatching at >100 m shows geography is still a key blocking signal when present.
- LLM-generated labels with two-model agreement (Overture) are a practical way to create extra labelled pairs, e.g., for an unseen country in test.

### Gaps
- **Google** (Maps/Places dedup, Knowledge Graph reconciliation): no primary technical publication with architecture or accuracy was found in this session.
- **Uber** POI conflation: no primary engineering blog post was found; search results were irrelevant.
- **Amazon business identity / seller-business matching**: no Amazon Science publication was found.
- **Foursquare's own production matcher**: its blog only says it combines "billions of inputs" with proprietary ML and human "Superuser" verification; no metrics — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Vendor accuracy claims (Quantexa 99%, Zingg 30–50 labels) have no disclosed methodology.
- The D&B Salesforce PDF (A/B/F/Z definitions, 4–10 range guidance) was seen only as a search snippet.

---

## Q4. Common tricks that gave large gains

### Takeaway
The recurring high-return tricks are: multi-retriever candidate unions sized by measured max-IoU; a cheap filter model; many string-similarity features plus per-anchor context features; a GBDT + transformer cross-encoder ensemble (stacking the transformer probability into the GBDT); graph post-processing over the predicted match graph; hard-negative mining for embedding models; and group-size-aware weighting/thresholds. Pseudo-labelling was not a major reported driver in these write-ups.

### Cited Findings
- **Graph post-processing gains**: 1st place neighbourhood merge (>50% overlap) + re-score CV +0.0056 ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)); 9th neighbour-average +0.005 LB ([9th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415)); 11th Dijkstra +0.002 ([11th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335924)); 13th GNN with IoU loss +0.022 ([13th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124)); 7th: post-processing lifts the ceiling from 0.9778 to 0.9935 max IoU ([7th](https://future-architect.github.io/articles/20220720a/)); Shopee 2nd betweenness pruning +0.001 and mutual-edge filter ([Shopee 2nd](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)).
- **Second-stage features from first-stage predictions (stacking)**: 2nd place fed transformer probability into XGBoost ([2nd, colum2131 part](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090)); 13th GNN used LGB/XLM-R/mDeBERTa predictions as edge features ([13th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124)); 4th tried a 3rd LightGBM on 2nd-stage score + "number of high-scored neighbours" but kept group-size thresholds instead ([4th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810)); Shopee 2nd used top-K similarity statistics and PageRank as graph features (+0.001) ([Shopee 2nd](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)).
- **GBDT + transformer ensemble**: 1st place CV 0.875 (LGB) / 0.878 (CatBoost) → 0.907 (mDeBERTa) → 0.911 ensemble ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)).
- **Per-anchor context features**: similarity max/min/mean per id and ratios ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)); rank among the id's 28 candidates ([6th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/348399)); value-pair target encoding ([9th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415)).
- **Hard-negative mining for the retriever**: 8th place second fine-tuning round with self-mined negatives raised candidate max IoU 0.97 → 0.986 ([8th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335928)); Facebook hard sampling raised recall@K 0.7534 → 0.8434 ([arXiv 1910.04861](https://arxiv.org/abs/1910.04861)).
- **Missing-field imputation from neighbours**: +0.009 LB ([11th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335924)).
- **Text normalisation**: unidecode as good as translation; pykakasi for Japanese ([4th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810)); normalised English SBERT beat multilingual SBERT at this data size ([8th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335928)).
- **Group-size awareness**: thresholds adapted to merged group sizes ([4th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810)); sample weight 1/(group size)^0.4 ([Shopee 2nd](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)); forcing predicted group-size distribution to match train was the largest single trick for Shopee 18th ([Shopee tricks summary](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/329472)).
- **Decision-rule tricks on embeddings**: min2, union of per-signal match sets, iterative neighbourhood blending: 0.724 → 0.793 on Shopee public LB, larger than encoder improvements ([Shopee 1st](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)).
- **Adversarial training / EMA for cross-encoders**: FGM + EMA in the 1st place's best single model (CV 0.907) ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)).
- **Engineering**: GPU ForestInference ~100x faster GBDT inference ([7th blog](https://future-architect.github.io/articles/20220720a/)); "40 min … became 2 min" ([Shopee 2nd](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)).
- **Validation design**: Psi sized the validation candidate pool like test (600k records with unique POIs) because retrieval difficulty scales with pool size ([3rd](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338112)); folds grouped by entity (GroupKFold on point_of_interest) were standard ([13th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124); [9th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415)).

### Inferences
- A ranked list of what to borrow for our task, by reported gain per unit of effort: (1) a candidate union measured by max per-entity F0.5; (2) per-reference context features (rank, ratio-to-best, number of close candidates); (3) a GBDT + cross-encoder ensemble with the cross-encoder probability stacked into the GBDT; (4) transitivity / neighbourhood-merge post-processing through S2–S3 links with length-dependent thresholds; (5) group-size / entity-size aware thresholds or weights for the macro metric; (6) hard-negative-mined fine-tuning of the name+address embedding retriever.
- Validation should mimic test: fold by reference entity, and keep the candidate pool (number of S1 references per country) at test scale, since Psi and the Foursquare CV–LB gaps show retrieval difficulty depends on pool size.
- Pseudo-labelling: none of the top Foursquare or Shopee write-ups credited pseudo-labelling with a gain; Overture's LLM double-labelling is the closest industrial equivalent.

### Gaps
- No component-level ablation on a leak-free Foursquare test set exists; gains above are from each team's own CV or the overlapping LB.
- No published measurement of how much coordinates contribute vs name/address in Foursquare solutions (no one ablated removing lat/lon), so the cost of our no-coordinates setting cannot be quantified from these sources.
