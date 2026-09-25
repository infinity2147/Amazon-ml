# Blocking / Candidate Generation for Entity Resolution at 1-10M+ Record Scale

Context for the reader: the system being improved currently retrieves candidates with sparse TF-IDF (word tokens + character 5-grams), top-k in both directions, per country; then a LightGBM ranker keeps 20 candidates per reference entity; plus an exact (name, house number) key view. Pair recall at 20 candidates/entity (share of true matching pairs that survive into the 20-candidate list): US 0.990, India 0.973. India misses fall into four groups: common business names with empty or truncated addresses, locality substitutions, pseudo-word replacement names, and native-script names transliterated to Latin.

Terms used below (defined once):
- **Recall / Pair Completeness (PC)**: share of true matching pairs that appear in the candidate set.
- **Precision / Pair Quality (PQ)**: share of candidate pairs that are true matches.
- **CSSR (candidate set size ratio)**: candidate pairs / |A x B| (all possible pairs). **Reduction Ratio (RR)** = 1 - CSSR.
- **top-k blocking**: for each record, keep its k most similar records (as opposed to a similarity threshold).
- **max IoU** (Foursquare competition): the best score the final matcher could reach if it classified every retrieved candidate perfectly, so it measures how complete the candidate set is.

## 1. Dense retrieval for ER blocking (DeepBlocker, Sudowoodo, SC-Block, UniBlocker, multilingual encoders): recall@k vs TF-IDF/BM25, and at what scale

### Takeaway
On published benchmarks, *unsupervised or self-supervised* dense blockers (DeepBlocker, Sudowoodo, UniBlocker, off-the-shelf sentence embeddings) do **not** beat a well-tuned top-k TF-IDF/BM25 blocker; Sparkly (Lucene BM25, character 3-grams) beats DeepBlocker's best models on all 15 datasets tested. A *supervised* contrastive bi-encoder trained on labelled match pairs (SC-Block, a fine-tuned RoBERTa) is the one dense method with clear wins over BM25, and its advantage *grows* with vocabulary size and scale (WDC-B up to 2.1M records: about 16 points higher recall than BM25 at k=5). Dense and sparse retrieval find partly different matches, so combining them (union) raises recall.

### Cited Findings
**DeepBlocker (Thirumuruganathan et al., PVLDB 14, 2021)**
- Tests 8 deep-learning blockers built on fastText embeddings. Autoencoder is best on structured and dirty data and Hybrid (Autoencoder + cross-tuple training) is best on textual data. Both beat AutoBlock and beat RBB (an industrial rule-based blocker) on textual/dirty data, and are "comparable" on structured data — [DeepBlocker, PVLDB 14](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- On structured Hospital and Song-Song (1M tuples), recall is comparable to RBB, but the DL candidate set is "much larger", so "it is not clear whether DL will consistently outperform RBB on structured datasets" — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- DL and non-DL blockers learn "highly overlapping but not quite the same" concepts. Union(DL, RBB) gives better recall with a minimal increase in output size — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- FAISS GPU vector pairing took under 1 minute at K=100 on all datasets except Song-Song (1M tuples), which took about 35 minutes (V100 GPU). Top-K cosine beats threshold cosine and is "much better than LSH" — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf); code (Python) at [qcri/DeepBlocker](https://github.com/qcri/DeepBlocker)

**Sparkly (Paulsen, Govind, Doan, PVLDB 16(6):1507-1519, 2023)**
- Method: Lucene BM25 over a bag of character 3-grams of concatenated attributes, top-k from the larger table into an index of the smaller one, distributed on Spark — [Sparkly paper](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Across 15 datasets, Sparkly-Manual beats DeepBlocker Autoencoder and Hybrid at every recall level. Example: Amazon-Google structured at 98% recall, Sparkly CSSR 2.5% vs 10% for both DL blockers. The gaps are bigger on textual data and smaller but "still quite significant" on dirty data. Restricting the DL blockers to the same attributes does not change this: Sparkly still wins on 14 of 15 datasets — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Sparkly recall: 92.5-100% at k=10, 96.4-100% at k=20, 98.7-100% at k=50. Output is capped at k x |B|. Songs (1M x 1M) structured: k=20 gives 20.0M pairs at 97.9% recall and k=50 gives 99.3%. Dirty Songs: k=20 gives 96.4% and k=50 gives 98.8% — [Sparkly Table 2](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Other blockers are unpredictable. JedAI PBW produced 4.2 billion pairs on Songs, and ran out of memory with >100 GB RAM on dirty Songs. JedAI JD recall ranges 35.4-96.4%. Union(DL,RBB) recall ranges 83-99.9% — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- The most important design decision is top-k rather than a threshold, because "similarity scores of many matching tuple pairs can be quite low" but matches still sit within each other's top-k. Top-k in both directions was judged to "significantly increase runtime yet improve recall only" marginally. Probing from the larger table gives higher recall at the same k — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Compared with kNN-cosine over 5-grams: Sparkly wins on 10 datasets, ties on 1 and is slightly worse on 4. Cosine over 3-grams is comparable to cosine over 5-grams, and both beat Jaccard — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Scale: 10M-tuple tables in under 100 minutes on 10 AWS m5.4xlarge nodes (16 cores and 64 GB each, $12.5). 26M tuples (WDC) in under 130 minutes on 30 nodes. A 10M-tuple index is 1.3-2 GB — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)

**Sudowoodo (Wang, Li, Wang; ICDE 2023; self-supervised contrastive RoBERTa)**
- Blocking, recall / number of candidate pairs vs DL-Block (DeepBlocker): Abt-Buy 88.6 / 3,276 vs 87.2 / 21,600. Amazon-Google 97.3 / 48,390 vs 97.1 / 68,200. DBLP-ACM 99.6 / 11,470 vs 99.6 / 13,100. DBLP-Scholar 98.4 / 257,052 vs 98.1 / 392,400. Walmart-Amazon 95.0 / 44,148 vs 92.2 / 51,100 — [Sudowoodo arXiv 2207.04122, Table VII](https://arxiv.org/pdf/2207.04122); code at [megagonlabs/sudowoodo](https://github.com/megagonlabs/sudowoodo)

**SC-Block (Brinkmann, Shraga, Bizer; arXiv 2303.03132, 2023)**
- Supervised contrastive loss over labelled match/non-match pairs, roberta-base encoder, FAISS nearest-neighbour search — [SC-Block](https://arxiv.org/pdf/2303.03132)
- Recall at fixed k=5 (test sets): Abt-Buy 99.5 vs BM25-trigram 98.1 vs BM25-word 92.2. Amazon-Google 97.4 / 94.4 / 93.6. Walmart-Amazon 95.9 / 96.9 / 95.9. WDC-B small (10k records) 71.5 / 52.5 / 59.4. WDC-B medium (205k) 66.4 / 46.6 / 53.8. WDC-B large (2.1M records, 6.9M unique tokens) 56.7 / timeout (>48h) / 41.7 — [SC-Block Table 3](https://arxiv.org/pdf/2303.03132)
- On the same benchmark, self-supervised dense blockers collapse at scale. SimCLR drops to 2.7% and Barlow Twins to 12.6% recall on WDC-B large. DeepBlocker Auto/CTT run out of memory on WDC-B large. Supervised SBERT (pair-wise loss) averages 39.8% recall. On WDC-B, BM25 misses on average 16.3% more pairs than SC-Block, which the authors attribute to the larger vocabulary — [SC-Block](https://arxiv.org/pdf/2303.03132)
- With k tuned to reach 99.5% recall on validation, SC-Block candidate sets are smaller. The full pipeline on WDC-B large went from 30 h to 8 h (SC-Block training took 5 minutes) — [SC-Block](https://arxiv.org/pdf/2303.03132)

**Off-the-shelf sentence embeddings, no fine-tuning (Zeakis, Papadakis, Skoutas, Koubarakis; PVLDB 16(9), 2023)**
- 12 language models over 17 datasets with FAISS HNSW kNN blocking. SentenceBERT-family models are best and S-GTR-T5 ranks first. On 8 of 10 datasets, S-GTR-T5 recall is 15% higher than DeepBlocker Autoencoder — [Zeakis et al. arXiv 2304.12329](https://arxiv.org/pdf/2304.12329)
- Scale test (synthetic 10K to 2M entities): S-GTR-T5 recall falls from 0.962 to 0.800 (-17%). fastText falls from 0.901 to 0.415. S-MiniLM and S-DistilRoBERTa fall from about 0.75 to about 0.45. BERT/RoBERTa/DistilBERT without sentence training fall "far below 0.2" at 2M — [Zeakis et al.](https://arxiv.org/pdf/2304.12329)

**UniBlocker (Wang et al., arXiv 2404.14831, 2024), dense blocking without domain training**
- Contrastive pre-training on GitTables tables, starting from a sentence-transformers checkpoint. Mean over 17 datasets at the smallest k reaching about 90% PC: UniBlocker PC 89.86 / PQ 20.29 / mean k 32.35. Sparkly-3gram: 89.73 / 22.58 / 33.24, with mAP 51.97 vs 49.60 for UniBlocker. Ensemble UniBlocker + Sparkly: PC 91.72 at a lower mean k of 23.53. The ensemble improves PC by up to 5% (cora). "Sparkly surpasses UniBlocker by about 2% mAP on average" — [UniBlocker Table 4](https://arxiv.org/pdf/2404.14831)
- Same table: DeepBlocker mean PC 80.03 needs mean k 61.8. Sudowoodo PC 82.62 needs k 53.2. Plain sentence-transformer PC 85.85 needs k 48.5 — [UniBlocker Table 3](https://arxiv.org/pdf/2404.14831)
- Sparse 3-gram blocking is faster for small inputs, but "the difference decreases as the number of records increases" (tested up to 10^6), because the number of 3-gram posting-list pairs grows non-linearly — [UniBlocker](https://arxiv.org/pdf/2404.14831)

**Multilingual encoders (licence and size; all fit the MIT/Apache-2.0, <=8B constraint)**
- Hugging Face model API: multilingual-e5-small (MIT, 118M params), multilingual-e5-base (MIT, 278M), multilingual-e5-large-instruct (MIT, 560M), LaBSE (Apache-2.0, 471M), gte-multilingual-base (Apache-2.0, 305M), paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M), BGE-M3 (MIT) — [HF API e5-small](https://huggingface.co/api/models/intfloat/multilingual-e5-small), [LaBSE](https://huggingface.co/api/models/sentence-transformers/LaBSE), [gte-multilingual-base](https://huggingface.co/api/models/Alibaba-NLP/gte-multilingual-base), [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- BGE-M3: XLM-RoBERTa based, 1024-dim, max 8192 tokens, 100+ languages. It produces dense, sparse (lexical weights) and ColBERT multi-vector outputs from one model — [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- BGE-M3 paper, MIRACL average nDCG@10: BM25 31.9, mE5-large 66.6, M3 dense 69.2, M3 sparse 53.9, M3 dense+sparse 70.4, all three 71.5. MKQA cross-lingual Recall@100: BM25 39.9, mE5-large 70.9, M3 dense 75.1, dense+sparse 75.3 — [BGE-M3 arXiv 2402.03216](https://arxiv.org/html/2402.03216)

**Neural LSH blocking**
- NLSHBlock (arXiv 2401.18064, Jan 2024) fine-tunes a pre-trained LM with an LSH-based loss to act as the hash function. The abstract claims "significant performance improvements" with no numbers — [NLSHBlock](https://arxiv.org/abs/2401.18064)

### Inferences
- The current TF-IDF (word + char 5-gram) design already matches what Sparkly found to be the strongest unsupervised blocker (char n-grams, top-k). Swapping it for an *unsupervised* dense retriever is unlikely to help. The evidence favours **adding** a *supervised* contrastive bi-encoder (SC-Block style) trained on the labelled US/India match pairs, and taking the **union** with TF-IDF candidates before the LightGBM ranker. Both DeepBlocker (Union(DL,RBB)) and UniBlocker (+5% PC ensemble) report gains from union.
- The India miss groups are the cases lexical overlap cannot solve by construction: transliteration, locality synonyms, and pseudo-word replacement names. A multilingual encoder (multilingual-e5 or BGE-M3) fine-tuned with in-batch negatives on known matching pairs is the natural fit. In the BGE-M3 cross-lingual results, dense Recall@100 is 75.1 vs BM25 39.9.
- Off-the-shelf embeddings lose recall steeply as the database grows (Zeakis: -17% for the best model, 10K to 2M). Expect a non-fine-tuned encoder to underperform at 1-10M per country. Fine-tuning (as in SC-Block) is the reason dense retrieval scales in that paper.
- Sparkly reports that searching top-k in both directions adds runtime for only small recall gains. The current system already does this, so the gain from raising k in the reverse direction is probably small.

### Gaps
- No paper found that benchmarks dense vs sparse blocking on business/POI names with addresses at 1-10M per country. Benchmarks are mostly products and bibliographic data of 1K-2M records.
- Sudowoodo and NLSHBlock report no numbers at >1M scale. The NLSHBlock abstract gives no recall figures.
- No published recall numbers found for LaBSE or multilingual-e5 used specifically as ER blockers.

## 2. Hybrid sparse+dense, learned blocking, meta-blocking, canopy, sorted neighbourhood, LSH/MinHash: reported PC / RR

### Takeaway
Classic hash/token blocking with meta-blocking (JedAI) produces unpredictable candidate-set sizes and recall at million-record scale. For a fixed per-entity budget, top-k similarity search is the most predictable choice. Hybrid (union of sparse and dense) consistently adds a few points of recall. Learned supervised meta-blocking re-ranks and prunes existing candidates rather than finding new ones, which is the role the LightGBM 20-cap already plays.

### Cited Findings
- Survey reference: Papadakis, Skoutas, Thanos, Palpanas, "Blocking and Filtering Techniques for Entity Resolution: A Survey", ACM Computing Surveys 53(2):31, 2020. It covers standard blocking, sorted neighbourhood, canopy clustering, q-gram/suffix blocking, LSH, meta-blocking and filtering, evaluated by PC, PQ and RR — [arXiv 1905.06167](https://arxiv.org/abs/1905.06167); [CSUR PDF](https://helios2.mi.parisdescartes.fr/~themisp/publications/csur20-blockingfiltering.pdf)
- Meta-blocking "goes beyond comparison propagation" by discarding redundant comparisons and "the vast majority of the superfluous ones" — [survey](https://arxiv.org/abs/1905.06167)
- Generalized Supervised Meta-blocking (Gagliardelli, Papadakis, Simonini, Bergamaschi, Palpanas; arXiv 2204.08801, 2022) combines many weighting features in a probabilistic classifier whose scores feed any pruning algorithm. The abstract gives no numbers — [arXiv 2204.08801](https://arxiv.org/abs/2204.08801)
- JedAI methods measured in the Sparkly paper. PBW recall 74.5-100%, output up to 4.2 billion pairs (Songs 1M x 1M), out of memory with >100 GB RAM on dirty Songs. DBW recall down to 84.7%, output up to 454.5M. JD output is reasonable but recall is 35.4-96.4% — [Sparkly Table 2](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- LSH: DeepBlocker found top-K cosine "much better than LSH" for pairing embeddings — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- Hybrid: UniBlocker + Sparkly ensemble reaches mean PC 91.72 at mean k 23.5, vs 89.9 / 89.7 for each alone — [UniBlocker](https://arxiv.org/pdf/2404.14831). BGE-M3 dense+sparse beats dense alone on MIRACL (70.4 vs 69.2 nDCG@10) — [BGE-M3](https://arxiv.org/html/2402.03216). DeepBlocker Union(DL,RBB) had the best recall — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)

### Inferences
- Canopy clustering, sorted neighbourhood and MinHash-LSH are mainly useful when the pair budget must be driven by thresholds, or when no ANN index is available. With a 64-core CPU and a GPU, exact or near-exact top-k search over sparse and dense vectors is feasible per country (see section 3). The literature above suggests top-k search dominates these methods in predictability.
- A cheap hybrid is to keep separate top-k lists per retriever (TF-IDF name, TF-IDF name+address, dense), take their union, and let the existing LightGBM ranker pick 20. The 7th-place Foursquare solution (section 4) used exactly this "multiple small k lists" design.

### Gaps
- Could not retrieve per-dataset PC/RR numbers from the Papadakis survey or the supervised meta-blocking paper within the call budget. Only abstracts were read.
- No quantitative head-to-head found for canopy or sorted-neighbourhood vs top-k TF-IDF at >1M records.

## 3. Practical tooling: FAISS (IVF/HNSW, GPU/CPU) at 10M vectors; sparse_dot_topn; encoder throughput on a 24 GB GPU

### Takeaway
At 1-10M vectors of 384-1024 dimensions, FAISS runs flat (exact) search in batches on one GPU, or IVF/HNSW on CPU, with query latency under 1 ms at recall@10 of 0.92-0.95. Vector search will not be the bottleneck; encoding the records with a transformer will be. For sparse TF-IDF top-k, sparse_dot_topn (multi-threaded with OpenMP) is the standard tool, used by ING on about 10M x 20M company names.

### Cited Findings
- FAISS GPU is "5x - 10x faster than the corresponding CPU implementation". k and nprobe must be <= 2048 on GPU, and performance "suffers" for k above about 512. float16 is supported and "Recall@N seems mostly unaffected by float16". Typical batch sizes are around 8192 — [Faiss wiki: Faiss on the GPU](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- FAISS + NVIDIA cuVS (H100, May 2025) at 95% recall@10. 100M x 96-dim: IVF-Flat build 37.9 s, search 0.39 ms. 5M x 1536-dim: IVF-Flat build 15.2 s, IVF-PQ build 9.0 s, CAGRA graph build 89.7 s (12.3x faster than CPU HNSW), CAGRA search 0.15 ms — [Meta Engineering](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- FAISS library guidance: graph (HNSW) indexes suit cases without a memory limit. Beyond 10M vectors, build time becomes limiting, and IVF is the only option when compression is needed — [Faiss paper arXiv 2401.08281](https://arxiv.org/html/2401.08281v4)
- One third-party benchmark (10M random 768-dim vectors, CPU): HNSW p95 latency 0.42 ms vs IVF 0.83 ms vs Flat 44.7 ms. HNSW used 3x the RAM and built 2.3x slower. IVF with nprobe=32 reaches recall@10 >0.92 at 0.41 ms. Treat this as indicative only (vendor blog, random vectors) — [Markaicode](https://markaicode.com/benchmarks/faiss-production-benchmark-latency/)
- DeepBlocker: FAISS GPU pairing at K=100 took about 35 minutes for Song-Song (1M x 1M) on a V100 — [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- sparse_dot_topn (ING, Apache-2.0): fused sparse matrix multiply + top-n selection with OpenMP multi-threading. It is up to 6x faster (8 cores, top-10, 20k x 193k TF-IDF matrices, Apple M2 Pro), and the new version is 1.5-4.0x faster than the old one. ING's use case is about 10M reference names x 20M names to match (2x10^14 comparisons) — [sparse_dot_topn GitHub](https://github.com/ing-bank/sparse_dot_topn); [ING Medium](https://medium.com/wbaa/yet-even-faster-string-comparison-d0fb2c7ae7a4)
- Sparkly alternative: Lucene/PyLucene BM25, which is optimised for top-k and not for thresholds. A 10M-tuple 3-gram index is 1.3-2 GB — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- Sentence-transformers efficiency docs: FP16 + Flash Attention 2 + input unpadding gives a median 3.87x speedup over FP32 without loss of quality. Very small models on a fast GPU "can see little gain or even a slight slowdown". The docs give no absolute sentences/second — [sbert.net efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)

### Inferences
- Rough compute estimate (not measured): exact inner-product top-k for 5M x 5M at 384 dims is about 2 x 5e6 x 5e6 x 384 ≈ 1.9e16 FLOPs. That is minutes of FP16 tensor-core time on an A5000 class GPU, so per-country exact flat search (batched, float16) is feasible and avoids ANN recall loss. The 384-dim vectors take 5M x 384 x 2 bytes ≈ 3.8 GB in fp16, which fits in 24 GB even on a shared card.
- Encoding: short strings (name + address, under 64 tokens) with a 118M-param model in fp16 plausibly run at thousands of records per second on one A5000, so 10M records take roughly an hour or less. This must be measured, because no source gave A5000 numbers.

### Gaps
- No published sentences/second figures for multilingual-e5-small/base or BGE-M3 on an RTX A5000 were found.
- No FAISS benchmark on an A5000 specifically. The cuVS numbers are from an H100.

## 4. Kaggle Foursquare Location Matching (2022): how top solutions generated candidates

### Takeaway
Every top solution used multi-source candidate generation. Sources were geographic kNN (haversine on lat/lon), TF-IDF name similarity (word and character level), and fine-tuned name embeddings, each with small k (4-20), followed by a cheap first-stage LightGBM or Transformer filter that shrinks candidates before the expensive matcher. Candidate-stage max IoU was around 0.98-0.99. This is structurally the same as the system under study; the main difference is that Foursquare had coordinates, which let name-only records be retrieved by location.

### Cited Findings
- Dataset: about 1.5M place entries "heavily altered to include noise, duplications, and incorrect information" — [search summary of the competition](https://www.kaggle.com/competitions/foursquare-location-matching)
- 7th place (Future Architect): 32 candidates per record from five retrievers. Haversine distance gave 4, haversine + embedding regression 12, name word-level 4, name character-level 8, name embedding cosine 4. Retrieval-only max IoU 0.9778 at 32 candidates, 0.9935 after post-processing. Search ran on GPU in TensorFlow, and the fusion weights were learned with a ranking loss rather than logistic regression. About 600k test records x 32 gives about 19.2M pairs for LightGBM — [Future Architect blog (Japanese)](https://future-architect.github.io/articles/20220720a/)
- Team 2:30: coarse candidates by "geographic proximity and TFIDF scores", then "a Transformer-based blocking stage to effectively reduce candidates while maintaining high recall". Final ensemble of LightGBM + XGBoost + BERT models — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Team re:waiwai: candidates by text similarity and geographic proximity. A light-feature LightGBM cut candidates "in a memory-efficient way". The top 40 per id went to a second LightGBM with rich features (Levenshtein, Jaro-Winkler, lat/lon distance, SVD name embeddings) plus BERT models — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Yuki Uehara: text similarity + geo candidates, then LightGBM filtering, transformers and GNN post-processing. The score improved from 0.907 to 0.946 — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Philipp Singer: ArcFace metric-learning model for candidates, then a bi-encoder NLP model on those candidates — [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- Other write-ups: distance kNN (k=10) plus kNN over fine-tuned SentenceBERT embeddings (k=20), with embedding kNN described as "very important". One solution reached max IoU >0.99 with concatenated SBERT embeddings in kNN. Another used 12 candidates/id at max IoU about 0.983 — [search summary of Kaggle write-ups, e.g. 21st place](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/bulian-ai-21st-place-solution)
- **Conflict on who placed 1st:** a web summary credits Team 2:30 with 1st place. The Foursquare blog lists Philipp Singer first. The Kaggle write-up URL is titled "re-waiwai-1st-place-solution" — [Kaggle 1st place write-up URL](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution); [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/). Rank attributions here are unverified. The Kaggle page did not render, so its text could not be read.

### Inferences
- The "several small top-k lists, then union, then learned filter" pattern is well validated. The 7th-place split (char-level name 8, word-level name 4, embedding 4, geo) is a direct template for adding a dense name retriever and a name-only retriever next to the existing TF-IDF.
- Foursquare solutions relied on fine-tuned (ArcFace or contrastive) name embeddings, not off-the-shelf ones. This matches the SC-Block finding in section 1.

### Gaps
- The Kaggle write-up pages (1st, 16th, 21st) are rendered with JavaScript and could not be fetched. Exact per-stage recall and candidate counts for the top 3 teams remain unverified.
- No write-up found that reports recall *by country*, for example for India.

## 5. Records with very little information (name only, empty/truncated address): how others retrieve candidates

### Takeaway
No ER paper found that addresses this case head-on. In practice the fix is a *separate* retrieval channel whose key does not depend on the missing field, with the per-name candidate budget controlled by name frequency. Foursquare solutions used coordinates. For the India transliteration misses, standard options are transliteration to one shared script (IndicXlit, MIT) or a cross-lingual dense encoder.

### Cited Findings
- Foursquare solutions ran geo kNN (haversine) as a retriever independent of text, together with name-only character and word TF-IDF and name-embedding kNN — [Future Architect](https://future-architect.github.io/articles/20220720a/)
- Top-k retrieval works even when similarity scores of true matches are low, because matches stay within each other's top-k. Thresholds "kill off many matches" — [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
- BM25 fails when vocabularies are large and token overlap is small, while a supervised dense blocker recovers about 16% more pairs — [SC-Block](https://arxiv.org/pdf/2303.03132)
- IndicXlit (AI4Bharat, MIT licence, 11M-parameter transformer) transliterates in both directions (Roman to Indic and Indic to Roman) for 21 Indic languages. Top-1 accuracy on Aksharantar Indic-to-English ranges from 11.76% (Bengali) to 84.64%. On Dakshina English-to-Indic it ranges from 42.12% (Urdu) to 84.85% (Marathi) — [IndicXlit GitHub](https://github.com/AI4Bharat/IndicXlit)
- BGE-M3 (MIT) cross-lingual MKQA Recall@100 is 75.1 dense vs 39.9 BM25, which is evidence that multilingual dense encoders bridge scripts and languages where lexical matching fails — [BGE-M3](https://arxiv.org/html/2402.03216)

### Inferences
- For common business names with empty or truncated addresses, the likely fix is a dedicated **name-only channel restricted to a coarse location key** (city, pincode or locality, or lat/lon if available), with k set by name frequency. For example, retrieve all same-name entities in the same city when that group is small, and fall back to a dense name+locality query otherwise. Without a location key, a common name ("Sharma Medical Store") has too many exact twins to fit into 20 candidates, so recall there is limited by the budget, not by the retriever.
- For native-script names in Latin letters, index a **phonetic or skeleton key** (lowercase, strip vowels, collapse doubled letters, then char n-grams), or transliterate native-script records into Latin with IndicXlit before TF-IDF. Top-1 accuracy is uneven (11-85%), so use top-n transliterations or fuzzy keys rather than a single output.
- For locality substitutions and pseudo-word replacement names, a fine-tuned dense encoder trained on India match pairs (hard negatives drawn from the TF-IDF top-k) is the retriever most likely to recover these. Its candidates should be unioned with the others, and the LightGBM ranker needs features that tell it which channel proposed each candidate.
- Because the ranker caps at 20, gains from new retrievers can be lost at the cap. Recall should be measured both *before* the cap (union recall) and *after* it, to see which stage loses each missed pair.

### Gaps
- No peer-reviewed benchmark was found for blocking recall on name-only or address-less records, or on Latin/Indic mixed-script business names.
- No source found that quantifies how often phonetic keys (Soundex, Metaphone, Indic-specific schemes) recover transliteration misses in ER blocking.
