# Cross-lingual / transliteration-robust entity matching and transfer to an unseen country (France)

Context for the report writer: the target task is business entity resolution (name + address). The training data covers the US and India, and India includes business names written in Indic scripts (Devanagari, Telugu, Tamil, Kannada, Bengali, Gujarati...) that must match Latin-script English forms. The test data adds France, for which there are no labels. Constraints: tools and models should be MIT or Apache-2.0 (others are flagged), at most 8B parameters, and no external lookup data (gazetteers, company registries) at inference.

Glossary for the metrics quoted below:
- **F1**: harmonic mean of precision and recall on the "match" class, computed on a held-out test split of pairs.
- **MRR**: mean reciprocal rank. For each query name, take 1/(rank of the correct name) in the retrieved list, then average over queries.
- **R@k**: fraction of queries whose correct name appears in the top-k results.
- **"NoDA"**: a model trained only on the labelled source dataset and applied unchanged to the target dataset, with no domain adaptation.

---

## Q1. Romanisation / transliteration tools for Indic scripts, and their licences

### Takeaway
There are three practical deterministic or learned romanisers. uroman (MIT) is context-aware and has improved Indic support. anyascii (ISC, a permissive licence that is not MIT/Apache and should be flagged) works character by character and drops the inherent vowels, so "महासमुंद" becomes "mhasmumd" instead of "Mahasamund". IndicXlit (MIT, about 11M parameters, trained on 26M Aksharantar pairs) is a learned native-to-Roman and Roman-to-native transliterator for 21 Indic languages. Of the three, IndicXlit gives the most natural romanisations, at the cost of running a neural model.

### Cited Findings
- **uroman** is MIT-licensed, copyright Ulf Hermjakob, USC ISI (2015–2020). Projects that use it are asked to acknowledge it in publications. — [Search summary of isi-nlp/uroman licence](https://github.com/isi-nlp/uroman); [HF Space copy of LICENSE.txt](https://huggingface.co/spaces/Afrinetwork/sts1/blob/bc747e61c9c717c85078eaef6ffcfd3bbe93e9c2/uroman/LICENSE.txt)
- uroman now has a Python version (v1.3.1.1, June 2024, `pip install uroman`) besides the legacy Perl version. It covers Devanagari, Telugu, Tamil, Bengali and other Indic scripts. v1.3.1 added "improved support for ... several Indian languages" including "better final schwa deletion". It uses "m-to-n character mappings, context, and a user-provided language code". Its own caveat is that "a romanizer is not a full transliterator". — [isi-nlp/uroman GitHub](https://github.com/isi-nlp/uroman)
- uroman's system paper is Hermjakob et al., ACL 2018 demo, "Out-of-the-box Universal Romanization Tool uroman". — [ACL Anthology P18-4003](https://aclanthology.org/P18-4003/)
- **anyascii** uses the ISC licence. "Text is converted character-by-character without considering the context". It covers 124k of 155k Unicode characters. The documented example is Hindi "महासमुंद" → "mhasmumd", where the conventional romanisation is "Mahasamund": the inherent schwa vowels are lost. — [anyascii GitHub](https://github.com/anyascii/anyascii)
- **IndicXlit** is a transformer-based multilingual transliteration model of about 11M parameters. It covers 21 Indic languages in both directions (Roman→native and native→Roman) and was trained on the Aksharantar dataset (26M word pairs, 20 Indic languages). The code and models are MIT-licensed. — [AI4Bharat/IndicXlit GitHub](https://github.com/AI4Bharat/IndicXlit)
- Aksharantar has 26M transliteration pairs ("21 times larger than existing datasets") across 21 Indic languages, 3 language families and 12 scripts. Its test set has 103k word pairs in 19 languages. IndicXlit improves accuracy by 15% on the Dakshina test set. — [Aksharantar, arXiv 2205.03018 / Findings of EMNLP 2023](https://arxiv.org/abs/2205.03018); [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.4/)
- In a cross-script historical-name benchmark (316 items, Mongol-era names), exact normalised match reached 48.42% accuracy, Unicode Levenshtein 53.16%, and Unidecode→ASCII followed by Levenshtein 59.81%. A multilingual embedding model (Gemini Embedding 2, a closed API that does not meet the constraints) reached 64.87% on names alone. So ASCII-folding transliteration recovers a real but limited amount of signal. — [When Names Cross Scripts, arXiv 2608.23507](https://arxiv.org/html/2608.23507)
- A contrastive-learning write-up on cross-script retrieval reports that about 70% of its cross-script evaluation queries share no characters with the Latin-indexed names, so classical string matching scores close to zero unless the text is transliterated first. — [Towards Data Science, Jumle, Apr 2026](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)

### Inferences
- For Indic→Latin business names, a reasonable feature stack is to compute string similarity on (a) the uroman output, (b) the anyascii output and (c) the IndicXlit native→Roman output against the Latin name, and let the matcher learn which one to trust. anyascii's missing vowels ("mhasmumd") suggest adding a vowel-stripped "consonant skeleton" comparison on both sides: "Mahasamund" becomes "mhsmnd", and "mhasmumd" becomes "mhsmmd". Anusvara (ं) is romanised as "m" rather than "n", which is why the two skeletons still differ. A consonant-class phonetic key would remove this last difference (see Q2).
- All three tools are self-contained rule tables or model weights. They are not lookup services for entities, so they appear compatible with the "no external lookup data" rule.

### Gaps
- ICU transforms (`Any-Latin`, `Devanagari-Latin`) are under **UNICODE LICENSE V3**, which is permissive but not MIT or Apache, so they are flagged ([unicode-org/icu LICENSE](https://github.com/unicode-org/icu/blob/main/LICENSE)). No accuracy figures were found for ICU on Indic business names.
- **Unidecode** (the tool used in the arXiv 2608.23507 baseline) is under the **GNU GPL v2** ([avian2/unidecode LICENSE](https://github.com/avian2/unidecode/blob/master/LICENSE)). That conflicts with the MIT/Apache constraint, so use anyascii or uroman instead.
- The IndicXlit README confirms: "The IndicXlit code (and models) are released under the MIT License" ([README](https://github.com/AI4Bharat/IndicXlit)).
- No published head-to-head accuracy of uroman vs anyascii vs IndicXlit on *business-name* matching was found.
- IndicXlit per-language top-1 accuracy numbers and the Aksharantar *dataset* licence were not retrieved; the GitHub raw files were unreachable from this environment.

---

## Q2. Phonetic keys for Indic names

### Takeaway
There are only small-scale academic Soundex variants for Indian names, plus one open-source implementation (libindic/soundex) that maps all Indic scripts and English into a shared phonetic code. None of them has a large-scale benchmark.

### Cited Findings
- libindic/soundex implements Soundex for English and a modified Soundex for Indian languages. It "supports all indian languages and English" and provides intra-Indic string comparison, meaning the same name written in Devanagari and in Malayalam gets comparable codes. — [libindic/soundex GitHub](https://github.com/libindic/soundex); [Santhosh Thottingal blog (IndicSoundex design)](https://thottingal.in/blog/2009/07/26/indicsoundex/)
- An improved Soundex for Hindi and Gujarati names was reported to reach "100% accuracy" on only 100 test words. The evaluation is too small to be reliable. — [IJCSEA paper PDF](https://www.airccse.org/journal/ijcsea/papers/4314ijcsea03.pdf); [ResearchGate](https://www.researchgate.net/publication/269672873_Improvement_of_Soundex_Algorithm_for_Indian_Language_Based_on_Phonetic_Matching)
- "Need for Customized Soundex based Algorithm on Indian Names" argues that standard English Soundex fails on Indian-name variation. — [ResearchGate 316891126](https://www.researchgate.net/publication/316891126_Need_for_Customized_Soundex_based_Algorithm_on_Indian_Names_for_Phonetic_Matching)
- In the byte-level cross-script study, the Double Metaphone baseline had a Latin vs non-Latin R@10 gap of 0.88–0.94 (almost total failure across scripts). The learned byte-level model's gap was 0.096. — [TDS, Jumle 2026](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)

### Inferences
- English phonetic keys such as Metaphone or Soundex only become useful after romanisation. A practical custom key is: romanise → lower-case → collapse aspirated consonants (bh→b, dh→d, kh→k, th→t) → merge v/w, sh/s, z/j → drop vowels except the first letter → drop repeated letters. This is my own suggestion and has not been evaluated in any source found.

### Gaps
- The libindic/soundex licence could not be verified (the GitHub API was rate-limited). Libindic projects are often LGPL/GPL, so it needs a flag and a check.
- No quantitative benchmark of phonetic keys on Indic↔Latin *business* names was found.

---

## Q3. Character/byte-level models and multilingual encoders for cross-script name similarity

### Takeaway
Models under 8B parameters with MIT or Apache licences exist: LaBSE, multilingual-e5, BGE-M3, XLM-R, mDeBERTa-v3, mmBERT, MuRIL, IndicBERT v2, ByT5 and CANINE. Almost none of them has published accuracy on *transliterated proper-name* matching. The only quantitative cross-script name result found comes from a small (about 4M-parameter) byte-level contrastive encoder trained on synthetic name pairs, which reached MRR 0.775 and R@10 0.897. That suggests fine-tuning a small byte-level or character-level model on the competition's own positive pairs is the most promising route.

### Cited Findings
- **Licences and sizes, taken from the Hugging Face API** ([HF model API](https://huggingface.co/api/models/sentence-transformers/LaBSE), queried per model):
  - LaBSE (`sentence-transformers/LaBSE`): Apache-2.0, about 471M parameters
  - multilingual-e5-large and -large-instruct: MIT, about 560M
  - BGE-M3: MIT
  - XLM-RoBERTa-large: MIT, about 561M
  - mDeBERTa-v3-base: MIT
  - ByT5-small: Apache-2.0
  - CANINE-c: Apache-2.0, about 132M
  - MuRIL-base-cased: Apache-2.0
  - IndicBERTv2-MLM-only: MIT
  - gte-multilingual-base: Apache-2.0, about 305M
  - mmBERT-small: MIT
  - **l3cube-pune/indic-sentence-similarity-sbert: CC-BY-4.0 (not MIT or Apache, so flagged)**
- **mmBERT-base** (Sept 2025, arXiv 2509.06888): MIT, 307M total parameters (110M non-embedding), trained on 1800+ languages, Gemma-2 tokenizer, 8192-token context. The card says it "significantly outperforms previous generation models like XLM-R on classification, embedding, and retrieval tasks". No name-matching numbers are given. — [HF model card jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
- **Byte-level contrastive name encoder** (Jumle, Apr 2026). It is a custom 6-layer, 8-head, 256-dimensional transformer over raw UTF-8 bytes (vocabulary of 256, about 4M parameters), not ByT5 or CANINE.
  - Training data: 4.67M positive pairs built from 119,040 Wikidata person entities. Latin phonetic variants came from Llama-3.1-8B and transliterations into 8 scripts (including Hindi/Devanagari) came from Qwen3-30B.
  - Test results: MRR 0.775, R@1 0.546, R@10 0.897. R@10 was above 0.95 for Arabic, Russian, Hebrew, Hindi and Greek, 0.666 for Chinese and 0.728 for Korean.
  - Baselines were Levenshtein, Double Metaphone, BM25 and transliteration; LaBSE, e5 and BGE-M3 were not compared.
  - Code: GitHub `vedant-jumle/cross-language-phonetic-text-alignment` (licence not verified).
  - This is a blog post, not peer-reviewed.
  — [Towards Data Science](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)
- **RomanSetu** (AI4Bharat, ACL 2024): romanised Indic text reduces token fertility by 2–4x and matches or beats native script on NLU, NLG and MT tasks. Embeddings of romanised text align more closely with their English translations than embeddings of native-script text. — [RomanSetu, arXiv 2401.14280](https://arxiv.org/html/2401.14280v2)
- L3Cube IndicSBERT reached a Spearman correlation of 0.82 on Hindi–English sentence similarity vs 0.72 for LaBSE. This is sentence-level, not name-level, and the model is CC-BY-4.0. — [L3Cube-IndicSBERT summary](https://lacuna.tiptreesystems.com/work/l3cube-indicsbert-a-simple-approach-for-learning-cross-lingual-sentence/wrk_c78a17f9043a349998d5f261db5c9133)
- A lexical-sharing study that transliterated Devanagari↔Gujarati found multilingual MT models "already reasonably robust to script differences" between closely related Indic scripts. — [arXiv 2305.03207](https://arxiv.org/pdf/2305.03207)
- On historical cross-script names, even a strong embedding model scored 0/5 on "identical-surface different-person" cases with names alone and 24/25 with context. Name similarity alone cannot tell apart different entities that share a name, and address or context features are required. — [arXiv 2608.23507](https://arxiv.org/html/2608.23507)

### Inferences
- Sentence encoders such as LaBSE, e5 and BGE-M3 are trained on translation pairs, so they are likely to judge semantically translated names well (for example "Sharma General Store" ↔ "शर्मा जनरल स्टोर", where the words "general store" are themselves transliterated). They are likely weaker on pure phonetic transliteration of rare proper names. Fine-tuning a small MIT or Apache encoder (CANINE, ByT5-small encoder, mmBERT-small) contrastively on the training set's own Indic↔Latin positive name pairs is the most direct route, since the byte-level result above shows that about 4M parameters are enough.
- Romanising first (RomanSetu) and then using an English-strong encoder is a cheap alternative that has evidence behind it.

### Gaps
- No peer-reviewed benchmark was found that compares LaBSE, multilingual-e5, BGE-M3, XLM-R, mDeBERTa, mmBERT, ByT5 or CANINE on *transliterated name* matching or cross-lingual entity linking with Indic scripts.
- Exact parameter counts for BGE-M3 (believed to be about 568M, since it is XLM-R-large based), mDeBERTa, ByT5-small and MuRIL were not returned by the API query.

---

## Q4. Learned transliteration dictionaries mined from parallel pairs

### Takeaway
Classic transliteration mining (Sajjad, Fraser, Schmid, Durrani) uses EM to fit an interpolated model of "transliteration pair vs noise", and it works unsupervised on noisy word pairs. Aksharantar applied large-scale mining to build its 26M-pair Indic corpus. The same idea can be applied to the competition's matched Indic↔Latin name pairs to learn character n-gram correspondences.

### Cited Findings
- Sajjad et al. proposed a generative model that interpolates a transliteration sub-model and a non-transliteration (noise) sub-model. It is trained with EM on noisy unlabelled data and supports unsupervised, semi-supervised and supervised mining. — [Computational Linguistics 43(2)](https://direct.mit.edu/coli/article/43/2/349/1568/Statistical-Models-for-Unsupervised-Semi); [ACL 2012 paper](https://www.cis.uni-muenchen.de/~schmid/papers/sajjad_acl2012.pdf)
- An earlier heuristic unsupervised mining algorithm (ACL 2011) was applied to word alignment. — [Sajjad, Fraser, Schmid ACL 2011](https://alexfraser.github.io/pubs/sajjad_acl2011.pdf)
- Sajjad and Durrani compared unsupervised mining with a rule-based method for extracting Hindi/Urdu transliteration pairs from a parallel corpus. — [IJCNLP 2011, I11-1015](https://aclanthology.org/I11-1015.pdf)
- Durrani et al. (EACL 2014) integrated an unsupervised transliteration model into SMT to transliterate out-of-vocabulary words. — [E14-4029](https://aclanthology.org/E14-4029.pdf)
- The Aksharantar release includes its *mining scripts* and transliteration guidelines under open-source licences. — [arXiv 2205.03018](https://arxiv.org/abs/2205.03018)

### Inferences
- For the competition, labelled Indic↔Latin matched pairs from India training data can be tokenised into words. Align words by position or by best edit similarity after romanisation, then keep a word-level dictionary (for example "स्टोर"↔"store", "एंटरप्राइजेज"↔"enterprises") and a character n-gram mapping table. Both are learned from training labels, not from external lookup data. The dictionary can then add a "dictionary-translated name similarity" feature.

### Gaps
- No 2020–2026 paper was found that applies transliteration mining specifically to business-name entity resolution. The quantitative mining precision figures in the Sajjad papers were not extracted.

---

## Q5. Domain adaptation / zero-shot transfer for entity matching

### Takeaway
Transferring a matcher to an unseen domain without labels is well studied.
- **DADER (SIGMOD 2022, BERT feature extractor):** on target datasets the model never saw labels for, the best alignment method improved F1 over no adaptation by 0–27 points when source and target were similar and by 11–44 points when they were different. MMD alignment was the most stable. Gains were small (−1.5 to +8.3) when the datasets already looked alike.
- **AnyMatch (GPT-2, 124M, trained leave-one-dataset-out):** 81.96 mean zero-shot F1, within 4.4% of GPT-4.
- **Cross-language learning (Peeters and Bizer):** adding 7,200 English pairs to 1,800 German pairs raised multilingual BERT from 87.7 to 91.4 F1, which is directly relevant to a French target.

### Cited Findings
**DADER — Tu, Fan, Tang, et al., SIGMOD 2022** ([paper PDF](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/3514221.3517870); code: github.com/ruc-datalab/DADER)
- Setup: the feature extractor is BERT-base (12 layers, max sequence length 256) and the matcher is a fully connected layer with softmax. There are six "feature aligners":
  - MMD and K-order (discrepancy-based)
  - GRL, InvGAN and InvGAN+KD (adversarial)
  - ED (reconstruction)
- The source dataset is labelled and the target dataset is unlabelled. The metric is F1 on the target test set. Hyper-parameters were chosen on a target validation set, which is a caveat: that validation set is labelled.
- Similar domains (Table 3), NoDA → best DA:
  - Walmart-Amazon→Abt-Buy: 65.8 → 72.6 (MMD), +6.8
  - Abt-Buy→Walmart-Amazon: 56.9 → 71.1, +14.2
  - DBLP-ACM→DBLP-Scholar: 77.8 → 92.3, +14.5
  - Fodors-Zagats→Zomato-Yelp: 47.6 → 75.0 (InvGAN+KD), +27.4
  - DBLP-Scholar→DBLP-ACM: 97.2 → 97.2, +0.0
- Different domains (Table 4):
  - Book2→Fodors-Zagats: 49.6 → 93.5, +43.9
  - iTunes-Amazon→DBLP-Scholar: 68.2 → 89.1, +20.9
  - RottenTomatoes-IMDB→Walmart-Amazon: 38.4 → 49.4, +11.0
- WDC product categories (the same website and the same title vocabulary): gains ranged from −1.5 to +8.3, because "the domain shift may not be significant". In some cases NoDA already beat models trained in-domain.
- Finding 2: DA works better when the source is closer to the target, measured by MMD distance on BERT features. The authors suggest selecting source data by that distance.
- Finding 3: MMD converges stably. Adversarial InvGAN+KD can oscillate; for example, it peaked at epoch 6 and then collapsed at learning rate 1e-5, and needs a smaller learning rate plus a labelled validation set. Plain InvGAN often did worse than NoDA (29.7 vs 47.6 on Fodors-Zagats→Zomato-Yelp), and ED was inferior almost everywhere.
- Finding 6: feature-level DA beat instance reweighting of source pairs.
- Finding 7: with a few target labels (active learning, 200 labels per round for 4 rounds), InvGAN+KD beat Ditto and DeepMatcher.
- Licence: a fetch of the VLDB demo paper returned "CC BY-NC-ND 4.0" together with the PyPI package `dader`. This is most likely the *paper's* PVLDB licence, not the code's, so the code licence needs verification. — [DADER VLDB demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)
- The VLDB demo reports that DA improved Walmart→Amazon by 14–20% using Abt-Buy as the source. — [VLDB demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)

**AnyMatch — Zhang et al., 2024** ([arXiv 2409.04073](https://arxiv.org/html/2409.04073v1); code [Jantory/anymatch](https://github.com/Jantory/anymatch))
- Protocol: leave one dataset out. To test on dataset X, fine-tune on the other 8 datasets, with no examples from X.
- Base model: GPT-2, 124M parameters, decoder-only.
- Serialisation: "Record A is <p>COL [value1], COL [value2]</p> ... Given the attributes of the two records, are they the same?"
- Data selection:
  - an AutoML filter that keeps hard pairs (for example, false negatives of an AutoML model)
  - extra attribute-level pairs, meaning single-attribute value pairs as additional examples
  - a 2:1 negative:positive ratio
- Results: mean F1 of 81.96, second best overall, within 4.4% of MatchGPT with GPT-4 (86.36) and above Ditto (66.05). Per-dataset: BEER 96.55, FOZA 100.00, ITAM 90.91. Inference cost is 3,899x lower than GPT-4.
- Ablations: removing attribute-level instances costs −3.14%, removing the AutoML filter −1.62%, removing flipped pairs −0.72%. Swapping the base model to T5 costs −2.38% and to BERT −9.04%.

**Unicorn — Fan et al., SIGMOD 2023** ([GitHub ruc-datalab/Unicorn](https://github.com/ruc-datalab/Unicorn); [SIGMOD Record](https://sigmodrecord.org/publications/sigmodRecord/2403/pdfs/12_unicorn-fan.pdf))
- A single encoder (DeBERTa-base) with a mixture-of-experts layer and a binary matcher is trained jointly on 7 matching tasks: entity matching, column type annotation, entity linking, string matching, schema matching, ontology matching and entity alignment. It supports zero-shot prediction on new tasks, and the authors report it beats task-specific models on most tasks. The zero-shot numbers and the repository licence were not retrieved.

**Cross-language learning — Peeters, Bizer and Glavaš, WWW 2022 Companion** ([arXiv 2110.03338](https://arxiv.org/abs/2110.03338))
- Setup: German test set of mobile-phone offer pairs (150 seed phones). Each model is fine-tuned on 1,800 German pairs, with or without 7,200 English pairs (Table 1). F1 on the German test set, averaged over 3 runs:

| Model | F1 without English pairs | F1 with English pairs | Change |
|---|---|---|---|
| SVM | 71.00 | 71.05 | +0.05 |
| English BERT | 65.27 | 74.29 | +9.02 |
| German BERT (gBERT) | 73.43 | 89.83 | +16.40 |
| mBERT | 87.69 | 91.44 | +3.75 |
| XLM-R-base | 73.40 | 86.98 | +13.58 |

- Table 2 (mBERT, varying the amounts of data):
  - With 450 German pairs, adding 7,200 English pairs lifts F1 from 67.11 to 87.97 (+20.86).
  - With 3,600 German pairs the gain shrinks to +0.83 (93.63 → 94.46).
- The authors conclude that choosing a transformer pre-trained on diverse multilingual text is crucial for cross-language transfer.

**Learning from natural-language explanations** ([arXiv 2406.09330](https://arxiv.org/abs/2406.09330))
- Entity matching is recast as conditional generation, and LLM-written explanations are distilled into smaller models. The paper reports a 10.85% F1 out-of-domain improvement where standalone generative methods struggle. Model sizes were not captured.

### Inferences
- For France, the US+India→France shift is closest to DADER's "similar domain, different textual style" case: same attributes (name, address), but different language, street-type vocabulary and legal suffixes. DADER's results suggest:
  - MMD feature alignment between labelled US+India pairs and *unlabelled* French candidate pairs is the safer choice.
  - The expected gain is modest when the representations already overlap. Gains were small on WDC, where the vocabulary was shared.
  - The source-to-target MMD distance can be used to check whether alignment is worth trying.
- The Peeters and Bizer result supports using a *multilingual* pre-trained encoder (mBERT or XLM-R class, or newer mmBERT) rather than an English-only encoder, since the English BERT lost 22 F1 points to mBERT on German when trained with no English pairs.
- AnyMatch's ablations suggest two things for a small transferable matcher: attribute-level pair augmentation (name-only pairs and address-only pairs as extra training rows) and hard-negative filtering.
- All DADER numbers use a labelled target validation set to tune hyper-parameters. With no French labels, the stable MMD variant and fixed hyper-parameters are preferable to the adversarial variants.

### Gaps
- No published EM transfer study was found for *country or language* shift in business or POI (point-of-interest) name+address matching specifically. The closest are Zomato-Yelp and Fodors-Zagats (US restaurants) and cross-language product matching.
- Unicorn zero-shot F1 numbers, and the Unicorn and DADER code licences, were not verified.
- Licence of GPT-2 weights (used by AnyMatch): not verified in this session.
- "Transfer learning for entity matching" (Kasai et al. ACL 2019, "Low-resource Deep ER with Transfer and Active Learning") was located but not read. — [Semantic Scholar](https://www.semanticscholar.org/paper/Low-resource-Deep-Entity-Resolution-with-Transfer-Kasai-Qian/c57757597d0664a0f66d40108e50bf696044c9fe)

---

## Q6. Synthetic data, noise augmentation, self-training/pseudo-labelling on unlabelled target data, and monotone constraints

### Takeaway
Pseudo-labelling on unlabelled pairs is the single most effective component in Sudowoodo. Removing it cost about 10 F1 points, and fully unsupervised Sudowoodo averaged 74.3 F1, within about 4 points of its 500-label version. Synthetic LLM-generated transliteration and phonetic variants were enough to train a strong cross-script name encoder. No entity-matching paper on monotone constraints was found.

### Cited Findings
- **Sudowoodo** (Megagon Labs, ICDE 2023) learns similarity-aware representations with contrastive self-supervision, with no labels. Its pseudo-labelling step labels an unlabelled candidate pair positive if cos(emb(x), emb(y)) > θ+ and negative if it is < θ−. Balancing the positive:negative ratio matters, and the unsupervised mode needs the *prior positive-label ratio* as a dataset statistic. — [arXiv 2207.04122](https://arxiv.org/pdf/2207.04122); [GitHub megagonlabs/sudowoodo](https://github.com/megagonlabs/sudowoodo)
  - Semi-supervised (500 labels) average F1 over AB, AG, DA, DS and WA: Sudowoodo 78.3 vs Rotom(500) 72.3, Ditto(500) 69.9, and DeepMatcher trained on the full label set 78.6. Without pseudo-labelling, Sudowoodo drops to 68.5, the largest ablation drop.
  - Unsupervised (0 labels) average F1: Sudowoodo 74.3 vs ZeroER 66.6 and Auto-FuzzyJoin 65.4. On Walmart-Amazon: Sudowoodo 61.2 vs ZeroER 51.0.
- In general domain-adaptation literature, self-training "utilizes the most confident labeled data from target domain to augment the source-domain labeled dataset", but pseudo-labels "can be unreliable due to distribution shifts". — [Self-Training with Label-Feature-Consistency, Springer](https://link.springer.com/chapter/10.1007/978-3-031-30678-5_7)
- DADER's related work classes pseudo-labelling of target data as instance-level DA. In DADER's own experiments, instance *reweighting* (the "Reweight" baseline) was beaten by feature-level DA (Finding 6). — [DADER SIGMOD 2022](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf)
- Synthetic training pairs: 4.67M LLM-generated phonetic and transliteration variants (Llama-3.1-8B, Qwen3-30B) of 119k Wikidata names were enough to train a 4M-parameter byte-level encoder to R@10 0.897 across 9 scripts. — [TDS, Jumle 2026](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)
- AnyMatch's attribute-level augmentation and hard-pair filtering together added about 4.8% F1 in zero-shot transfer (see Q5). — [arXiv 2409.04073](https://arxiv.org/html/2409.04073v1)

### Inferences
- A concrete French pipeline suggested by these results:
  1. Train on US+India.
  2. Score French candidate pairs.
  3. Take the most confident top and bottom pairs as pseudo-labels, keeping the positive ratio close to the ratio observed in training.
  4. Retrain with the pseudo-labels.
  Hand-built French synthetic positives are also possible, for example the same name with and without "SARL", or "AV" vs "AVENUE". These are generated by rules, not by external lookup.
- Monotone constraints, as in LightGBM or XGBoost `monotone_constraints` (for example, forcing the match probability to be non-decreasing in name similarity), are a plausible way to keep a gradient-boosted matcher from learning country-specific quirks that invert on France. No quantitative EM evidence was found for this.

### Gaps
- No peer-reviewed EM paper was found that evaluates monotone constraints for robustness.
- The Ditto and Rotom augmentation operators (span deletion, token shuffle and so on) were not re-verified in this session. Ditto's code licence (believed to be Apache-2.0) was not verified.

---

## Q7. French address and name normalisation (street types, legal forms, departments)

### Takeaway
There are open, rule-type resources: La Poste's official abbreviation tables under AFNOR XP Z10-011 (246 street types, plus tables for titles, building words and legal forms), libpostal (MIT) with French `expand_address`, the GLEIF ISO 20275 Entity Legal Forms list (CC0, more than 2,100 forms across more than 90 jurisdictions, including French SARL and SAS) and cleanco (MIT) for stripping legal suffixes. All of them are normalisation rules rather than entity lookup data. However, libpostal's parser also bundles place-name dictionaries, which needs a check against the "no gazetteer" rule.

### Cited Findings
- La Poste publishes the authorised abbreviations under **AFNOR XP Z 10-011**: a 246-item table of street types plus tables for civil titles, building designations and legal entity forms. Examples: AVENUE→AV, BOULEVARD→BD, CHEMIN→CHEM, FAUBOURG→FAUB, PLACE→PL, IMPASSE→IMP, ROUTE→RTE. La Poste states that "l'abréviation reste une exception" (abbreviation remains an exception), and the last line (postcode + city) is preferably written in capitals. — [La Poste abbreviation page](https://www.laposte.fr/envoyer/abreviation-adresses-postales)
- Other French postal abbreviations: ALLEE→ALL, RESIDENCE→RES, ROND-POINT→RPT. — [Search summary citing La Poste list](https://www.laposte.fr/envoyer/abreviation-adresses-postales)
- UPU (Universal Postal Union) documents the French postcode format and position. — [UPU France addressing PDF](https://www.upu.int/UPU/media/upu/PostalEntitiesFiles/addressingUnit/fraEn.pdf)
- **libpostal** is MIT-licensed. `expand_address` supports French; for example, "Quatre vingt douze R. de l'Église" becomes "92 rue de l eglise", with numbers spelled out, abbreviations expanded and accents stripped. The model download is about 1.8 GB. It was trained on more than 1 billion addresses from OpenStreetMap and OpenAddresses. — [openvenues/libpostal](https://github.com/openvenues/libpostal)
- libpostal's abbreviations are language-specific: "St." becomes "Saint" in French and "Sankt" in German. Its normalisation uses multilingual address dictionaries *and* place-name dictionaries compiled from GeoNames and OSM. — [Mapzen: Inside libpostal](https://www.mapzen.com/blog/inside-libpostal/)
- A known issue reports bad parsing for some French street names. — [libpostal issue #357](https://github.com/openvenues/libpostal/issues/357)
- **GLEIF ISO 20275 ELF code list** contains more than 2,100 legal forms in more than 90 jurisdictions, in native language with local abbreviations (SARL, SAS, SA, GmbH...). Each form has a 4-character code. It is published as CSV/Excel under **CC0 1.0**. v1.6 was dated 2026-02-19. — [GLEIF ELF page](https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list); [CSV v1.6](https://www.gleif.org/media/pages/lei-data/code-lists/iso-20275-entity-legal-forms-code-list/875c1c78fc-1771833695/2026-02-19-elf-code-list-v1.6.csv)
- A cleaned derivative with "name matching" exists: Reg-Data/open-legal-forms (licence not verified). — [GitHub](https://github.com/Reg-Data/open-legal-forms)
- **cleanco** (MIT) strips legal-form suffixes using "a database of organization type terms" and suggests countries from the term found (for example "Oy"→Finland). French coverage was not confirmed on the README page. — [psolin/cleanco](https://github.com/psolin/cleanco)
- French legal forms to strip or normalise include SARL, SAS, SA and EURL, among others; the official guide lists the choice of legal forms. — [Service Public Entreprendre, legal forms](https://entreprendre.service-public.gouv.fr/vosdroits/F23844?lang=en); [Stripe: business legal types in France](https://stripe.com/resources/more/business-legal-types-france)

### Inferences
- A lightweight, rule-only French normaliser that meets the constraints:
  1. Unicode NFKD, strip accents, upper-case.
  2. Expand AFNOR street-type abbreviations to one canonical form (AV/AVE/AVENUE → AVENUE; BD/BLVD → BOULEVARD; R → RUE only when it sits in the street-type slot).
  3. Normalise "ST"/"STE" → SAINT/SAINTE, number words, and BIS/TER suffixes.
  4. Strip legal forms at the start or end of the name (SARL, SAS, SASU, SA, EURL, SNC, SCI, SELARL...) using the GLEIF list filtered to the French jurisdiction as a *static rule list*.
  5. Keep the 5-digit postcode as a strong blocking key. Its first two digits give the département, for example 75 = Paris.
  This last point about département codes comes from common knowledge, not from a source retrieved here, and should be verified.
- Whether a static legal-form list (GLEIF) or libpostal's bundled GeoNames/OSM dictionaries count as "external lookup data" is a judgement call for the competition rules. A hand-written legal-form list of about 20 French terms avoids the question.

### Gaps
- The exact French postcode→département rule, Corsica (2A/2B), CEDEX handling and overseas departments were not sourced in this session.
- No published quantitative study of French-specific normalisation gains in entity resolution was found.
- cleanco's French term coverage and the licence of the Reg-Data open-legal-forms repository were not verified.
