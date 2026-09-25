# Collective / Global Entity Resolution vs Independent Pairwise Decisions

Context these notes serve: S1 is a clean, already-deduplicated reference of 1.7M businesses. S2 and S3 are noisy sources with about 10M records in total. Each S2/S3 record belongs to at most one S1 entity. An S1 entity has 0 to 11 matching records, and about 26-40% of S2/S3 records are "decoys" that belong to no S1 entity. Current system: a LightGBM model scores each (S1, candidate record) pair; "sibling" features measure how well a candidate agrees with the S1's other confident candidates; a "one-owner" step gives each record only to its best-scoring S1; then, for each S1, the set of records to output is picked to maximise expected F0.5.

Terminology used below (plain language):
- **Pairwise decision**: judge each (A, B) pair alone, for example "score > 0.5 means match".
- **Similarity graph**: nodes are records and edges are pairs the matcher scored; an edge weight is the match score.
- **Transitive closure / connected components (CC)**: if A matches B and B matches C, put A, B and C in one cluster, even if A-C was never judged a match. It is cheap but one wrong edge merges two whole groups.
- **Clean-Clean ER**: linking two sources that are each duplicate-free. **Dirty ER**: finding duplicates inside one source. **Multi-source ER**: more than two sources. Definitions follow [Christophides et al. survey](https://arxiv.org/pdf/1905.06397). Our setting is a hybrid: S1 is clean, but S2 and S3 can each hold several records of the same S1 entity (0 to 11 in total), so S2/S3 are "dirty".
- **Source-consistent cluster**: a cluster with at most one record from each source. FAMER, CLIP and GraLMatch assume this. **It does not hold for our S2/S3**, which limits how directly those methods transfer.

---

## Q1. Correlation clustering, constrained transitive closure, pruned connected components, GNNs, "match-then-cluster": what gains are reported over pairwise decisions?

### Takeaway
Plain connected components over pairwise matches is consistently the worst clustering choice, because a few false-positive edges merge whole groups. Collective methods help most when (a) clusters are large or (b) groups can only be joined through transitive paths. The best documented gains come from targeted pruning of "bridge" edges (min-cut, betweenness, strong-link rules): +5 to +24 F1 points over naive transitive closure, and up to about +25 F1 points over pairwise baselines in relational settings. With small clusters like ours (≤12 records), the choice of clustering algorithm matters little (Draisbach et al. 2019), so gains will more likely come from the constraints and decoy handling than from swapping the clustering algorithm.

### Cited Findings
**FAMER comparison of 8 clustering schemes (Saeedi, Nentwig, Peukert, Rahm, CSIMQ 2018)**
- The 8 schemes are Connected Components, CCPivot (a correlation-clustering approximation), Center, Merge Center, Star-1, Star-2, SplitMerge and CLIP, all run on the same input similarity graph — [Saeedi et al. 2018, FAMER](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- Datasets: DS1 geography, 3,054 entities from 4 sources; DS2 music (MusicBrainz with heavy synthetic corruption), 20,000 entities from 5 sources; DS3 persons (NC voter data plus GeCo corruption), 5M entities from 5 sources and 10M entities from 10 sources. The metric is pairwise precision, recall and F-measure computed from the clusters (a cluster of m entities counts as m(m-1)/2 match pairs) — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- "Connected Components reaches the lowest F-Measure for all datasets and almost all threshold values because it suffers from very poor precision values." Merge Center behaves similarly — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- Center, Star-2 and CCPivot "can exceed the F-Measure of the input graph in only few cases". CCPivot (correlation clustering) "improves precision over the input similarity graph but suffers from lower recall so that F-Measure is not improved" — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- CLIP and SplitMerge "outperform all previous algorithms (and the input similarity graph) in terms of precision and F-Measure for all three datasets". On the very noisy DS2, most schemes "could mostly not exceed the quality of the input similarity graph (with a maximal F-Measure of about 0.75)" — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- Runtimes on 10M records (DS3, 10 sources) with 16 workers on Apache Flink: Connected Components 79 s, CLIP 228 s, Split 278 s, Star-2 173 s, Center 423 s, CCPivot 1,303 s, SplitMerge 2,819 s. CCPivot ran out of memory with fewer workers — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- How CLIP works: it first takes the transitive closure of **strong links**, meaning edges that are the highest-weight edge (per source) for *both* endpoints. It then removes **weak links**, which are the best edge for *neither* endpoint, recomputes components and splits clusters so that each holds at most one entity per source — [Nentwig & Rahm, Incremental Clustering on Linked Data](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/incremtal-dmc2018-final.pdf)

**Draisbach, Christen, Naumann (ACM JDIQ 2019/2020): pairwise duplicates to clusters**
- "The commonly used Transitive Closure is inferior to most of the other clustering algorithms, especially for the precision of results. In scenarios with small cluster sizes, the choice of the clustering algorithm has no or little effect on the overall result. In scenarios with larger clusters, [Extended Maximum Clique Clustering] is, together with Markov Clustering, one of the two best performing" — [Draisbach et al. 2019](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2019_draisbach_transforming.pdf)
- "No overall winner emerges" across the real-world datasets they tested — [Draisbach et al. 2019](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2019_draisbach_transforming.pdf)
- The earlier Stringer benchmark (Hassanzadeh, Chiang, Miller, Lee, PVLDB 2009) found that "some clustering algorithms that have never been considered for duplicate detection perform extremely well in terms of both accuracy and scalability" — [Hassanzadeh et al. 2009](http://www.vldb.org/pvldb/vol2/vldb09-1025.pdf)

**GraLMatch (De Toni et al., EDBT 2025): match-then-clean-then-cluster for companies and securities**
- Pipeline: (1) fine-tune a language model (DistilBERT) as a pairwise matcher; (2) score the blocked pairs; (3) Graph Cleanup deletes likely false-positive edges using the edge's connected component; (4) output the connected components of the cleaned graph as groups — [GraLMatch, EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- Cleanup uses the **minimum edge cut** (the fewest edges whose removal splits a component) for some component sizes and **edge betweenness centrality** (how many shortest paths run through an edge) for others. A size threshold γ picks the technique, and components are cut until they are no larger than μ, where μ is set to the number of sources. The authors note this is suited to data with at most one record per source and "will not be ideal" where group sizes vary — [GraLMatch 2025](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- Transitive closure destroys precision. On Synthetic Companies (DistilBERT-ALL), pairwise F1 on blocked pairs was 78.18, but group-level F1 including implied transitive matches was **0.00** before cleanup (precision 0.00, recall 82.26). After GraLMatch cleanup it was **60.03** (precision 98.76, recall 43.31). With DistilBERT-15K, pre-cleanup group F1 was 0.02 and post-cleanup 72.34 — [GraLMatch 2025, Table 4](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- On Real Companies (DITTO-128), pairwise F1 was 38.24 and pre-cleanup group F1 was 0.10. After cleanup group F1 was **99.06** (precision 99.86, recall 98.23), with inference time about 6.7 min — [GraLMatch 2025, Table 4](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- Cleanup cannot fix a useless matcher: on Real Securities, DITTO post-cleanup group F1 was about 18-20. The authors conclude that "precision is the key to achieving a good entity group matching especially with large volumes of records" — [GraLMatch 2025](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)

**TransClean (arXiv 2506.04006, June 2025): transitive-consistency cleanup**
- TransClean measures whether the pairwise model agrees on *all* pairs inside each connected component it produced. It labels edges from minimum cuts and shortest paths in inconsistent components, removes edges and then re-adds removed edges that keep consistency. Datasets: Synthetic Companies (868K records, 5 sources), MusicBrainz (19K records), Camera, Monitor, WDC Products — [TransClean 2025](https://arxiv.org/html/2506.04006)
- Results: an average **+24.42 F1** from before to after TransClean (this setting includes a small amount of manual edge labelling); it removes 48-97% of false positives. With the CLER matcher on Synthetic Companies F1 went from 86.61 to 92.22, and with DistilBERT on MusicBrainz from 91.93 to 98.14 — [TransClean 2025](https://arxiv.org/html/2506.04006)

**Collective relational ER (Bhattacharya & Getoor, ACM TKDD 2007)**
- Relational clustering (RC-ER) resolves entities jointly, using co-occurrence relations (for example co-authors) as well as attributes. F1 against the attribute-only pairwise baseline: CiteSeer 0.980 → 0.995, arXiv 0.976 → 0.985, BioBase 0.568 → 0.818 — [Bhattacharya & Getoor 2007](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf) (numbers as summarised in search results for that paper; not re-verified in the PDF tables)
- The survey notes that iterative collective ER (merge-based such as Swoosh, or relationship-based) "involve[s] several iterations until they converge", so "collective ER is hard to scale" — [Christophides et al., End-to-End ER for Big Data survey (ACM CSUR 2021)](https://arxiv.org/pdf/1905.06397)

**GNNs**
- HierGAT+ (SIGMOD 2022) extends a hierarchical graph attention network from pairwise matching to "collective ER where multiple candidate matches exist". The authors report it as best on all their datasets — [Yao et al., HierGAT, SIGMOD 2022](https://dl.acm.org/doi/pdf/10.1145/3514221.3517872) (from search-result summary; exact numbers not extracted)
- GNNs have also been used to detect inconsistent clusters in incremental ER — [arXiv 2105.05957](https://arxiv.org/pdf/2105.05957)
- The newest GNN-for-correlation-clustering paper (Nerini, Bonchi, Khan, Panisson, CIKM 2026) is *not* evaluated on ER. It reports inference up to 5 orders of magnitude faster, with an approximation ratio within about 10% of the best baseline — [arXiv 2608.27153](https://arxiv.org/abs/2608.27153)

**Tooling**
- Splink turns pairwise predictions into clusters with connected components at a chosen probability threshold. It provides cluster-quality metrics (density, centralisation, bridges, cluster size) and a Cluster Studio dashboard to find clusters "containing inaccurate links" — [Splink docs, cluster evaluation](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/overview.html)

### Inferences
- Our pipeline already avoids the worst failure mode. It never forms record-record transitive closures, because each record attaches directly to an S1 hub, and one-owner stops chaining through S1. The results where transitive closure collapses (GraLMatch pre-cleanup F1 ≈ 0) therefore describe the risk of *adding* naive S2-S3 clustering, not a gain we are missing.
- Our "sibling" features are a soft, learned version of what CLIP, TransClean and GraLMatch do with hard rules: they check whether a candidate is consistent with the rest of its group. The literature suggests the next step is **per-edge graph-context features** fed back into the scorer, for example: is this edge a bridge / in a min-cut of the S1's candidate subgraph, and what is its edge betweenness. This matches the "Graph-based Active Learning for Entity Cluster Repair" line (arXiv 2401.14992), which trains a classifier on graph metrics to label edges correct or incorrect.
- Draisbach's "small clusters, algorithm choice matters little" finding applies directly (≤12 records per S1). Expect small returns from swapping clustering algorithms, and larger returns from decoy-aware features and constraints (Q3, Q5).
- CPU feasibility: per-S1 candidate subgraphs are tiny, so min-cut, betweenness and even exact correlation clustering on ≤12-node subgraphs are cheap (milliseconds each) on a single machine. Flink-scale machinery (FAMER) is unnecessary.

### Gaps
- I found no study with our exact topology (clean hub source plus dirty spoke sources plus a high decoy rate) reporting collective-vs-pairwise gains.
- Exact HierGAT+ collective-ER numbers and a CPU-scale GNN-for-ER result at 10M records were not verified. The GNN evidence here is thin and mostly GPU-oriented.
- FAMER's figure-level F-measure numbers per algorithm are only shown as plots in the paper; only the qualitative ranking and the "max ≈ 0.75 on DS2" text statement could be extracted.

---

## Q2. Should we cluster S2+S3 records among themselves (S2-S3 and S2-S2 links) first and then link clusters to S1? How is it done?

### Takeaway
The multi-source ER literature (FAMER, CLIP, incremental "Max-Both Merge", GraLMatch) supports building a record-record similarity graph across sources and clustering it. But nearly all of it assumes duplicate-free sources (at most one record per source per cluster), which our S2/S3 violate. Its strongest reported gains come from (a) exploiting that constraint and (b) repairing clusters rather than from clustering itself. The closest analogue to "link clusters to a reference, with some clusters matching nothing" is joint entity-linking-plus-NIL-clustering (ArboEL, NAACL 2022), which reports significant gains from using mention-to-mention affinities alongside mention-to-entity scores.

### Cited Findings
- FAMER's workflow is: blocking, then pairwise match classification into a similarity graph over *all* sources, then clustering. Clusters are preferred over pairwise links because pairwise mappings grow quadratically with the number of sources (200 sources would need 19,900 mappings) and clusters allow incremental addition of new records "by comparing them with the set of previously determined clusters" — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- FAMER assumes "all sources are duplicate-free", so final clusters "should thus be source-consistent … at most one entity from each input data source". CLIP and SplitMerge's advantage is partly that "they ensure source-consistent clusters" — [FAMER 2018](https://dbs.uni-leipzig.de/files/research/publications/2018-11/pdf/FAMER-2407-8454-1-SM.pdf)
- Incremental multi-source ER (Saeedi, Peukert, Rahm, ESWC 2020) adds new entities to existing clusters. **Max-Both Merge** assigns a new entity to the cluster it has the highest similarity with only if that link is also the cluster's best link from that source; entities matching no existing cluster "form a new cluster". **n-Depth Reclustering** re-clusters the neighbourhood of affected clusters. It was tested on the same geography (3,054), music (19,375) and persons (10M, 10 sources) data. n-Depth Reclustering was "as effective as the batch approach" regardless of insertion order, while Max-Both Merge showed "substantially lower F-measure values with unfavorable insertion orders" — [Saeedi et al. 2020 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7250616/)
- Cluster repair for *dirty* sources: a 2024 approach builds an edge classifier from graph metrics of the similarity graph plus active learning, and "outperforms existing cluster repair methods without distinguishing between duplicate-free or dirty data sources" — [Graph-based Active Learning for Entity Cluster Repair, arXiv 2401.14992 (2024)](https://arxiv.org/abs/2401.14992)
- GraLMatch shows why clustering noisy records helps recall. Some groups "can only be discovered" transitively, for example when only one record carries the acquisition information that links the others. It also shows the danger: false edges between groups create "numerous false transitive matches" — [GraLMatch 2025](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- Entity linking plus discovery (Agarwal, Angell, Monath, McCallum, NAACL 2022) builds minimum arborescences (directed spanning trees) over *mentions and KB entities together*. It uses mention-to-mention affinities to make linking decisions and clusters mentions with no KB entity into new entities (NIL clustering). It reports "significant improvements in performance for both entity linking and discovery" over identically parameterised models on Zero-Shot EL and MedMentions — [ArboEL, arXiv 2109.01242](https://arxiv.org/abs/2109.01242)
- Survey framing: incremental and query-time ER keep "representative entity descriptions for each set of already resolved descriptions", and each query description "corresponds either to descriptions already resolved to a distinct real-world entity, or to a new one" — [Christophides et al. survey](https://arxiv.org/pdf/1905.06397)
- JedAI implements CNC, RSR, BMC, EXC, KRC, UMC and related clustering methods for Clean-Clean ER in Java — [Papadakis et al., arXiv 2112.14030](https://arxiv.org/pdf/2112.14030)

### Inferences
- Our problem maps almost exactly onto **entity linking with NIL clustering**: S1 is the knowledge base, S2/S3 records are the mentions, and decoys are NIL mentions that belong to absent businesses. ArboEL's recipe suggests a concrete design. Build one graph with S1 nodes and record nodes. Its edges are record→S1 edges (existing LightGBM) and record↔record edges (a second model). Take a minimum spanning arborescence or forest over it. A record tree that attaches to an S1 root is linked; a tree with no S1 root (or whose best root edge is weak) is a decoy group.
- A practical CPU version that avoids full clustering of 10M records: build record-record edges **only among records that share at least one candidate S1** (the S1's candidate pool, ≤ tens of records). Cluster inside each pool, then decide per record-group rather than per record. Pool-local graphs stay tiny, so exact methods are affordable.
- Training labels for a record-record model come free from the training ground truth. Two records assigned to the same S1 are a positive pair; two records assigned to different S1s are a negative pair. Pairs involving decoys are unknown, because decoys have no ID telling whether two decoys are the same absent business.
- Do **not** use FAMER/CLIP/GraLMatch as-is. Their source-consistency constraint (≤1 record per source per cluster, max cluster size = number of sources) would wrongly split legitimate S1 groups that have several S2 records. Only the "strong link" idea survives, on the S1 side only: a record's best S1 edge.

### Gaps
- I found no paper reporting the gain from "cluster the noisy sources first, then link clusters to a clean reference" versus "link each record to the reference directly" on business/POI data with a high decoy rate.
- ArboEL's exact linking-accuracy and discovery numbers were not extracted (only the abstract-level claim).
- Whether JedAI's multi-source or dirty-ER clustering scales to 10M records on one CPU box was not verified.

---

## Q3. Assignment / bipartite / min-cost-flow formulations with an "unmatched" option; 1-1 constraints (Jaro 1989, Sadinle 2017); Hungarian and alternatives at scale

### Takeaway
Enforcing exclusivity constraints consistently beats unconstrained thresholding. Microsoft's movie ER found 1-1 methods reach about 90% precision and recall at the knee versus about 81%/69% for many-to-many, and the constraint made results robust to a weaker scorer. However, **exact max-weight matching (Hungarian) is not the best rule**: a simple global greedy (sort edges, accept if both ends are free) or mutual-best rule matched or beat it and runs in O(E log E). In our one-to-many setting, the exclusivity lies only on the record side, which is exactly our one-owner rule, so the classic 1-1 gains are largely already captured.

### Cited Findings
- Jaro (1989) runs a **linear sum assignment** (optimal 1-1 pairing) *before* applying the Fellegi-Sunter decision rule, because Fellegi-Sunter alone can link several records in one file to the same record in the other. This was used for the 1985 Tampa test census and the 1990 US Census — [Sadinle 2017 (arXiv 1601.06630)](https://arxiv.org/pdf/1601.06630); [Winkler, application to the 1990 Census](https://www.researchgate.net/publication/2507428_An_Application_Of_The_Fellegi-Sunter_Model_Of_Record_Linkage_To_The_1990_US_Decennial_Census)
- Sadinle (JASA 2017, vol. 112, pp. 600-612) argues that the independence of pair match statuses assumed by Fellegi-Sunter "is unreasonable". It treats the whole bipartite matching as the parameter, derives Bayes point estimates under several loss functions, and proposes "**partial Bayes estimates that allow uncertain parts of the bipartite matching to be left unresolved**" (a reject option). It "outperforms the traditional methodology" in challenging simulated scenarios. The R package is BRL — [Sadinle 2017](https://arxiv.org/abs/1601.06630); [BRL on GitHub](https://github.com/msadinle/BRL)
- Gemmell, Rubinstein, Chandra (Microsoft Research TR-2011-100) compared, on IMDB vs Netflix movie catalogues:
  - ManyMany: plain threshold, no constraint.
  - FirstChoice: one-to-many, each record takes its best partner (= our one-owner).
  - MutualFirstChoice: both sides' best.
  - Greedy: sort all edges by score and accept an edge if neither endpoint is used.
  - MaxWeight: max-weight bipartite matching.

  Findings: "globally constrained resolution uniformly improves upon unconstrained resolution"; "constrained one-to-one ER is significantly more effective than one-to-many ER"; and "maximizing matching weight does not necessarily optimize performance". On the hard Boundary test set, the 1-1 methods reach precision and recall close to 90% at the knee, versus about 83%/81% for FirstChoice and 81%/69% for ManyMany. Greedy agreed with Freebase on 12,695 of 13,005 pairs (98%). The algorithms are used in production at a major search engine — [Gemmell et al. 2011](https://arxiv.org/pdf/1108.6016)
- Gemmell et al. also found that the 1-1 constraint makes results robust to a poorly tuned score function. The IMDB-Netflix model applied without retraining to AMG-iTunes degraded least for the 1-1 algorithms and most for ManyMany, so practitioners "can put less effort into the underlying scoring if they make use of one-to-one constraints" — [Gemmell et al. 2011](https://arxiv.org/pdf/1108.6016)
- A near-twin example from Gemmell et al. (their Fig. 12): "The Wonderful Wizard of Oz" (1987) and "The Marvelous Land of Oz" (1987) share cast and nearly equal runtimes, with scores 0.96/0.99/0.93. FirstChoice adopts two arcs, one of them wrong; MutualFirstChoice keeps only one, losing a true match; Greedy matches both correctly. MaxWeight can pick "diagonal" wrong arcs when a bonus-material or making-of entry scores high (their Fig. 13) — [Gemmell et al. 2011](https://arxiv.org/pdf/1108.6016)
- Papadakis, Efthymiou, Thanos, Hassanzadeh (VLDB Journal 2023; arXiv 2112.14030) evaluated 8 bipartite matching algorithms for Clean-Clean ER on 10 real datasets and more than 700 similarity graphs: CNC, RSR, RCA, BAH, BMC, EXC, KRC and UMC. The Hungarian algorithm was excluded as O(n³), as was a min-cost-flow method at O(n² log n) — [Papadakis et al. 2023](https://arxiv.org/pdf/2112.14030)
  - **UMC (Unique Mapping Clustering)** prunes edges below t, then greedily accepts edges in decreasing weight when both endpoints are unmatched (the same principle as Gemmell's Greedy). **EXC** keeps only mutual-best pairs above t, "a strict version of the reciprocity filter". **KRC** is a 3/2-approximation to maximum stable marriage, O(n + m log m). **BMC** is best-match (stable-marriage-like) — [Papadakis et al. 2023](https://arxiv.org/pdf/2112.14030)
  - Conclusions: the best algorithm "mainly depends on the type of edge weights and the portion of duplicates". "UMC is the best choice for balanced entity collections." EXC "consistently achieves (close to) the maximum F1 over scarce and one-sided entity collections" and is "the best choice for applications requiring both high effectiveness and efficiency". KRC achieves very high or the highest effectiveness in most cases, at higher run-time — [Papadakis et al. 2023](https://arxiv.org/pdf/2112.14030)
- Simple threshold-based matching "does not guarantee that each source entity can be matched with at most one other entity", and a single global threshold ignores that "similarity scores vary significantly depending on the characteristics of the entities" — [Papadakis et al. 2023](https://arxiv.org/pdf/2112.14030)

### Inferences
- Our problem is **one-to-many**: an S1 entity takes 0-11 records and a record takes 0-1 S1. The binding exclusivity is therefore only "one record, one owner", which is Gemmell's FirstChoice seen from the record side. Classic 1-1 gains (Jaro, Sadinle, Gemmell's 90% vs 83%) come mostly from *two-sided* exclusivity, which does not exist here. The residual gain available is the kind Gemmell's Oz example shows: record-side exclusivity decided **globally in score order** (Greedy/UMC) rather than per-record argmax. This matters only when one record's top two S1s are close and the runner-up S1 has no other good candidates.
- An "unmatched" option maps naturally to a **min-cost flow**: source → each record (capacity 1) → candidate S1s (edge cost = −log-odds) or → a dummy "decoy" sink (cost = decoy threshold) → S1 → sink (capacity 11, or a soft cost per extra record). This is only worthwhile if a per-S1 capacity or diminishing-returns term is actually informative. Otherwise it reduces to one-owner plus threshold. With about 10M record nodes and a few candidate edges each, network-simplex or cost-scaling solvers (for example OR-tools `SimpleMinCostFlow`) are feasible on CPU, but I found no ER paper benchmarking this at 10M.
- Sadinle's **partial Bayes estimate with a reject option** is the statistics-literature analogue of our expected-F0.5 set selection. Leaving uncertain links unresolved is the right behaviour under F0.5, which weights precision more than recall.

### Gaps
- No source quantifies one-owner (FirstChoice) versus global greedy on a one-to-many problem with a large decoy fraction.
- No ER benchmark of min-cost flow or auction algorithms at 10M-record scale was found. Papadakis et al. explicitly excluded Hungarian and min-cost flow for complexity reasons.
- I did not verify Sadinle's simulation numbers (only the abstract claim that the method "outperforms the traditional methodology").

---

## Q4. Set-level / metric-aware decision rules (expected F-measure optimisation): do collective decisions help for a macro per-entity F-score?

### Takeaway
Choosing each S1's output set to maximise *expected* F0.5 is the "decision-theoretic approach" (DTA). Under independent labels, its optimum is always "take the top-k candidates by probability" for some k, which is exact and cheap (O(n²), n = candidates per S1). Ye et al. (ICML 2012) show that DTA beats threshold tuning when the probability model is good, especially for rare positives and under domain shift (relevant to a new test country such as France). Threshold tuning is more robust when probabilities are miscalibrated. The main "collective" upgrade left is to drop the independence assumption: sibling records' labels are correlated, and the optimum under dependent labels can differ from top-k.

### Cited Findings
- Two paradigms: "the empirical utility maximization (EUM) approach learns a classifier having optimal performance on training data" (for example tuning a score threshold), while "the decision-theoretic approach learns a probabilistic model and then predicts labels with maximum expected F-measure" — [Ye, Chai, Lee, Chieu, ICML 2012](https://icml.cc/2012/papers/175.pdf)
- Results: "Given accurate models … the two approaches are asymptotically equivalent given large training and test sets. … EUM … appears to be more robust against model misspecification, and given a good model, the decision-theoretic approach appears to be better for handling rare classes and a common domain adaptation scenario." The rare-class benefit is stated for *small* datasets, and the paper's theory holds only for large sets — [Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
- EUM optimises instance classifiers (one label per instance), while DTA optimises **set classifiers** (a function from a set of instances to a set of labels). The optimal DTA classifier maximises expected F over the set — [Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
- Assuming independent labels, Lewis (1995) showed that in the optimal prediction no excluded item has a higher probability than an included item, so the optimum is a top-k set. Chai (2005) gave an O(n³) exact algorithm, Jansche (2007) an O(n⁴) one, and Ye et al. a new O(n²) algorithm that handles "tens of thousand instances within seconds" — [Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
- Dembczyński et al. (NIPS 2011) give an exact O(n³) F-maximiser *without* the independence assumption. It needs n²+1 parameters of the joint label distribution: for each item i and each count s, P(y_i = 1, Σy = s), plus P(Σy = 0). They showed independence "can lead to bad performance in the worst case", but on their practical datasets independence-based methods were "at least as good" — [as summarised in Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
- Jansche (HLT/EMNLP 2005) trains logistic regression to maximise a non-convex approximation of expected F directly — [cited in Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
- A caution on F-measure for record linkage: Hand & Christen (Statistics and Computing 2018) is "A note on using the F-measure for evaluating record linkage algorithms", which argues that F-measure has problematic properties as an evaluation measure for record linkage — [cited in Draisbach et al. 2019](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2019_draisbach_transforming.pdf)

### Inferences
- With ≤ about 30 candidates per S1, exact expected-F0.5 over top-k prefixes is trivial. The collective gain available is not in the optimiser but in **the probabilities it is fed**. Two concrete upgrades follow from the literature:
  1. **Calibrate per-pair probabilities after one-owner.** A record's probability for S1 A should be reduced by the probability that it belongs to a competing S1 B. One-owner currently makes this a hard 0/1 cut.
  2. **Model label dependence.** Dembczyński's joint-count parameters P(y_i = 1, Σy = s) could be approximated by a second-stage model predicting "how many true matches does this S1 have" (0, 1, 2, …). Combining it with per-pair probabilities gives an S1-level "count prior". This directly targets S1 entities with 0 true matches, where any output costs the full per-entity F0.5.
- Ye et al.'s domain-adaptation finding supports keeping DTA (expected F) rather than a single tuned threshold for an unseen test country (France), *provided* probabilities stay calibrated there. Their robustness finding supports keeping a tuned-threshold fallback and comparing both on a held-out country.
- How per-entity F0.5 is defined for an S1 with no true matches and an empty prediction (1, 0, or excluded) changes the optimal rule for "output nothing". This must be checked against the competition's metric definition. The literature does not settle it.

### Gaps
- No paper was found that measures expected-F set selection versus thresholding specifically for **macro per-entity** F in ER. Ye et al. and Dembczyński are about set/multi-label F.
- Quantitative gains of Ye et al.'s DTA versus EUM (table numbers) were not extracted, only the qualitative conclusions.

---

## Q5. Evidence on detecting decoy clusters (coherent groups of records linked to no reference entity)

### Takeaway
There is no ER paper on "decoys" as such, but three bodies of work apply:
1. NIL clustering in entity linking: mentions that match no KB entity are clustered into new entities, and mention-mention affinity helps (ArboEL).
2. Incremental multi-source ER: new records that do not reach an existing cluster through a mutual-best link start a new cluster (Max-Both), and neighbourhood re-clustering fixes order effects.
3. Graph-structure diagnostics: bridges, min-cuts, betweenness and density flag edges that wrongly join two coherent groups (GraLMatch, TransClean, Splink).

For near-twins (same name, house number changed), the only directly relevant evidence is Gemmell's "Oz" case, which is resolved correctly by global greedy assignment and incorrectly by per-record argmax.

### Cited Findings
- ArboEL links a mention to a KB entity or leaves it in a mention-only cluster (a new entity) by partitioning a minimum arborescence over mentions and entities. Mention-mention edges supply the "these unlinked records belong together" evidence. It reports significant gains in both linking and discovery — [Agarwal et al., NAACL 2022](https://arxiv.org/abs/2109.01242)
- NASTyLinker is another NIL-aware, transformer-based linker (2023) — [arXiv 2303.04426](https://arxiv.org/pdf/2303.04426) (not read in detail)
- In incremental multi-source ER, a new entity joins an existing cluster only through a max-both link, and otherwise forms "a new cluster". Neighbourhood re-clustering (n-Depth Reclustering) makes results as good as batch clustering — [Saeedi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7250616/)
- False-positive edges "are often the only link between sets of densely connected nodes". Minimum edge cut and edge betweenness find them. Min-cut always disconnects the component, while betweenness is expected to remove fewer true edges but runs slower — [GraLMatch 2025](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- Transitive consistency, meaning whether the pairwise model agrees on all pairs inside a component, detects 48-97% of false positives — [TransClean 2025](https://arxiv.org/html/2506.04006)
- Splink recommends density, centralisation, bridge and cluster-size distributions to find suspicious clusters — [Splink docs](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/overview.html)
- Near-twin case: two different 1987 Oz films with the same cast and similar runtimes. Per-record best choice (FirstChoice) produced a false positive; global greedy produced both correct matches — [Gemmell et al. 2011](https://arxiv.org/pdf/1108.6016)
- Sadinle's partial Bayes estimates leave uncertain links unresolved rather than forcing a match — [Sadinle 2017](https://arxiv.org/abs/1601.06630)

### Inferences (proposed features and rules for our pipeline, built from the findings above)
- **Competing-group feature.** Inside an S1's candidate pool, cluster the records with the record-record model (Q2). Suppose the pool splits into two or more coherent groups, for example group G1 at "12 Main St" and group G2 at "14 Main St", both with the same name. Then at most one group is the S1's true group, and the other is likely a near-twin decoy group. Features to add: number of coherent groups in the pool, this record's group size, this group's mean S1 score minus the best other group's mean S1 score, and house-number agreement of the group's majority value with S1. This is the same "bridge between two dense groups" signal GraLMatch and TransClean exploit, applied around an S1 hub.
- **Group-level decoy score.** Aggregate over each record-group: the maximum S1 score, the share of members whose best S1 is the same S1, and within-group agreement on the fields that decoys perturb (house number, phone, postcode). A tight group whose consensus address disagrees with every candidate S1 is the "NIL cluster" pattern and should be output for no S1.
- **Global greedy instead of per-record argmax for contested records.** When a record's top-2 S1 scores are close, resolve assignments in global score order (UMC/Greedy) so that an S1 already well covered by a strong group does not also absorb a near-twin, and vice versa. This is Gemmell's Oz case. Measure it on out-of-fold predictions before adopting.
- Training and evaluation must be done on held-out folds grouped by S1 entity (and ideally by geography), so that group-level features do not leak labels.

### Gaps
- I found no published evaluation of decoy or NIL-cluster detection in record linkage with a known decoy rate (26-40%) or with synthetic near-twins produced by perturbing house numbers.
- No quantitative evidence was found on how often decoys form multi-record coherent groups versus singletons. This must be measured on our own training data (for example: among decoy records, what fraction have a high-scoring record-record neighbour that is also a decoy).
- ArboEL's gains are in text entity linking (Zero-Shot EL, MedMentions). Transfer to structured business records is plausible but not demonstrated.
