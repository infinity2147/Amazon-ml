# Label/prior shift, calibration and per-entity F0.5 decoding when the test set has more decoy records

Context these notes are written for (from the task brief and the project's own playbook skill,
`amazon-ml-entity-resolution/SKILL.md`):
- Pipeline: blocking produces candidate pairs (one S1 entity × one S2/S3 record). A LightGBM model
  scores each pair; isotonic regression, fitted on out-of-fold (OOF) predictions, turns scores into
  calibrated P(match). Then, for each S1 entity separately, a decoder picks the top-k candidates
  (k may be 0) that maximise the *expected* F0.5 of that entity, computed exactly by a Poisson-binomial
  dynamic programme under an assumption that the pair labels are independent.
- Metric: F0.5 = 1.25·P·R / (0.25·P + R) per S1 entity, averaged over all S1 entities. An S1 with no
  true match ("singleton") scores 1 if we predict the empty set and 0 otherwise.
- Shift: about 40% of S2/S3 records are unmatched ("decoys") in test vs about 26% in train, so the test
  has roughly twice the decoy density. Cross-validation (CV) score 0.978 vs public leaderboard (LB)
  0.9655. Test also contains France, which has no training rows (a covariate shift, separate from the
  decoy shift).

Terms used below:
- **Label shift / prior shift**: P(y) changes between train and test while P(x | y) stays the same.
  Here y is the pair label (match / non-match); x is the pair's feature vector.
- **Prior π**: the fraction of candidate pairs that are true matches. π_s = in train (source), π_t = in test (target).
- **MLLS**: maximum-likelihood label-shift estimation; the EM algorithm of Saerens et al. is one way to compute it.
- **BBSE**: Black Box Shift Estimation, a confusion-matrix (moment-matching) estimator.

---

## Q1. Label-shift estimation and correction without target labels (EM/MLLS, BBSE, RLLS, bias-corrected calibration), when the assumption holds, and how to apply it to per-entity pair groups

### Takeaway
The strongest, simplest tool is EM/MLLS (Saerens 2002) run on *calibrated* probabilities, with the
source prior set to the mean calibrated OOF probability (Alexandari et al., ICML 2020). It beat BBSE
and RLLS across their benchmarks and needs no retraining: estimate the test pair prior π_t from
unlabelled test pair probabilities, rescale every pair's odds by (π_t/π_s)/((1−π_t)/(1−π_s)), and then
decode. The catch for this task: "P(x | y) unchanged" only holds if decoys look like training
negatives *and* the pair features do not depend on how many other candidates an entity has. Rank or
gap features, candidate counts and France all break it.

### Cited Findings
- **Saerens, Latinne & Decaestecker (2002)**, "Adjusting the Outputs of a Classifier to New a Priori
  Probabilities: A Simple Procedure", *Neural Computation* 14(1):21–41. An EM procedure re-estimates
  the class priors of new data and adjusts the classifier's posteriors to match, with no new labels needed. — [MIT Press](https://direct.mit.edu/neco/article/14/1/21/6577/Adjusting-the-Outputs-of-a-Classifier-to-New-a); [PubMed](https://pubmed.ncbi.nlm.nih.gov/11747533/)
- EM update equations, as restated by Alexandari et al. (2020), for N unlabelled test points x_k:
  - E-step: q^(s)(y=i | x_k) = [q^(s)(y=i)/p(y=i)] · p(y=i | x_k) / Σ_j [q^(s)(y=j)/p(y=j)] · p(y=j | x_k)
  - M-step: q^(s+1)(y=i) = (1/N) Σ_k q^(s)(y=i | x_k)
  — [Alexandari et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1901.06852)
- **Alexandari, Kundaje & Shrikumar (ICML 2020)**, "Maximum Likelihood with Bias-Corrected Calibration
  is Hard-To-Beat at Label Shift Adaptation":
  - MLLS combined with a calibration method that has per-class bias terms (they call it
    "bias-corrected calibration") "outperforms both BBSL and RLLS across diverse datasets and distribution
    shifts". The maximum-likelihood objective is concave, and no retraining with importance weights is needed. — [arXiv 1901.06852](https://arxiv.org/abs/1901.06852)
  - Calibrators compared: Temperature Scaling (TS, one parameter), Vector Scaling (VS, per-class scale
    and bias), Bias-Corrected Temperature Scaling (BCTS = TS plus per-class biases, a multiclass
    generalisation of Platt scaling), and No-Bias Vector Scaling (NBVS). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1901.06852)
  - Source-prior recommendation: "set p̂(y=i) to the average value of p̂(y=i|x) over the source-domain
    validation set" rather than the empirical label frequency. This makes EM return the original priors
    when there is no shift. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1901.06852)
  - Example numbers, CIFAR-10 with Dirichlet shift (α=0.1, n=2000 test points), mean squared error of
    the estimated weights: EM+BCTS 0.00037; EM+VS 0.00061; BBSL-soft+TS 0.0031; RLLS-soft+TS 0.00312.
    That is about 8× lower error for EM with bias-corrected calibration. Accuracy gains after adaptation
    were about 5.8% for BCTS/VS vs about 5.6% for TS. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1901.06852)
  - Reference implementation: the `abstention` package (`pip install abstention`) implements EM
    (Saerens), BBSL, RLLS and the calibrators Platt, **Isotonic**, TS, VS, BCTS and NBVS. Its companion
    notebooks are at `blindauth/labelshiftexperiments`. — [GitHub kundajelab/abstention](https://github.com/kundajelab/abstention)
- **Lipton, Wang & Smola (ICML 2018)**, "Detecting and Correcting for Label Shift with Black Box
  Predictors" (BBSE):
  - Assumptions: (A.1) label shift, p(x|y)=q(x|y); (A.2) support, q(y)>0 ⇒ p(y)>0; (A.3) the expected
    confusion matrix C = p(f(x), y) is invertible. The predictor may be biased or uncalibrated. — [ar5iv 1802.03916](https://ar5iv.labs.arxiv.org/html/1802.03916); [arXiv](https://arxiv.org/abs/1802.03916)
  - Estimator: ŵ = Ĉ⁻¹ μ̂_ŷ. Here Ĉ is the joint confusion matrix (predicted label × true label) on
    held-out *source* data, μ̂_ŷ is the distribution of predicted labels on unlabelled *target* data,
    and ŵ_y = q(y)/p(y). Target prior: μ̂_y = diag(ν̂_y)·ŵ. Correction: importance-weighted ERM,
    min Σ_i w_{y_i} ℓ(y_i, x_i). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.03916)
  - Detection: under A.1–A.3, q(y)=p(y) if and only if p(ŷ)=q(ŷ), so a two-sample test (KS,
    Anderson-Darling, MMD) on the model's *outputs* detects label shift. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.03916)
- **Azizzadenesheli, Liu, Yang & Anandkumar (ICLR 2019)**, "Regularized Learning for Domain Adaptation
  under Label Shifts" (RLLS): a regularised weight estimator for small samples that accounts for
  uncertainty in the estimated weights, followed by weighted training. It improves accuracy "especially
  in the low sample and large-shift regimes" on CIFAR-10 and MNIST. — [arXiv 1903.09734](https://arxiv.org/abs/1903.09734)
- **Garg, Wu, Balakrishnan & Lipton (NeurIPS 2020)**, "A Unified View of Label Shift Estimation":
  BBSE is roughly MLLS with a particular coarse calibration. MLLS is empirically better because BBSE
  loses information through that coarse calibration. MLLS error decomposes into a miscalibration term
  and a finite-sample estimation term. Both methods need calibration plus invertibility to be
  consistent. — [arXiv 2003.07554](https://arxiv.org/abs/2003.07554)

### Inferences
Everything in this block is our reasoning applied to the task. None of it is a published result.
- **Concrete binary recipe (EM on pairs).** Let p_k be the isotonic-calibrated P(match) of test pair k.
  Clip it to [1e-4, 1−1e-4], because isotonic outputs can be exactly 0 or 1 and those break the ratio.
  1. Set π_s = mean of calibrated OOF probabilities over all training candidate pairs. Pool
     Adjacent Violators makes the fitted isotonic values average to the label mean on the fitting set,
     so this is close to the empirical match rate, which is what Alexandari et al. recommend.
  2. Start with π = π_s. Repeat until converged:
     a = (π/π_s)·p_k; b = ((1−π)/(1−π_s))·(1−p_k); p'_k = a/(a+b); then π ← mean_k p'_k.
  3. Decode each S1 entity with the p'_k values in place of p_k.
  This is the Saerens EM specialised to two classes. On the logit scale it is a constant shift of
  log[(π_t/(1−π_t)) / (π_s/(1−π_s))] added to every pair.
- **Worked example of how much this matters (illustrative numbers, not measured on our data).** Say
  20% of training candidate pairs are matches (π_s = 0.20, odds 0.25) and extra decoys push the test
  share down to 14% (π_t = 0.14, odds 0.163). Every pair's odds are multiplied by 0.65, a logit shift
  of −0.43. A pair at p = 0.55 becomes p' = 0.44.
  - For an S1 with that single candidate, the decoder compares E[F0.5 | predict it] = p with
    E[F0.5 | predict nothing] = 1 − p, so the decision flips from "predict" to "empty".
  - This is exactly where extra decoys hurt us: singleton S1s that pick up a plausible decoy score 0
    instead of 1.
- **Pair-level BBSE in binary form** (a cross-check that does not trust calibration). Pick a working
  threshold and measure TPR and FPR on OOF pairs. Then π_t = (q̂(ŷ=1) − FPR)/(TPR − FPR), where
  q̂(ŷ=1) is the fraction of test pairs predicted positive at that threshold. This is ŵ = Ĉ⁻¹μ̂ written
  out for 2×2. If EM and BBSE agree, calibration is probably fine on test. If they disagree, suspect
  that P(x|y) itself has shifted.
- **When the label-shift assumption breaks in this pipeline:**
  1. **Group-relative features.** Rank within the S1's candidate list, the gap to the best candidate,
     the number of candidates, and "is mutual best match" all change for true matches when more decoys
     enter the same block. So P(x | match) changes, not just P(match), and a single global prior
     correction will under- or over-correct.
  2. **Decoy type.** If test decoys are "harder" than training negatives (near-duplicates such as chain
     branches or the same name at a different address), P(x | non-match) shifts toward the match region.
     That is covariate/concept shift inside the negative class, and it needs the Q4 tools.
  3. **France.** No training data at all, so the France pair-feature distribution differs from US and
     India. Run EM per country (US, India, France separately) so that France's prior is not averaged
     with the others.
  4. **Blocking changes the prior.** π counts candidate *pairs*, so it depends on blocking. More decoys
     means more candidates per S1, which lowers π_t even if the number of true matches per S1 is unchanged.
- **Two-level variant for the per-entity structure.** The metric is decided mostly by whether a
  singleton S1 gets the empty set, so a second prior matters: ρ = P(an S1 entity has ≥1 match).
  - Estimate ρ_t on test as the mean over S1 of 1 − Π_k(1 − p'_k), which is the probability, under the
    decoder's independence assumption, that at least one candidate is a true match.
  - Compare it with the training singleton rate and with the LB "all-empty" probe (Q2).
  - If ρ_t from the EM-adjusted probabilities still exceeds the probe's value, the pair-level correction
    is too weak for singletons. An extra entity-level logit offset tuned on the simulation in Q2 is then justified.
- **Why not RLLS or importance-weighted retraining first.** Alexandari et al. found post-hoc EM better
  than both, and retraining LightGBM with class weights only changes the model's scores before
  isotonic recalibration anyway. Post-hoc reweighting is cheaper and can be undone, so it is the right first test.

### Gaps
- No published work was found that applies MLLS/EM specifically to entity-resolution candidate pairs,
  where the "instances" are grouped by entity and the features depend on the group. The failure modes
  above are reasoned, not measured.
- Not verified here: whether EM with isotonic calibration specifically (as opposed to BCTS) has been
  benchmarked. The `abstention` library supports isotonic, but I did not find reported numbers for
  EM+isotonic in the pages fetched.
- Our actual π_s, π_t and per-country shift are unknown until they are computed on our OOF and test predictions.

---

## Q2. How to validate a shift correction offline (simulated shift) and with a few leaderboard probes

### Takeaway
Rebuild the train→test shift inside the OOF data, then check that the correction recovers the
lost score before trusting it.
- Offline: create "shifted validation folds" with two protocols. Knock-out removes matched S2/S3
  records; decoy injection adds unmatched records from other folds or regions. Tune both until about
  40% of S2/S3 records are unmatched, then measure macro F0.5 with and without EM correction.
- On the leaderboard: probe the singleton fraction directly with an all-empty submission, and probe a
  small grid of prior offsets.

### Cited Findings
- BBSE was evaluated with three simulated shift protocols:
  - **knock-out**: remove a fraction δ of samples of one class;
  - **tweak-one**: give one class probability ρ and spread the rest uniformly;
  - **Dirichlet**: draw p(y) ~ Dir(α), where smaller α means larger shift.
  — [Lipton et al. 2018 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1802.03916)
- Alexandari et al. evaluated with Dirichlet shift (for example α = 0.1) and a fixed number of test
  samples (n = 2000), reporting MSE of the estimated weights and the change in accuracy. — [ar5iv 1901.06852](https://ar5iv.labs.arxiv.org/html/1901.06852)
- Shift *detection*: a two-sample test on a trained classifier's outputs (black-box shift detection)
  performed best among the detectors compared. Domain-discriminating classifiers were most useful for
  *characterising* what shifted. — [Rabanser, Günnemann & Lipton, NeurIPS 2019, arXiv 1810.11953](https://arxiv.org/abs/1810.11953)
- Estimating target performance without labels: ATC (Average Thresholded Confidence). It learns a
  confidence threshold on source data and predicts target accuracy as the fraction of unlabelled
  target examples above it. It was reported 2–4× more accurate than prior methods. The authors also
  prove that "identifying the accuracy is just as hard as identifying the optimal predictor", so any
  such estimate rests on assumptions about the shift. — [Garg et al., ICLR 2022, arXiv 2201.04234](https://arxiv.org/abs/2201.04234)
- Project playbook: an all-empty submission scores exactly the singleton fraction of the public test
  subset, which gives a direct label-free measurement of the entity-level prior. — local file
  `~/.claude/skills/synced/.../amazon-ml-entity-resolution/SKILL.md` (section on probes)

### Inferences
Everything in this block is our reasoning applied to the task. None of it is a published result.
- **Offline protocol A: knock-out (fewer true matches survive).** Inside each OOF validation fold,
  delete a random subset of *matched* S2/S3 records and re-run blocking and decoding.
  - Arithmetic for a fold with N S2/S3 records, 26% unmatched: deleting m matched records gives an
    unmatched fraction of 0.26N/(N − m). Setting that to 0.40 gives m ≈ 0.35N, which is about 47% of
    the matched records.
  - Effect: many S1 entities become singletons, and the remaining unmatched records act as decoys.
  - This matches the "knock-out shift" of Lipton et al.
- **Offline protocol B: decoy injection (extra distractors).** Add unmatched records d so that
  (0.26N + d)/(N + d) = 0.40, which gives d ≈ 0.233N.
  - Take them from S2/S3 records whose true match is in *another* fold, with that match removed from
    the current fold's S1 set. They are then genuine decoys with realistic noise.
  - Run a variant that draws the decoys from the same postal code or city as the fold's S1s, so they are hard.
  - This matches the case where the test set simply contains more distractor businesses.
- **Why run both.** Knock-out mostly raises the singleton rate. Injection mostly adds hard negatives to
  non-singleton S1s.
  - The LB all-empty probe tells you which is closer to the truth. If the public singleton fraction is
    much higher than train's, knock-out is the right model. If it is similar, the extra test decoys are
    distractors and injection is the right model.
- **What to report on each shifted fold (plain definitions):**
  - OOF macro F0.5 without correction;
  - the same with EM correction;
  - the same with an "oracle" correction that uses the true π of the shifted fold;
  - EM's estimated π_t vs the true π_t;
  - the score split into singleton S1s and matched S1s.

  A correction is worth submitting if it closes most of the gap to the oracle on both protocols and
  does not lower the *unshifted* OOF score by more than noise.
- **Leaderboard probes (spend only a few submissions):**
  1. **All-empty.** Gives the public singleton fraction ρ_LB exactly. Compare it with the training
     singleton rate and with the model's implied ρ_t (Q1).
  2. **Prior-offset grid.** Submit decodes with a global logit offset δ ∈ {0, δ_EM, 1.5·δ_EM}, where
     δ_EM is the EM-estimated shift. A peak at δ_EM supports the label-shift story. A peak further out
     suggests hard decoys (P(x|y) shift), and group-feature redesign or the covariate tools in Q4 are needed.
  3. **Per-country check.** If allowed, submit a version where France alone gets a different offset.
     Public/private splits are noisy, so trust only differences much larger than the LB's resolution.
- **Unlabelled monitors to compute on test before any submission:**
  - histogram of calibrated p;
  - predicted singleton rate per country;
  - mean candidates per S1;
  - mean predicted set size.

  Then run a KS test between the OOF p distribution and the test p distribution, per country, as
  black-box shift detection (Rabanser et al.).

### Gaps
- No published protocol was found for simulating decoy shift in entity resolution specifically. The
  knock-out and injection designs above adapt the generic label-shift protocols.
- The competition's public/private split size is unknown, so the noise level of a single LB probe
  cannot be quantified here.

---

## Q3. Decision-theoretic optimisation of per-group F-beta under uncertainty, robustness to miscalibration, and threshold vs expected-utility decoding

### Takeaway
The current decoder is the right structure: pick top-k by probability and choose k to maximise
expected F under independent labels. This is the classic plug-in or "decision-theoretic approach" (DTA).
- Ye et al. (2012) show DTA and threshold tuning (EUM) are asymptotically equivalent with accurate
  models. DTA wins for rare classes and under domain adaptation. EUM is more robust when the model is
  misspecified.
- So the decoder's optimality depends on calibration being correct *on test*. Under decoy shift it is
  not, which is why the prior correction (Q1) must come before decoding. A small tuned "safety offset"
  is the EUM-style hedge.
- Dembczyński et al. give an exact decoder (GFM) that does not need the independence assumption.

### Cited Findings
- **Jansche (ACL 2007)**, "A Maximum Expected Utility Framework for Binary Sequence Labeling", pp.
  736–743. Shows that the number of hypotheses whose expected F-score must be evaluated grows only
  linearly with sequence length, and gives a framework for efficiently computing expected
  utilities, including F. — [ACL Anthology P07-1093](https://aclanthology.org/P07-1093/)
- **Nan Ye, Chai, Lee & Chieu (ICML 2012)**, "Optimizing F-measure: A Tale of Two Approaches".
  - The two approaches: EUM (empirical utility maximisation, i.e. tune a classifier or threshold to
    maximise F on training data) and DTA (fit a probability model, then predict the labels that
    maximise expected F).
  - "Given accurate models, our results suggest that the two approaches are asymptotically equivalent
    given large training and test sets."
  - EUM is more robust to model misspecification. DTA does better with a good model, especially for
    rare classes and in domain adaptation.
  — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)
- **Dembczyński, Waegeman, Cheng & Hüllermeier (NeurIPS 2011)**, "An Exact Algorithm for F-Measure
  Maximization" (the General F-measure Maximizer, GFM).
  - No closed-form maximiser exists, and earlier algorithms were approximate and "rely on additional
    assumptions regarding the statistical distribution of the binary response variables".
  - GFM is exact for any distribution and needs only a quadratic number of parameters of the joint
    distribution: m²+1 parameters and O(m³) operations for m labels.
  — [NeurIPS page](https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization); [search summary of complexity](https://www.semanticscholar.org/paper/An-Exact-Algorithm-for-F-Measure-Maximization-Dembczynski-Waegeman/211229be3e804881118ec8ab95da511df65fb4df)
- **Dembczyński, Jachnik, Kotłowski, Waegeman & Hüllermeier (ICML 2013)**, "Optimizing the F-Measure in
  Multi-Label Classification: Plug-in Rule Approach versus Structured Loss Minimization". A plug-in
  rule estimates all parameters needed for the Bayes-optimal prediction with a set of multinomial
  regression models. It is proven *consistent*, whereas structured SVM approaches are not, and it
  performed well in a large experimental study. — [PMLR v28](https://proceedings.mlr.press/v28/dembczynski13.html)
- **Waegeman, Dembczyński, Jachnik, Cheng & Hüllermeier (JMLR 2014)**, "On the Bayes-Optimality of
  F-Measure Maximizers", JMLR 15:3513–3568. Surrogate-loss methods (including approaches that assume
  label independence) can have substantial worst-case regret. The exact algorithm is Bayes-optimal
  for all distributions with quadratic parameters. — [JMLR](https://jmlr.org/papers/v15/waegeman14a.html)
- **Lipton, Elkan & Narayanaswamy (2014)**, "Thresholding Classifiers to Maximize F1 Score": for a
  calibrated classifier, the F1-optimal threshold equals half the optimal F1. For an uninformative
  classifier the F1-optimal policy is to predict everything positive. — [arXiv 1402.1892](https://arxiv.org/abs/1402.1892)

### Inferences
Everything in this block is our reasoning applied to the task. None of it is a published result.
- **Threshold view of F0.5.** Generalising the Lipton et al. argument to F_β gives an optimal
  threshold for calibrated scores of F*_β/(1+β²). For β = 0.5 that is 0.8·F*. With F* ≈ 0.97 the
  implied pair threshold is about 0.78, which is why the decoder is so conservative. We did not fetch
  the paper that states the F_β generalisation (Koyejo et al., NeurIPS 2014), so check the formula
  before relying on it.
- **Sensitivity to miscalibration.** For S1 entities with one plausible candidate, the decision is
  simply "predict if p > 0.5" (see the Q1 worked example). Any upward bias in p on test, which is what
  a higher decoy rate produces, converts correct empties on singletons (score 1) into wrong
  predictions (score 0). That is a loss of 1.0 per affected entity. In macro-F, an error rate of 1.3%
  on singletons alone would explain the entire 0.978 → 0.9655 drop *if* singletons were the whole set.
  The real attribution needs the per-group breakdown on the LB (Q2).
- **When independence fails.** The Poisson-binomial decoder assumes candidate labels within an S1 are
  independent. Under decoy shift they are *negatively* dependent: several near-identical candidates
  for one S1 usually mean at most one is real, or none is.
  - GFM (Dembczyński 2011) removes the independence assumption, but it needs P(y_i = 1, |y| = s) for
    each candidate i and set size s.
  - One could estimate that with a second-stage multinomial model per candidate over s = total true
    matches of the S1, as the ICML 2013 plug-in rule does. This is a heavier project. Try it only if
    simulation shows the independent decoder is the bottleneck.
- **Hedge between DTA and EUM (cheap and robust).** Keep expected-F decoding, but add two scalar knobs:
  a global logit offset δ (the prior correction) and an extra cost λ on non-empty predictions for S1s
  whose best candidate is below some confidence. Tune both on the *shifted* OOF folds from Q2. This
  follows Ye et al.'s finding that empirical tuning is more robust to misspecification, while keeping
  DTA's per-entity optimality.
- **Calibrate on shifted data.** After EM correction, a second isotonic fit is not possible on test
  because it has no labels. Instead, fit a one-parameter recalibration (δ, or δ plus a temperature) on
  the shifted OOF folds, then check that the "oracle δ" on those folds is close to the EM-estimated δ.

### Gaps
- The complexity of the Ye et al. 2012 DTA algorithm (the page cites an O(n²)/O(n³) dynamic programme)
  and its exact regret numbers could not be pulled; only the abstract-level conclusions are cited.
- I found no published comparison of expected-F decoding vs thresholding specifically under
  label shift for *set-valued per-entity* outputs like ours.
- The F_β threshold formula F*/(1+β²) is stated from derivation, not from a fetched source.

---

## Q4. Covariate shift, importance weighting, domain-adversarial methods, and test-time calibration with unlabelled test statistics

### Takeaway
If simulation shows EM prior correction does not close the gap, the shift is not pure label shift.
Likely causes are harder decoys or France.
- The standard next step is density-ratio importance weighting from a *domain classifier* that tells
  train pairs from test pairs (Bickel et al. 2007). Use the weights w(x) to reweight the isotonic
  calibration fit and, optionally, LightGBM training.
- Domain-adversarial training (DANN) targets neural encoders and is mostly relevant to a cross-encoder
  feature, not to LightGBM itself.
- Test-time calibration without labels is limited to prior correction (EM), distribution matching and
  performance estimation such as ATC. No unlabelled method can identify arbitrary shift.

### Cited Findings
- **Bickel, Brückner & Scheffer (ICML 2007)**, "Discriminative Learning for Differing Training and Test
  Distributions", pp. 81–88. Covariate-shift weights are learned discriminatively by training a
  probabilistic classifier to tell training instances from test instances. The resulting weights
  correct the learner (kernel logistic regression in the paper). — [dblp](https://dblp.org/rec/conf/icml/BickelBS07.html); [ACM DL](https://dl.acm.org/doi/10.1145/1273496.1273507)
- Rabanser et al.: domain-discriminating classifiers are best at *characterising* a shift (which
  examples or features drive it), and two-sample tests on model outputs are best at *detecting* it. — [arXiv 1810.11953](https://arxiv.org/abs/1810.11953)
- Garg et al. (ICLR 2022): estimating target accuracy from unlabelled data is provably as hard as
  finding the optimal target predictor, so every label-free method depends on assumptions about the
  shift. ATC works well empirically on natural shifts. — [arXiv 2201.04234](https://arxiv.org/abs/2201.04234)
- Garg et al. (NeurIPS 2020): MLLS consistency requires calibration on the target. Its error
  splits into a miscalibration term and an estimation term, so a model that is well calibrated on
  source but miscalibrated on target (the covariate-shift case) biases the prior estimate. — [arXiv 2003.07554](https://arxiv.org/abs/2003.07554)
- Domain-adversarial training of neural networks (DANN): Ganin et al., JMLR 2016. — [arXiv 1505.07818](https://arxiv.org/abs/1505.07818)
  (cited for identification only; its contents were not fetched in this session)

### Inferences
Everything in this block is our reasoning applied to the task. None of it is a published result.
- **Domain-classifier recipe for our pairs.**
  1. Build one table of candidate-pair feature vectors: all OOF training pairs labelled d = 0, all test
     pairs labelled d = 1. Exclude features that trivially identify the country, or also run it within country.
  2. Train a small LightGBM to predict d with cross-fitting.
  3. Compute w(x) = [P(d=1|x)/P(d=0|x)]·(n_train/n_test), and clip it, for example to [0.1, 10].
  4. Use w as sample weights when fitting the isotonic calibrator on OOF pairs (weighted PAV). The
     calibration map then reflects the test mix of easy and hard negatives.
  5. Diagnose with the AUC of the domain classifier. AUC near 0.5 means pair features look alike and
     the gap is label shift, so use EM. A high AUC, driven by features such as rank or gap or by
     France, means covariate shift, so use the weights and redesign those features.
- **Separating label shift from covariate shift.** Train the domain classifier on *negatives only*,
  meaning OOF pairs with label 0 vs test pairs with p' < 0.05.
  - If even these are separable, the test decoys are a different kind of negative (P(x|non-match)
    shifted). A prior correction cannot fix that. Harder training negatives (Q2 injection) or the
    cross-encoder feature are needed.
- **Make features shift-invariant.** Group-relative features (rank, gap to max, candidate count)
  change by construction when decoy density doubles. Options:
  1. Recompute them on OOF folds after decoy injection and retrain on that augmented data. This is
     training-time augmentation to the test regime.
  2. Replace them with features that do not depend on how many decoys an entity has, for example
     mutual-best-match computed within a source, or similarity to S1 only.
- **DANN** is only worth considering for the cross-encoder, where a gradient-reversal domain head can
  push US/India and France text embeddings to be similar. It does not apply to gradient-boosted trees.
  The playbook's monotone constraints and leave-one-country-out (LOCO) validation are the tree-model
  analogues for robustness.
- **Label-free test-time statistics that are still valid:**
  - EM-estimated π_t per country;
  - the domain-classifier AUC;
  - the ATC-style estimate. Choose a threshold t on OOF so that the fraction of S1s with max-E[F] above
    t equals the OOF macro F0.5, then apply the same t to test.

  If the ATC estimate on test comes close to the LB (0.9655), the model's confidence is still
  informative and the loss comes from decision/prior mismatch. If it stays near 0.978, the model is
  overconfident on test, i.e. miscalibrated, and needs the Q4 covariate tools.

### Gaps
- No published results were found for importance-weighted *isotonic* calibration or domain-classifier
  weighting applied to entity-resolution pair classifiers. The recipe is an adaptation.
- The DANN paper content was not fetched. It is cited for identification only.
- Whether the 0.978 → 0.9655 gap comes mainly from decoy density, France, or public-LB noise cannot be
  determined from the literature. It needs the probes and simulations in Q2.
