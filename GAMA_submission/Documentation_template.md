# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** GAMA  
**Team Members:** Aayush Prasad (Team Leader), Gaurav Damor, Arshad Khatib, Manav Dhamecha  
**Submission Date:** 29 September 2026

**Final submission:** public leaderboard macro F0.5 **0.990107**.
`output/matching_results.tsv` (md5 `b760fdbc7e6753571509a0ed40805aed`) and
`output/candidate_pairs.tsv` (md5 `5aff25825ae44012e935c10571ea4cba`), produced by the code in
`code/business_entity_resolution/` (git commit `06532db`). Appendix B traces the file to the exact steps
and settings that produced it.

---

## 1. Executive Summary

Our solution is a cascade that secures recall early and spends every later stage on precision, because
F0.5 punishes a wrong merge about 2.5 times more than a missed match.

1. **Candidates:** four TF-IDF searches plus a fine-tuned multilingual bi-encoder (bge-m3) find 99.92% of
   true pairs; a LightGBM pruner cuts them to 4.86 candidates per entity while keeping 99.62%.
2. **Scoring:** a fine-tuned cross-encoder (bge-reranker-v2-m3) scores every candidate. A two-level LightGBM
   stacker combines it with 69 pairwise features (level 1) and 32 collective features that compare each
   pair with its competitors (level 2).
3. **Test realism:** the stackers are trained, and the decision rule is tuned, on "stress worlds": training
   copies with 19% of the Source-1 owners removed, which reproduces the test set's higher distractor density.
4. **Decisions:** each Source-2/3 record keeps at most one owner, and each Source-1 entity gets the match set
   that maximises its *expected* F0.5, including the empty set.
5. **France:** France has no training labels. A final rule-based step, applied only to countries absent from
   training, removes decoy look-alikes and restores same-site "filler" copies. Each rule was checked on our own
   predictions with label-free tests before use.

Every learned component is cross-fitted, so validation is out-of-fold over all 2,206,821 training entities:
macro F0.5 **0.99234**, and **0.99180** on a held-out stress world. Public leaderboard:
0.985007 (first full run) → 0.987282 (stress-world stacking) → **0.990107** (France post-processing).

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale**

| | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | US 60%, India 40% |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | US 38%, India 47%, France 15% |

France never appears in training.

**Ground truth (verified on the training data)**
- Of 7,638,365 links, no S2/S3 record belongs to two clusters and none crosses countries. This justifies one
  owner per record and searching within each country.
- 5.6% of S1 entities are singletons; each S1 has 0–5 S2 and 0–6 S3 matches, 3.46 on average.
- 26% of S2/S3 records match no S1.

**Shift between train and test.** Test has 5.75 S2+S3 records per S1 against 4.68 in train, 23% more in
both the US and India. This matches about 19% of S1 owners removed from the test's Source 1 while their
records stay behind as look-alike distractors.

**Name noise:** case changes and repeated tokens ("Laxmi Laxmi Technologies"); reordering; legal-form
changes, moves and brackets ("63/95 Top [Inc]"); filler words ("Center", "Services"; in France "Groupe",
"Développement", "& Fils", "& Associés", which often *replace* a word: "Nantes Club SARL" → "Nantes SARL &
Associés"); typos and OCR-style swaps ("A1titude"); web domains ("uptowncoffeecenter.com", about 3%);
`X dba Y` aliases (about 73K, Source 3 only); phone and store numbers; completely replaced names; Hindi,
Tamil, Telugu, Kannada or Malayalam script (23.5% of Indian S2 names, 13% of Indian S3 names; S1 is always
Latin script).

**Address noise:** empty addresses (about 3% of S2/S3); abbreviations and full words used interchangeably;
states as codes, full names or native script (ಕರ್ನಾಟಕ, महाराष्ट्र); inserted or changed cities; corrupted house
numbers (9914 → 914/991, 218 → 218C, "6774 1/2"); reordered components; no postal codes anywhere.

**Precision traps**
- Namesakes: more than 20 S1 entities are named exactly "Summit Inc".
- Decoys that belong to no S1: the same street with the house number shifted *up* by one of
  {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, usually with a changed legal form or one extra word ("Arnold
  Empire-Newbridge Partners, 9720 Stardust Trl" is not "Arnold Empire Newbridge, 9707 Stardust Trail"), or the
  same address with the business-type word replaced (in the French test data, for example, "Creme Ecole SCI"
  at the address of "Creme Sportive SCI").
- True-copy house-number noise is symmetric: among true training pairs, a shift of +k (k in the set above)
  occurs 27,415 times and −k 27,494 times in the US (India: 24,147 and 25,238). An excess of upward shifts
  among predicted pairs therefore measures accepted decoys without labels.

**What the metric implies.** Per-entity F0.5 = 1.25·TP / (0.25·T + k) for T true and k predicted records,
averaged over all entities. For an entity with about 3.4 true records, one wrong merge costs 0.19 and one miss
costs 0.08, and a singleton scores 1.0 only for an empty prediction. So: protect recall early (a pair never
retrieved can never be matched), spend the rest on precision, choose each entity's set size separately, and
give every record at most one owner.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-pass lexical blocking plus fine-tuned dense retrieval, a GBDT pruner, a
fine-tuned cross-encoder, a two-level GBDT stacker, a metric-aware decision layer, and label-free rules for the
country without training labels.

**Core Innovation:**
1. **Cross-fitting end to end.** The bi-encoder and cross-encoder use 2 folds and the GBDTs 5 folds, all
   grouped by S1 entity. Every training pair gets a score from a model that never saw it, so every downstream
   model trains on test-like inputs for all 2.2M entities.
2. **Stress worlds.** Training copies with 19% of S1 owners removed, where everything that depends on which
   S1s exist is recomputed. Stackers train on them, and the decision rule is tuned on a held-out one.
3. **Collective (level-2) features.** Each pair is judged next to the S1's other candidates and the other S1s
   that want the same record: competition, house-number drift, coherence, namesake arbitration, record twins.
4. **An expected-F0.5 decision layer** that chooses each entity's set size, including "none", under the
   one-owner constraint.
5. **Label-free checks for the unseen country.** A symmetric-noise test and training base rates measured
   which French errors the model makes; four rules correct them, gated to countries absent from training.

```
records ─► prepare (two views per record) ─► retrieval: 4× TF-IDF + fine-tuned bge-m3 (union ~84/S1, recall 0.9992)
        ─► 61 pair features ─► LightGBM pruner (4.86/S1, recall 0.9962) ─► candidate_pairs.tsv
        ─► fine-tuned bge-reranker-v2-m3 on every candidate
        ─► stack level 1 (69 features) ─► stack level 2 (+32 collective features), trained on stress worlds
        ─► calibrate → one owner per record → expected-F0.5 set per S1
        ─► France rules (countries without training labels) ─► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used.** All searches run within each country value; no country list is hard-coded, so
  France is handled like any other country.
  1. **Name words:** TF-IDF on the core name (legal forms, stopwords and "dba" removed), sublinear term
     frequency, one vectorizer per country.
  2. **Address words:** TF-IDF on canonical address tokens; finds copies whose name was replaced or aliased.
  3. **Name + address:** the two vectors joined with weights √0.6 and √0.4, so one dot product equals
     0.6 · name cosine + 0.4 · address cosine.
  4. **Name character 3-grams** (within word boundaries): typos, concatenations, domains
     ("uptowncoffeecenter").
  5. **Fine-tuned bge-m3 bi-encoder** (MIT, 568M parameters) on the text "name | address": exact GPU
     inner-product search, top 50 per S1 per country. On test, the two fold models' embeddings are joined and
     scaled by 1/√2, so the search ranks by their mean cosine.

  Each lexical search keeps the top 20 records per S1. Words present in more than 3,000 records (10,000 for
  trigrams) are skipped during search, and each S1 searches with its 10 rarest trigrams, which keeps the sparse
  products tractable at 10 million records.
- **Bi-encoder training.** Symmetric InfoNCE (S1 → record and record → S1), temperature 0.02; batches of 256
  S1s from one country, each with one true record and 2 hard negatives sampled from its top-10 lexical
  non-matches; other true records of the same S1 in a batch are masked, never used as negatives; learning rate
  1e-5, one epoch, bf16. Model F trains on the S1s outside fold F, never sees fold F's records (not even as
  negatives), and retrieves only for fold F.
- **Recall (out-of-fold, train):** the four TF-IDF searches together 0.780; bi-encoder recall@10 0.9926,
  recall@50 0.9988; union **0.9992** at about 84 candidates per S1.
- **Pruning.** A LightGBM model (5 folds grouped by S1, fitted on 35% of S1s, out-of-fold scores for all)
  scores every union pair from the 61 pair features. Per S1 it keeps the 3 best candidates plus any candidate
  among the best 32 with score ≥ τ; τ is the largest threshold whose out-of-fold recall of retrieved true pairs
  is still ≥ 99.7% (bisection): τ = 0.0116. Pruned recall **0.9962** at 4.73 candidates per S1 on train;
  oracle F0.5 ceiling 0.9989.
- **Candidate pairs generated (test):** **8,412,076** pairs for 1,732,544 entities, **4.86 per entity**,
  written to `output/candidate_pairs.tsv`. This is exactly the set the cross-encoder scores, and every
  predicted match is one of these pairs.
- **How we ensured true matches were not lost:**
  - The searches fail on different records: name-only search covers empty addresses; address search covers
    replaced names and aliases; trigrams cover typos and domains; the bi-encoder covers scripts,
    transliteration and heavy rewording.
  - Every union pair receives every score (the dense cosine for lexical-only pairs, exact TF-IDF cosines for
    dense-only pairs), so no pair is judged on partial evidence.
  - Recall is measured for each search, the union and the pruned set; the pruner's recall target is explicit
    and tuned out-of-fold, and the top 3 per S1 are always kept.

---

## 4. Matching Model

### 4.1 Preprocessing and representations

Each record gets two views, because two kinds of models read it:
- **Model text** (for the neural models): NFKC; combining marks removed only after Latin letters (Indic vowel
  signs are also combining marks); lower case; `<NULL>` junk removed; original scripts kept; "name | address"
  with the name first, so truncation cuts the end of long addresses rather than the name. For French rows,
  "& Fils" / "& Associés" add-ons are stripped (they are French-only noise that the US/India-trained models read
  as a different business).
- **Lexical tokens** (for TF-IDF and features): "doing business as" → dba; web prefixes and domain endings,
  phone numbers and store numbers removed; legal forms canonicalised (incorporated → inc, private → pvt,
  limited → ltd, French forms kept); Indic tokens mapped to Latin; address abbreviations to one form
  (road → rd, street → st, north → n, suite → ste); every digit run kept in order (the first is usually the house
  number); state = the last address component found in a (country, alias) table. Three name views: legal forms,
  core name (without legal forms, stopwords and repeated tokens) and the alias after "dba".
- **Maps mined from training matches only:** an Indic → Latin token map from position-aligned (S1 name, record
  name) pairs of equal length (≥ 3 votes, ≥ 50% majority; generic romanisation as a fallback), and state aliases
  from short or native-script address components of matched records (≥ 20 votes, ≥ 90% majority).
- **Flags and frequencies:** domain, DBA, phone, Indic name, Indic address, empty address; how many S1s and
  records share the core name in the country.

### 4.2 Features used (61 pairwise features; country is never a feature)

- **Name features:** RapidFuzz ratio, partial ratio, token-set, token-sort and Jaro-Winkler on the core name;
  ratio with spaces removed (domains); token-set similarity against the "dba" alias; shared and extra tokens on
  each side, Jaccard, first-token match, exact core-name match; legal forms shared / equal / missing on either
  side; name frequency (S1s and records sharing the name) on both sides.
- **Address features:** RapidFuzz ratio, token-set and partial ratio; token overlap and Jaccard; digit-run
  overlap; first number equal, one containing the other (dropped digits), log absolute difference; state equal,
  or unknown when either state is unrecognised (never "mismatch").
- **Retrieval features:** dense cosine and rank; rank in each TF-IDF search, best lexical rank, number of
  searches that found the pair; exact TF-IDF cosines (name words, address words, trigrams, combined).
- **Context features:** candidates per S1; S1s that retrieved the record; the record's dense rank among them
  and gap to its best S1; gap to the S1's best dense score.
- **Record features:** source (S2 or S3); domain, DBA, phone, Indic and empty-address flags.

### 4.3 Model type

- **Cross-encoder:** `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M parameters), one output, binary
  cross-entropy. Input: the S1 text and the record text prefixed with "[s2]" or "[s3]", up to 192 tokens.
  Training pairs per S1: every true record in its pruned set, its 10 hardest non-matches by pruner score, and the
  true records that retrieval or pruning lost. 40,000 steps × 128 pairs per fold, learning rate 2e-5, bf16;
  2 folds, cross-fitted. It scores every candidate pair (10.4M train pairs out-of-fold, 8.41M test pairs).
- **Stacker, level 1:** LightGBM (255 leaves, learning rate 0.05, up to 3,000 rounds with early stopping after
  100 on an S1-disjoint 10% slice, 80% feature and row sampling), 5 folds grouped by S1, on 69 features: the 61,
  pruner score and rank, the cross-encoder logit and its context (rank within the S1, gap to the S1's best,
  number of positive logits in the S1, rank among the S1s claiming the record, gap to the record's best claimant).
- **Stacker, level 2:** the level-1 logit plus 32 collective features computed from the level-1 probabilities of
  all pairs:
  - *competition* among the S1's candidates and among the S1s that want the same record (counts, sums, ranks,
    best rival, margin, share, per-source counts);
  - *house-number drift*: whether a record's number differs while a strong sibling's matches exactly;
  - *coherence*: name/address similarity and house agreement with the S1's best other record;
  - *namesake arbitration*: among S1s sharing a name, which one already wins the address-bearing records;
  - *record graph*: records sharing the name, or the name and house number (copies of one business share both).

  Level 2 is used only if it beats level 1 on the held-out stress world; it did (0.99180 against 0.99074).
- **Stress worlds.** Drop 19% of S1 entities (chosen by a salted hash), keep all records, and recompute every
  S1-dependent input: records' retrieval context, S1 name frequencies, the pruner score (saved pruner models,
  out-of-fold), the cross-encoder context and all level-2 features. Both levels train on the full training set
  plus stress worlds 11 and 12; stress world 5 is held out.
- **Test time.** Each cross-encoder fold model's logits pass through both stack levels separately and the
  probabilities are averaged, so the stackers see single-model logits as in training.

### 4.4 Threshold selection method

- Calibration p' = σ(a · logit p + b).
- One owner: each S2/S3 record keeps only its highest-scoring S1.
- For each S1, candidates sorted by p', choose k ∈ {0, …, n} maximising expected F0.5, estimated with
  256 Monte-Carlo draws of Bernoulli(p') outcomes; k = 0 scores 1.0 in draws where no candidate is true.
- The rule and its parameters are chosen by grid search (thresholds 0.20–0.95, a closed-form approximation and
  the Monte-Carlo rule, each with a ∈ {0.75, 1, 1.25, 1.5}, b ∈ {−1, −0.5, 0, 0.5}) on the out-of-fold scores of
  the held-out stress world.

### 4.5 France post-processing (`frpost`, countries without training labels)

The model learned from US/India that one word changed at the same address is almost always a true copy (the
generator's filler noise, "Services", "Center"). In France the generator uses the same surface operation for
decoys, swapping the business-type word ("Club" → "École"). The model also learned its tolerance for
house-number changes from US/India, where true copies keep the number in 87% (US) and 76% (India) of pairs;
among our accepted French pairs it is kept in 98%, so a small shift is much stronger evidence of a decoy there.

Evidence measured without labels on our own test predictions:
- **Symmetry:** accepted French pairs shifted *up* by a decoy offset: 3,957; shifted *down* by the same amounts:
  1,783. True noise is symmetric (§2.1: US 27,415 vs 27,494 in training), so about 2,170 were decoys
  (about 75% at +1 and +2).
- **Base rates:** pairs where both the legal form and the house number differ are 0.18% of true US training
  pairs; 2,026 accepted French pairs had that shape.

Rules, applied only to test countries absent from the training data (here France), on the final match lists:
1. **Drop** if both legal forms are present and differ and both first house numbers are present and differ
   (2,026 pairs).
2. **Drop** if the record's house number is the S1's plus one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
   (3,957 pairs; 2,209 not already dropped by rule 1).
3. **Drop** same-site business-word swaps: legal forms ignored, exactly one name token replaced by exactly one
   other, same house number (or none); the new token is a sector word (alphabetic, ≥ 4 letters, in ≥ 200 French
   S1 names, not a filler word); Levenshtein similarity < 0.5 and neither token a letter-subsequence of the other,
   so abbreviations ("Cie" / "Compagnie") and typos are excluded (12,535 pairs).
4. **Add** a candidate record that no S1 claims when it sits at the same site (same house number, street-token
   Jaccard ≥ 0.6, a shared city token) and its name adds exactly one filler word (groupe, développement,
   associés, services), drops at most one word and adds no legal form, and it qualifies for exactly one S1
   (12,753 pairs). Example: "Chercheurs Services SARL, 38 Rue du Petit Bois, Pessac" and "Chercheurs SARL
   Développement, 38 R Du Petit Bois, Pessac".

Net: 16,770 French pairs dropped and 12,753 added; US and India unchanged; every change and its rule is logged
in `work/frpost/changes.tsv`. Additions come only from `candidate_pairs.tsv`, so matches stay a subset of the
candidates. On the public leaderboard this step moved the score from 0.987282 to 0.990107.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold, all training S1 entities, singletons included):**

  | World | Macro F0.5 | US | India | Singletons | Matched S1 |
  |---|---|---|---|---|---|
  | Full training set (2,206,821 S1) | **0.99234** | 0.99154 | 0.99355 | 0.99791 | 0.99201 |
  | Held-out stress world (1,787,116 S1) | **0.99180** | 0.99099 | 0.99302 | 0.99540 | 0.99159 |

- **Ablations (out-of-fold, full training set):**

  | Variant | Macro F0.5 |
  |---|---|
  | Final stacker (level 2) | 0.99234 |
  | Level 1 only | 0.99114 |
  | Without cross-encoder features | 0.98922 |
  | Lexical and string features only (no cross-encoder, dense or pruner features) | 0.98763 |

- **Public leaderboard:**

  | Submission | Public F0.5 |
  |---|---|
  | First full run (level-1 stacker, decisions tuned at training density) | 0.985007 |
  | + French add-on stripping, stress worlds, level-2 stacker, per-fold test scoring | 0.987282 |
  | + full French normalisation rules (tested, not kept) | 0.987067 |
  | + France post-processing (`frpost`) — **final** | **0.990107** |

- **Common false positives (wrong merges):**
  - decoy neighbours on the same street, with the house number shifted up by a small offset and the legal form
    or one word changed;
  - same-address decoys with the business-type word swapped (French names follow "place + business type +
    legal form", so these are very close in text);
  - namesakes: common names shared by many S1 entities.
- **Common false negatives (missed matches):**
  - French copies where a word was replaced by a filler word the model never saw in training
    ("Services" → "Développement");
  - replaced names on records with an empty address (no evidence left);
  - heavily garbled names and truncated long addresses.
- **Tried and not used:**
  - *Full French normalisation* (street short forms, "N°"/"#", leading zeros, bis/ter, ordinals, CEDEX,
    département → région, dotted legal forms): a label-free token-separability check improved
    (AUC 0.858 → 0.909), but the leaderboard fell to 0.987067. The neural models already absorbed the formatting
    noise; the remaining French errors were decoys. Kept as an option, off (`data.fr_rules`).
  - *A uniform decision shift for unseen countries* (raising France's matches per S1 to the US/India rate): it
    adds the easiest pairs first, and in France many of those are decoys. Not submitted.
  - *Ensembles of our own weaker variants* (stacks without the cross-encoder or without neural features, fed into
    level 2, and a residual LightGBM starting from the final stacker's logit): no gain on the held-out stress
    world, because the variants share the base model's errors. Not used.
  - *A transformer-free pipeline* (per-S1 posting budgets, reverse searches, a hashed n-gram MLP encoder): it
    fixed lexical recall (0.779 → 0.988 on an Indian subset) but cost about 3 hours of CPU search, and the neural
    pipeline already reached 0.9992. Not used.

---

## 6. Conclusion

Recall is solved early: fine-tuned dense retrieval plus lexical searches reach 0.9992, and pruning keeps 99.62%
at under five candidates per entity. The rest is precision, and the largest gains came from making training look
like the test set (stress worlds, collective level-2 features, decision tuning on a held-out stress world,
consistent test-time inputs) and from handling the unseen country with rules that were measured before they were
used. Lessons: validation that ignores the test's distractor density overestimates the score; symmetric-noise
tests measure precision without labels; for a country without labels, targeted, verified rules beat blanket
normalisation or threshold shifts.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/ber/`: one module per step, run as `python -m ber <step>`: `prepare`, `sparse`, `train-retriever`,
  `dense`, `features`, `prune`, `train-ce`, `score-ce`, `stack`, `frpost` (plus `residual`, an optional
  ensemble step that is not part of the final submission).
- `configs/default.yaml`: every setting, overridable as `key=value`.
- `hpc/`: SLURM scripts for environment setup, model download, preflight and single-step submission.
- `tests/`: 36 unit tests, including the France rules.
- `README.md`, `requirements.txt` (pinned versions) and `requirements.lock.txt` (the complete environment of
  the run), and `run_amazon_ml.sh`, which runs the whole pipeline as one job.
- `tools/make_package.py`: builds the submission zip.

**Reproduction** (with the provided data copied to `data/student_resource/dataset/` at the package root):
```
cd code/business_entity_resolution
bash run_amazon_ml.sh setup     # environment and model download, once
sbatch run_amazon_ml.sh full    # prepare -> sparse -> train-retriever (2 folds) -> dense -> features -> prune
                                # -> train-ce (2 folds) -> score-ce -> stack -> frpost
```
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Seeds are fixed, folds are stable
hashes of entity ids, and each step saves its resolved configuration to `work/runs/`.

**Hardware, software and runtime.** One shared node with an NVIDIA H100 NVL (93 GB), 128 CPU cores and 125 GB RAM;
torch 2.14.0 (CUDA 13.0), transformers 5.17.0, LightGBM 4.7.0, polars 1.44.1, scikit-learn 1.9.1.
Approximate step times: prepare 4 min; sparse 8 min; bi-encoder training 2.6 h per fold; dense retrieval 2.2 h;
features 24 min; pruner 1.7 h; cross-encoder training 1.7 h per fold; cross-encoder scoring 2 h; stacking about
2.5 h; France rules 1 min.

### B. Final submission provenance

1. **Base pipeline (public 0.987282):** the full pipeline above with the French add-on stripping and
   `data.fr_rules=false`; stacker trained with worlds [0, 11, 12], held-out world 5, level 2 enabled
   (`stack.second_stage=true`), per-fold test scoring (`stack.per_fold_test=true`), no ensemble members
   (`stack.members=[]`), no base-rate shift (`stack.unseen_rate_match=false`). Output `matching_results.tsv`,
   md5 `c16d82f372ad532bd8be46a27ff74d02`.
2. **France rules:** `python -m ber frpost` with the default `frpost` settings (all four rules on,
   `swap_min_df=200`, offsets {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}) applied to that output: 16,770 French pairs
   dropped, 12,753 added. Output `matching_results.tsv`, md5 `b760fdbc7e6753571509a0ed40805aed`: the file that
   scored **0.990107**. The same command reproduced this file byte for byte on the cluster and locally.
3. `candidate_pairs.tsv` (md5 `5aff25825ae44012e935c10571ea4cba`) is the pruner's output from step 1 and is
   unchanged by step 2. Final file: 5,846,967 matched pairs over 1,732,544 S1 rows.
4. Not part of the final submission: the optional `residual` ensemble step, `stack.members`, the full French
   normalisation rules and the base-rate shift.

### C. Compliance

- **Models:** `BAAI/bge-m3` (MIT, 568M parameters) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M
  parameters), 1.1B parameters in total, both fine-tuned only on the provided training data. LightGBM (MIT).
- **Data:** the final submission and every component described in this document use only the provided TSV
  files. No external databases, registries, geocoding, APIs or web lookups.
- **Word lists:** the legal forms, stopwords, street abbreviations and US/Indian state names in
  `src/ber/lexicon.py` are hand-written general knowledge. The Indic → Latin token map and native-script state
  aliases are mined from the training matches.
- **Countries:** country is never a model feature and no country list is hard-coded. The France rules are gated on
  "country present in test Source 1 but absent from training", not on the name France.

### D. Limitations and future work

- Put decoy knowledge into the model (signed house-number difference, decoy-offset membership, legal-form swap
  direction, business-word swaps as features) and train on US/India, whose training data also contains decoys, so
  hand rules become learned weights.
- French self-training: pseudo-label confident French test pairs and refit the stacker.
- Stress worlds re-score only pairs kept in the full world; re-pruning each stress world would expose
  look-alikes that rank high only once their owner is gone.
- Level-2 features can use a rival's score from a fold model that saw the true owner's pair, so the out-of-fold
  score may be slightly optimistic; a nested split removes this.
- The Monte-Carlo decision flips about 0.3% of near-tie decisions between runs; an exact Poisson-binomial
  computation removes that noise.
