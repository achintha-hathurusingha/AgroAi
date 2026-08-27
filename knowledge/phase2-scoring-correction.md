# Phase 2 Scoring Correction: Ground-Truth Vocabulary & Multi-Value Field Bugs

Triggered by a request to audit everything after Stage F, before trusting further
results. Two real bugs were found and fixed; both are pure rescoring of already-collected
data (no new model runs). Full detail on discovery: see the "review everything" turn in
session history. Fix: [server-scripts/scoring.py](server-scripts/scoring.py). Rescore
scripts: [server-scripts/rescore_diagnosis_accuracy.py](server-scripts/rescore_diagnosis_accuracy.py)
(local, no GPU), [server-scripts/rescore_retrieval.py](server-scripts/rescore_retrieval.py)
(re-embeds saved queries, re-ranks against existing corpus embeddings, run on qbits CPU
since the GPU was occupied by another user's job).

## Bug 1: ground truth was never normalized to corpus vocabulary

`ground_truth_disease` = raw PlantVillage class label, lowercased. The corpus (AgMMU
facts) uses different wording for the same disease in several classes: PV "bacterial
spot" vs corpus "bacterial leaf spot" (3 classes), PV "tomato mosaic virus" vs corpus
"mosaic virus", PV "cercospora leaf spot gray leaf spot" vs corpus "cercospora leaf
spot". **6 of 16 confident_match classes (37.5%, ~90/240 cases)** could never score
"correct" under exact-string match even with flawless retrieval and diagnosis.

One additional near-match, `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` vs corpus's
"ascochyta leaf blight", was investigated and **deliberately excluded** from the alias
fix -- Isariopsis/Pseudocercospora and Ascochyta are plausibly different pathogens, not
just different wording for the same one, so treating them as equivalent risked
introducing a *new*, scientifically wrong error. Left unresolved, scored as before.

**Fix**: `scoring.py`'s `PV_ALIAS_MAP` maps each PV label to its accepted alias set (PV
label + corpus canonical term, only where confirmed same-disease).

## Bug 2: Recall@K used full-string equality, undercounting multi-value fields

7.6% of corpus facts bundle multiple diseases in one comma-separated `disease` field
(e.g. `"apple anthracnose, black rot, burrknot, powdery mildew"`). The original
Recall@K walk (`corpus[idx]["disease"] == target`) never credited these even when the
VLM correctly extracted the right disease from the list (confirmed: 12/37 correct Stage
D diagnoses came from exactly this situation).

**Fix**: `fact_matches_ground_truth()` splits the field into a term set and checks
membership, combined with the alias fix above.

## Result: diagnosis accuracy barely moved; retrieval recall moved more; the narrative held and sharpened

### Diagnosis accuracy (exact model-output match, all cases where model must literally produce an accepted string)

| Experiment | Group | Strict | Alias-corrected | Δ |
|---|---|--:|--:|--:|
| Main ablation, VLM-only | confident_match | 0.4% | 0.4% | +0.0 |
| Main ablation, RAG-BGE | confident_match | 3.7% | 4.2% | +0.4 |
| Main ablation, RAG-rerank | confident_match | 2.9% | 4.6% | +1.7 |
| Oracle Retrieval | confident_match | 73.3% | 78.7% | +5.4 |
| Stage D, Visual RAG | confident_match | 15.4% | 16.7% | +1.2 |

Diagnosis accuracy moved by low single digits everywhere except Oracle. This makes
sense in hindsight: the VLM tends to echo whatever vocabulary its evidence uses, so
fixing the ground-truth string only rescues a case if the model happened to output
exactly the alias string -- most wrong-vocabulary-class cases were simply wrong for
other reasons too. **None of the main ablation's or Stage D's headline conclusions
flip.**

### Retrieval Recall@K (confident_match, n=240) -- this moved more

| Retrieval mode | Strict R@1 | Alias R@1 | Strict R@5 | Alias R@5 |
|---|--:|--:|--:|--:|
| Oracle (text) | 81.2% | **87.5%** | 81.2% | **93.8%** |
| Current pipeline (Qwen text query) | 1.3% | 1.3% | 5.4% | 6.2% |
| SigLIP visual (Stage C) | 3.3% | 3.3% | 17.1% | 20.0% |
| Hybrid α=0.25 (Stage F) | 5.8% | **8.3%** | 20.4% | **24.6%** |

Oracle's true retrieval ceiling is **93.8% R@5**, not 81.2% -- confirming the corpus
essentially always contains the right fact when queried well; the strict metric was
hiding ~20% of genuinely-successful retrievals behind a wording mismatch. Hybrid got
the largest relative correction (R@1 +43% relative) -- its real advantage over pure
retrieval modes was understated more than the others.

**The current pipeline's Qwen-generated query barely moved at all (R@1 1.3%→1.3%,
unchanged).** This is the important negative result: it rules out "the query-failure
finding was itself a scoring artifact." The query genuinely fails to retrieve relevant
evidence almost regardless of how leniently you score the retrieval -- this was never
an artifact.

### Error taxonomy recount (confident_match, n=240)

| Category | Original (strict) | Corrected (alias) |
|---|--:|--:|
| QUERY_FAILURE | 74.6%* | **86.2%** (207/240) |
| RETRIEVAL_CORPUS_FAILURE | ~18-19%* | **6.2%** (15/240) |
| REASONING_FAILURE | — | 3.3% (8/240) |
| CORRECT | 3.7%* | 4.2% (10/240) |

*(original percentages were computed against the 240-case total including CORRECT; see
[phase2-extension-error-taxonomy.md](phase2-extension-error-taxonomy.md) for the exact
original breakdown.)*

Fixing the scoring bugs **sharpened rather than overturned** the original finding: with
the corrected oracle_hit gate (using the true 93.8% R@5 ceiling), fewer cases are
genuinely un-findable in the corpus (RETRIEVAL_CORPUS_FAILURE drops from ~19% to 6.2%),
and nearly all of those reclassify into QUERY_FAILURE, since the current pipeline's
query still fails to find them. **The bottleneck is even more overwhelmingly query
construction than originally reported (86.2% vs 74.6%), not missing corpus content.**

## What this changes going forward

- All *relative* comparisons from Stage C/D/F (visual > text retrieval, hybrid > pure
  visual, RAG helps if retrieval succeeds) hold, and in most cases the true gaps are
  larger than originally reported, not smaller.
- The Oracle ceiling is higher than reported (93.8% R@5 vs 81.2%), meaning there is more
  achievable headroom than previously thought if retrieval quality keeps improving.
- The currently-running hybrid-retrieval diagnosis job on devon will be rescored with
  the corrected `scoring.py` once it lands, for an apples-to-apples comparison against
  the corrected Stage D numbers above.
- A separate, real, and still-open issue -- **SigLIP silently truncates ~18% of corpus
  facts to 64 tokens before embedding** (confirmed directly against the actual
  tokenizer and corpus: mean 43.8 tokens/fact, 17.8% exceed the 64-token limit) -- was
  not addressed by this rescore. This is a modeling limitation, not a scoring bug, and
  is the leading candidate for the next round of retrieval-quality work (see literature
  review: caption-then-retrieve is the best-precedented mitigation).
