# Final Research Findings

**Status: consolidation pass, complete.** This document is the single authoritative
summary of the AgroAi / AgriVision-RAG research project. It supersedes any accuracy
number quoted in an earlier per-experiment document that conflicts with the numbers
here — every number below was recomputed directly from the raw saved experiment outputs
using the corrected scoring logic (see [phase2-scoring-correction.md](phase2-scoring-correction.md)),
and the recomputation is itself reproducible: [server-scripts/build_final_results_table.py](server-scripts/build_final_results_table.py)
regenerates [results/phase2/final_results_table.json](results/phase2/final_results_table.json)
from the raw JSONL outputs with no new model calls. Per-experiment documents remain in
the repo as the historical record of how each finding was reached (including now-corrected
intermediate numbers) — this document is where to look for what's actually true after
every correction.

## Research Question

> Can retrieval-augmented multimodal reasoning improve the fine-grained diagnosis and
> reliability of vision-language models for agricultural disease identification, compared
> with zero-shot VLM diagnosis?

Decided in [phase-1-plan.md](phase-1-plan.md), replacing a broader, less falsifiable
original framing. The explicit commitment made at the time: "if it turns out RAG
*doesn't* meaningfully help, that's still a valid, reportable finding" — a commitment
this document honors; several of the strongest results below are negative results.

## Experimental Timeline

| Stage | Question answered | Decision it caused |
|---|---|---|
| Phase 1A (corpus inspection/preprocessing) | What does the AgMMU fact set actually contain, and how should it be structured? | Split into a 17,583-fact diagnostic corpus + 17,924-fact context corpus; dedup with traceability; keep `species_raw` unnormalized |
| Phase 1B Q1/Q2 (400-fact pool) | Which text embedder retrieves best? Which query-construction prompt works? | BGE-micro-v2 provisionally selected; Prompt C provisionally selected — **both later found not to hold at full scale** |
| Milestone 3 control failure | Does the small-pool result generalize to the real corpus? | No — same query, same code: rank 1 in 400 facts, rank 40 in 17,583. Triggered full-scale re-validation of Q1 and Q2 |
| Phase 1B.5 (full-scale Q1/Q2) | Do BGE and Prompt C actually hold at 17,583-fact scale? | BGE still wins overall but ties at R@1 (not "clearly wins"); **no prompt variant (A–E) works at full scale, none robustly beats another** — the small-pool Prompt C finding does not hold |
| Bottleneck diagnosis / CPJ refinement | Is the Q2 failure fixable by better query engineering? | No single intervention (species hints, iterative judge-refine) rescues the hardest cases; bottleneck is what Qwen3-VL's description *contains*, not its phrasing |
| Q6 (vector store benchmark) | NumPy vs FAISS vs Qdrant at real scale? | NumPy exact cosine fastest on every measure; adopted |
| Corpus freeze + leakage audit + eval protocol | Is everything ready for a comparable Phase 2 ablation? | Corpus frozen as `agmmu_phase2_v1` with checksums; zero image-leakage collisions found; ground-truth hierarchy (PlantVillage primary, AgMMU secondary) frozen |
| Phase 2 main ablation (5 arms, 465 primary cases) | Does the built RAG pipeline improve diagnosis over VLM-only? | No detectable improvement at this sample size on the full eval set; but a real, unresolved-at-the-time signal on the disease-in-corpus subgroup |
| Oracle Retrieval | If retrieval always finds the right evidence, does RAG actually help? | **Yes, decisively** — the main ablation's null result is a query-construction artifact, not evidence against RAG |
| Retrieval Decomposition | Where exactly does the current pipeline's query fail? | Confirms Oracle: ~62x R@1 gap between the current text query and the oracle query, same embedder, same corpus |
| Error Taxonomy | What fraction of failures are query-construction vs corpus vs reasoning? | Query-construction failure dominates (86.2% corrected) — motivates trying a different image→query representation |
| Visual-Data Audit | Can true image-to-image retrieval be built against the corpus? | No (no paired corpus images, ~500GB never downloaded) — cross-modal SigLIP retrieval (image tower vs corpus text tower) proposed instead |
| Stage C (visual retrieval-only) | Does cross-modal image-embedding retrieval beat the text-query pipeline? | Yes, ~2.5–3x on Recall@k — proceed to a full diagnosis run |
| Stage D (visual RAG diagnosis) | Does that retrieval gain survive into diagnosis accuracy? | **Yes, large and statistically significant** — the strongest practical (non-oracle) result in the project |
| Stage F (hybrid retrieval-only) | Does blending text + visual similarity retrieve better than visual alone? | Yes, on Recall@k (after correction) |
| Stage E follow-up (hybrid RAG diagnosis) | Does that retrieval gain survive into diagnosis accuracy? | **No** — not statistically distinguishable from pure-visual RAG |
| Scoring audit (post-Stage F) | Were the eval metrics themselves measuring the right thing? | Two real bugs found and fixed (ground-truth vocabulary mismatch, multi-value field undercounting); all results rescored |
| Stage G (truncation fix, caption-then-retrieve) | Can two literature-motivated retrieval fixes beat the hybrid/visual approach? | **No** — both underperform on Recall@5, the metric that predicts diagnosis accuracy; retrieval-representation experimentation closed |

## Frozen Dataset and Evaluation Protocol

- **Corpus**: `agmmu_phase2_v1`, frozen with SHA-256 checksums ([phase2-corpus-freeze.md](phase2-corpus-freeze.md)). Diagnostic corpus: 17,583 facts (disease/pest-bearing). Context corpus: 17,924 facts (species-only, used only in the "full corpus" ablation arm). Source: AgMMU's 45,096-entry fact set, deduplicated.
- **Query images**: 465 primary cases sampled from PlantVillage (38 classes), stratified into `primary_confident_match` (n=240, disease exists in corpus with ≥5 matching facts), `primary_healthy` (n=180), `primary_negative_control` (n=45, disease provably absent from corpus — tests hallucination resistance). A 4th group, `supplementary_agmmu` (n=40, drawn from AgMMU's own eval images), was **excluded from all analysis** — its ground truth was contaminated by non-diagnostic AgMMU question types (management tips scored as if they were disease names). This is a data-construction bug in the eval-set builder, documented and excluded, not a model result.
- **Ground truth**: hierarchical, PlantVillage primary / AgMMU secondary, frozen before any inference ([phase2-evaluation-protocol.md](phase2-evaluation-protocol.md)).
- **Leakage audit**: 0 image collisions (SHA-256, byte-for-byte) between PlantVillage query images and AgMMU knowledge-source images; code review confirmed no ground-truth path into query construction ([leakage-audit-results.md](leakage-audit-results.md)).
- **Scoring bugs found and fixed after Stage F** ([phase2-scoring-correction.md](phase2-scoring-correction.md)):
  1. `ground_truth_disease` was the raw PlantVillage label, never normalized to the corpus's own disease vocabulary. 6 of 16 confident_match classes had a wording mismatch (e.g. PV "bacterial spot" vs corpus "bacterial leaf spot") that made them unscoreable as correct regardless of model quality.
  2. Recall@K used full-string equality, undercounting the 7.6% of corpus facts that bundle multiple diseases in one comma-separated field.
  - Both fixed via an alias-set matcher ([server-scripts/scoring.py](server-scripts/scoring.py)). One additional near-match (`Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` vs corpus's "ascochyta leaf blight") was investigated and **deliberately left unresolved** — plausibly a different pathogen, not just different wording, so aliasing it risked introducing a new, scientifically wrong error rather than fixing one.

## Phase 1 Findings

**Q1 (embedding model)** — [phase1b-q1q2-results.md](phase1b-q1q2-results.md), [phase1b5-fullscale-validation-results.md](phase1b5-fullscale-validation-results.md). ⚠️ **Superseded**: the original 400-fact-pool result ("BGE clearly wins," R@1=0.681 vs sentence-transformers' 0.574) does not hold at full corpus scale. **Current, final**: at 17,583-fact scale, BGE-micro-v2 and sentence-transformers **tie exactly at R@1 (0.426 each)**; BGE wins on every other metric (R@3=0.596, R@5=0.617, R@10=0.702, R@20=0.787, MRR=0.526, all best of three). Decision (BGE) unchanged, margin corrected.

**Q2 (query construction)** — [phase1b5-q2-fullscale-results.md](phase1b5-q2-fullscale-results.md). ⚠️ **Superseded**: the original small-pool finding ("Prompt C wins, 28.6% R@1 vs Prompt A's 0%") does not hold at full scale. **Current, final**: at full scale, **every prompt variant (A–E) scores 0% R@1–R@5**, and no variant is a consistent winner across diseases — Prompt D wins for apple scab/black rot, Prompt A wins for powdery mildew (the exact case C "solved" at small scale), Prompt C wins for septoria/bacterial spot, nothing gets cedar apple rust or early blight usable. Kept as the practical baseline, explicitly documented as **unvalidated-as-optimal**, not proven best. This is the earliest evidence for the project's central finding: query-phrasing tweaks don't fix retrieval when the underlying visual *content* Qwen3-VL extracts doesn't contain the discriminating feature.

**Q3 (reranking)**: deferred to Phase 2 as its own ablation arm, not assumed to help — motivated by the bottleneck diagnosis showing correct facts sometimes rank thousands of positions down, which a reranker over a bounded candidate set cannot rescue by construction.

**Q4 (ground truth)**: hierarchical policy frozen (PlantVillage primary, AgMMU secondary, disagreements flagged not resolved, undecidable cases excluded).

**Q5 (preprocessing)**: 45,096 raw entries → 17,583 diagnostic + 17,924 context facts, deduplicated with `source_faq_ids` traceability preserved.

**Q6 (vector store)** — [phase1-q6-vector-store-results.md](phase1-q6-vector-store-results.md). NumPy exact brute-force cosine fastest on every measure at both 17,583- and 35,507-fact scale (build ~0ms vs FAISS 13–28ms vs Qdrant 4.5–8.8s; query mean 0.343–0.390ms vs FAISS 0.7–2.1ms vs Qdrant 2.9–3.6ms). Adopted.

**Methodological lesson established here and repeatedly confirmed later**: small-scale benchmark results in this project did not reliably predict full-scale behavior (discovered independently at least three times — Milestone 3's control-case failure, the Q1/Q2 full-scale re-validation, and later the Phase 2 scoring-bug discovery). Any number reported at reduced scale in this project should be treated as provisional.

## Phase 2 Main Ablation

[phase2-main-ablation-results.md](phase2-main-ablation-results.md), rescored with corrected ground-truth/alias matching. 465 primary cases, 5 arms, 0 errors, v2 diagnosis prompt (see [phase2-diagnosis-prompt-iteration.md](phase2-diagnosis-prompt-iteration.md) — two earlier prompt versions were superseded after discovering the model was exploiting "healthy"/"unknown" as escape hatches; this is documented, not hidden, since the low resulting accuracy is itself informative, not an artifact to explain away).

| Arm | Overall (n=465) | Disease-in-corpus (`confident_match`, n=240) |
|---|--:|--:|
| 1. VLM only | 37.8% | 0.4% (1/240) |
| 2. VLM + RAG (BGE text) | 39.6%* | 4.2% (10/240) |
| 3. VLM + RAG (SigLIP text-mediated)** | 39.1%* | 4.2% (10/240) |
| 4. VLM + RAG (full corpus, diagnostic+context) | 39.4%* | 3.3% (8/240) |
| 5. VLM + RAG + rerank | 39.1%* | 4.6% (11/240) |

*Alias-corrected; original strict-scoring numbers in [phase2-main-ablation-results.md](phase2-main-ablation-results.md) were 39.4% / 39.1% / 39.4% / 38.3% — differences are ≤1pp, headline conclusion unchanged.
**Arm 3 uses SigLIP's **text tower** on the Qwen-generated query — still text-mediated, not the cross-modal image-tower retrieval used later in Stage D. This distinction matters for interpreting the central finding below.

**On the full 465-case set, no arm reaches significance against VLM-only** (McNemar p=0.096–0.803 in the original run). **Restricting to the confident_match subgroup specifically, Arm 2 vs Arm 1 does reach nominal significance** (McNemar p=0.016, recomputed on corrected scoring) — a real but small effect (10 cases right vs 1), consistent with the original document's framing that RAG's apparent benefit "lives entirely in the one group where it's structurally possible to help," even though it's diluted below significance when the healthy/negative-control groups (where nothing differs between arms) are included in the same test.

Every arm is near-ceiling on healthy cases (95–97%) and near-floor on negative controls (0.0%, confirming RAG doesn't hallucinate a plausible wrong answer when the disease genuinely isn't in the corpus). Calibration is poor everywhere (ECE ≈ 0.43–0.45 across all 5 arms) — ***not*** rescored (ECE and faithfulness are computed from confidence scores and judge output, independent of the ground-truth-string bug, so these numbers are unaffected by the correction).

## Oracle Retrieval

[phase2-oracle-retrieval-results.md](phase2-oracle-retrieval-results.md), rescored. Ground-truth disease name used **only** to construct the retrieval query, never passed to the diagnosis call — measures the ceiling if retrieval always succeeds.

| Group | VLM only | Text RAG (BGE) | **Oracle** |
|---|--:|--:|--:|
| Healthy (180) | 97.2% | 96.7% | 98.3%* |
| **Disease-in-corpus (240)** | **0.4%** | **4.2%** | **78.7%** |
| Overall (465) | 37.8% | 39.6% | **78.7%** (366/465) |

*Healthy-group oracle figure not separately rescored (alias correction only affects disease-name matching, not "healthy" classification) — carried from [phase2-oracle-retrieval-results.md](phase2-oracle-retrieval-results.md)'s 98.3%. The overall figure (78.7%, 366/465) is exact, recomputed directly from raw data by [server-scripts/build_final_results_table.py](server-scripts/build_final_results_table.py).

**Statistical significance** (McNemar, confident_match, n=240, recomputed): Oracle vs VLM-only, 188 cases right where VLM-only was wrong vs **0** the reverse, p=2.4×10⁻⁴². Oracle vs text-RAG, 179 vs **0**, p=2.2×10⁻⁴⁰. In every comparison across this entire project, the discordant count in "current pipeline right, oracle wrong" is exactly zero — oracle is never worse on any individual case.

**This is the single most important result in the project**: the main ablation's null result is not evidence against RAG — it is evidence that the specific query-construction mechanism in the tested pipeline fails to retrieve useful evidence. This is a **ceiling measurement, not a deployable technique** (it requires knowing the answer to construct the query) — see Limitations.

## Error Taxonomy

[phase2-extension-error-taxonomy.md](phase2-extension-error-taxonomy.md), recomputed with corrected oracle/current retrieval hits ([phase2-scoring-correction.md](phase2-scoring-correction.md)). Classifies all 240 confident_match cases.

| Category | Original (strict) | **Corrected (final)** |
|---|--:|--:|
| Query failure | 74.6% (179/240) | **86.2% (207/240)** |
| Retrieval/corpus failure | 18.8% (45/240) | **6.2% (15/240)** |
| Reasoning failure | 2.9% (7/240) | **3.3% (8/240)** |
| Correct | 3.8% (9/240) | **4.2% (10/240)** |

The correction **sharpened, not overturned**, the original finding: fixing the scoring bug revealed that fewer cases are genuinely unfindable in the corpus (retrieval/corpus failure dropped) and correspondingly more are attributable specifically to query construction. **Query construction is the dominant, addressable failure mode by a wide margin (86.2%, >13x reasoning failure).**

Overlay findings (not rescored — independent of the ground-truth-string bug; a 1-case shift in the "correct" set does not materially affect these): hallucination (diagnosis not grounded in evidence) present in 13.0% of wrong cases; **calibration failure (≥70% stated confidence despite being wrong) present in 96.1% of wrong cases** — when this pipeline is wrong, it is confidently wrong almost every time.

## Visual-First Retrieval

**Stage C (retrieval-only)** — [phase2-stageC-visual-retrieval-results.md](phase2-stageC-visual-retrieval-results.md), corrected. Query images embedded directly with SigLIP's **image tower**, retrieved against the corpus's existing SigLIP **text-tower** embeddings — true cross-modal retrieval bypassing Qwen3-VL text generation for the retrieval step entirely (no paired corpus images exist to enable literal image-to-image retrieval, confirmed in [phase2-visual-data-audit.md](phase2-visual-data-audit.md)).

| Query representation | confident_match R@1 | R@5 |
|---|--:|--:|
| Current pipeline (Qwen text query → BGE) | 1.3% | 6.2%* |
| **SigLIP visual (image tower → corpus text tower)** | **3.3%** | **20.0%*** |
| Oracle (ground-truth text) | 87.5%* | 93.8%* |

*Alias-corrected; original strict numbers were R@5 5.4%/17.1%/81.2% respectively.

**Stage D (full diagnosis run)** — [phase2-stageD-visual-rag-diagnosis-results.md](phase2-stageD-visual-rag-diagnosis-results.md), corrected. Same retrieval, fed into the diagnosis step.

| Arm | confident_match accuracy |
|---|--:|
| VLM-only | 0.4% |
| Text RAG (BGE) | 4.2% |
| **Visual RAG (Stage D)** | **16.7% (40/240)** |

McNemar (recomputed): visual RAG vs VLM-only, p=2.9×10⁻⁹ (40 corrected, 1 broken). Visual RAG vs text RAG, p=3.0×10⁻⁷ (31 corrected, 1 broken). **This is the strongest practical (non-oracle) result in the project** — swapping only the retrieval query representation (text description → direct image embedding), with the identical corpus and diagnosis model, produces a ~4x accuracy improvement with high statistical confidence.

## Hybrid Retrieval

**Stage F (retrieval-only)** — [phase2-stageF-hybrid-retrieval-results.md](phase2-stageF-hybrid-retrieval-results.md), corrected. Per-query min-max-normalized weighted blend of BGE text similarity and SigLIP visual similarity, α swept 0–1.

| Method | confident_match R@1 | R@5 |
|---|--:|--:|
| SigLIP visual alone | 3.3% | 20.0% |
| **Hybrid (α=0.25)** | **8.3%** | **24.6%** |

**Stage E follow-up (full diagnosis run)** — [phase2-hybrid-diagnosis-results.md](phase2-hybrid-diagnosis-results.md).

| Arm | confident_match accuracy | vs Stage D (McNemar) |
|---|--:|---|
| Visual RAG (Stage D) | 16.7% | — |
| **Hybrid RAG (α=0.25)** | **17.5% (42/240)** | **p=0.81 — not significant (10 vs 8 discordant)** |

**This is a clean, informative negative result, preserved as such**: hybrid's retrieval-ranking improvement (R@1 nearly 2.5x pure-visual) did **not** translate into a diagnosis-accuracy improvement. Interpretation: the diagnosis step reads all top-5 evidence regardless of rank order, so a metric that predicts diagnosis accuracy is "is the right fact anywhere in the top-5" (R@5: 24.6% vs 20.0%, a real but much smaller gap than the R@1 numbers suggested), not "is it ranked first." This closes the retrieval-blending direction as diminishing returns.

**Stage G (two further retrieval alternatives, both negative)** — [phase2-stageG-retrieval-alternatives-results.md](phase2-stageG-retrieval-alternatives-results.md). Retrieval-only, not extended to diagnosis given the weak R@5 results below:

| Method | confident_match R@1 | R@5 |
|---|--:|--:|
| SigLIP visual, truncation-safe shortened text | 3.7% | **10.0%** (worse than full-length text's 20.0%) |
| Caption-then-retrieve (Qwen caption → BGE) | 6.2% | **10.8%** (still below hybrid's 24.6%) |

Confirmed directly against the corpus and SigLIP's own tokenizer: corpus facts average 43.8 tokens, 17.8% exceed SigLIP's 64-token text-tower limit and are silently truncated. Fixing this by aggressively shortening every fact **made R@5 worse, not better** (over-shortening — mean 8.9 words/fact — lost more signal than truncation cost, since ~51% of corpus facts are pest-only entries with no symptom text to draw on). Caption-then-retrieve (literature-precedented) improved R@1 but not R@5, with high variance by disease (near-perfect for visually distinctive diseases like powdery mildew, near-total failure for cedar apple rust and mosaic virus). **Neither beats hybrid or pure-visual retrieval; retrieval-representation experimentation was closed on this evidence.**

## Final Results

Authoritative table, all numbers alias-corrected, reproducible via [server-scripts/build_final_results_table.py](server-scripts/build_final_results_table.py) → [results/phase2/final_results_table.json](results/phase2/final_results_table.json).

| Architecture | Overall accuracy (n=465) | confident_match accuracy (n=240) | Retrieval R@1 / R@5 (confident_match) |
|---|--:|--:|--:|
| VLM-only | 37.9% (176/465) | 0.4% [0.1, 2.3%] | n/a (no retrieval) |
| Text RAG (BGE) | 39.6% (184/465) | 4.2% [2.3, 7.5%] | 1.3% / 6.2% |
| SigLIP visual RAG (Stage D) | 45.2% (210/465) | **16.7%** [12.5, 21.9%] | 3.3% / 20.0% |
| Hybrid RAG (α=0.25) | 45.4% (211/465) | 17.5% [13.2, 22.8%] | 8.3% / 24.6% |
| Oracle RAG (ceiling, non-deployable) | 78.7% (366/465) | **78.7%** [72.9, 83.6%] | 87.5% / 93.8% |

All figures recomputed directly from raw saved outputs by [server-scripts/build_final_results_table.py](server-scripts/build_final_results_table.py) → [results/phase2/final_results_table.json](results/phase2/final_results_table.json) — no hand-carried or approximated numbers in this table.

## Statistical Analysis

All p-values below: McNemar's test, paired (same 240 confident_match cases through both conditions), continuity-corrected, alias-corrected ground truth. Recomputed in this consolidation pass (not just carried from per-stage docs) — see [server-scripts/build_final_results_table.py](server-scripts/build_final_results_table.py) output.

| Comparison | Discordant (A right / B right) | p-value | Significant? |
|---|--:|--:|---|
| Text RAG (BGE) vs VLM-only | 10 / 1 | 0.016 | Yes (nominally, on this subgroup only — not on the full 465-case set) |
| Visual RAG (Stage D) vs VLM-only | 40 / 1 | 2.9×10⁻⁹ | **Yes, strongly** |
| Visual RAG (Stage D) vs Text RAG | 31 / 1 | 3.0×10⁻⁷ | **Yes, strongly** |
| Hybrid RAG vs Visual RAG (Stage D) | 10 / 8 | 0.81 | **No** |
| Hybrid RAG vs Text RAG | 33 / 1 | 1.1×10⁻⁷ | **Yes, strongly** |
| Oracle RAG vs VLM-only | 188 / 0 | 2.4×10⁻⁴² | **Yes, overwhelmingly** |
| Oracle RAG vs Text RAG | 179 / 0 | 2.2×10⁻⁴⁰ | **Yes, overwhelmingly** |

## Failure Mechanisms

From the corrected error taxonomy (n=240, confident_match), classified using already-collected evidence, not a new judge pass:

| Mechanism | Rate | What it means |
|---|--:|---|
| **Query-construction failure** | **86.2%** | Oracle retrieval finds the right evidence; the actual generated query doesn't. The dominant, addressable bottleneck. |
| Retrieval/corpus failure | 6.2% | Not findable even with a perfect query — a genuine corpus-coverage ceiling, unaffected by any query-representation fix. |
| Reasoning failure | 3.3% | Correct evidence was retrieved; the diagnosis was still wrong. Rare — the final generation step is not where the bottleneck lives once given genuinely relevant evidence. |
| Correct | 4.2% | — |
| Hallucination (overlay, among wrong cases) | 13.0% | Diagnosis not grounded in the retrieved evidence, per Qwen3-VL's own faithfulness judge. |
| Calibration failure (overlay, among wrong cases) | 96.1% | ≥70% stated confidence despite being wrong — near-universal when the pipeline is wrong at all. |

## Main Findings

1. **RAG can dramatically improve fine-grained agricultural disease diagnosis — but only if the image-to-query representation avoids a lossy text bottleneck.** Oracle retrieval (78.7% confident_match accuracy) proves the corpus and diagnosis model are capable; the gap to any deployable arm is a retrieval-query problem, not a corpus or model-capacity problem.
2. **Text-mediated retrieval (VLM-generated caption → text embedder) is the weak link, and swapping the text embedder alone does not fix it.** Arms 2 and 3 (BGE-text vs SigLIP-text, same Qwen-generated query) score identically (4.2% each) — the embedding *model* isn't the bottleneck, the fact that the query is *text derived from a lossy VLM description* is.
3. **Bypassing that text bottleneck — direct cross-modal image embedding — closes much of the gap.** Stage D (16.7%) is ~4x Arm 2's accuracy using the same SigLIP model that scored identically to BGE when used in text-mediated mode (Arm 3, 4.2%). The representation, not the model brand, is what mattered.
4. **Better ranking within retrieval does not reliably translate to better diagnosis.** Hybrid retrieval improved Recall@1 substantially over pure-visual (8.3% vs 3.3%) but did not improve diagnosis accuracy (p=0.81) — the diagnosis step consumes the whole top-5 evidence set, so recall breadth (R@5) matters more than rank-1 precision.
5. **Two further retrieval-representation attempts (truncation fix, caption-then-retrieve) both failed to beat what was already found**, and are preserved as negative results, not discarded.
6. **The dominant, quantified failure mode across this entire project is query construction (86.2% of remaining errors)**, not corpus coverage (6.2%) or model reasoning (3.3%).
7. **Calibration is broken independent of everything else** — ECE ≈0.44 across every arm, 96.1% of wrong diagnoses are confidently wrong. Fixing retrieval does not fix trustworthiness.

### On the proposed central claim

> "RAG effectiveness is strongly dependent on the representation used to connect the image to the knowledge base."

**The evidence supports this claim, with a more precise mechanism than the general phrasing implies.** The cleanest single piece of evidence is the Arm 3 vs Stage D comparison: **the same SigLIP model**, at 4.2% accuracy when used to embed a Qwen-generated text description (text-mediated) vs 16.7% when used to embed the image directly (cross-modal) — a 4x difference from changing only the representation, holding the embedding model, corpus, and diagnosis model fixed. Combined with the Oracle ceiling (78.7%, using the ground-truth text as a hypothetically-perfect representation) and Q2's full-scale finding that no amount of *prompt* reformatting fixes a fundamentally lossy text description, the evidence is consistent and multi-angled, not a single result.

A more precise version, if a stricter formulation is wanted for the thesis:

> **RAG effectiveness in this pipeline is bottlenecked by how the image is converted into a retrieval query, not by the retrieval backend, embedding-model brand, corpus scale, or reranking.** Representations that avoid routing the image through a free-text VLM-generated description — direct cross-modal image embedding, or (as an upper bound) a hypothetically perfect text query — substantially outperform text-mediated retrieval, even when the text-mediated version uses the identical embedding model.

This is more precise than the original phrasing because it names the specific mechanism (avoiding the lossy text-generation step) rather than "representation" in the abstract, and it's directly falsifiable against the Arm 2 vs Arm 3 same-query-different-embedder comparison (which shows embedder choice alone does *not* explain the effect).

## Limitations

- **Evaluation-set composition**: 465 primary cases across 3 groups is not large by deep-learning standards; per-arm confident_match subgroup is n=240, and several individual disease classes within it have only ~15 images each — individual-class conclusions (e.g. "hybrid is worse for X disease") are not statistically powered, only the aggregate comparisons above are.
- **Disease-in-corpus dependence**: every positive RAG result in this project is conditional on the disease existing in the corpus with reasonable coverage (`confident_match`, ≥5 facts). On corpus-absent diseases (negative_control, n=45) every arm scores 0% by construction — this is a designed, informative negative control, not a shortcoming, but it means the reported gains do not generalize to out-of-corpus diagnosis.
- **PlantVillage/domain limitations**: query images are lab-photographed, single-leaf, largely clean-background images (PlantVillage's known characteristic) — real-world field photos (cluttered background, multiple leaves, variable lighting) were not tested and would likely be harder for both the diagnosis and retrieval steps.
- **AgMMU corpus limitations**: community-sourced facts, uneven field coverage (only 19% of raw entries have an explicit disease field; many facts are pest-only or context-only), inconsistent disease naming (the exact bug this consolidation pass fixed for evaluation, but the underlying corpus inconsistency itself remains — e.g. "bacterial spot" vs "bacterial leaf spot" as genuinely different strings in different facts is a corpus data-quality property, not just a scoring artifact) — 7.6% of facts bundle multiple diseases in one field, another corpus-construction property inherited from the source data.
- **Visual/text embedding alignment**: SigLIP was not fine-tuned on agricultural imagery or on this corpus's structured `Species:/Disease:/Symptoms:` text format; its text tower's 64-token limit silently truncates ~18% of corpus facts. This project tested two mitigations (truncation-safe shortening, caption-bridging) and both underperformed the untruncated baseline — the alignment gap looks structural, not fixable by off-the-shelf embedding tricks (see Stage G).
- **Model-judged reliability metrics**: faithfulness, groundedness, and hallucination rate are all scored by Qwen3-VL judging its own output — the same model, same weights. The CPJ refinement experiment directly demonstrated this self-judging bias (the single hardest case in the project scored 8.0/10 "grounded" from the judge despite the judge's own written feedback naming a missing diagnostic detail). These numbers should be read as directional, not as ground truth.
- **Sample size / statistical power**: the main ablation's full-465-case McNemar tests could not distinguish "RAG helps a little" from "RAG doesn't help at all" for several arm comparisons (p=0.10–0.80) — only the confident_match-restricted tests and the oracle/Stage-D comparisons reach clear significance. Absence of significance elsewhere is not evidence of no effect.
- **Oracle retrieval is not deployable**: it requires knowing the ground-truth disease name to construct the query, which is unavailable at real inference time. It is reported strictly as a ceiling/upper-bound measurement, never as a candidate architecture.
- **No paired image-text training data exists for learned cross-modal alignment**: the ~500GB `images_ft` corpus that would pair the diagnostic facts with images was deliberately never downloaded (Phase 1A/data-finding-plan decision). This forecloses fine-tuning SigLIP (or a similar dual encoder) on this exact corpus without either downloading that data or sourcing paired data elsewhere — the literature review's suggested remedies (Long-CLIP, GME, domain-adapted dual encoders) all assume paired training data this project does not have.
- **Single VLM, single embedding-model family tested**: Qwen3-VL-8B-Instruct (4-bit) and SigLIP-base are the only VLM/vision-embedder combination evaluated end-to-end; results may not transfer to other VLMs (GPT-4V-class, LLaVA, InternVL) or other vision-language embedders (CLIP variants, ALIGN).

## Final Architecture

**Recommended for the thesis writeup, in order of what each component is for:**

1. **VLM-only baseline** — `Image → Qwen3-VL → Diagnosis`. Establishes the zero-shot floor (0.4% confident_match accuracy). Necessary as a baseline, not viable standalone.
2. **Text-RAG baseline** — `Image → Qwen3-VL query → BGE retrieval → Qwen3-VL diagnosis`. Establishes that naive RAG, as commonly implemented (VLM-generated text query, off-the-shelf text embedder), provides only a marginal, largely-noise-level improvement (4.2%) over the zero-shot floor — an important negative/cautionary baseline for anyone assuming RAG "just works" once bolted onto a VLM.
3. **Visual-RAG — the best practical (deployable) architecture found**: `Image → SigLIP image-tower embedding → cross-modal retrieval against SigLIP-text-embedded corpus → Qwen3-VL diagnosis with retrieved evidence`. 16.7% confident_match accuracy, ~4x text-RAG, statistically robust (p<10⁻⁶ against both baselines). No ground-truth leakage, no reranking dependency, deployable as-is.
4. **Oracle-RAG — upper bound, not deployable**: `ground-truth-disease-name → BGE retrieval → Qwen3-VL diagnosis`. 78.7% confident_match accuracy. Used exclusively to characterize how much headroom remains and to prove the corpus/diagnosis-model combination is capable — never proposed as a real system.
5. **Hybrid-RAG — experimentally tested alternative, not recommended over Visual-RAG**: `Image → [SigLIP visual + BGE text on Qwen query, α=0.25 blend] → retrieval → Qwen3-VL diagnosis`. 17.5% confident_match accuracy — nominally higher than Visual-RAG but **not statistically distinguishable from it** (McNemar p=0.81). Documented as a legitimate, tested alternative with no proven diagnostic benefit over the simpler Visual-RAG architecture, and with added complexity (two embedding models, a blend weight to tune, no formal dev/test split to select α on). **Recommendation: Visual-RAG over Hybrid-RAG on parsimony grounds, given no measured accuracy benefit.**

Reranking (originally Arm 5 / the plan's "Q3") was evaluated on the main ablation's text-RAG pipeline (4.6% confident_match, not significantly different from unreranked text-RAG) and never retested on top of Visual-RAG — an explicit open item, not a finding (see Open Future Work).

## Conclusions

- The project's central, defensible conclusion is **not** "RAG works" or "RAG doesn't work" — it is that **RAG's effectiveness in this domain is gated by a specific, identifiable, and partially-fixable bottleneck: how the query connecting the image to the text knowledge base is constructed**, and that this bottleneck accounts for the great majority (86.2%) of remaining diagnostic failures once genuine corpus-coverage gaps (6.2%) and model-reasoning failures (3.3%) are separated out.
- This conclusion is supported by convergent evidence at multiple levels: a controlled representation-only comparison (Arm 3 vs Stage D, same embedding model), a large statistically overwhelming ceiling measurement (Oracle, p<10⁻⁴⁰), a quantified failure-mode breakdown (error taxonomy), and a negative result ruling out a plausible alternative explanation (reranking/blending, Stage E/F, doesn't close the gap the way changing representation does).
- The negative results are as scientifically load-bearing as the positive ones: naive text-mediated RAG barely helps (a real, useful caution against assuming RAG "just works"); reranking/hybrid-blending, truncation-fixing, and caption-bridging all failed to improve on the simplest cross-modal approach already found — ruling out several plausible "easy fixes" rather than leaving them untested.
- The remaining gap between the best deployable architecture (Visual-RAG, 16.7%) and the ceiling (Oracle, 78.7%) is large and, on current evidence, not closeable by further off-the-shelf retrieval-representation swaps — it looks like it needs either paired training data for a fine-tuned cross-modal aligner (not available in this project) or a fundamentally different query-construction mechanism not yet tried.

## Open Future Work

Explicitly **not** pursued in this project, listed for a future phase, not implied as necessary follow-on work for this thesis:

1. **Reranking on top of Visual-RAG** (rather than on top of the weaker text-RAG pipeline, which is all that was tested) — untested combination.
2. **Fine-tuning a cross-modal aligner** (SigLIP or similar) on paired agricultural image-text data, if such data is sourced — the literature review's suggested remedy for the embedding-alignment gap, blocked in this project by the absence of paired training data.
3. **A different query-construction mechanism** not yet tried — e.g. a VLM fine-tuned specifically to produce retrieval-optimized (not human-readable) queries, or a learned query encoder trained end-to-end against retrieval success rather than prompted zero-shot.
4. **External (non-self) judge model** for faithfulness/groundedness/hallucination scoring, to remove the self-judging bias documented throughout (CPJ test, main ablation's groundedness caveat).
5. **Testing on field-realistic images** (cluttered background, multiple leaves, variable lighting) rather than PlantVillage's lab-photographed images, to test whether the Visual-RAG advantage generalizes beyond this dataset's visual distribution.
6. **A second VLM/embedder family** (e.g. LLaVA or InternVL in place of Qwen3-VL; CLIP or ALIGN in place of SigLIP) to test whether the central finding (representation matters more than model brand) generalizes beyond the one model pair tested here.
7. **Corpus-quality remediation** — resolving the corpus's internal disease-naming inconsistencies (the same disease under different strings in different facts) at the source, rather than working around it at evaluation time via the alias table built in this consolidation pass.
