# Thesis Outline Scaffold

Structure agreed 2026-08-27. This is a skeleton pre-filled with verified numbers,
statistical results, and source-document pointers for each claim — not draft prose.
Every number here is traceable to [final-research-findings.md](final-research-findings.md)
and, beneath that, to a raw saved experiment output. Write the actual chapter prose from
this scaffold; do not introduce numbers here that aren't already in the final findings
doc — if a chapter needs a number that isn't verified yet, that's a signal to check
`final-research-findings.md` first, not to estimate one.

**Primary results are frozen** (per the "no more changes to primary results" decision).
Any new number this thesis needs that isn't already in `final-research-findings.md`
should be flagged before writing around it, not assumed.

---

## Chapter 1 — Introduction

- Agricultural disease diagnosis problem: crop losses, smallholder impact (see AgriRAG
  paper's cited FAO figure — ~40% of global crop production losses to pests/disease —
  in [knowledge/papers/Marques_AgriRAG...pdf](papers/Marques_AgriRAG_Training-Free_Retrieval-Augmented_Generation_for_Agricultural_Disease_Diagnosis_with_Vision-Language_CVPRW_2026_paper.pdf)).
- VLM opportunity / VLM limitations: cite the project's own concrete evidence, not just
  literature — the Cedar Apple Rust case (Qwen3-VL got species + disease-vs-healthy
  right, specific disease wrong; [AgriVision-RAG-architecture.md](AgriVision-RAG-architecture.md)
  "First real-data pipeline test") is the motivating anecdote referenced in
  [phase-1-plan.md](phase-1-plan.md)'s research-question framing.
- RAG as proposed mechanism; the representation problem as the gap.
- **Research gap** (as drafted): "the effectiveness of retrieval augmentation depends on
  how visual observations are converted into retrieval representations... insufficiently
  characterized" — this is exactly what [final-research-findings.md](final-research-findings.md)'s
  Main Findings §"On the proposed central claim" evaluates and refines. Use that
  section's more precise formulation as the thesis's actual claim-under-test:
  > "RAG effectiveness in this pipeline is bottlenecked by how the image is converted
  > into a retrieval query, not by the retrieval backend, embedding-model brand, corpus
  > scale, or reranking."

## Chapter 2 — Literature Review

- 2.1 Plant disease diagnosis — PlantVillage lineage, prior CNN-classifier work.
- 2.2 Vision-language models — Qwen3-VL family, general VLM capability/limitation framing.
- 2.3 Agricultural multimodal datasets — **AgMMU** (arXiv:2504.10568, NeurIPS 2025) is
  the corpus source; its own stated future-work note ("vision-centric multimodal RAG...
  under-explored yet promising") directly motivates this thesis — cite it as
  positioning, not just data provenance. See [final-research-findings.md](final-research-findings.md)
  Limitations for the exact corpus-construction caveats (uneven field coverage, naming
  inconsistency) to disclose here, not discover later in a review.
- 2.4 Retrieval-Augmented Generation — standard RAG framing, RAG-Triad/RAGAS
  (faithfulness/groundedness/context-relevance), cited in [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md).
- 2.5 Multimodal / visual retrieval — cross-modal retrieval without paired images:
  KAT/REVIVE (Gui et al.) as the validated precedent for "text-only KB, image-tower
  query at inference," and the AgriRAG paper (verified real, in `knowledge/papers/`) as
  the closest published analogue to this project's own architecture — same base corpus
  family (AgMMU), same VLM (Qwen3-VL-8B, 4-bit), SigLIP-based retrieval. Worth an
  explicit comparison paragraph: AgriRAG reports 83.7% MCQ accuracy on a
  multiple-choice benchmark with a 74,611-fact fused corpus; this thesis evaluates
  open-ended free-text diagnosis accuracy (a harder task than MCQ selection) on a
  17,583-fact single-source corpus — the two results are not directly comparable
  numbers, and the thesis should say so explicitly rather than imply equivalence.
- 2.6 Evaluation of RAG systems — Recall@K conventions (qrels/ranx-style set membership
  vs strict string equality — this thesis's own scoring correction is a worked example,
  see Chapter 4/6), TriviaQA-style answer-alias matching for multi-value ground truth.
- 2.7 Research gap — position as in Chapter 1.

## Chapter 3 — Methodology

- **Dataset**: AgMMU 45,096 raw entries → 17,583 diagnostic + 17,924 context facts,
  frozen as `agmmu_phase2_v1` with SHA-256 checksums
  ([phase2-corpus-freeze.md](phase2-corpus-freeze.md)). Preprocessing decisions and
  rationale in [phase1a-preprocessing-decisions.md](phase1a-preprocessing-decisions.md).
  PlantVillage (465 primary evaluation cases, 3 groups) as query images — construction
  detail in [phase2-execution-plan.md](phase2-execution-plan.md) §1.
- **Baseline** (VLM-only): `Image → Qwen3-VL → Diagnosis`.
- **Text-RAG**: `Image → Qwen3-VL (Prompt C) → BGE embed → NumPy top-k → Qwen3-VL + evidence → Diagnosis`.
  Note for methodology rigor: explicitly state Prompt C was **not** validated as optimal
  at full corpus scale ([phase1b5-q2-fullscale-results.md](phase1b5-q2-fullscale-results.md))
  — it's the practical baseline, documented as such, not claimed as a proven-best choice.
- **Visual-RAG**: `Image → SigLIP image tower → cross-modal retrieval vs SigLIP text-tower corpus embeddings → Qwen3-VL + evidence → Diagnosis`.
  State plainly why this design was chosen over literal image-to-image retrieval: no
  paired corpus images exist ([phase2-visual-data-audit.md](phase2-visual-data-audit.md)).
- **Oracle**: `ground-truth disease name → BGE retrieval → Qwen3-VL + evidence → Diagnosis`.
  State plainly and early that this is a ceiling measurement, not a candidate system —
  avoids a reader mistaking it for a proposed architecture later in the thesis.
- **Evaluation protocol**: hierarchical ground truth (Q4,
  [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md)), leakage audit result
  (0 image collisions, [leakage-audit-results.md](leakage-audit-results.md)), corrected
  scoring methodology (alias-set matching — [phase2-scoring-correction.md](phase2-scoring-correction.md),
  present this as a methodological contribution, see Chapter 6 note below).
- **Statistics**: Wilson score CIs, McNemar's paired test (same cases through every
  arm) — methodology fixed in [phase2-execution-plan.md](phase2-execution-plan.md) §9.

## Chapter 4 — Phase 1: Retrieval System Investigation

Table (verbatim from [final-research-findings.md](final-research-findings.md)'s
Experimental Timeline, expand each row into a subsection):

| Question | Experiments | Decision |
|---|---|---|
| Q1 (embedding model) | MiniLM / BGE-micro-v2 / SigLIP-text, 400-fact pool then full 17,583-fact corpus | BGE-micro-v2 — but **R@1 ties with MiniLM exactly (0.426 each) at full scale**; wins on R@3–R@20/MRR. Original small-pool "clearly wins" framing corrected. |
| Q2 (query construction) | Prompts A–E, 400-fact pool then full-scale | No robust winner at full scale — **every variant scores 0% R@1–R@5**; small-pool Prompt C result does not hold |
| Q3 (reranking) | — | Deferred to Phase 2 as its own ablation arm |
| Q6 (vector store) | NumPy vs FAISS vs Qdrant, real benchmark at 17,583/35,507-fact scale | NumPy exact cosine — fastest on every measure |
| Corpus | Dedup, inspection, freeze | `agmmu_phase2_v1`, checksummed |
| Leakage | Image hash + code audit | Clean — 0 collisions, no ground-truth path into query construction |

**Key methodological narrative for this chapter**: the Milestone 3 control-case failure
([milestone3-control-failure-diagnosis.md](milestone3-control-failure-diagnosis.md)) —
same query, same code, rank 1 at 400-fact scale vs rank 40 at full 17,583-fact scale —
is the pivot that triggered full-scale re-validation of Q1/Q2. Frame this explicitly as
*good methodology* (a discovered measurement flaw, corrected before it propagated into
Phase 2), not as an embarrassing false start. This same "small-scale results don't
generalize" lesson recurs at least twice more later in the project (Stage F rescoring
in Ch. 6) — worth stating once here as an established pattern, then referencing it by
name in Ch. 6 rather than re-explaining.

## Chapter 5 — Main Evaluation

Numbers (from [final-research-findings.md](final-research-findings.md) "Phase 2 Main
Ablation" — **use alias-corrected figures**, not the original per-experiment doc's
pre-correction numbers):

| Arm | Overall (n=465) | Disease-in-corpus (n=240) |
|---|--:|--:|
| VLM-only | 37.8% (176/465) | 0.4% (1/240) |
| Text RAG (BGE) | **39.6%** (184/465) | 4.2% (10/240) |
| RAG (SigLIP-text, still text-mediated) | 39.1% (182/465) | 4.2% (10/240) |
| RAG (full corpus) | 39.4% (183/465) | 3.3% (8/240) |
| RAG + rerank | 39.1% (182/465) | 4.6% (11/240) |

No arm reaches significance against VLM-only on the full 465-case set (McNemar
p=0.10–0.80, original run). Text RAG vs VLM-only **does** reach nominal significance
restricted to the confident_match subgroup (p=0.016, recomputed) — state both results;
don't present only the full-set null result or only the subgroup-significant one, since
the *reason* for the discrepancy (healthy/negative-control groups where nothing differs
between arms, diluting the paired test) is itself part of the methodological story in
this chapter.

Subgroup structure to present visually (bar chart or table): healthy ≈95–97% flat
across every arm; negative control 0.0% flat across every arm (confirms no
hallucinated-disease behavior); disease-in-corpus 0.4–4.6% — this is where the entire
"why is overall accuracy misleading" argument for the chapter comes from.

Calibration: ECE ≈0.43–0.45 across all 5 arms — state this here as a standing,
unresolved finding that recurs in Ch. 6/9, not something RAG fixes.

## Chapter 6 — Diagnosing the RAG Failure

This is the pivot chapter. Structure:

1. **Oracle result**: 0.4% (VLM-only) → 4.2% (text RAG) → **78.7%** (oracle,
   confident_match). McNemar oracle-vs-text-RAG: p=2.2×10⁻⁴⁰, 179 discordant cases in
   oracle's favor, **zero** the other way, across the entire project. State this as the
   thesis's strongest single statistical result.
2. **Retrieval decomposition**: same mechanism from a different angle — Qwen-query R@1
   1.3% vs oracle-query R@1 87.5% (alias-corrected) on confident_match. A ~65x gap from
   changing only the query text.
3. **Error taxonomy** (the quantitative payoff of this chapter):

   | Category | Rate (n=240) |
   |---|--:|
   | Query-construction failure | **86.2%** |
   | Retrieval/corpus failure | 6.2% |
   | Reasoning failure | 3.3% |
   | Correct | 4.2% |

   State clearly that this is the **corrected** figure (up from an original 74.6%) —
   worth one paragraph on *why* it moved, since that paragraph doubles as your
   methodological-rigor demonstration: two scoring bugs (ground-truth vocabulary
   mismatch, multi-value field undercounting) were found by auditing after later
   experiments (Ch. 7), fixed, and every affected number recomputed and reported here
   as corrected — this is worth foregrounding as evidence of rigor, not burying as an
   erratum. Source: [phase2-scoring-correction.md](phase2-scoring-correction.md).
4. **Overlay findings**: hallucination in 13.0% of wrong cases; calibration failure
   (confidently wrong) in **96.1%** of wrong cases — flag as a separate, RAG-independent
   reliability problem, foreshadowing Ch. 9/10.

Closing argument for the chapter (near-verbatim from
[final-research-findings.md](final-research-findings.md)):
> The original RAG null result is not evidence that agricultural knowledge is
> ineffective for diagnosis — it is evidence that the specific query-construction
> mechanism tested fails to retrieve that knowledge.

## Chapter 7 — Visual-RAG

- Motivation: visual-data audit finding — no paired corpus images exist, so true
  image-to-image retrieval is infeasible ([phase2-visual-data-audit.md](phase2-visual-data-audit.md));
  cross-modal SigLIP retrieval (image tower query vs text-tower corpus) proposed
  instead, exploiting SigLIP's shared embedding space rather than needing paired data.
- Retrieval-only result first (Stage C): R@1 3.3% / R@5 20.0% vs current pipeline's
  1.3% / 6.2% — present as the "does this direction look promising before committing to
  a full diagnosis run" gate, mirroring the thesis's own actual research process.
- Full diagnosis result (Stage D): **16.7%** (40/240) vs text-RAG's 4.2% (10/240).
  McNemar p=2.9×10⁻⁹ vs VLM-only, p=3.0×10⁻⁷ vs text-RAG.
- **The controlled comparison that makes this chapter's argument rigorous, not just a
  bigger number**: Arm 3 (SigLIP *text* tower on the Qwen-generated query — still
  text-mediated) scores identically to Arm 2 (4.2%). Same embedding model family,
  different representation, 4x different outcome. This isolates representation from
  embedding-model choice — make this comparison explicit and prominent, it's the
  chapter's (and arguably the thesis's) best piece of evidence.

## Chapter 8 — Hybrid and Negative Experiments

Present as a deliberate falsification exercise, not an afterthought:

- **Hybrid retrieval** (Stage F/E): retrieval R@1 improved substantially (8.3% vs 3.3%
  pure-visual) but diagnosis accuracy did not (17.5% vs 16.7%, McNemar p=0.81 — not
  significant). This is the chapter's headline finding:
  $$\text{Retrieval metric improvement} \not\Rightarrow \text{diagnostic improvement}$$
  Explain the mechanism: diagnosis reads the whole top-5 evidence set regardless of
  rank order, so R@5 (breadth) predicts diagnosis accuracy better than R@1 (rank-1
  precision) — R@5 gap (24.6% vs 20.0%) is real but much smaller than the R@1 gap
  suggested.
- **Truncation-safe SigLIP re-embedding** (Stage G-1): confirmed SigLIP's 64-token text
  limit silently truncates 17.8% of corpus facts (mean 43.8 tokens/fact) — then showed
  that *fixing* this by shortening every fact **halved R@5** (20.0%→10.0%), since
  over-aggressive shortening (mean 8.9 words/fact) discarded more signal than
  truncation cost. Worth foregrounding as a counter-intuitive result: the "obvious fix"
  made things worse.
- **Caption-then-retrieve** (Stage G-2): literature-precedented (caption image → text
  query → text retrieval), improved R@1 (6.2%) but not R@5 (10.8%, still well below
  hybrid's 24.6%), with high variance by disease (near-perfect for powdery mildew,
  near-total failure for cedar apple rust / mosaic virus).
- **Reranking** (Phase 2 main ablation, Arm 5): 4.6% confident_match, not significantly
  different from unreranked text-RAG (4.2%) — note explicitly that reranking was never
  retested on top of Visual-RAG, an open gap, not a completed negative result.
- **Prompt iteration** (v1/v2 diagnosis-prompt fixes,
  [phase2-diagnosis-prompt-iteration.md](phase2-diagnosis-prompt-iteration.md)): worth
  one paragraph as a methodology note — the original 80% "healthy" collapse was a
  prompt-design artifact (model exploiting "healthy"/"unknown" as escape hatches), not
  a decoding-parameter issue as first hypothesized; documenting the wrong-hypothesis
  detour is itself good practice to model in the thesis.

## Chapter 9 — Discussion

Structure as RQ/answer pairs (as drafted), each answer traceable to a specific number
already established in Ch. 5–8 — do not introduce new figures here.

- **RQ1** (Does RAG improve diagnosis?) → Not reliably with VLM-generated text queries
  (39.6% vs 37.8% overall, not significant on the full set); direct visual retrieval
  provides a significant improvement (16.7% vs 0.4%/4.2%, p<10⁻⁶).
- **RQ2** (Does retrieval representation affect performance?) → Yes — the Arm 3 vs
  Stage D controlled comparison (Ch. 7) is the central evidence: same embedding model,
  4x different outcome from representation alone.
- **RQ3** (Does relevant knowledge have diagnostic value?) → Yes, decisively — Oracle's
  78.7% (p<10⁻⁴⁰ vs every deployable arm, zero counter-examples across the project).
- **RQ4** (What limits practical RAG performance?) → Query-construction failure (86.2%
  of remaining errors), not corpus coverage (6.2%) or reasoning (3.3%).

Also address directly (examiners will ask): why is 16.7% "good" when it's still low in
absolute terms? Answer using the framing already established in
[final-research-findings.md](final-research-findings.md)'s closing note — the
contribution is the *demonstrated, measured progression* (VLM-only → Text-RAG → Oracle
→ Visual-RAG) and the *identified, quantified bottleneck*, not a claim of a
production-ready diagnostic accuracy number.

## Chapter 10 — Limitations and Future Work

Pull directly from [final-research-findings.md](final-research-findings.md)'s
Limitations and Open Future Work sections — both are already written at
thesis-appropriate precision and don't need re-deriving:

**Limitations** (8 items in the source doc): eval-set size/composition,
disease-in-corpus dependence, PlantVillage domain gap, AgMMU corpus quality, SigLIP
alignment gap (64-token truncation, confirmed not merely suspected), self-judged
reliability metrics, statistical power gaps in the full-465 McNemar tests, oracle
non-deployability, absence of paired training data, single VLM/embedder pair tested.

**Future work** (7 items in the source doc) — the "learned cross-modal alignment"
direction the user proposed as Project B maps directly onto future-work item #2
(fine-tuning a cross-modal aligner on paired data, explicitly blocked in this project
by data absence); Project C (structured attribute reasoning) and Project D (retrieval +
differential diagnosis) are **not** currently in the source doc's future-work list —
add them there (or here) as thesis-appropriate forward-looking proposals, clearly
labeled as proposals, not findings.

---

## After submission: Research Extension (separate phase, not part of this thesis)

Per the agreed sequencing — do not start until the thesis is frozen/defended/submitted,
and start from a clean copy of the repository:

- Project B: learned agricultural cross-modal alignment (needs paired image-knowledge
  data not currently available — first task would be sourcing or constructing it).
- Project C: structured diagnostic-attribute reasoning (species / lesion morphology /
  color / distribution / texture / severity / differential diagnoses) instead of a
  single free-text diagnosis call.
- Project D: retrieve-multiple-candidates-then-differential-diagnose, rather than
  single-best-guess RAG.
- Stronger/domain-adapted visual encoder as a prerequisite or alternative to Project B.
- Possibly a new, purpose-built paired dataset if none of the above can be done on
  existing data.
