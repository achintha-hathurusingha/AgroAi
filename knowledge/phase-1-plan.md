# AgriVision-RAG — Phase 1 Plan

**How we're working from here:** you learn the concepts and make the design decisions;
I explain things in depth on request, execute infra/ops you delegate (SSH, downloads,
environment setup), and review your calls rather than making them for you.

**Scope finalized 2026-08-26.** After reviewing the original all-in-one architecture
(crop detection + disease + quality + segmentation + severity + RAG + hallucination
detection + benchmarking + deployment, all at once), you deliberately narrowed it. The
reasoning: that architecture risked becoming "a half-finished collection of features"
rather than a project with a real, defensible research contribution. The sections below
marked **(decided)** are locked in; sections marked **(open)** are still yours to decide.

## Core research question (decided)

> Can retrieval-augmented multimodal reasoning improve the fine-grained diagnosis and
> reliability of vision-language models for agricultural disease identification,
> compared with zero-shot VLM diagnosis?

This replaces the vaguer original framing ("build a multimodal agricultural diagnosis
system"). It's motivated by real evidence already in hand: the Cedar Apple Rust test,
where Qwen3-VL got species and disease-vs-healthy right but the specific disease wrong —
precisely the failure mode retrieval grounding is meant to fix. If it turns out RAG
*doesn't* meaningfully help, that's still a valid, reportable finding.

## Finalized core pipeline (decided)

```
IMAGE
  │
  ▼
Vision Model (SigLIP / Qwen3-VL) ── candidate diseases
  │
  ▼
Vector Search (Qdrant) ── relevant agronomic evidence
  │
  ▼
Qwen3-VL
  │
  ├── Disease
  ├── Severity
  └── Explanation
  │
  ▼
JSON Response
```

Model responsibilities are now explicit rather than asking SigLIP to do everything:

| Task | Model |
|---|---|
| Visual embedding / retrieval | SigLIP |
| General image understanding, final reasoning | Qwen3-VL |
| Knowledge retrieval | Qdrant |

(Object detection, segmentation, and specialized classifiers are explicitly **not**
part of Phase 1 — see "Deferred" below.)

## Ablation design (decided)

The actual deliverable is a comparison across these variants, not a single system:

1. **VLM only** — `Image → Qwen3-VL → Diagnosis` (the baseline, already partially tested).
2. **VLM + RAG** — `Image → Retrieval → Qwen3-VL → Diagnosis`.
3. **VLM + RAG + SigLIP retrieval** — using SigLIP-based similarity for the retrieval step specifically.
4. **VLM + RAG + structured knowledge** — retrieving the structured (species/disease/symptoms/management) facts rather than unstructured text.
5. **VLM + RAG + structured knowledge + reranking** — adding a reranking step on top.

Which components actually help is the experimental question — not assumed in advance.

## Evaluation metrics (decided)

| Metric | Measures |
|---|---|
| Disease accuracy / Top-3 accuracy | Diagnostic correctness |
| Retrieval Recall@1 / @3 / @5, MRR | Does retrieval surface the right evidence? |
| Groundedness / evidence faithfulness | Does the answer actually follow the retrieved evidence? |
| Hallucination rate | Invented diseases, treatments, symptoms, unsupported claims |
| Confidence calibration | If it says 95% confident, is it right ~95% of the time? |
| Latency | Image processing + retrieval + LLM inference, broken down |
| Resource usage | VRAM, RAM, GPU utilization, throughput |

This directly resolves the old "what counts as success" question: success is a
**quantified comparison across the ablation table above**, not a single accuracy number.

## Explicitly deferred — not Phase 1 (decided)

These are real, valuable extensions, but starting on them now is what caused scope
creep. Revisit only after the core ablation study produces results:

- Agricultural Foundation Model (Agri-FM-style domain adaptation) — too large for the core FYP.
- Electronic-nose / non-visual multimodal fusion — changes hardware/data requirements substantially.
- Standalone hallucination-detector *product* (hallucination *rate* stays as an eval metric — the difference is building a whole detection system vs measuring it).
- Full disease segmentation + severity estimation (SAM2/Grounding DINO pipeline) — potentially a project by itself.
- LoRA fine-tuning of Qwen3-VL — don't fine-tune just because it's possible; only pursue if the RAG-only result needs it.
- Fruit **quality/freshness grading** ("Freshness: 78%", "Commercial grade: B") — dropped for a concrete reason: there's no reliable ground-truth source for these labels in the data we have. Manufacturing precise-looking numbers from the LLM without a measurable ground truth would be scientifically hollow. If revisited later, it needs either a real labeled dataset or a coarser, defensible categorical scale (Fresh / Acceptable / Spoiled) instead of a fabricated percentage.

## Still open — yours to decide

The finalized scope resolves *what* to build and *how to evaluate it*, but not every
implementation detail:

1. ~~Text embedding model~~ — **decided: `TaylorAI/bge-micro-v2`, re-validated at full
   scale with a corrected margin.** Original 400-fact-pool result in
   [phase1b-q1q2-results.md](phase1b-q1q2-results.md) (R@1=0.681, clear win). A Milestone
   3 control-case failure revealed that pool was too small to represent real retrieval
   difficulty (same query, same code: rank 1 in the 400-fact pool, rank 40 in the full
   17,583-fact corpus — see [milestone3-control-failure-diagnosis.md](milestone3-control-failure-diagnosis.md)),
   so Q1 was re-run at full scale: [phase1b5-fullscale-validation-results.md](phase1b5-fullscale-validation-results.md).
   BGE still wins overall (R@3=0.596, R@5=0.617, R@10=0.702, R@20=0.787, MRR=0.526, all
   best of the three) but **ties sentence-transformers exactly at R@1 (0.426 each)** at
   full scale — the original "clearly wins" framing overstated the margin.
2. **Exact retrieval query construction — REOPENED.** Was recorded as "decided: Prompt
   C" based on the small 400-fact-pool isolation experiment (C: 28.6% R@1 vs A: 0%). A
   Milestone 3 control failure prompted re-testing at full 17,583-fact scale (same
   Prompt-A/B/C/D/E texts, no new generation, BGE-micro-v2 fixed) — see
   [phase1b5-q2-fullscale-results.md](phase1b5-q2-fullscale-results.md). **Result: at
   full scale, every variant scores 0% R@1 through R@5, and no variant wins consistently
   across diseases** — A wins for powdery mildew (the exact case C had "solved" at rank 1
   in the small pool), D wins for apple scab and black rot, C wins for septoria and
   bacterial spot, nothing gets cedar apple rust or early blight close to usable. The
   original Prompt C conclusion does not hold and should not be relied on. Current best
   read: the bottleneck isn't query *phrasing* — it's that Qwen3-VL's descriptions rarely
   contain the specific feature that would distinguish the true disease from thousands of
   textually-similar alternatives in the full corpus, and no reformatting of the same
   generic visual content ("brown spots," "irregular margin") fixes that. A follow-up
   bottleneck diagnosis ([phase1b5-bottleneck-diagnosis-results.md](phase1b5-bottleneck-diagnosis-results.md))
   found the corpus/embedding representation itself is capable (hand-authored queries hit
   rank 1-3), so the gap is specifically what Qwen's description contains — and species
   context turned out to be a double-edged signal (40x better for some diseases, 6x worse
   for others), not a simple fix. **Closed for Phase 1 as: Prompt C kept as the practical
   baseline query strategy, explicitly documented as unvalidated-as-optimal rather than
   proven best** — no tested strategy (A-E) is a robust winner at full scale, and that
   itself is the finding, not a gap still being chased.
3. ~~Reranking method~~ — **decided: deferred to Phase 2, not part of the Phase 1
   baseline.** See [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md). Ablation
   arms 1-4 run without a reranker; arm 5 (structured knowledge + reranking) is where
   reranking gets evaluated on its own merits in Phase 2 — not assumed to help, given the
   bottleneck diagnosis showing correct facts sometimes rank thousands of positions down,
   which a reranker over a limited candidate set can't rescue by construction.
4. **Ground-truth source per experiment** — proposed in [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md)
   (PlantVillage labels primary, AgMMU's 770-entry eval set secondary) but flagged there
   as a proposal needing your confirmation, not a freeze — this is a real design choice,
   unlike the corpus freeze and leakage audit in the same document, which are checks.
5. ~~Preprocessing of the 45,096 facts~~ — **decided**, see
   [phase1a-preprocessing-decisions.md](phase1a-preprocessing-decisions.md): split into a
   ~17,760-entry diagnostic corpus and a ~27,336-entry context corpus (kept, not deleted,
   so retrieval-population choice stays a separate later experiment); dedupe the
   retrieval corpus while preserving `source_faq_ids` traceability; normalize (not
   exclude) the 3 dict-shaped multi-species answers; keep `species_raw` unnormalized.
6. ~~FAISS vs Qdrant vs plain numpy~~ — **decided: NumPy exact cosine.** See
   [phase1-q6-vector-store-results.md](phase1-q6-vector-store-results.md): a real
   benchmark (standalone Qdrant binary as a genuine server, no Docker/root needed) showed
   NumPy fastest on every measure at both 17,583- and 35,507-fact scale — build ~0ms vs
   FAISS's 13–28ms vs Qdrant's 4.5–8.8s; query mean 0.343–0.390ms vs FAISS's 0.7–2.1ms vs
   Qdrant's 2.9–3.6ms. FAISS/Qdrant remain viable future swaps if corpus size grows much
   larger, or if persistence across restarts / metadata filtering / multi-process shared
   access becomes an actual requirement — not needed for Phase 1's single-process pipeline.

## Concepts worth understanding before/while building this

1. **Embeddings & similarity search** — what a vector embedding represents, why
   cosine/dot-product similarity finds "similar" items.
2. **RAG (Retrieval-Augmented Generation)** — retrieve-then-generate, so the LLM grounds
   its answer instead of relying purely on memorized training data. The
   [AgriRAG paper](papers/Marques_AgriRAG_Training-Free_Retrieval-Augmented_Generation_for_Agricultural_Disease_Diagnosis_with_Vision-Language_CVPRW_2026_paper.pdf)
   is the direct reference.
3. **Vector databases** — what Qdrant/FAISS actually do differently from looping over
   embeddings in Python (indexing structures, approximate vs exact search, metadata filtering).
4. **Quantization** — what `bitsandbytes` 4-bit (NF4) loading does to model weights, and
   the accuracy/memory tradeoff.
5. **Prompt construction for grounded generation** — how retrieved facts get formatted
   and inserted into the Qwen3-VL chat prompt.
6. **Evaluation methodology** — MCQ-format scoring (objective, easy) vs open-ended
   generation scoring (harder, closer to real use); the SMART paper's finding that
   "structured captions outperform LLM-generated narratives" is directly relevant to
   open question 5 above.
7. **Retrieval evaluation** — what Recall@k and MRR actually measure, since they're now
   part of the locked-in metrics table.
8. **Calibration** — what it means for a confidence score to be "calibrated," and how
   it's measured (e.g. reliability diagrams / Expected Calibration Error).

## What already exists (context, not homework)

- Downloaded: SigLIP (`google/siglip-base-patch16-224`), Qwen3-VL-8B-Instruct (4-bit),
  PlantVillage (54,305 images, 38 classes), AgMMU eval set (770 image+MCQ entries) and
  its 45,096-entry fact set (`agmmu_ft_hf1.json`).
- Running on `devon` (192.248.10.68), `agrivision` conda env, RTX 4090 (24GB VRAM, ample
  for this scope — more headroom than the shared qbits 4080 SUPER).
- Documented in [AgriVision-RAG-architecture.md](AgriVision-RAG-architecture.md) and
  [data-finding-plan.md](data-finding-plan.md) — those still describe the original wider
  architecture; treat this document as the current source of truth for scope until they're
  updated to match.

## Suggested milestones

1. Explain the finalized pipeline and ablation design back in your own words/diagram.
2. Resolve the six open questions above and write down *why* — useful for the resume/FYP
   writeup later, and lets me sanity-check the reasoning if you want a second opinion.
3. Build the smallest possible retrieval step (embed the facts, embed one query, get
   top-k results); sanity-check results by eye before building anything on top.
4. Wire retrieval into the Qwen3-VL prompt; re-run the Cedar Apple Rust case — does
   grounding fix the wrong-diagnosis problem?
5. Run the full ablation table (arms 1–5) against whichever ground-truth source(s) you
   picked, computing all seven metrics.
6. Write up the results — including any arm that *doesn't* help, since the ablation's
   value is showing what actually matters, not confirming everything works.

## How I'll help vs what's yours

- **Mine on request:** deep explanations of any concept above, code review, running
  infra/data tasks you delegate, catching bugs, keeping the knowledge base in sync with
  decisions you make.
- **Yours:** the six open questions above, writing the pipeline and evaluation code, and
  interpreting the ablation results.
