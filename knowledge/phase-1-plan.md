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

1. ~~Text embedding model~~ — **decided: `TaylorAI/bge-micro-v2`.** See
   [phase1b-q1q2-results.md](phase1b-q1q2-results.md): won every metric on the controlled
   47-query benchmark (R@1=0.681, R@3=0.809, R@5=0.830, MRR=0.751), beating both
   sentence-transformers/all-MiniLM-L6-v2 and SigLIP's text tower.
2. ~~Exact retrieval query construction~~ — **provisionally decided, not proven optimal.**
   See [phase1b-q2-isolation-results.md](phase1b-q2-isolation-results.md): raw open-ended
   VLM captioning is rejected outright (0% R@1, and a confident wrong species claim was
   shown to actively damage retrieval — same image, same target disease, rank 1 vs rank
   15 depending only on whether species was falsely claimed). Direction selected instead:
   **structured visual-evidence extraction with species treated as optional/unknown**,
   not a plain candidate-disease guess or an unconstrained description. A structured
   "Species: unknown + bulleted symptoms" prompt recovered most of the lost performance
   (28.6% R@1) with no extra token budget needed. A remaining gap to the hand-authored
   benchmark (62–100% R@1) traces partly to the VLM's descriptions missing each disease's
   *specific* diagnostic feature (e.g. cedar apple rust's orange tube structures, early
   blight's concentric rings) even when not hallucinating — being validated further with
   a diagnostic-feature-focused prompt variant on a small hard-case set.
3. **Reranking method** for ablation arm 5 — a cross-encoder, an LLM-as-reranker prompt,
   or something else.
4. **Ground-truth source per experiment** — PlantVillage folder labels give species +
   disease-vs-healthy for free; AgMMU's 770-entry eval set gives MCQ-format ground truth
   with richer background context. Decide which drives the headline accuracy numbers, or
   how to combine both.
5. ~~Preprocessing of the 45,096 facts~~ — **decided**, see
   [phase1a-preprocessing-decisions.md](phase1a-preprocessing-decisions.md): split into a
   ~17,760-entry diagnostic corpus and a ~27,336-entry context corpus (kept, not deleted,
   so retrieval-population choice stays a separate later experiment); dedupe the
   retrieval corpus while preserving `source_faq_ids` traceability; normalize (not
   exclude) the 3 dict-shaped multi-species answers; keep `species_raw` unnormalized.
6. **FAISS vs Qdrant vs plain numpy** — the finalized architecture names Qdrant, but at
   45K facts even brute-force cosine similarity in numpy runs in well under a second.
   Worth deciding whether Qdrant is worth standing up for Phase 1 or whether that's
   premature infrastructure — your call, since the diagram names it but the reasoning for
   *why* over the simpler options wasn't spelled out yet.

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
