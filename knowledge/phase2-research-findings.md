# Phase 2 Research: Evaluation Methodology & a Concrete Lead on Q2

Web research to inform Phase 2 design, done before finalizing the ablation methodology
and before deciding whether Q2's unresolved query-construction problem is worth one more
targeted experiment before starting the main ablation.

## Most directly relevant finding: CPJ (Caption-Prompt-Judge)

**"CPJ: Explainable Agricultural Pest Diagnosis via Caption-Prompt-Judge with LLM-Judged
Refinement"** — https://arxiv.org/html/2512.24947

This is startlingly close to our own Q2 experiments — agricultural VQA using structured
captions that "explicitly exclude crop or disease names, focusing on objective
morphological and symptomatic features," exactly the spirit of our Prompt C/B. But it
adds a step we never tried:

- **Iterative caption refinement**: an initial caption gets scored by an LLM judge on
  accuracy/completeness/neutrality (threshold τ=8.0/10); captions scoring below that
  threshold get regenerated with targeted refinement instructions, rather than accepting
  the first-pass output the way our entire Q2 experiment series did.
- **Reported result**: using a stronger captioning model (GPT-5-mini) improved downstream
  disease classification by **+22.7 percentage points** over baseline, and knowledge-QA
  score by +19.5 points.
- **Error analysis**: ~8% of failures traced to genuinely low-quality captions (blur,
  occlusion, poor lighting) — the rest presumably to caption *content* quality, matching
  what our own bottleneck diagnosis found (generic captions that don't capture the
  disease-discriminating feature).
- Scoring approach: keyword matching for disease/crop identification; GPT-4-as-judge
  (1–10, normalized to 100) for open-ended knowledge QA.

**Why this matters for us**: every one of our Q2 experiments (A/B/C/D/E) was a
*single-shot* caption generation — we never gave Qwen3-VL a chance to revise a weak
caption. Our bottleneck diagnosis showed the corpus/retrieval system works fine given a
*good* query (human-authored queries hit rank 1–3) — CPJ's refinement loop is a concrete,
literature-backed mechanism for closing exactly that gap without hand-authoring queries
ourselves, which obviously doesn't scale to a real system.

## RAG evaluation methodology (for the Phase 2 groundedness/faithfulness metrics)

- **The "RAG Triad"** (TruLens) and **RAGAS**: three metrics, each gating a different
  failure surface — **groundedness** (is the answer supported by retrieved context),
  **faithfulness** (proportion of atomic claims entailed by context vs. contradicted/
  unsupported), **answer relevance** (does the answer address the question), **context
  relevance** (is what was retrieved actually relevant to the question).
- **Methodology**: faithfulness is typically operationalized via NLI (natural language
  inference) — decompose the generated answer into atomic claims, classify each as
  entailed/neutral/contradicted against the retrieved context, aggregate as a proportion.
- **Practical thresholds cited in industry practice**: groundedness < 0.80 → flag for
  human review; faithfulness < 0.70 → block before reaching a user. Useful as a reference
  point for Phase 2's reliability metrics, not something to adopt blindly without
  checking whether they fit this project's evaluation scale.
- Sources: [RAG Evaluation Metrics 2026](https://futureagi.com/blog/rag-evaluation-metrics-2025/),
  [Openlayer: Groundedness/Faithfulness/Retrieval Quality](https://www.openlayer.com/blog/rag-pipeline-evaluation-groundedness-faithfulness),
  [CCRS: Zero-Shot LLM-as-a-Judge RAG Evaluation](https://arxiv.org/pdf/2506.20128).

## Calibration measurement

- **Expected Calibration Error (ECE)** is the standard metric — average gap between
  predicted confidence and observed accuracy across confidence bins. Confirms Phase
  2 protocol's planned calibration metric is the right standard choice, not something to
  reinvent.
- A medical-VQA calibration study found combining multi-strategy prompting with an
  auxiliary judge model reduced ECE by ~40% — another data point suggesting a
  judge/refinement step (not just a single VLM call) is a recurring theme in what
  actually improves multimodal diagnostic reliability. Source:
  [Confidence Calibration for Multimodal LLMs: Medical VQA study](https://link.springer.com/chapter/10.1007/978-3-032-04978-0_9).

## Graduated diagnosis scoring (vs. binary correct/incorrect)

A medical LLM-as-judge methodology uses a 3-tier rubric instead of binary accuracy:
**exact match** (10 pts) / **clinically relevant differential** (5 pts, e.g. correctly
identifying "fungal leaf spot disease" without the exact pathogen) / **complete miss** (0
pts). This is directly adaptable to our disease-diagnosis scoring — several of our own
Phase 1B results (e.g. Qwen3-VL correctly saying "fungal leaf spot disease" when the
truth was a specific rust) would score partial credit under this scheme instead of
counting as flat failures, which is arguably a fairer reflection of real diagnostic
usefulness. Source: [LLM-as-a-Judge in Healthcare](https://arxiv.org/pdf/2605.25273).

## What this changes about the Phase 2 plan

1. **Before running the full 5-arm ablation**, it's worth testing whether a CPJ-style
   caption-refinement step closes some of the Q2 gap found in Phase 1B — cheap to test
   (reuses the exact same 4 hard cases and infrastructure already built), and if it
   works, it changes what "Prompt C" as the baseline query strategy should actually mean
   going into the ablation.
2. **Groundedness/faithfulness** should follow the RAG-Triad/NLI-decomposition approach
   rather than a single ad-hoc "supported: yes/no" judgment.
3. **Diagnosis accuracy scoring** should consider the 3-tier graduated rubric instead of
   pure binary correct/incorrect, alongside the existing plan to also track retrieval
   success and diagnostic success as separate outcomes.
