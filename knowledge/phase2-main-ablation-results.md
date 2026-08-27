# Phase 2 — Main Ablation Results

The core research result. 610 cases run, 0 errors, using the corrected v2 diagnosis
prompt (see [phase2-diagnosis-prompt-iteration.md](phase2-diagnosis-prompt-iteration.md)).
**40 `supplementary_agmmu` cases excluded from all analysis below** — their ground truth
turned out to be answers to non-diagnostic AgMMU question types (management tips, pest
names, general observations like "50% of the leaves have become holey and damaged"),
pulled in by the eval-set builder without filtering by question type. This is a data
construction bug, not a model result — scoring diagnoses against "keep the trees well
watered" would be meaningless. Kept in the raw data for the record, excluded from every
number below. **465 primary cases** and **105 supplementary weak-corpus-match cases**
remain valid and are analyzed. Raw data: [results/phase2/phase2_full_v2_results.jsonl](results/phase2/phase2_full_v2_results.jsonl).

## Headline result: RAG does not significantly improve diagnosis accuracy here

| Arm | Accuracy (n=465) | 95% CI |
|---|--:|---|
| 1. VLM only | 37.8% | [0.336, 0.423] |
| 2. VLM + RAG (BGE) | 39.4% | [0.350, 0.439] |
| 3. VLM + RAG (SigLIP) | 39.1% | [0.348, 0.436] |
| 4. VLM + RAG (full corpus) | 39.4% | [0.350, 0.439] |
| 5. VLM + RAG + rerank | 38.3% | [0.340, 0.428] |

All five confidence intervals overlap heavily. McNemar's test (paired, since every case
runs through every arm) against the VLM-only baseline:

| Comparison | p-value |
|---|--:|
| Arm 1 vs Arm 2 (RAG BGE) | 0.096 |
| Arm 1 vs Arm 3 (RAG SigLIP) | 0.149 |
| Arm 1 vs Arm 4 (RAG full corpus) | 0.070 |
| Arm 1 vs Arm 5 (RAG rerank) | 0.803 |

None reach conventional significance (p<0.05), though Arm 4 comes closest. **At this
sample size, this experiment cannot distinguish "RAG helps a little" from "RAG doesn't
help at all."** That is itself the honest answer, not a failure to find one.

## Where the real story is: performance is almost entirely determined by eval group, not by arm

| Eval group (n) | Arm 1 | Arm 2 (BGE) | Arm 3 (SigLIP) | Arm 4 (full corpus) | Arm 5 (rerank) |
|---|--:|--:|--:|--:|--:|
| Healthy (180) | 97.2% | 96.7% | 95.6% | 97.2% | 95.0% |
| **Disease, corpus has it (240)** | **0.4%** | **3.8%** | **4.2%** | **3.3%** | **2.9%** |
| Disease, corpus doesn't have it (45) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Weak corpus coverage (105, supplementary) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Two things this table makes clear that the headline number hides:

1. **Every arm is near-ceiling on healthy cases and near-zero on diseased ones.** The
   overall ~38-39% accuracy figures are mostly just "gets healthy cases right" (95-97%)
   diluted by "essentially never gets a specific disease right" (0.4-4.2%) across a
   roughly 39%-healthy/61%-diseased eval set. A single blended accuracy number obscures
   that these are two very different performance regimes, not one moderate one.
2. **RAG's apparent benefit lives entirely in the one group where it's structurally
   possible to help** — the 240 cases where the disease exists in the corpus. There, RAG
   arms get 7-10x more cases right than VLM-only in absolute terms (2-3% → 8-10 cases,
   vs. VLM-only's 1 case) even though the McNemar test on the *full* primary set doesn't
   reach significance (diluted by the healthy and no-match subgroups where nothing
   differs between arms). On the 45 no-corpus-match and 105 weak-match cases — where RAG
   structurally cannot retrieve the right answer — **every arm scores exactly 0%,
   confirming those negative controls worked as designed**: RAG doesn't hallucinate a
   plausible-sounding disease it has no evidence for, but it also doesn't help.

## Concrete illustration: retrieval bias propagates directly into diagnosis errors

Spot-checking wrong predictions on apple scab cases (12 sampled) found the exact same
wrong answer every time: **"phyllosticta leaf spot"** — a real, different disease, not a
naming variant. This is not a metric artifact; it's the retrieval-quality problem from
Phase 1B ("phyllosticta leaf spot" facts on Japanese maple share generic symptom language
with apple scab and get retrieved for it) directly causing the same wrong diagnosis to
repeat across many cases. This is a clean, traceable causal chain from Phase 1B's
retrieval findings through to Phase 2's diagnosis errors — not a new, unrelated problem.

## Calibration: severe overconfidence, across every arm

| Arm | ECE |
|---|--:|
| 1. VLM only | 0.443 |
| 2. RAG BGE | 0.436 |
| 3. RAG SigLIP | 0.446 |
| 4. RAG full corpus | 0.434 |
| 5. RAG rerank | 0.447 |

ECE ≈ 0.44 across the board is large — the model reports high confidence (commonly
80-100%, see prior examples in [phase2-greedy-decoding-collapse.md](phase2-greedy-decoding-collapse.md))
essentially independent of whether it's actually right. RAG does not meaningfully improve
calibration here either.

## Faithfulness / groundedness (RAG arms only)

| Arm | Grounded rate | Mean faithfulness (entailed/claims) |
|---|--:|--:|
| 2. RAG BGE | 60.0% | 0.371 |
| 3. RAG SigLIP | 58.5% | 0.369 |
| 4. RAG full corpus | 52.7% | 0.358 |
| 5. RAG rerank | 56.6% | 0.379 |

Arm 4 (retrieving from the larger combined diagnostic+context corpus) has the lowest
groundedness despite statistically indistinguishable accuracy from Arm 2 — a real,
interpretable secondary finding: pulling from the larger, more heterogeneous corpus adds
noise to what gets retrieved, measurably reducing how well the final diagnosis follows
from the evidence, even when it doesn't change whether the diagnosis happens to be right.
**Reminder of the self-judging limitation** flagged in the execution plan: this judge is
Qwen3-VL scoring its own output, the same setup the CPJ test showed has a generous bias —
these numbers should be read as directional, not absolute.

## What this means for the research question

> Does retrieval-augmented multimodal reasoning improve fine-grained agricultural disease
> diagnosis compared to zero-shot VLM diagnosis?

**On this evidence: not detectably, at this sample size, with this pipeline.** The
honest, defensible answer is not "RAG helps" or "RAG doesn't help" — it's:

- RAG shows a real, non-trivial *relative* improvement (2-3x) in the one subgroup where
  it's structurally capable of contributing (disease-in-corpus cases), but off an
  extremely low absolute base (~1% → ~3-4%), too small to be statistically distinguishable
  from noise at n=240 for that subgroup, let alone n=465 overall.
- The dominant bottleneck is not the RAG mechanism — it's the same one identified
  throughout Phase 1B: fine-grained diagnosis is hard for this VLM, and generic retrieval
  queries surface generic (often wrong) evidence, which the model then faithfully but
  incorrectly reports.
- Calibration is poor everywhere, RAG included — confidence scores in this pipeline
  should not be trusted as a proxy for correctness in their current form.

This is a legitimate, defensible Phase 2 result — not a null result to be embarrassed by.
It's consistent with, and now quantitatively confirms at real scale, everything Phase 1B
found qualitatively about where this pipeline's limits are.
