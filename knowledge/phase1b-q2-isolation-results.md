# Phase 1B — Q2 Caption-Isolation Experiment Results

Follow-up to [phase1b-q1q2-results.md](phase1b-q1q2-results.md), isolating why the
original `vlm_caption` queries scored 0% Recall@1 across all three embedders. Same 7
PlantVillage images as the original experiment, three new prompt variants, scored with
BGE-micro-v2 (the Q1 winner) against the same frozen pool. Raw artifacts:
[results/q2_isolation_results.json](results/q2_isolation_results.json) (full caption
text), [results/q2_isolation_scored.json](results/q2_isolation_scored.json) (retrieval detail).

**One deviation from the originally proposed design, flagged rather than silently made:**
the planned "corrected caption" variant (manually strip the wrong species claim from the
original Prompt A caption, keep everything else) turned out not to be meaningful for
most of the 7 cases — re-reading the full Prompt A captions showed most were truncated
by the 128-token limit *before* reaching any real symptom content (e.g. D1 cuts off
mid-word at "reddish-brown to"; D2/D3/D4/D7 cut off still mid-species-discussion, never
reaching symptoms at all). Stripping the wrong species clause from these would leave
almost nothing to keep. Ran the three cleaner isolation variants (B/C/D below) instead,
which test the same two hypotheses more directly.

## Prompt variants tested

- **B — visual-evidence-only**: explicitly forbids naming species or disease; asks only
  for shape/color/lesions/texture. 128 tokens (same budget as the original A).
- **C — structured, anti-inference**: forces `Species: unknown` plus a bulleted symptom
  list; explicitly told not to infer species/disease unless visually unambiguous. 128 tokens.
- **D — same wording as the original A prompt**, but 400 tokens instead of 128 — isolates
  the token-budget hypothesis from the prompt-wording hypothesis.

## Result 1: both hypotheses were real, but prompt wording is the dominant cause

| Variant | R@1 | R@3 | R@5 | MRR |
|---|--:|--:|--:|--:|
| Original Prompt A (open-ended, 128 tok) | 0.000 | 0.000 | 0.000 | 0.060 |
| B (visual-only, 128 tok) | 0.143 | 0.286 | 0.429 | 0.264 |
| **C (structured, 128 tok)** | **0.286** | **0.429** | 0.429 | **0.366** |
| D (same wording as A, 400 tok) | 0.143 | 0.429 | 0.429 | 0.312 |

All three variants substantially beat the original (0.000 R@1) — so both the prompt
wording and the token budget were genuinely contributing to the original failure. But
**C, which fixes only the prompt wording (still just 128 tokens), does best overall** —
wording matters more than budget.

**D (budget fix alone, same prompt as A) is inconsistent — it does not reliably stop the
hallucination.** Rereading the actual D captions: D3 correctly identified *Malus
domestica* (apple) this time and reached rank 1; but D1 now claims *Betula* (birch) —
a *different* wrong species than A's original *Sorbus* guess — and D7 still confidently
claims *Quercus* (oak). More budget lets the model produce more content, but it doesn't
reliably stop it from confidently committing to a wrong species early in that content.

## Result 2: a wrong species claim actively hurts retrieval, more than omitting it helps

D7 (actual: peach, bacterial spot) is the clearest single case:
- Prompt B (no species claimed): retrieved rank **1**
- Prompt C (no species claimed): retrieved rank **1**
- Prompt D (confidently claims "*Quercus* spp., most likely *Quercus robur*... oak"): retrieved rank **15**

Identical underlying image, identical target disease — the only difference is whether
the caption asserts a wrong species. This is fairly direct evidence that a confident
wrong claim is worse for retrieval than no claim at all, not merely neutral noise.

## Result 3: even the best variant is far below hand-authored query performance

C's 28.6% R@1 (n=7) is much lower than the hand-authored query types' 62.5–100% R@1 from
the main experiment. Two of the 7 cases (D1 cedar apple rust, D6 early blight) failed to
reach the top-5 in *any* of the three variants, including the best ones. Reading those
captions specifically:

- **D1 (cedar apple rust)**: all three variants described "reddish-brown spots" — accurate
  as far as it goes, but never mentioned the disease's actually distinctive visual feature
  (the orange gelatinous tube-like structures / aecia) that the hand-authored queries
  explicitly included. Generic "brown spots" language is common across many diseases in
  the pool, so it isn't distinctive enough to retrieve the right one.
- **D6 (early blight)**: all three variants described dark spots but — notably — Prompt D
  explicitly stated "There is no clear pattern to the distribution of the spots," which is
  the *opposite* of early blight's actual defining trait (concentric "target/bullseye" rings).
  Qwen missed the specific feature that would have made retrieval easy.

This points to a **second, distinct bottleneck beyond hallucination**: even an honest,
non-hallucinating caption still needs to happen to mention the disease's specific
diagnostic visual feature, not just generic symptom language, for text retrieval to
disambiguate it from similar-looking diseases in the corpus.

## What this does and doesn't resolve

- **Confirms**: an open-ended "describe the species and disease" prompt is a bad way to
  generate a retrieval query — it induces unreliable species hallucination that actively
  damages retrieval, worse than not claiming a species at all.
- **Confirms**: prompt wording change alone (forbid species/disease naming, force a
  symptom-focused structured or unstructured description) recovers most of the lost
  performance without needing a longer token budget.
- **Does not resolve**: how to close the remaining gap between real-VLM-caption
  performance (~14–29% R@1 even in the best variant) and hand-authored-query performance
  (62–100% R@1). Possible directions this experiment surfaces but doesn't test: prompting
  Qwen to specifically look for disease-diagnostic features rather than generic
  appearance; a multi-turn/iterative captioning approach; combining the caption with the
  candidate-disease-name pathway from the original architecture rather than using either
  alone. Which of these (if any) to pursue is still an open decision.
