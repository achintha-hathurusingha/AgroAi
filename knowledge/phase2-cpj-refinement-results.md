# CPJ-Style Caption Refinement Test — Results

Tested the idea from [phase2-research-findings.md](phase2-research-findings.md) (CPJ paper,
arxiv 2512.24947): give Qwen3-VL a chance to revise its own caption when a judge scores
it below threshold, instead of accepting single-shot output like every prior Q2
experiment did. Same 4 hard cases as the bottleneck diagnosis, same corpus, same
BGE-micro-v2 scoring. Raw data: [results/cpj_refinement_results.json](results/cpj_refinement_results.json).
Script: [server-scripts/cpj_refinement_test.py](server-scripts/cpj_refinement_test.py).

## Result: real improvement where it triggered, but not enough, and the gate itself is unreliable

| Case | Initial rank | Final rank | Refinements | Judge score (initial) |
|---|--:|--:|--:|--:|
| Cedar apple rust | 2,165 | 2,165 | 0 | 8.0 (met threshold — no refinement attempted) |
| Early blight | 267 | **100** | 1 | 7.0 (triggered refinement) |
| Bacterial leaf spot | 26 | 26 | 0 | 8.0 (met threshold — no refinement attempted) |
| Powdery mildew | 80 | **34** | 1 | 7.0 (triggered refinement) |

## Finding 1: when refinement actually happens, it works — 2.4–2.7x rank improvement

For the two cases where the judge score fell below the 8.0 threshold and refinement
triggered, the refined caption ranked meaningfully better: early blight 267→100 (2.7x),
powdery mildew 80→34 (2.4x). The refined captions are visibly more specific — e.g. early
blight's refinement added "lesions concentrated along the midrib," "coalescing into
larger patches," and explicitly noted the *absence* of fungal sporulation/oozing, which
is itself diagnostically useful information the first-pass caption omitted entirely. This
validates that the core CPJ idea has real value, not just a paper claim.

## Finding 2: the self-judging threshold gate is unreliable — it let through the worst case in the whole project

Cedar apple rust — the single hardest case throughout every experiment in this
project — got a judge score of **8.0 (above threshold, no refinement triggered)**, even
though the judge's own qualitative feedback in the same response explicitly named a
missing diagnostic detail: *"does not mention the reddish-brown coloration of the leaf's
veins or the presence of small, scattered necrotic spots that appear to be clustered near
the leaf's midrib."* The numeric score and the judge's own stated criticism are
inconsistent — a generous self-scoring bias, unsurprising since the same model is judging
its own output, but it means **the exact case that most needed refinement never got a
second attempt.** This is a real limitation of using one VLM as both captioner and judge,
not a minor implementation detail — worth fixing (e.g. a stricter threshold, or gating on
whether the judge's feedback field is non-empty rather than trusting its own numeric
score) before relying on this mechanism further.

## Finding 3: even the successful refinements aren't good enough for practical retrieval

Rank 100 and rank 34 are real improvements, but neither would appear in a Top-5, Top-10,
or even Top-20 retrieval result in a real system. The refinement loop helps, but on this
evidence it's a mitigation, not a solution to Q2 — consistent with the earlier bottleneck
diagnosis finding that no single intervention tested so far (hybrid feature-injection,
oracle species, and now iterative refinement) gets any of the genuinely hard cases to a
usable rank on its own.

## What this means for Phase 2

- Worth keeping as a documented technique with real, measured benefit — but not a
  justification to delay Phase 2 further chasing a fully-solved Q2, since even a fixed
  version of this (stricter/more reliable gating) would likely still leave cedar
  apple rust and similar hard cases short of a usable rank.
- If pursued further, the fix to test first would be the judge-gating logic itself, not
  another prompt variant — e.g. requiring the judge to output a boolean "has_missing_detail"
  alongside the score, and refining whenever that's true regardless of the numeric score.
- Cedar apple rust's defining feature (the orange gelatinous tube structures) has never
  been mentioned by Qwen3-VL in any experiment across this entire project — not in
  Prompts A/B/C/D/E, not in the hybrid/oracle-species diagnosis, not here. That's worth
  flagging on its own: this may not be a prompting problem at all, but a case the model's
  visual representation genuinely can't resolve from this particular image, regardless of
  how it's asked to describe it.
