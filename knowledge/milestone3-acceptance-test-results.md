# Milestone 3 — Acceptance Test Result

First complete, real end-to-end run of the pipeline: `Image → Qwen3-VL (Prompt C) →
BGE-micro-v2 → NumPy exact cosine over the 17,583-fact deduplicated diagnostic corpus →
Top-5`. Pipeline module: [server-scripts/agrivision_pipeline.py](server-scripts/agrivision_pipeline.py),
test: [server-scripts/milestone3_acceptance_test.py](server-scripts/milestone3_acceptance_test.py),
raw output: [results/milestone3_acceptance_result.json](results/milestone3_acceptance_result.json).

**Result: the acceptance test failed.** The pipeline ran mechanically end-to-end without
errors, but did not retrieve the correct disease in the Top-5 for the test image.

## What happened

Test image: the same Cedar Apple Rust PlantVillage image used from the very first
real-data test in this project.

Generated query (Prompt C):
```
Species: unknown
Visible symptoms:
- Leaf is green with irregular brown spots
- Some spots appear necrotic or discolored
- Leaf margin is slightly curled or irregular
- Veins are visible and appear unaffected
- Leaf is attached to a short petiole
```

Top-5 retrieved (none correct):

| Rank | Disease | Species | Similarity |
|--:|---|---|--:|
| 1 | leaf spot | marigold | 0.8428 |
| 2 | phyllosticta leaf spot | japanese maple | 0.8314 |
| 3 | impatiens necrotic spot | primrose | 0.8248 |
| 4 | leaf spot disease | tree | 0.8230 |
| 5 | fungal leaf spots | laurel | 0.8225 |

Ground truth (cedar apple rust) did not appear anywhere in the Top-5.

## This is not a new failure — it's the same known limitation, now confirmed at full scale

Cedar apple rust has now failed retrieval in **every single query variant tested across
all of Phase 1B**: the original open-ended caption, Prompt B, Prompt C, Prompt D (all in
the isolation experiment), and Prompt E — and now again here, in the real full-scale
pipeline (17,583 candidate facts, not the small ~400-fact toy pool used in earlier
experiments). This is the specific, previously-identified limitation from
[phase1b-q2-isolation-results.md](phase1b-q2-isolation-results.md): the generated
description ("brown spots," "necrotic," "irregular") is accurate but generic — it never
mentions the disease's actual distinguishing visual feature (orange-yellow coloring,
orange gelatinous tube-like structures on the leaf underside). The retrieval mechanism is
working correctly — it found other genuinely "spot"-themed diseases with high textual
similarity — it just had no way to distinguish cedar apple rust from marigold leaf spot,
japanese maple phyllosticta leaf spot, etc. from the information actually present in the
query. At full corpus scale there's more textually-similar competition (many more "leaf
spot" facts across many species) than the small toy pool had, which likely makes this
specific failure mode even more visible than it was in the smaller-scale experiments.

## What this does and doesn't mean

- **Does not mean** the pipeline is broken — mechanically, every step (query generation,
  embedding, retrieval, ranking) executed correctly and the retrieved results are
  sensible given the query's actual content.
- **Does mean** this specific acceptance-test image is, based on everything measured so
  far, the single hardest case encountered in this project's data — not representative of
  typical pipeline performance. Every other disease tested with Prompt C earlier (powdery
  mildew, bacterial spot, black rot) succeeded, several at rank 1.
- **Open question, not decided here**: whether to also run the acceptance test on a case
  known to succeed (e.g. powdery mildew or bacterial spot, both rank-1 with Prompt C
  earlier) to confirm the pipeline works correctly end-to-end when the case isn't
  inherently hard — since relying on only the hardest known case risks concluding "the
  pipeline doesn't work" when the more precise conclusion is "the pipeline works, but
  this particular disease's retrieval-relevant visual signature is one Qwen3-VL
  consistently fails to capture, regardless of prompt strategy tried so far."
