# Phase 2 Extension — Stage A: Error Taxonomy

Classifies every wrong diagnosis in the `primary_confident_match` group (240 cases — the
group where retrieval is structurally able to help) using evidence already collected
across Experiments 1 and 2, rather than a new LLM-judge pass. No new model calls needed.
Script: [server-scripts/error_taxonomy.py](server-scripts/error_taxonomy.py). Full
classification: [results/phase2/error_taxonomy_full.json](results/phase2/error_taxonomy_full.json)
(all 240 cases). Stratified sample for qualitative review (52 cases):
[results/phase2/error_taxonomy_sample.json](results/phase2/error_taxonomy_sample.json).

## Taxonomy definition and how it's operationalized from existing data

| Category | Meaning | How it's determined |
|---|---|---|
| **Query failure** | A good query would have found the right evidence; the actual generated query didn't | Oracle retrieval (Experiment 2) *did* find the correct disease in its top-5 for this case, but the current pipeline's actual Arm-2 retrieval (using the real Qwen-generated query) did not |
| **Retrieval/corpus failure** | Not findable even with a perfect query | Oracle retrieval also failed to find the correct disease in its top-5 — the limitation is in the corpus/embedding, not the query |
| **Reasoning failure** | Correct evidence was retrieved, diagnosis still wrong | The current pipeline's actual top-5 retrieval *did* contain the correct disease, but the final diagnosis didn't use it correctly |
| **Hallucination** (overlay, any category) | Diagnosis not supported by the evidence | The faithfulness judge (already collected in the main ablation) marked `grounded: False` |
| **Calibration failure** (overlay, any category) | High confidence despite being wrong | Confidence ≥ 70 on an incorrect diagnosis |

Hallucination and calibration failure are independent, overlaid flags, not mutually
exclusive with the three primary categories — a case can be both a "query failure" and a
"calibration failure" simultaneously.

## Results: query failure dominates, decisively

| Category | Count | % of 240 |
|---|--:|--:|
| **Query failure** | **179** | **74.6%** |
| Retrieval/corpus failure | 45 | 18.8% |
| Correct | 9 | 3.8% |
| Reasoning failure | 7 | 2.9% |

**Three of every four disease-in-corpus cases fail specifically because the query
construction step failed to retrieve evidence that was findable and available.** This is
not a marginal contributing factor — it is the dominant failure mode by a wide margin,
more than 4x larger than the next category (corpus/retrieval limitations) and 25x larger
than reasoning failures. This matches the retrieval-decomposition finding exactly: the
18.8% retrieval/corpus failure rate here lines up almost precisely with Experiment 2's
finding that oracle retrieval itself only reaches ~81.2% R@1 on this subgroup (an ~18.8%
miss rate) — internal consistency between two independently-computed numbers.

## Overlay findings

| Overlay | Rate (among 231 wrong cases) |
|---|--:|
| Hallucination (not grounded) | 13.0% |
| **Calibration failure (confident despite wrong)** | **96.1%** |

Hallucination is a real but secondary issue. **Calibration failure is nearly universal**
— when the current pipeline gets a diagnosis wrong, it is confidently wrong (≥70%
stated confidence) 96.1% of the time. This is a much starker way of stating the ECE≈0.44
finding from the main ablation: it's not that confidence is loosely correlated with
correctness, it's that being wrong barely moves the model's stated confidence at all.

## Representative examples

**Query failure** (evidence existed and was findable, actual query missed it):
```
Apple scab       -> diagnosed "Phyllosticta leaf spot"   (confidence 75%)
Bacterial spot   -> diagnosed "Phyllosticta leaf spot"   (confidence 75%)
Powdery mildew   -> diagnosed "Mosaic leaf virus"         (confidence 70%)
```

**Retrieval/corpus failure** (even oracle retrieval couldn't find it):
```
Tomato mosaic virus              -> diagnosed "healthy"           (confidence 100%)
Cercospora/gray leaf spot (corn) -> diagnosed "Leaf spot disease" (confidence 70%)
```

**Reasoning failure** (correct evidence retrieved, still wrong — the rarest category):
```
Powdery mildew  -> diagnosed "anthracnose"      (confidence 70%)
Leaf scorch     -> diagnosed "Phyllosticta leaf spot" (confidence 70%)
```

**Correct cases** cluster heavily around one disease — every sampled correct case was
**squash powdery mildew**, the single highest-volume disease in the corpus (306 matching
facts, by far the most of any of the 8 originally-studied diseases) and one with a highly
visually distinctive symptom (white powdery coating). This suggests correctness here
tracks corpus density and visual distinctiveness, not a generally-working pipeline.

## Implications for architecture (per the plan: report only, don't act yet)

- **Query construction is confirmed as the primary lever**, quantitatively, not just via
  the oracle-ceiling comparison. Three-quarters of fixable failures are attributable to
  it specifically.
- **Corpus/retrieval limitations (18.8%) are a real but secondary ceiling** — even a
  perfect visual-first or hybrid retrieval approach would still be bounded by this,
  since it reflects cases where the disease isn't findable in the corpus at all, not a
  query-representation problem.
- **Reasoning failures are rare (2.9%)** — the final diagnosis step is not where the
  bottleneck lives, once given genuinely relevant evidence.
- **Calibration is broken almost everywhere the diagnosis is wrong**, independent of
  which of the three failure modes applies — worth keeping in mind that fixing query
  construction won't automatically fix trustworthiness/calibration, which appears to be
  a separate, pipeline-wide issue.

This directly and quantitatively supports moving to Stage B (Visual-First Retrieval) as
the next experiment, since the dominant, addressable failure mode (query failure, 74.6%)
is exactly what a different image-to-retrieval representation is meant to bypass.
