# Phase 1B — Q1/Q2 Embedding + Retrieval-Query Experiment (FROZEN)

**Status: FROZEN, approved for execution.** Reviewed against the actual 32 hand-authored
queries; 4 minor wording corrections applied (D1-Q5 "nearby" removed, D3-Q5 "starting
from" → "associated with" cankers, D6-Q5 narrowed to potato-only to match the selected
PlantVillage domain, D7-Q5 "stone fruit" → "peach and other stone fruit"). No further
changes to the disease list, pool construction, or query text once results start coming
in — per the explicit no-post-hoc-adjustment rule agreed on to avoid experimenter bias.

Note on scope: this experiment resolves Q1 (embedding model choice) directly via the
Recall@k/MRR numbers, and gives evidence toward Q2 (retrieval query construction) via the
per-query-type breakdown — but doesn't fully settle Q2 on its own, since the real system
still needs a decision on what Qwen3-VL's actual retrieval query should look like
(disease candidate alone vs. visual description vs. a combination). That remains a
downstream decision informed by, not automatically solved by, these results.

Full machine-readable spec: [server-scripts/q1q2_query_spec.json](server-scripts/q1q2_query_spec.json).

## Methodological corrections incorporated (per your review)

1. **Ground truth is declared, not inferred.** Every query has a `target_disease` field
   set explicitly in the spec *before* any query text was written from it. Evaluation
   code must look up `target_disease` directly — it must never parse or infer the target
   from `query_text`. This is what makes the D7 naming-mismatch test (query says
   "bacterial spot," `target_disease` is the AgMMU string "bacterial leaf spot") a valid
   test of embedding robustness rather than a data-leak that happens to work.
2. **Exact vs. semantic relevance kept separate.** The quantitative metric uses only
   exact `disease` field string match (case-insensitive) between pool facts and
   `target_disease` — objective and reproducible. Cases where a human might judge a
   different disease string as related (e.g. "apple rust" vs "cedar apple rust" showing
   up as a nearby distractor) get noted during qualitative inspection, never folded into
   the quantitative ground truth.
3. **Full retrieval records saved, not just ranks** — every top-5 result logged as
   `{query_id, embedder, rank, fact_id, similarity, disease}` so score distributions can
   be examined later, not just pass/fail.
4. **Automated scoring, qualitative inspection afterward** — Recall@1/@3/@5 and MRR
   computed programmatically from the saved records; manual eyeballing of a subset
   happens only after the numbers are in, as a separate qualitative pass.

## Disease set (unchanged from proposal, 8 diseases)

| # | Target disease (exact string used for ground truth) | Pool count (diagnostic corpus) | PlantVillage image class | Why included |
|---|---|--:|---|---|
| 1 | cedar apple rust | 25 | `Apple___Cedar_apple_rust` | real baseline failure case (already tested) |
| 2 | apple scab | 94 | `Apple___Apple_scab` | strong image reference, high volume |
| 3 | black rot | 28 | `Apple___Black_rot` | second apple disease, distinct symptoms |
| 4 | powdery mildew | 308 | `Cherry_(including_sour)___Powdery_mildew` | highest-volume case |
| 5 | septoria leaf spot | 66 | `Tomato___Septoria_leaf_spot` | different species (tomato) |
| 6 | early blight | 24 | `Potato___Early_blight` | different species (potato) |
| 7 | bacterial leaf spot | 25 | `Peach___Bacterial_spot` | **naming-mismatch stress test** |
| 8 | fire blight | 87 | *none* | **no-image stress test — pure text retrieval** |

## Query set — 47 total, frozen in `q1q2_query_spec.json`

6 query types × 8 diseases, minus the `vlm_caption` type for fire blight (no source
image) = 47:

1. `disease_name` — the literal disease name (trivial, sanity-check baseline)
2. `symptoms` — hand-authored symptom description
3. `species_symptoms` — species + symptom description
4. `description_no_name` — fuller natural sentence, disease name deliberately omitted
5. `management` — how-to-manage phrasing, disease name omitted
6. `vlm_caption` — filled in at execution time from Qwen3-VL's actual output on a real
   sampled image of that class (not written in advance — this is the point: it captures
   genuine model noise, not my clean phrasing)

The 32 hand-authored queries (types 2–5 × 8 diseases) are the ones that need your
review — they encode plant-pathology claims I wrote from general knowledge (standard,
well-documented symptoms for these 8 common diseases), and if any is inaccurate it
corrupts that query's result, not just a cosmetic issue. Please read through
`q1q2_query_spec.json` and flag anything that looks wrong before I run this.

## Sample pool construction (~400 facts)

1. For each of the 8 target diseases: dedupe matching diagnostic-corpus facts (via the
   same fingerprint method from the Phase 1A inspection), cap at 20 unique facts per
   disease → up to 160 target facts.
2. Add ~240 random deduped distractor facts from the diagnostic corpus, **excluding**
   entries whose disease matches any of the 8 targets → ~400 total pool.
3. Pool composition (fact_id, disease, full canonical record) gets saved alongside
   results for reproducibility — the exact same 400 facts get used for all three
   embedders, so the comparison is apples-to-apples (pun noted).

## Execution pipeline

```
1. Build sample pool (target facts + distractors) — frozen, saved to disk
2. For each of 7 diseases with a PlantVillage class: sample one real image,
   run Qwen3-VL, record its raw output as that disease's Q6 query text
3. Embed all 47 queries + all ~400 pool facts, separately for each of 3 embedders:
   - sentence-transformers
   - TaylorAI/bge-micro-v2
   - SigLIP text tower
4. For each embedder × each query: cosine similarity against all pool facts,
   take top-5, log {query_id, embedder, rank, fact_id, similarity, disease}
5. Score: Recall@1/@3/@5, MRR -- per embedder overall, and per embedder × query_type
6. Qualitative pass: manually inspect a subset of results, note any semantic-but-not-
   exact relevance cases (e.g. related disease strings) separately from the score
```

## What this resolves

- **Q1** (embedding model): decided from the Recall@k/MRR table across all three candidates.
- **Q2** (retrieval query construction): the per-query-type breakdown directly shows
  whether a clean disease-name guess, a fuller description, or genuine noisy VLM output
  retrieves best — informing how the real pipeline should construct its retrieval query.

## Before running

Confirm the 32 hand-authored queries in `q1q2_query_spec.json` are accurate, or correct
them. After that confirmation, the design is frozen — no changes to disease list, pool
construction, or query text once results start coming in.
