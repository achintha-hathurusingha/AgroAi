# Phase 2 Execution Plan

Written for overnight unattended execution. Every genuine judgment call below is flagged
explicitly as **[JUDGMENT CALL]** rather than silently decided, so it's reviewable in the
morning even though there wasn't time for the usual freeze-then-review cycle used in
Phase 1B. Where Phase 1 already decided something (Q1–Q6), it's referenced, not redecided.

## 1. Evaluation-set construction

Cross-referenced PlantVillage's 38 classes against the frozen `agmmu_phase2_v1`
diagnostic corpus's disease vocabulary (script:
[server-scripts/check_pv_corpus_coverage.py](server-scripts/check_pv_corpus_coverage.py),
data: [results/pv_corpus_coverage.json](results/pv_corpus_coverage.json)):

| Group | Classes | Meaning |
|---|--:|---|
| Confident corpus match (≥5 matching facts) | 16 | RAG can plausibly help — this is the main comparison group |
| Healthy | 12 | Tests false-positive/hallucinated-disease behavior, not retrieval |
| No corpus match (0 facts) | 3 (corn common rust, corn northern leaf blight, grape esca) | **Negative control** — RAG structurally cannot help; tests whether RAG arms correctly fail gracefully vs. hallucinate a plausible-sounding wrong answer |
| Weak/unreliable match (1–4 facts, several hedged) | 7 | **[JUDGMENT CALL] Excluded from primary analysis** — matches like "possibly tomato yellow leaf curl virus" (count=1) aren't reliable enough to treat as ground-truth-corpus alignment; reported as a separate supplementary group, not part of headline numbers |

**Sampling**: **[JUDGMENT CALL]** 6 images per class (fixed seed=42, deterministic,
reproducible) across the 16+12+3=31 primary classes → **186 primary cases**. The 7
weak-match classes get the same treatment as a labeled supplementary set, not mixed into
primary numbers. Images already used in prior Phase 1 hard-case experiments (cedar apple
rust etc.) are excluded from this sample to avoid overlap with cases the pipeline was
implicitly tuned against — reported as a small separate "previously-studied" group for
continuity instead.

**Supplementary generalization check**: **[JUDGMENT CALL]** also draw 20 cases from
AgMMU's own 770-entry eval image set (`copied_images/`), to actually exercise the Q4
ground-truth hierarchy's secondary path (PlantVillage-primary is otherwise vacuous when
every eval image *is* a PlantVillage image) and test whether findings generalize beyond
the PlantVillage image distribution.

Ground truth per case recorded per the frozen Q4 policy
([phase2-evaluation-protocol.md](phase2-evaluation-protocol.md)): `ground_truth_source =
"plantvillage"` for the 186 primary cases, `"agmmu_eval"` for the 20 supplementary ones.

## 2. Five-arm architecture

The original architecture doc's arms 3–4 ("SigLIP retrieval," "structured knowledge")
predate the Q1/Q5 decisions and no longer map cleanly onto what was actually built —
**[JUDGMENT CALL]**, redefined to use only validated Phase 1 components and to test
questions Phase 1 raised but didn't answer at the diagnosis level (Q1 only measured
retrieval metrics, never final diagnosis accuracy):

| Arm | Pipeline | Tests |
|---|---|---|
| 1. VLM only | Image → Qwen3-VL → Diagnosis | Baseline |
| 2. VLM + RAG (BGE) | Image → Qwen3-VL (Prompt C) → BGE embed → NumPy top-k over diagnostic corpus (17,583) → Qwen3-VL + evidence → Diagnosis | Main treatment |
| 3. VLM + RAG (SigLIP) | Same as 2, but SigLIP text tower instead of BGE for embedding/retrieval | Does Q1's retrieval-only embedder result (BGE > SigLIP) actually translate to better *diagnosis*, not just better Recall@k? |
| 4. VLM + RAG (full corpus) | Same as 2, but retrieval over diagnostic+context combined (35,507 facts) instead of diagnostic-only | Does the context corpus (kept-not-deleted per the Phase 1A decision) help or add noise — the experiment that decision explicitly deferred |
| 5. VLM + RAG + reranking | Same as 2, but retrieve top-20 then rerank to top-k | Q3, evaluated on its own merits per the Phase 1 closure decision |

## 3. Inference procedure

- **Decoding**: **[JUDGMENT CALL]** greedy (`do_sample=False`) for the main run, not the
  sampling used throughout Phase 1B — reproducibility matters more here than the
  natural-variance information sampling would add, and Phase 1B already established that
  re-running with sampling produces meaningfully different outputs (a confound we don't
  want inside the main ablation).
- **Query generation** (arms 2–5): one Prompt C call per case, **shared across arms 2–5**
  (retrieval backend/corpus differs, not the query) — cuts redundant VLM calls roughly in
  half versus generating separately per arm.
- **Final diagnosis prompt** (all arms): structured output —
  ```
  Diagnosis: <disease name, or "healthy" or "unknown">
  Confidence: <0-100>
  Reasoning: <one to two sentences>
  ```
  Arms 2–5 additionally receive the retrieved evidence text before this instruction; arm
  1 does not.

## 4. What information each arm sees

- Arm 1: image + diagnosis-prompt instruction only. No retrieval, no evidence.
- Arms 2–5: image + diagnosis-prompt instruction + top-k retrieved evidence text. The
  query-generation step (shared) sees **only the image** — no ground truth, consistent
  with the leakage audit already performed.
- No arm ever sees `ground_truth_disease` at any stage before scoring — enforced by the
  same code-structure principle verified in the leakage audit (score-only access, never
  passed into a generation call).

## 5. Retrieval Top-k

**[JUDGMENT CALL]** k=5 as primary (matches the original architecture's stated design and
the R@5 metric already tracked throughout Phase 1B), k=10 as a secondary comparison —
cheap to also compute since retrieval itself is sub-millisecond (Q6 result); only the
generation call's context length changes. Arm 5's reranking operates on a top-20 initial
candidate pool, reranked down to top-k.

## 6. Ground-truth mapping

Per the frozen Q4 policy — see [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md).
For this eval set specifically: 186 primary cases use PlantVillage folder labels
(species + disease, "healthy" where applicable); 20 supplementary cases use AgMMU eval
annotations. No case in this design triggers the "both exist and disagree" adjudication
path, since primary and supplementary cases are drawn from disjoint sources by
construction — worth noting as a limitation of this particular eval-set design, not a
general property of the Q4 policy.

## 7. Hallucination / groundedness scoring

**[JUDGMENT CALL, with a known limitation flagged upfront]**: no external judge model is
available (no GPT-4/GPT-5 access), so Qwen3-VL is used as its own judge — the same setup
that the CPJ refinement test already showed produces an unreliable self-scoring bias
(the judge let the hardest case in the project through unrefined despite naming the
missing detail itself). This limitation is **inherited into Phase 2's groundedness/
hallucination numbers** and should be read with that caveat, not treated as
ground-truth-quality judgment.

- **Faithfulness**: for arms 2–5, a judge call decomposes the diagnosis's `Reasoning`
  field into atomic claims, classifies each against the retrieved evidence text as
  entailed/neutral/contradicted, reports proportion entailed — per the RAG-Triad/NLI
  methodology in [phase2-evaluation-protocol.md](phase2-evaluation-protocol.md).
- **Hallucination flag**: a claim classified as contradicted by both the retrieved
  evidence *and* not verifiable from the image description counts as hallucinated.
- **Groundedness**: aggregate judgment from the same judge call — does the diagnosis as a
  whole follow from the evidence.

## 8. ECE / calibration computation

Diagnosis prompt requires a `Confidence: 0-100` field (all arms). Cases are binned into
10 confidence deciles; ECE = weighted average of |mean confidence − accuracy| across bins,
weighted by bin population. Computed separately per arm.

## 9. Statistical comparison between arms

- **Paired accuracy comparison** (same cases run through every arm): **McNemar's test**
  for each pairwise arm comparison — appropriate for paired binary outcomes, avoids the
  independence assumption a plain two-proportion z-test would wrongly make here.
- **Confidence intervals**: Wilson score interval for each arm's accuracy (more reliable
  than a normal approximation at this sample size).
- **Retrieval metrics** (R@k, MRR, arms 2–5 only): bootstrap confidence intervals (1000
  resamples) rather than a closed-form formula, for consistency with the accuracy CIs.

## 10. Compute / storage requirements

- Estimated ~10 VLM calls per primary case (1 shared query-gen + 5 diagnosis calls across
  arms + 4 judge calls for arms 2–5) at an estimated 10–20s/call on the RTX 4090 (4-bit
  Qwen3-VL, consistent with Phase 1B timings) → roughly 30–60s/case → **186 cases ≈
  1.5–3 hours**, well within an overnight window, with real throughput to be confirmed by
  the pilot run below before committing to the full 206 (186+20).
- Storage: negligible — images already local, corpus already frozen, results as JSONL
  (append-only, one line per completed case) plus a final summary JSON. No new large
  artifacts expected (<50MB total).

## 11. Acceptance criteria and failure handling

Given Phase 1B's repeated experience of implementation surprises (JSON shape bugs, scale
effects, judge inconsistency) that only showed up when things were actually run, this
overnight run is **not** a single blind full-scale execution:

1. **Pilot run first**: 10 cases through all 5 arms end-to-end, inspected manually before
   scaling up. Acceptance: no crashes, per-case latency within 2x of the estimate above,
   output fields parse correctly (`Diagnosis`/`Confidence`/`Reasoning`), judge output
   parses correctly. If the pilot fails any of these, stop and fix before the full run —
   not "fix and continue in place," since a bug caught 10 cases in is far cheaper than one
   caught 150 cases in.
2. **Checkpointing**: results appended to a JSONL file per-case as they complete, not
   held in memory until the end — a crash at case 150 of 186 still leaves 150 cases of
   usable, analyzable data, and the run is resumable from the last completed case.
3. **Per-case error isolation**: a single case's exception (OOM, malformed output,
   generation timeout) is caught, logged with the case ID and traceback, and the run
   continues — never let one bad case kill the whole overnight job.
4. **Morning review checklist**: how many cases completed vs. failed, pilot-run sanity
   check results, any judgment calls above that need revisiting given what the pilot or
   partial full run actually showed.
