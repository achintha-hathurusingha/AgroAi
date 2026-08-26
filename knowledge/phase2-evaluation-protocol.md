# Phase 2 Evaluation Protocol (FROZEN)

Written to close Phase 1 per the final checklist: freeze corpus, audit leakage, freeze
evaluation protocol. This is the frozen protocol for the main VLM-vs-RAG ablation. All
sub-decisions, including ground-truth source (Q4), are now confirmed and closed — see
below and [phase-1-plan.md](phase-1-plan.md). Updated after Phase 2 research
([phase2-research-findings.md](phase2-research-findings.md)) to incorporate graduated
diagnostic scoring and the full RAG evaluation dimensions (faithfulness, groundedness,
context relevance) rather than a single ad-hoc "supported: yes/no" judgment.

## Ablation arms (unchanged from the original scope)

1. VLM only
2. VLM + RAG
3. VLM + RAG + SigLIP retrieval
4. VLM + RAG + structured knowledge
5. VLM + RAG + structured knowledge + reranking

**Q3 decision, per this closure pass:** no reranker in the Phase 1 baseline arms (1–4).
Reranking is arm 5 specifically, evaluated in Phase 2 as its own ablation step, not
folded into the baseline. This also reflects the bottleneck diagnosis finding — since
retrieval failures often placed the correct fact thousands of positions down, a reranker
over a limited candidate set can't be assumed to fix that, so its actual contribution
needs to be measured on its own rather than assumed.

## Primary metric

**Disease identification accuracy**: correct disease predictions / total cases, compared
across the five arms above. Stays primary — graduated scoring (below) is secondary
analysis, not a replacement, so the headline number stays simple and comparable.

$$
\text{Accuracy} = \frac{\text{correct disease predictions}}{\text{evaluated cases}}
$$

## Graduated diagnostic scoring (secondary analysis)

Binary correct/incorrect treats every wrong answer as equally wrong, which loses
information a real evaluation should keep — "predicted a different fungal leaf-spot
disease" and "predicted a nutrient deficiency" are very different failure modes for the
same wrong-disease outcome. Adopted from medical LLM-as-judge methodology
([phase2-research-findings.md](phase2-research-findings.md)):

| Tier | Meaning |
|---|---|
| Exact match | Correct disease |
| Clinically relevant differential | Wrong exact disease but diagnostically relevant/closely related (same pathogen family, overlapping symptom presentation, etc.) |
| Complete miss | Incorrect and not diagnostically useful |

Reported alongside, not instead of, primary accuracy.

## Retrieval metrics (evaluates the retrieval component independently of final diagnosis)

- Recall@1, Recall@3, Recall@5 (kept from Phase 1B for continuity)
- MRR

Retrieval success and diagnostic success are tracked as **separate, non-redundant
outcomes** — a correct fact can be retrieved and the VLM can still diagnose incorrectly
from it, and the VLM can diagnose correctly without the retrieved evidence containing the
exact matching disease fact. Both cases get recorded distinctly, not collapsed into one
pass/fail number.

## RAG-quality dimensions (a correct diagnosis does not imply good RAG)

A correct final diagnosis and good retrieval-augmented reasoning are not the same
achievement — the model can land on the right disease with irrelevant retrieved evidence,
or produce the wrong diagnosis despite well-retrieved evidence. Following the RAG
Triad/RAGAS framework ([phase2-research-findings.md](phase2-research-findings.md)):

- **Context relevance**: is what was retrieved actually relevant to the query/image?
- **Faithfulness**: decompose the generated diagnosis into atomic claims, classify each
  against the retrieved context as entailed / neutral / contradicted; report as the
  proportion of entailed claims. This replaces a single ad-hoc "is it supported: yes/no"
  judgment with something closer to auditable, per-claim scoring.
- **Groundedness**: does the final diagnosis, taken as a whole, actually follow from the
  retrieved evidence — the aggregate judgment faithfulness's claim-level scores feed into.

Reference thresholds from industry practice (not adopted blindly, but useful calibration
points): groundedness < 0.80 typically flagged for review, faithfulness < 0.70 typically
treated as a blocking failure.

## Reliability metrics

- Hallucinated disease/species claims (unsupported by the image or retrieved evidence)
- Confidence calibration: **Expected Calibration Error (ECE)** — average gap between
  stated confidence and observed accuracy across confidence bins, where the system reports
  a confidence score
- Abstention/uncertainty handling, where applicable

## Query-generation mechanism: Prompt C only — CPJ refinement is exploratory, not baseline

Per [phase2-cpj-refinement-results.md](phase2-cpj-refinement-results.md): a CPJ-style
judge-and-refine loop showed real benefit where it triggered (2.4–2.7x rank improvement
on 2 of 4 tested cases), but the self-judging threshold gate proved unreliable — it let
the single hardest case in the project through unrefined despite the judge's own written
feedback naming the missing detail, and even the successful refinements landed well short
of a usable rank. **Decision: Phase 2's baseline query generation is plain Prompt C,
single-shot, no refinement loop.** CPJ-style refinement is documented as a validated-but-
unreliable exploratory mitigation, not adopted into the baseline — worth revisiting only
if Phase 2 results show retrieval-query quality is clearly the limiting factor on RAG's
overall benefit, at which point it becomes a targeted follow-up experiment (starting with
fixing the judge-gating logic, not another prompt variant).

## Ground-truth source — FROZEN

**Q4 closed.** A hierarchical policy, not a blended pool — chosen specifically to keep
evaluation truth cleanly separated from AgMMU, which is simultaneously the retrieval
knowledge source (using AgMMU's own eval annotations as the *primary* truth would blur
that separation for exactly the cases where it matters most).

| Situation | Ground truth | Role |
|---|---|---|
| Evaluation image has a valid PlantVillage disease class | PlantVillage | **Primary** |
| No appropriate PlantVillage class exists | AgMMU evaluation annotation | Secondary |
| Both exist and disagree | — | **Flagged for adjudication, never silently resolved** |
| Neither provides a defensible disease label | — | **Excluded from the accuracy denominator, reported separately** |

**Non-negotiable rule**: the ground-truth label is fixed *before* inference and is never
derived from the model's prediction, the retrieved fact, or the query text — consistent
with the same principle already applied to `target_disease` throughout Phase 1B (declared
independently of query text, never inferred from it).

**Provenance tracking**: every evaluation case records `ground_truth_source` as either
`"plantvillage"` or `"agmmu_eval"`. Results are reported both overall and broken out by
source:

| Subset | Accuracy |
|---|--:|
| PlantVillage ground truth | — |
| AgMMU ground truth | — |
| Overall | — |

This exists specifically to preempt the legitimate question of how much reported accuracy
depends on labels drawn from the same ecosystem as the retrieval knowledge base.

## Retrieval success vs. diagnostic success stay separate outcomes

Already established as a design principle above — restated here because it's essential to
interpreting the ablation correctly: retrieving the correct fact and diagnosing correctly
are not the same event. A case can have `retrieval=success, diagnosis=failure` (the right
fact was retrieved but the VLM still got it wrong) or `retrieval=failure, diagnosis=success`
(the VLM diagnosed correctly without the exact fact being retrieved). Both get recorded as
distinct, valid outcomes — never collapsed into a single pass/fail number.

## Leakage audit (performed as part of this closure pass)

- **Image leakage**: hashed every PlantVillage image (used as query images) against every
  downloaded AgMMU eval image (part of the knowledge-corpus source data) — see
  [leakage-audit-results.md](leakage-audit-results.md) for the result.
- **Query-construction leakage**: code-reviewed `agrivision_pipeline.py`'s
  `generate_prompt_c_query(model, processor, image)` — takes only `model`, `processor`,
  `image`; the fixed `PROMPT_C` template contains no ground-truth placeholder. Confirmed
  across call sites (`milestone3_acceptance_test.py`, `milestone3_control_test.py`) that
  `GROUND_TRUTH_DISEASE` is defined but only ever used *after* query generation, for
  scoring — never passed into the generation call. No leakage path found in the pipeline
  code as written.

## Frozen corpus reference

Phase 2 must use the frozen corpus version, not re-derive it from the raw AgMMU file —
see [phase2-corpus-freeze.md](phase2-corpus-freeze.md) for the version identifier,
checksums, and exact counts. If corpus content ever needs to change, that requires a new
version identifier and re-running any experiment being compared against it, specifically
to avoid the "did the model improve or did the corpus change" ambiguity this freeze
exists to prevent.
