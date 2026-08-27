# Experiment 1 — Oracle Retrieval: Full Results (465 cases, statistically definitive)

Full run, same 465 primary Phase 2 cases as the main ablation, 0 errors. The retrieval
query is the ground-truth disease name — used **only** to construct the retrieval query,
never passed to the final diagnosis call, which sees only the image and retrieved
evidence, identical to how Arm 2 works. This measures a ceiling: if retrieval is
guaranteed to find the right evidence, does the VLM actually diagnose better? Raw data:
[results/phase2/oracle_retrieval_results.jsonl](results/phase2/oracle_retrieval_results.jsonl).
Script: [server-scripts/run_oracle_retrieval.py](server-scripts/run_oracle_retrieval.py).

## Headline: yes, dramatically, and it's not close

| Group (n) | VLM only (Arm 1) | Current RAG (Arm 2) | **Oracle retrieval** |
|---|--:|--:|--:|
| Healthy (180) | 97.2% | 96.7% | 98.3% |
| **Disease, in corpus (240)** | **0.4%** | **3.8%** | **73.3%** |
| Disease, not in corpus (45) | 0.0% | 0.0% | 0.0% |
| **Overall (465)** | **37.8%** | **39.4%** | **75.9%** [95% CI 71.8–79.6%] |

On the group where retrieval can structurally help — 240 cases where the disease exists
in the corpus — accuracy goes from essentially zero (VLM-only) or negligible (current
RAG) to **73.3%** [95% CI 67.4–78.5%] with a perfect query. Everywhere else stays
consistent with the main ablation (healthy ceiling ~97-98%, negative-control floor 0%,
confirming those groups behave exactly as designed regardless of query quality).

## Statistical significance: not marginal, not a fluke

McNemar's test (paired, same cases through both conditions):

| Comparison | Discordant (oracle right, other wrong) | Discordant (other right, oracle wrong) | p-value |
|---|--:|--:|--:|
| Oracle vs. Arm 1 (all 465) | 177 | **0** | 5.97 × 10⁻⁴⁰ |
| Oracle vs. Arm 2 (all 465) | 170 | **0** | 2.02 × 10⁻³⁸ |
| Oracle vs. Arm 2 (confident_match only, n=240) | 167 | **0** | 9.12 × 10⁻³⁸ |

**In every comparison, the discordant count in the "current pipeline right, oracle
wrong" direction is exactly zero** — there is not a single case in this entire eval set
where the existing pipeline's query beat the oracle query. Oracle retrieval is not just
better on average, it's never worse on any individual case. Combined with p-values on
the order of 10⁻³⁸, this is about as unambiguous as a result gets — a ceiling this far
above the current pipeline, this consistently, could not plausibly be sampling noise.

## What this settles

This directly answers the question posed before running the experiment:

> Can RAG help if we give it good retrieval information? — **Yes, decisively.**

The Phase 2 main-ablation null result (no significant difference between VLM-only and
the current RAG pipeline) is **not evidence that retrieval-augmented diagnosis doesn't
work**. It's evidence that **the specific query-construction mechanism in the current
pipeline (a single-shot Qwen3-VL description) fails to retrieve useful evidence, and
that failure — not a limitation of RAG itself — is what the main ablation actually
measured.** This is Outcome A from the pre-registered interpretation table: query
generation is the major bottleneck, not a fundamental ceiling on what external
agricultural knowledge can contribute to diagnosis.

## Residual gap worth noting, not explaining away

73.3% oracle accuracy is high, not perfect. Two contributing factors, both already
identified in prior experiments:
- Retrieval itself isn't 100% even with the oracle query — Experiment 2 found oracle
  retrieval hits R@1=81.2% on this exact subgroup, meaning ~19% of the time even a
  perfect-in-principle query still doesn't surface the exact right fact at rank 1 (likely
  corpus-internal naming inconsistency, not a retrieval-mechanism flaw).
- The generation-layer gap: 81.2% retrieval success → 73.3% diagnosis success is a real,
  separate ~8-point loss where the VLM had the correct evidence in front of it and still
  didn't land on the right diagnosis — worth investigating on its own, but small relative
  to the retrieval-layer gap this experiment isolates.

## What this doesn't solve

Oracle retrieval requires knowing the answer in advance to construct the query — it's a
ceiling measurement, not a deployable technique. The open question this motivates is
exactly Experiment 3 (Visual-First Retrieval) and future query-construction work: can a
retrieval query be constructed *without* knowing the ground truth that gets anywhere
close to this ceiling? The current pipeline's Prompt C-based approach clearly doesn't
(1.3% R@1 on the same subgroup, per Experiment 2) — but that doesn't mean no query
construction method can.
