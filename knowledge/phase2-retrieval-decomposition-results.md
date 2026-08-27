# Experiment 2 — Retrieval Decomposition Results

Run in parallel with Experiment 1 (Oracle Retrieval, diagnosis-level) on qbits, while
devon ran the diagnosis-level oracle experiment — pure embedding + retrieval, no Qwen3-VL
generation needed at all, so it completed in under 20 seconds. Same 465 primary Phase 2
cases. Compares retrieval quality (not final diagnosis) between the actual Qwen-generated
query (reused from the main ablation's `query_text` field) and the oracle query (ground
truth disease name). Script: [server-scripts/run_retrieval_decomposition.py](server-scripts/run_retrieval_decomposition.py).
Raw data: [results/phase2/retrieval_decomposition_results.json](results/phase2/retrieval_decomposition_results.json).

## Result: the retrieval-quality gap is enormous, and confirms Experiment 1 from a different angle

| Query type | R@1 | R@5 | R@20 | MRR |
|---|--:|--:|--:|--:|
| Qwen-generated (actual pipeline) | 0.6% | 2.8% | 6.7% | 0.021 |
| **Oracle (ground-truth disease name)** | **41.9%** | 41.9% | 41.9% | 0.419 |

On `primary_confident_match` specifically (the 240 cases where the disease genuinely
exists in the corpus):

| Query type | R@1 | R@5 | R@20 | MRR |
|---|--:|--:|--:|--:|
| Qwen-generated | 1.3% | 5.4% | 12.9% | 0.040 |
| **Oracle** | **81.2%** | 81.2% | 81.2% | 0.812 |

**A ~62x improvement in R@1** (1.3% → 81.2%) from switching only the retrieval query,
nothing else — same corpus, same embedder, same everything downstream.

## One structural detail worth noting: oracle R@1 = R@5 = R@20 exactly

When the oracle query succeeds, it succeeds at rank 1 — there's no case where oracle
retrieval finds the right fact at rank 3 or 15 but misses rank 1. This makes sense: an
exact or near-exact disease-name string match against a corpus indexed by the same
disease names is close to a direct lookup, not a fuzzy semantic search. The 18.8% of
`primary_confident_match` cases where oracle retrieval still fails (81.2% vs a theoretical
100%) reflects genuine corpus limitations — likely near-duplicate/inconsistent disease
naming within the corpus itself, not a retrieval-mechanism weakness.

## How this connects to Experiment 1

Experiment 1 (oracle retrieval → final diagnosis) found accuracy jumping from ~0.4-3.8%
to ~68.8% (pilot; full run pending) on the same subgroup. This experiment shows *why*
at the mechanism level: oracle querying doesn't just retrieve "better" evidence, it
retrieves the **correct** evidence at rank 1 in 81.2% of cases, vs. the current pipeline's
1.3%. The gap between 81.2% retrieval success and ~68.8% diagnosis success is a real,
separate generation-layer loss (the VLM sometimes still gets it wrong even when handed
the correct fact) — but it's a much smaller gap than the retrieval-layer one, confirming
retrieval/query-construction as the dominant bottleneck, not the final generation step.

## What remains open

This isolates the problem to query construction, but doesn't yet test *how* to fix it in
a scalable way (oracle queries require knowing the answer in advance, which is obviously
not available at real inference time). That's what Experiment 3 (Visual-First Retrieval)
and future query-construction work are for — this result is the evidence that motivates
pursuing them, not a deployable fix on its own.
