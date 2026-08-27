# Hybrid RAG Diagnosis: Full Run Results (Stage E follow-up)

Extends Stage F's retrieval-only hybrid result (α=0.25 text+visual blend) into an
actual diagnosis run, the same way Stage D extended Stage C. Full 465-case run on
devon, same v2 diagnosis prompt, same corpus, same evaluation as Stage D for direct
comparability. All numbers below use the corrected alias-aware scoring (see
[phase2-scoring-correction.md](phase2-scoring-correction.md)). Script:
[server-scripts/run_hybrid_rag_diagnosis.py](server-scripts/run_hybrid_rag_diagnosis.py).
Raw data: [results/phase2/hybrid_rag_diagnosis_results.jsonl](results/phase2/hybrid_rag_diagnosis_results.jsonl).

## Result: no further improvement over pure-visual RAG, despite better retrieval ranking

| Arm | confident_match accuracy | vs pure-visual RAG (McNemar) |
|---|--:|---|
| VLM-only | 0.4% (1/240) | — |
| RAG, BGE text query | 4.2% (10/240) | — |
| Stage D, pure visual RAG | 16.7% (40/240) | — |
| **Hybrid RAG (α=0.25)** | **17.5% (42/240)** | **p=0.81 (not significant)** |

Hybrid vs pure-visual: 10 cases hybrid got right that visual-only missed, 8 cases the
reverse — a coin flip, not a real difference. Hybrid vs RAG-BGE text-only remains
massively significant (p=1.1×10⁻⁷, 33 corrected vs 1 broken) — the visual signal is
still doing essentially all the work, whether alone or blended.

## Why this is a real, useful negative result, not a wasted run

Stage F's retrieval-only metrics showed hybrid clearly beating pure-visual (alias R@1
8.3% vs 3.3%, R@5 24.6% vs 20.0% on confident_match) -- a meaningful-looking retrieval
gain. This run tested whether that gain survives into diagnosis accuracy, and it
**doesn't, at this α**. Plausible explanation: the diagnosis step is given the same
top-5 evidence facts either way for most cases (`TOP_K=5`); hybrid's improvement is
concentrated in *rank-1 precision* (getting the single best fact ranked first), but
since the model reads all 5 candidates regardless of order, a modest reranking-within-5
doesn't change what the model has available to reason over. The retrieval metric that
predicts diagnosis accuracy is "is the right fact anywhere in the top-5," not "is it
ranked first" -- and hybrid and pure-visual are much closer on that measure (R@5 24.6%
vs 20.0%, a real but smaller gap than the R@1 numbers suggested).

## Implication for where to spend further effort

This closes the retrieval-*reranking* thread (alpha-tuning, hybrid blending) as a
diminishing-returns direction -- it moved the ranking metric more than it moved the
outcome that matters. The larger lever is still upstream: **how much useful evidence
enters the candidate pool at all**, which is bounded by embedding quality, not blend
weight. This is exactly why the SigLIP-truncation finding (18% of corpus facts
silently cut off at 64 tokens before embedding, confirmed against the corpus's own
tokenizer) is the more promising next target, not another retrieval-blend sweep --
Stage G-1/G-2 test this directly.
