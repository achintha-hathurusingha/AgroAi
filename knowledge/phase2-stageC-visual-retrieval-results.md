# Stage C — Visual-First Retrieval: Retrieval-Only Results

Per [phase2-visual-data-audit.md](phase2-visual-data-audit.md)'s proposed design: the 465
primary-case **query images** embedded directly with SigLIP's image tower, retrieved
against the diagnostic corpus's existing SigLIP text embeddings — cross-modal retrieval
that bypasses Qwen3-VL text generation entirely for the retrieval step. No diagnosis run
yet, no corpus changes, no new downloads. Script: [server-scripts/run_visual_retrieval.py](server-scripts/run_visual_retrieval.py).
Raw data: [results/phase2/visual_retrieval_results.json](results/phase2/visual_retrieval_results.json).

## Result: meaningful improvement over the current pipeline, well short of oracle

All four query representations tested so far, on `primary_confident_match` (n=240 — the
group where retrieval can structurally help):

| Query representation | R@1 | R@3 | R@5 | R@20 | MRR |
|---|--:|--:|--:|--:|--:|
| Qwen-generated text (current pipeline) | 1.3% | — | 5.4% | 12.9% | 0.040 |
| **Visual (SigLIP image tower)** | **3.3%** | — | **17.1%** | **35.4%** | **0.106** |
| Oracle (ground-truth disease name) | 81.2% | 81.2% | 81.2% | 81.2% | 0.812 |

Visual retrieval beats the current text pipeline by **~2.5x at R@1, ~3.2x at R@5, ~2.7x
at R@20, ~2.65x on MRR** — consistent, not a single-metric artifact. This is genuine
signal: bypassing the VLM's text description and embedding the image directly recovers
real retrieval quality the text-mediated pipeline was losing. It confirms part of the
hypothesis behind Stage B — information *is* being lost in the Image→Text step, and
skipping it helps.

It does **not** close the gap to oracle. 35.4% R@20 vs. 81.2% is still a large,
unclosed distance — visual-first retrieval alone is a real improvement, not a solved
problem. Two plausible (untested here) reasons for the remaining gap: SigLIP was never
fine-tuned on agricultural imagery specifically, and its text tower's alignment with the
short, catalog-style captions it was trained on may not transfer perfectly to the
longer, structured `Species: ... Disease/Issue: ... Symptoms: ...` format used in this
corpus's `retrieval_text`.

## Expected zeros, not new findings

`primary_healthy` and `primary_negative_control` both show exactly 0% across every
metric — expected and structural, not a new result. The diagnostic corpus contains no
fact with `disease == "healthy"` (by construction, since Phase 1A defined the diagnostic
corpus as disease/pest-bearing facts only), so no query type — text, oracle, or visual —
can ever retrieve a "healthy" match; R@k is undefined-as-zero for that group regardless
of method. Negative-control cases stay at zero for the same reason as every prior
experiment: the disease genuinely isn't in the corpus.

## Decision: proceed to Stage D

Per the plan's threshold ("only if Stage C produces meaningful retrieval"), this result
qualifies — a consistent 2.5-3x improvement across every recall level is not marginal.
Next: Stage D, full visual RAG end-to-end (visual retrieval → Qwen3-VL diagnosis with
that evidence), on the same 465 cases, to see whether this retrieval-quality gain
actually translates into better diagnosis accuracy — the same test Experiment 1 already
validated the mechanism for (retrieval quality → diagnosis quality), now with the visual
retrieval numbers as the input rather than oracle's.
