# Phase 2 — Greedy Decoding Collapse (run invalidated, cause found and fixed)

The first full Phase 2 run (610 cases, 0 crashes, completed successfully as an
*engineering* run) produced accuracy numbers that are **not a valid measurement** and
should not be cited. Documented here because it's a real, informative finding in its own
right, not just an error to bury.

## What happened

`run_phase2_ablation.py`'s `generate()` function used `do_sample=False` (greedy
decoding), added specifically for Phase 2 in the name of run-to-run reproducibility — a
judgment call flagged in [phase2-execution-plan.md](phase2-execution-plan.md) section 3.
That choice caused a severe degenerate collapse:

| Arm | "healthy" rate | "unknown" rate | True healthy rate |
|---|--:|--:|--:|
| 1 (VLM only) | **80.0%** | 6.9% | 29.5% |
| 2 (RAG BGE) | 35.2% | 28.5% | 29.5% |
| 3 (RAG SigLIP) | 39.7% | 21.8% | 29.5% |
| 4 (RAG full corpus) | 42.0% | 36.9% | 29.5% |
| 5 (RAG rerank) | 30.3% | 27.4% | 29.5% |

Arm 1's exact-match "accuracy" came out to 29.8% — almost identical to the eval set's
true healthy-rate (29.5%). Breaking it down confirmed why: **100% accuracy on the 180
genuinely-healthy cases, 0.5% accuracy on the 430 diseased cases** — the model was
essentially always predicting "healthy" regardless of image content, and getting
"credit" only by chance whenever the ground truth happened to also be healthy. This is
not a measurement of diagnostic capability; it's an artifact of a decoding-induced
collapse to a single safe-default output.

Arms with retrieved evidence didn't collapse to "healthy" as severely, but shifted
instead toward a different degenerate default — "unknown" (22–37% vs. 6.9% for Arm 1) —
suggesting the presence of retrieved text changes *which* safe-default the greedy
decoding collapses onto, not whether it collapses at all.

## Root cause

Every prior Phase 1B script (`test_real_pipeline.py`, `milestone3_acceptance_test.py`,
the entire Q1/Q2 experiment series) relied on Qwen3-VL's default generation config
(sampling enabled) and never showed anything like this — outputs were consistently
diverse and specific, even when wrong. `do_sample=False` was new to Phase 2's script,
added without testing whether it changed generation behavior qualitatively, not just
"more reproducible." It did the former, not just the latter.

## Fix

Reverted to the model's default sampling config (removed `do_sample=False`) — see the
inline comment left in `run_phase2_ablation.py` at the change site. Validated on a fresh
15-case pilot before relaunching the full run; see
[phase2-sampling-fix-validation.md](phase2-sampling-fix-validation.md) for that result.

## Disposition of the invalidated run

The 610-case greedy-decoding results
([results/phase2/phase2_full_results.jsonl](results/phase2/phase2_full_results.jsonl))
are kept for the record but must not be cited as Phase 2 findings — they measure a
decoding artifact, not the research question. The mechanical pipeline validation they
provided (0 errors across all 610 cases, all parsing correctly, judge output well-formed)
remains valid and useful; only the diagnosis *content* is compromised.
