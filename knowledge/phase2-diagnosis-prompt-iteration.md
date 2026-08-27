# Phase 2 — Diagnosis Prompt Iteration (the real fix, and what it revealed)

Follow-up to [phase2-greedy-decoding-collapse.md](phase2-greedy-decoding-collapse.md).
That doc's initial hypothesis (greedy decoding caused the collapse) turned out to be
**wrong** — reverting to sampling barely moved the healthy-collapse rate (80.0% →
78.4% on a diverse 51-case cross-section). The real cause and fix took two more
iterations to find. Documented in full because the end state — very low accuracy even
after fixing the measurement artifacts — is itself the most important finding here, not
something to explain away.

## Root cause (confirmed, not decoding)

Inspecting actual reasoning text showed the model *seeing* real symptoms and dismissing
them: one case explicitly noted "some brown spots and margin necrosis" but concluded "no
definitive signs of disease... natural senescence." The diagnosis prompt's format
template offered `"healthy"` and `"unknown"` as named, equally-weighted options right
next to the specific-disease-name slot — an easy escape hatch the model took constantly.

## Three iterations, same 51-case diverse cross-section for direct comparison

| Version | Healthy rate | Unknown rate | Cases committing to a specific diagnosis | Accuracy among committed |
|---|--:|--:|--:|--:|
| Original (greedy) | 78.4%* | ~4%* | ~2 | — |
| Sampling only (no prompt change) | 78.4% | 3.9% | 2/51 | — |
| Prompt v1 (discourage premature "healthy") | 41.2% | 56.9% | 1/51 | 0/1 = 0% |
| **Prompt v2 (also discourage "unknown", push best-guess)** | **37-41%** | **2-16%** | **22-31/51** | **0-4.5%** |

*(the "original" row is from the full 610-case run; the 51-case diverse subset happened
to closely match its aggregate rate)

## What v1 got right and wrong

v1 added: an instruction that most submitted images do show something, a forced
"Observed signs" field before the diagnosis verdict, and language discouraging dismissing
visible marks as "natural wear." This worked for its target (healthy-rate roughly halved)
but had a side effect: suppressing false "healthy" claims pushed the model toward
"unknown" as the new safe default instead of toward a real diagnosis — from 1/51 committed
specific diagnoses under sampling-only to still just 1/51 under v1. The escape hatch
moved, it didn't close.

## What v2 fixed

Added explicit language: *"If signs were observed, give your single best-guess specific
diagnosis even if uncertain — a specific guess is more useful than declining to answer.
Only answer 'unknown' if signs were observed and you truly have no plausible guess at
all."* This is a legitimate methodological choice, not just a workaround — Phase 2 is
measuring diagnostic accuracy, so a forced-choice-style push against excessive hedging is
appropriate (similar to how many benchmarks require a committed answer rather than
allowing free abstention).

Result: unknown-rate collapsed to 2–16%, commitment rate rose to 22–31 of 51 cases per
arm. **But conditional accuracy on those committed guesses is 0–4.5%** — the model
commits far more often now, and is right almost none of the time when it does.

## Why this is the real, trustworthy result — not a fourth bug to chase

This low accuracy is **consistent with everything else measured in this project**: Q1/Q2
showed real retrieval degrading sharply at full corpus scale, the bottleneck diagnosis
found the VLM's own descriptions routinely miss the disease-discriminating feature, and
the CPJ refinement test showed even iterative self-correction only partially closes that
gap. A VLM that's bad at fine-grained diagnosis when honestly measured (not hidden behind
"healthy"/"unknown" hedging) is the expected outcome given that accumulated evidence, not
a new anomaly. Overall accuracy across all five arms now clusters tightly at 27.5–29.4%
on the 51-case subset — close to the ~29.4% healthy base rate, meaning at this small
sample size there's no visible separation between VLM-only and RAG arms yet. Whether a
real (likely modest) difference exists is exactly what the full 610-case run is powered
to detect that a 51-case sample isn't.

## Disposition

The original 610-case run and both intermediate prompt-fix tests are kept for the record
but superseded — see [results/phase2/](results/phase2/) for all four datasets
(`phase2_full_results.jsonl` = original/invalid, `phase2_diverse_sampling_test.jsonl` =
sampling-only diagnostic, `phase2_prompt_fix_test.jsonl` = v1, `phase2_prompt_fix_v2_test.jsonl`
= v2). The full 610-case run with the v2 prompt is the one to treat as the actual Phase 2
result.
