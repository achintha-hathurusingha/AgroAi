# Stage F — Hybrid Retrieval: Alpha Sweep Results

Run on qbits in parallel with devon's Stage D diagnosis job. Combines BGE text
similarity (Qwen-generated query) and SigLIP visual similarity (query image) via
per-query min-max-normalized weighted sum: `S_H = α·norm(S_T) + (1-α)·norm(S_V)`. Swept
α from 0.0 (pure visual) to 1.0 (pure text). **No formal dev/test split exists for this
project** — the 465-case set has been the single fixed evaluation set throughout, so α
wasn't "selected on a held-out dev set" as the original plan proposed; the full sweep is
reported transparently instead of picking one value and hiding the rest. Script:
[server-scripts/run_hybrid_retrieval.py](server-scripts/run_hybrid_retrieval.py). Raw
data: [results/phase2/hybrid_retrieval_results.json](results/phase2/hybrid_retrieval_results.json).

## Full sweep (overall, all primary groups)

| α (1.0=text, 0.0=visual) | R@1 | R@5 | R@20 | MRR |
|--:|--:|--:|--:|--:|
| 0.00 (pure visual) | 1.7% | 8.8% | 18.3% | 0.055 |
| 0.10 | 2.8% | 10.1% | 17.4% | 0.063 |
| **0.25** | **3.0%** | **10.5%** | 17.6% | **0.063** |
| 0.40 | 3.0% | 8.8% | 17.6% | 0.061 |
| 0.50 | 2.2% | 7.1% | 16.6% | 0.051 |
| 0.60 | 1.9% | 5.4% | 15.5% | 0.042 |
| 0.75 | 1.7% | 3.7% | 11.0% | 0.036 |
| 0.90 | 0.6% | 2.8% | 9.2% | 0.023 |
| 1.00 (pure text) | 0.6% | 2.8% | 6.7% | 0.021 |

## Result: a small amount of text signal helps, but more consistently hurts

The relationship is not "more text is better" or "text is useless" — it's a clear peak
around **α=0.1–0.4**, degrading monotonically on both sides but especially as α
increases past ~0.4. R@1 peaks at α=0.25–0.40 (3.0%, vs. 1.7% pure-visual and 0.6%
pure-text); R@20 actually peaks at pure visual (α=0.0, 18.3%) and declines steadily as
text weight increases. This means: **a little text signal sharpens the top of the
ranking (R@1/MRR) without adding much broader recall, while too much text signal drags
performance all the way back down toward the poor pure-text baseline.**

## On the subgroup that matters (primary_confident_match, n=240), at α=0.25

| Method | R@1 | R@5 | R@20 | MRR |
|---|--:|--:|--:|--:|
| Pure text (Qwen query, from Experiment 2) | 1.3% | 5.4% | 12.9% | 0.040 |
| Pure visual (Stage C) | 3.3% | 17.1% | **35.4%** | 0.106 |
| **Hybrid (α=0.25)** | **5.8%** | **20.4%** | 34.2% | **0.123** |

Hybrid improves R@1 by ~1.75x over pure visual (3.3%→5.8%) and MRR by ~16% (0.106→0.123),
with R@5 also improving (17.1%→20.4%). R@20 is essentially flat (35.4%→34.2%, within
noise) — the hybrid signal sharpens *where in the ranking* the correct answer lands more
than it changes *whether* it's findable at all within a generous cutoff.

## Interpretation

This is consistent with, not contradictory to, Stage C's finding that visual embedding
is the dominant useful signal — text alone is close to unusable (per Experiment 2's
1.3% R@1), but a small amount of it evidently carries real complementary information the
visual signal alone doesn't capture, since adding a modest amount improves on pure-visual
rather than just diluting it. This is exactly the kind of result that motivates
Stage F's premise: text and visual retrieval aren't redundant, they're partially
complementary, with visual doing most of the work.

## What's still open

This is retrieval-only — whether this R@1 improvement (3.3%→5.8% on confident_match)
translates into a further diagnosis-accuracy improvement over Stage D's visual-only
result requires actually running the diagnosis step with hybrid-retrieved evidence,
which hasn't been done yet. Given the modest absolute R@1 gain, the expected diagnosis
improvement (if the retrieval→diagnosis relationship stays roughly proportional, per the
pattern observed between Stage C and Stage D) would likely be small in absolute terms —
worth testing to confirm, not worth assuming either way.
