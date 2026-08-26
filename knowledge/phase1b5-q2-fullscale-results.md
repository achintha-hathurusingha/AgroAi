# Phase 1B.5 — Q2 Full-Scale Validation: the small-pool result does not hold

Closes the scope gap flagged in [phase1b5-fullscale-validation-results.md](phase1b5-fullscale-validation-results.md):
re-scored the already-generated Prompt B/C/D/E caption texts (no new Qwen3-VL calls —
reused the cached text from the isolation experiment) plus Prompt A's already-known
full-scale ranks, all with BGE-micro-v2 against the full 17,583-fact diagnostic corpus.
Raw data: [results/q2_fullscale_summary.json](results/q2_fullscale_summary.json). Script:
[server-scripts/q2_fullscale_validation.py](server-scripts/q2_fullscale_validation.py).

## Result: at full scale, no prompt variant works, and none is clearly better than another

| Variant | n | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|--:|--:|--:|--:|--:|--:|--:|
| A (open-ended) | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.286 | 0.021 |
| B (visual-only) | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.143 | 0.016 |
| **C (structured)** | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.143 | 0.020 |
| D (longer budget) | 7 | 0.000 | 0.000 | 0.000 | 0.143 | 0.143 | 0.032 |
| E (diagnostic checklist) | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 |

**The small-pool finding that Prompt C clearly beat the others (28.6% R@1 vs. A's 0%) does
not survive full-scale testing.** Every variant scores 0% at R@1 through R@5. MRR values
are all tiny (0.013–0.032) and don't meaningfully separate the variants — D's MRR is
technically highest, but from a single R@10 hit (black rot, rank 6) among 7 queries, not
a systematic advantage.

## Per-disease ranks tell a more specific story than the aggregate numbers

| Disease | A | B | C | D | E |
|---|--:|--:|--:|--:|--:|
| D1 cedar apple rust | 3,126 | 1,798 | 1,755 | 2,736 | 681 |
| D2 apple scab | 181 | 718 | 672 | **81** | — |
| D3 black rot | 17 | 761 | 104 | **6** | — |
| D4 powdery mildew | **13** | 35 | 40 | 68 | 99 |
| D5 septoria leaf spot | 236 | 60 | **29** | 53 | — |
| D6 early blight | 2,932 | 199 | 228 | **134** | 281 |
| D7 bacterial leaf spot | 884 | 17 | **16** | 525 | 27 |

**No prompt variant wins consistently across diseases.** D wins for apple scab and black
rot; A wins for powdery mildew (the case Prompt C had "solved" at rank 1 in the small
pool — now A beats C at full scale for that exact disease); C wins for septoria and
bacterial spot; nothing gets cedar apple rust or early blight anywhere close to usable.
This directly contradicts the small-pool experiment's implied conclusion that Prompt C
was a general-purpose fix — it wasn't; different query phrasings apparently interact with
different diseases' textual neighborhoods in the full corpus in ways that don't reduce to
"more structured is better" or "avoid species claims is better."

## What this means, following the outcome taxonomy from before deciding to run this

This lands closest to **Outcome E**: not "C wins narrowly" or "C and A converge to a small
gap" — all five variants are simultaneously near-useless at realistic scale. That
matches the interpretation flagged as a real possibility going in: **the dominant
bottleneck is not query *formatting* — it's Qwen3-VL's ability to extract
retrieval-discriminating visual information from these images in the first place.** No
amount of reformatting the same underlying visual description (which consistently uses
generic language like "brown spots," "irregular margin," "mottled discoloration" across
all five prompt variants) can overcome the fact that the *content* rarely contains the
specific feature that would separate the correct disease from thousands of textually
similar alternatives in the full corpus.

## What this does and doesn't settle

- **Does not validate Prompt C as the Q2 answer.** The original small-pool justification
  no longer holds at real scale — this needs to be corrected in the project record, not
  quietly left as-is.
- **Does not identify a better alternative either** — none of A/B/D/E outperformed C in
  aggregate; D showed the most (still very weak) signal, but on different diseases than
  the ones C or B did best on.
- **Reframes what Q3 (reranking) would actually be for.** The user's earlier caution
  (don't use reranking as a band-aid for an unvalidated Q1/Q2 choice) was correct — but
  this result is different from that scenario. This isn't "retrieval works but ranks
  things imperfectly"; it's "the correct answer is often thousands of positions away,"
  which no k-limited reranker over an initial Top-N candidate set could rescue if the
  correct fact isn't even in that N. That points toward the bottleneck being upstream
  (what visual information Qwen3-VL actually extracts and expresses) rather than
  downstream (how the retrieval/ranking step processes that information) — a materially
  different problem than "tune the query prompt more."
