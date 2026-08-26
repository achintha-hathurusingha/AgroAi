# Phase 1B.5 — Full-Scale Q1/Q2 Validation Results

Re-ran the original 47-query benchmark against the full 17,583-fact diagnostic corpus
(not the 400-fact curated pool), per the decision that the control-case failure in
Milestone 3 invalidated the small-pool numbers as an estimate of real performance. Same
47 queries, same declared ground truth, same three embedders. Added R@10/R@20 and the
uncapped rank of the first relevant fact per query. Raw data:
[results/q1q2_fullscale_summary.json](results/q1q2_fullscale_summary.json),
[results/q1q2_fullscale_records.json](results/q1q2_fullscale_records.json). Script:
[server-scripts/q1q2_fullscale_validation.py](server-scripts/q1q2_fullscale_validation.py).

## Important scope note before the results

This experiment re-validates **Q1** (embedder comparison) fully and rigorously — same 47
queries as the original, now at real scale. It does **not** re-validate the specific Q2
finding from the isolation experiment ("Prompt C beats Prompt A/B/D/E"), because the
47-query spec's `vlm_caption` entries were generated with the *original open-ended Prompt
A*, not Prompt C — Prompt C was only ever tested in the separate, smaller, informal
isolation experiment (7 images, no formal 47-query-style spec). So this run confirms (and
strengthens) that raw open-ended captioning is bad at full scale, but whether Prompt C's
earlier advantage over it *also* holds at full scale is still untested. Flagging this
now rather than silently treating Q2 as fully closed.

## Q1 result: BGE still wins overall, but the margin was smaller than reported

| Embedder | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|--:|--:|--:|--:|--:|--:|
| sentence-transformers | 0.426 | 0.511 | 0.596 | 0.681 | 0.745 | 0.499 |
| **bge-micro-v2** | 0.426 | **0.596** | **0.617** | **0.702** | **0.787** | **0.526** |
| siglip-text | 0.298 | 0.511 | 0.553 | 0.617 | 0.638 | 0.420 |

Compare to the original 400-fact-pool numbers: BGE's R@1 lead over sentence-transformers
was +10.7pp (0.681 vs 0.574) there — **at full scale that gap is gone; they tie exactly
at R@1 (0.426 each)**. BGE still wins on every other metric (R@3 through R@20, and MRR),
so the overall Q1 decision doesn't flip, but "BGE clearly wins" needs to be corrected to
"BGE wins on most metrics, ties on R@1" — a real, measured difference from what was
originally reported, not a rounding note. siglip-text remains clearly worst throughout,
consistent with the original finding.

## Query-type breakdown — the vlm_caption catastrophe got worse, not better

| Query type | sentence-transformers R@1/R@5/R@20 | bge-micro-v2 R@1/R@5/R@20 | siglip-text R@1/R@5/R@20 |
|---|---|---|---|
| disease_name (n=8) | 1.000 / 1.000 / 1.000 | 0.875 / 0.875 / 1.000 | 0.625 / 0.875 / 0.875 |
| symptoms (n=8) | 0.250 / 0.375 / 0.625 | 0.250 / 0.500 / 0.750 | 0.125 / 0.375 / 0.750 |
| species_symptoms (n=8) | 0.250 / 0.750 / 0.875 | 0.375 / 0.625 / 0.875 | 0.250 / 0.750 / 0.750 |
| description_no_name (n=8) | 0.375 / 0.625 / 1.000 | 0.500 / 0.875 / 0.875 | 0.250 / 0.625 / 0.625 |
| management (n=8) | 0.625 / 0.750 / 0.875 | 0.500 / 0.750 / 0.875 | 0.500 / 0.625 / 0.750 |
| **vlm_caption (n=7)** | **0.000 / 0.000 / 0.000** | **0.000 / 0.000 / 0.286** | **0.000 / 0.000 / 0.000** |

Every hand-authored query type dropped noticeably from the 400-fact-pool numbers (e.g.
BGE's `disease_name` R@1 fell from a clean 1.000 to 0.875 — even the literal disease-name
string doesn't always win outright against 17,583 real competitors). But `vlm_caption`
went from "catastrophic" (0% R@1, some signal by R@5) to **essentially total failure at
every k up to 20** for two of three embedders, with BGE alone managing 2 of 7 queries
within the top 20 (still 0% at R@5). At full scale, an unedited open-ended VLM caption is
not just a weak retrieval query — it's close to non-functional.

## Rank distribution (BGE-micro-v2) — not uniform degradation, a mix of hard and easy cases

Full uncapped rank of the first relevant fact, per query:

| Disease | Q1 (name) | Q2 (symptoms) | Q3 (species+sympt.) | Q4 (description) | Q5 (management) | Q6 (real caption) |
|---|--:|--:|--:|--:|--:|--:|
| D1 cedar apple rust | 1 | **875** | 3 | 2 | 3 | **3,126** |
| D2 apple scab | 1 | 2 | 1 | 1 | 1 | 181 |
| D3 black rot | 1 | 33 | 6 | 2 | 25 | 17 |
| D4 powdery mildew | 1 | 1 | 1 | 1 | 1 | 13 |
| D5 septoria leaf spot | 1 | 11 | 3 | 4 | 13 | 236 |
| D6 early blight | 1 | 2 | 1 | 1 | 2 | **2,932** |
| D7 bacterial leaf spot | 8 | 7 | 65 | 129 | 1 | 884 |
| D8 fire blight | 1 | 1 | 7 | 1 | 1 | — (no image) |

Two things worth calling out, not just the vlm_caption column:

- **D1's symptom-only query (Q2) unexpectedly ranked 875th**, even though the same
  disease's species+symptoms (Q3, rank 3) and description (Q4, rank 2) queries did fine.
  The Q2 text is a long, detailed, hand-authored description of the orange tube
  structures — arguably the *most* diagnostically precise query in the whole benchmark —
  yet it did far worse than shorter queries that simply add "apple tree" to similar
  content. This suggests species context may matter more for this particular disease's
  retrievability than symptom specificity does, at least for this embedder — genuinely
  counter-intuitive and worth a second look rather than dismissing as noise.
- **D7 (the naming-mismatch stress test) got notably worse at full scale** for Q3/Q4
  (rank 65, 129) despite doing reasonably at small-pool scale — while Q5 (management)
  still nailed rank 1. Full-scale behavior for this disease looks meaningfully different
  from what the small pool suggested, not just "the same result, slightly worse."

## What this settles and what it doesn't

- **Q1 is now validated at real scale, with a corrected margin**: BGE-micro-v2 remains
  the best overall choice, but the earlier "clearly wins" framing overstated the R@1
  advantage — it's now a tie there, a real win elsewhere.
- **The vlm_caption (raw Prompt A) failure is now confirmed as more severe at scale**,
  not an artifact of the small pool.
- **Not yet answered**: whether Prompt C's earlier advantage over raw captioning (from
  the isolation experiment) holds up at this same full scale, since Prompt C was never
  tested against the full corpus. That would need the same isolation-experiment prompts
  (B/C/D/E) re-run against the 17,583-fact corpus rather than the 400-fact pool, to fully
  close Q2 with the same rigor Q1 now has.
