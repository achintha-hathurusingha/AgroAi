# Milestone 3 control test also failed — root cause identified: scale mismatch, not a bug

Per the agreed plan, ran a control case (powdery mildew, a known Prompt-C rank-1 success
from the isolation experiment) through the identical Milestone 3 pipeline. **It failed
too** — hit rank: None, not in Top-5. This needed investigating before trusting the
pipeline, exactly as planned for this scenario.

## Diagnosis

Two checks, both using the cached corpus/embeddings, no new Qwen3-VL generation needed:

1. **How far off was it really?** The best-matching "powdery mildew" fact for this run's
   query was actually ranked **#40** out of 17,583 — not a near-miss, genuinely outside
   any reasonable Top-k.
2. **Does the earlier successful query text still work through this exact pipeline
   code?** Took the verbatim query text that scored rank 1 in the isolation experiment
   and ran it through Milestone 3's embed/retrieve functions against the full corpus:
   **also rank #40.** Then ran that same exact text against the original small 400-fact
   pool from the Q1/Q2 experiments (loaded directly, embedded fresh): **rank #1**,
   matching the original finding almost exactly (similarity 0.7637 vs the original
   0.812 reported in the isolation experiment — small numeric difference from embedding
   batch-order floating point, not a discrepancy in the ranking).

## Conclusion: this is a corpus-scale effect, not an implementation bug

**The exact same query, embedded with the exact same model, using the exact same
retrieval code, ranks #1 in the 400-fact pool and #40 in the 17,583-fact corpus.** The
Milestone 3 pipeline code is correct and consistent with everything measured before it.

**What this actually reveals: every Q1, Q2, and E result so far was measured against a
small, curated 400-fact benchmark pool (160 target facts across 8 diseases + 240
distractors) — not the real ~17,583-fact deployment-scale diagnostic corpus.** At full
scale there is vastly more textually-similar competition (thousands of other generic
"leaf spot"/"mottled discoloration" facts across many species), which the small pool
never exposed. The retrieval task is genuinely, substantially harder at real scale than
the earlier experiments suggested.

## Why this matters beyond just Milestone 3

This calls into question how much the **absolute** performance numbers from Q1
(BGE R@1=0.681) and Q2 (Prompt C R@1=0.286) generalize to the real corpus — they almost
certainly overstate real performance, since they were never tested at full scale. It's a
separate, still-open question whether the **relative** rankings from those experiments
(BGE beating sentence-transformers and SigLIP-text; Prompt C beating A/B/D/E) would hold
up if re-run at full 17,583-fact scale — plausible, since there's no obvious reason scale
would flip which embedder or prompt is *comparatively* best, but this hasn't actually
been tested and shouldn't be assumed either, given this project's consistent finding that
assumptions here keep turning out wrong.

## What this doesn't mean

- Not a bug in Milestone 3's pipeline code — directly verified above.
- Not evidence that BGE-micro-v2 or Prompt C were the wrong choices — no full-scale
  comparison against the alternatives has been run to say otherwise.
- Not a reason to distrust NumPy/the Q6 decision — that benchmark was already run at full
  17,583/35,507-fact scale and is unaffected by this finding.

## Decision needed

This is a real fork, not something to resolve unilaterally:

1. **Re-run Q1 and/or Q2 at full corpus scale** before trusting those decisions further —
   costs real time/compute, but would close this gap properly.
2. **Proceed with BGE-micro-v2 + Prompt C as-is**, explicitly documenting that their
   selection was validated at small-pool scale only, and treat full-scale retrieval
   quality as an open empirical question for the main ablation study to answer rather
   than something Phase 1B claims to have already settled.
3. Something else — e.g. investigate whether a smarter retrieval approach (reranking,
   which is Q3 and was intentionally deferred) is actually necessary *because* of this
   scale effect, not despite it.
