#!/usr/bin/env python3
"""
Diagnose why the powdery mildew control failed. Two checks, no Qwen3-VL reload needed
(reuses cached corpus + embeddings):
1. How far down was the correct disease actually ranked? (near-miss vs total miss)
2. Does the EXACT query text that succeeded earlier (from the Q2 isolation experiment,
   captured verbatim) still retrieve correctly through this pipeline's code? If yes, the
   cause is generation non-determinism (Qwen3-VL sampling produced weaker text this run),
   not a bug in the corpus/embedding/retrieval implementation.
"""
import json
from pathlib import Path

import agrivision_pipeline as pipeline

CORPUS_CACHE = pipeline.CORPUS_CACHE
EMB_CACHE = pipeline.EMB_CACHE

# the exact Prompt C text that got rank 1 for powdery mildew in the earlier isolation experiment
EARLIER_SUCCESSFUL_QUERY = (
    "Species: unknown\n"
    "Visible symptoms:\n"
    "- Leaf is heart-shaped with a pointed tip\n"
    "- Surface has mottled green and lighter green patterns\n"
    "- Veins are visible and prominent\n"
    "- Edges appear slightly curled or irregular\n"
    "- Small brownish spot near the base of the petiole\n"
    "- Overall texture appears somewhat wrinkled or puckered"
)

# this run's actual query text (from milestone3_control_result.json)
THIS_RUN_QUERY = (
    "Species: unknown\n"
    "Visible symptoms:\n"
    "- Leaf is ovate with a pointed apex\n"
    "- Veins are prominent and form a clear network\n"
    "- Surface has mottled or blotchy discoloration\n"
    "- Some areas appear darker green or brownish\n"
    "- Margins show slight curling or irregularity\n"
    "- Base of leaf has a small, yellowish petiole remnant"
)


def full_rank_of_target(query_text, corpus, corpus_emb, target_disease):
    q_emb = pipeline.embed_query(query_text)
    sims = corpus_emb @ q_emb
    order = sims.argsort()[::-1]
    target = target_disease.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank, float(sims[idx]), rank / len(corpus)
    return None, None, None


def main():
    with open(CORPUS_CACHE) as f:
        corpus = json.load(f)
    import numpy as np
    corpus_emb = np.load(EMB_CACHE)
    print(f"corpus size: {len(corpus)}")

    n_powdery = sum(1 for f in corpus if (f.get("disease") or "").strip().lower() == "powdery mildew")
    print(f"number of 'powdery mildew' facts in corpus: {n_powdery}")

    print("\n=== check 1: how far down was the correct disease for THIS RUN's query? ===")
    rank, sim, pct = full_rank_of_target(THIS_RUN_QUERY, corpus, corpus_emb, "powdery mildew")
    print(f"  full rank of best powdery-mildew match: {rank} (similarity={sim}, percentile={pct})")

    print("\n=== check 2: does the EARLIER successful query text still work through this pipeline? ===")
    rank2, sim2, pct2 = full_rank_of_target(EARLIER_SUCCESSFUL_QUERY, corpus, corpus_emb, "powdery mildew")
    print(f"  full rank of best powdery-mildew match: {rank2} (similarity={sim2}, percentile={pct2})")

    print("\n=== interpretation ===")
    if rank2 is not None and rank2 <= 5:
        print("Earlier successful query text STILL retrieves correctly (rank <= 5) through this")
        print("pipeline's corpus/embedding/retrieval code. This means the implementation is")
        print("consistent with the earlier finding -- the control failure is attributable to")
        print("generation non-determinism (this run's Qwen3-VL sampling produced weaker/more")
        print("generic text than the earlier run), not a bug in Milestone 3's pipeline code.")
    else:
        print("Earlier successful query text does NOT retrieve correctly through this pipeline")
        print("even though it did in the isolation experiment. This points to an actual")
        print("difference in the implementation (corpus construction, embedding, or retrieval)")
        print("between the two experiments -- needs further investigation before trusting")
        print("Milestone 3's pipeline.")


if __name__ == "__main__":
    main()
