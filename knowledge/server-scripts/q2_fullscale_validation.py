#!/usr/bin/env python3
"""
Closes the Q2 scope gap: re-scores the already-generated B/C/D/E caption variants
(from the isolation experiment -- no new Qwen3-VL calls needed) against the full
17,583-fact diagnostic corpus with BGE-micro-v2 (fixed, per the Q1 full-scale result).
Also pulls Prompt A's already-computed full-scale numbers (from the vlm_caption rows of
the Q1 full-scale validation) so all five variants can be compared on equal footing.
"""
import json
from pathlib import Path

import numpy as np

import agrivision_pipeline as pipeline

ISOLATION_PATH = Path("/home/minura/agrivision-rag/data/q2_isolation_results.json")  # B, C, D for 7 images
PROMPT_E_PATH = Path("/home/minura/agrivision-rag/data/q2_prompt_e_results.json")  # E for 4 images
FULLSCALE_SUMMARY = Path("/home/minura/agrivision-rag/data/q1q2_fullscale_summary.json")  # has uncapped rank_by_query
OUT_PATH = Path("/home/minura/agrivision-rag/data/q2_fullscale_summary.json")

K_LIST = (1, 3, 5, 10, 20)


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(pipeline.BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.cpu().numpy().astype("float32")


def full_rank(query_emb, corpus_emb, corpus, target_disease):
    sims = corpus_emb @ query_emb
    order = np.argsort(-sims)
    target = target_disease.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank
    return None


def metrics_from_ranks(ranks, k_list=K_LIST):
    n = len(ranks)
    m = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in k_list}
    m["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    m["n"] = n
    return m


def main():
    print("loading diagnostic corpus + cached BGE embeddings...")
    corpus = pipeline.build_diagnostic_corpus()
    corpus_emb = np.load(pipeline.EMB_CACHE)
    print(f"  corpus: {len(corpus)} facts, embeddings: {corpus_emb.shape}")

    with open(ISOLATION_PATH) as f:
        isolation = json.load(f)  # list of {query_id, target_disease, prompt_b_text, prompt_c_text, prompt_d_text}
    with open(PROMPT_E_PATH) as f:
        prompt_e = json.load(f)  # list of {query_id, target_disease, prompt_e_text}

    variants = {"B": [], "C": [], "D": [], "E": []}
    meta = {"B": [], "C": [], "D": [], "E": []}
    for r in isolation:
        variants["B"].append(r["prompt_b_text"]); meta["B"].append(r)
        variants["C"].append(r["prompt_c_text"]); meta["C"].append(r)
        variants["D"].append(r["prompt_d_text"]); meta["D"].append(r)
    for r in prompt_e:
        variants["E"].append(r["prompt_e_text"]); meta["E"].append(r)

    results = {}
    per_query_ranks = {}

    for variant, texts in variants.items():
        print(f"\n=== embedding + retrieving Prompt {variant} ({len(texts)} queries) ===")
        q_emb = embed(texts)
        ranks = []
        for i, m in enumerate(meta[variant]):
            r = full_rank(q_emb[i], corpus_emb, corpus, m["target_disease"])
            ranks.append(r)
            print(f"  {m['query_id']} ({m['target_disease']}): rank={r}")
        per_query_ranks[variant] = {m["query_id"]: r for m, r in zip(meta[variant], ranks)}
        results[variant] = metrics_from_ranks(ranks)
        print(f"  {variant}: {results[variant]}")

    # pull Prompt A's already-computed TRUE UNCAPPED full-scale ranks for bge-micro-v2
    # (from the Q1 fullscale run's rank_by_query, not the top-20-only records file)
    print("\n=== extracting Prompt A (original open-ended) full-scale numbers from Q1 run ===")
    with open(FULLSCALE_SUMMARY) as f:
        fullscale = json.load(f)
    a_ranks_by_query = fullscale["rank_by_query"]["bge-micro-v2"]  # uncapped, e.g. D1-Q6: 3126
    vlm_caption_ids = [m["query_id"] for m in meta["B"]]  # D1-D7
    # the Q1 spec used "D{n}-Q6" as the vlm_caption query_id; map to bare "D{n}" for comparability
    a_ranks = [a_ranks_by_query.get(f"{qid}-Q6") for qid in vlm_caption_ids]
    results["A"] = metrics_from_ranks(a_ranks)
    per_query_ranks["A"] = dict(zip(vlm_caption_ids, a_ranks))
    print(f"  A (true uncapped ranks): {dict(zip(vlm_caption_ids, a_ranks))}")
    print(f"  A: {results['A']}")

    with open(OUT_PATH, "w") as f:
        json.dump({"results": results, "per_query_ranks": per_query_ranks}, f, indent=2)

    print("\n\n=== FINAL Q2 FULL-SCALE COMPARISON (BGE-micro-v2, 17,583-fact corpus) ===")
    for variant in ["A", "B", "C", "D", "E"]:
        m = results[variant]
        print(f"Prompt {variant} (n={m['n']}): R@1={m['recall@1']:.3f} R@3={m['recall@3']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} R@20={m['recall@20']:.3f} MRR={m['mrr']:.3f}")


if __name__ == "__main__":
    main()
