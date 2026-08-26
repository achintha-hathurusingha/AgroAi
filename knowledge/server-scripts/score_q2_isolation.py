#!/usr/bin/env python3
"""Embed the B/C/D caption variants with BGE-micro-v2 (the Q1 winner) and retrieve
against the same frozen pool used in the main Phase 1B experiment."""
import json
from pathlib import Path

import torch

POOL_PATH = Path("/home/minura/agrivision-rag/data/q1q2_pool.json")
CAPTIONS_PATH = Path("/home/minura/agrivision-rag/data/q2_isolation_results.json")
OUT_PATH = Path("/home/minura/agrivision-rag/data/q2_isolation_scored.json")
BGE_MODEL = "TaylorAI/bge-micro-v2"


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    del model
    torch.cuda.empty_cache()
    return emb.cpu()


def main():
    with open(POOL_PATH) as f:
        pool = json.load(f)
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f)

    pool_texts = [f["retrieval_text"] for f in pool]
    p_emb = embed(pool_texts)

    variants = ["prompt_b_text", "prompt_c_text", "prompt_d_text"]
    all_texts = []
    meta = []
    for r in captions:
        for v in variants:
            all_texts.append(r[v])
            meta.append({"query_id": r["query_id"], "target_disease": r["target_disease"], "variant": v})

    q_emb = embed(all_texts)
    sims = q_emb @ p_emb.T

    results = []
    for i, m in enumerate(meta):
        scores = sims[i]
        topk_vals, topk_idx = torch.topk(scores, k=20)
        target = m["target_disease"].strip().lower()
        first_rank = None
        top1_disease = pool[topk_idx[0].item()]["disease"]
        top1_sim = topk_vals[0].item()
        for rank, idx in enumerate(topk_idx.tolist(), start=1):
            if (pool[idx].get("disease") or "").strip().lower() == target:
                first_rank = rank
                break
        results.append({**m, "first_relevant_rank": first_rank, "top1_disease": top1_disease, "top1_sim": top1_sim})
        print(f"{m['query_id']} {m['variant']:16s} target={target!r:25s} first_rank={str(first_rank):5s} top1={top1_disease!r}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary by variant ===")
    for v in variants:
        rows = [r for r in results if r["variant"] == v]
        r1 = sum(1 for r in rows if r["first_relevant_rank"] == 1) / len(rows)
        r3 = sum(1 for r in rows if r["first_relevant_rank"] and r["first_relevant_rank"] <= 3) / len(rows)
        r5 = sum(1 for r in rows if r["first_relevant_rank"] and r["first_relevant_rank"] <= 5) / len(rows)
        mrr = sum((1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] else 0.0 for r in rows) / len(rows)
        print(f"{v:16s} n={len(rows)}  R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  MRR={mrr:.3f}")


if __name__ == "__main__":
    main()
