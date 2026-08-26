#!/usr/bin/env python3
"""
Phase 1B.5: full-scale Q1/Q2 validation. Same 47 queries, same declared ground truth,
same three embedders as the original Q1/Q2 experiment -- but searched against the full
~17,583-fact deduplicated diagnostic corpus instead of the original curated 400-fact
pool. Adds R@10/R@20 and the actual (uncapped) rank of the first relevant fact, to
distinguish "occasional hard case" from "systematic degradation" per query.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import agrivision_pipeline as pipeline

SPEC_PATH = Path("/home/minura/agrivision-rag/data/q1q2_query_spec_filled.json")
OUT_DIR = Path("/home/minura/agrivision-rag/data")
SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "TaylorAI/bge-micro-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"


def embed_sentence_transformers(model_name, texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    return emb.cpu().numpy().astype("float32")


def embed_siglip_text(texts, batch_size=32):
    from transformers import AutoProcessor, AutoModel
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        embs.append(feat.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(embs, dim=0).numpy().astype("float32")


def retrieve_full_rank_and_score(embedder_name, query_embs, corpus_emb, queries, corpus, k_list=(1, 3, 5, 10, 20)):
    sims = query_embs @ corpus_emb.T  # (n_queries, n_corpus)
    records = []
    per_query_hits = {}

    for qi, q in enumerate(queries):
        scores = sims[qi]
        order = np.argsort(-scores)  # full ranking, uncapped
        target = q["target_disease"].strip().lower()

        first_hit_rank = None
        for rank, idx in enumerate(order, start=1):
            if (corpus[idx].get("disease") or "").strip().lower() == target:
                first_hit_rank = rank
                break
        per_query_hits[q["query_id"]] = first_hit_rank

        # save top-20 for inspection
        for rank, idx in enumerate(order[:20], start=1):
            fact = corpus[idx]
            is_relevant = (fact.get("disease") or "").strip().lower() == target
            records.append({
                "query_id": q["query_id"], "embedder": embedder_name, "rank": rank,
                "fact_id": fact["fact_id"], "similarity": float(scores[idx]),
                "disease": fact.get("disease", ""), "relevant": is_relevant,
            })

    metrics = {}
    for k in k_list:
        hits = sum(1 for r in per_query_hits.values() if r is not None and r <= k)
        metrics[f"recall@{k}"] = hits / len(queries)
    metrics["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in per_query_hits.values()) / len(queries)

    by_type = defaultdict(list)
    for q in queries:
        by_type[q["query_type"]].append((q["query_id"], per_query_hits[q["query_id"]]))
    type_metrics = {}
    for qtype, pairs in by_type.items():
        ranks = [r for _, r in pairs]
        tm = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in k_list}
        tm["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)
        tm["n"] = len(ranks)
        type_metrics[qtype] = tm

    rank_by_query = {qid: r for qid, r in per_query_hits.items()}
    return records, metrics, type_metrics, rank_by_query


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)
    queries = spec["queries"]
    query_texts = [q["query_text"] for q in queries]

    print("loading diagnostic corpus (cached)...")
    corpus = pipeline.build_diagnostic_corpus()
    print(f"  {len(corpus)} facts")
    corpus_texts = [f["retrieval_text"] for f in corpus]

    all_records = []
    all_metrics = {}
    all_type_metrics = {}
    all_rank_by_query = {}

    print("\n=== sentence-transformers (all-MiniLM-L6-v2), full corpus ===")
    q_emb = embed_sentence_transformers(SENTENCE_TRANSFORMERS_MODEL, query_texts)
    c_emb = embed_sentence_transformers(SENTENCE_TRANSFORMERS_MODEL, corpus_texts)
    records, metrics, type_metrics, rank_by_query = retrieve_full_rank_and_score("sentence-transformers", q_emb, c_emb, queries, corpus)
    all_records += records
    all_metrics["sentence-transformers"] = metrics
    all_type_metrics["sentence-transformers"] = type_metrics
    all_rank_by_query["sentence-transformers"] = rank_by_query
    print(metrics)

    print("\n=== BGE-micro-v2, full corpus ===")
    q_emb = embed_sentence_transformers(BGE_MODEL, query_texts)
    c_emb_cache = pipeline.EMB_CACHE
    if c_emb_cache.exists():
        c_emb = np.load(c_emb_cache)
        print(f"  (reused cached BGE corpus embeddings, shape={c_emb.shape})")
    else:
        c_emb = embed_sentence_transformers(BGE_MODEL, corpus_texts)
    records, metrics, type_metrics, rank_by_query = retrieve_full_rank_and_score("bge-micro-v2", q_emb, c_emb, queries, corpus)
    all_records += records
    all_metrics["bge-micro-v2"] = metrics
    all_type_metrics["bge-micro-v2"] = type_metrics
    all_rank_by_query["bge-micro-v2"] = rank_by_query
    print(metrics)

    print("\n=== SigLIP text tower, full corpus ===")
    q_emb = embed_siglip_text(query_texts)
    c_emb = embed_siglip_text(corpus_texts)
    records, metrics, type_metrics, rank_by_query = retrieve_full_rank_and_score("siglip-text", q_emb, c_emb, queries, corpus)
    all_records += records
    all_metrics["siglip-text"] = metrics
    all_type_metrics["siglip-text"] = type_metrics
    all_rank_by_query["siglip-text"] = rank_by_query
    print(metrics)

    with open(OUT_DIR / "q1q2_fullscale_records.json", "w") as f:
        json.dump(all_records, f, indent=2)
    with open(OUT_DIR / "q1q2_fullscale_summary.json", "w") as f:
        json.dump({"overall": all_metrics, "by_query_type": all_type_metrics, "rank_by_query": all_rank_by_query,
                    "n_corpus": len(corpus), "n_queries": len(queries)}, f, indent=2)

    print("\n\n=== FULL-SCALE SUMMARY (overall, n_corpus=%d) ===" % len(corpus))
    for name, m in all_metrics.items():
        print(f"{name:20s}  R@1={m['recall@1']:.3f}  R@3={m['recall@3']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}  R@20={m['recall@20']:.3f}  MRR={m['mrr']:.3f}")

    print("\n=== FULL-SCALE SUMMARY (by query type) ===")
    for name, tm in all_type_metrics.items():
        print(f"\n{name}:")
        for qtype, m in tm.items():
            print(f"  {qtype:20s} n={m['n']:2d}  R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@20={m['recall@20']:.3f}  MRR={m['mrr']:.3f}")

    print("\n=== rank of first relevant fact, per query (bge-micro-v2) ===")
    for qid, r in all_rank_by_query["bge-micro-v2"].items():
        print(f"  {qid}: {r}")


if __name__ == "__main__":
    main()
