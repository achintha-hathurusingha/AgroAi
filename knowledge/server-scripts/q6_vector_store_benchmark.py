#!/usr/bin/env python3
"""
Q6: NumPy vs FAISS vs Qdrant, at real corpus scale, using BGE-micro-v2 (the Q1 winner).
Measures index-build time and per-query latency (mean/p50/p95) for each backend, at two
scales: the diagnostic corpus alone (~17.6K facts) and the full diagnostic+context corpus
(~40K+ facts) for growth headroom. Qdrant is a real standalone-binary server process, not
an embedded/in-memory shortcut.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

RAW_PATH = Path("/home/minura/agrivision-rag/data/agmmu/agmmu_ft_hf1.json")
BGE_MODEL = "TaylorAI/bge-micro-v2"
N_QUERY_TRIALS = 50
K = 5


def get_str(qa, field):
    v = qa.get(field, {})
    a = v.get("a", "") if isinstance(v, dict) else ""
    return a if isinstance(a, str) else ""


def normalize_entry(e):
    qa = e.get("finetuning qa", {})
    species = get_str(qa, "species")
    dict_fields = {}
    for field in qa:
        a = qa[field].get("a") if isinstance(qa[field], dict) else None
        if isinstance(a, dict):
            dict_fields[field] = a
    if not dict_fields:
        return [{
            "species_raw": species, "disease": get_str(qa, "disease/issue identification"),
            "pest": get_str(qa, "insect/pest"), "symptoms": get_str(qa, "symptom description"),
            "management": get_str(qa, "management instructions"),
        }]
    species_keys = set()
    for a in dict_fields.values():
        species_keys.update(a.keys())
    facts = []
    for sk in species_keys:
        facts.append({
            "species_raw": sk,
            "disease": dict_fields.get("disease/issue identification", {}).get(sk, "") if "disease/issue identification" in dict_fields else get_str(qa, "disease/issue identification"),
            "pest": dict_fields.get("insect/pest", {}).get(sk, "") if "insect/pest" in dict_fields else get_str(qa, "insect/pest"),
            "symptoms": dict_fields.get("symptom description", {}).get(sk, "") if "symptom description" in dict_fields else get_str(qa, "symptom description"),
            "management": dict_fields.get("management instructions", {}).get(sk, "") if "management instructions" in dict_fields else get_str(qa, "management instructions"),
        })
    return facts


def fingerprint(fact):
    parts = [fact.get(k, "") for k in ("species_raw", "disease", "pest", "symptoms", "management")]
    return "|".join(str(p).strip().lower() for p in parts)


def retrieval_text(fact):
    lines = []
    if fact.get("species_raw"):
        lines.append(f"Species: {fact['species_raw']}")
    if fact.get("disease"):
        lines.append(f"Disease/Issue: {fact['disease']}")
    if fact.get("pest"):
        lines.append(f"Pest: {fact['pest']}")
    if fact.get("symptoms"):
        lines.append(f"Symptoms: {fact['symptoms']}")
    if fact.get("management"):
        lines.append(f"Management: {fact['management']}")
    return "\n".join(lines)


def build_corpora():
    with open(RAW_PATH) as f:
        raw = json.load(f)
    all_facts = []
    for e in raw:
        all_facts.extend(normalize_entry(e))

    seen = {}
    for f in all_facts:
        fp = fingerprint(f)
        if fp not in seen:
            f["fact_id"] = f"fact_{len(seen):06d}"
            seen[fp] = f
    deduped_all = list(seen.values())

    diagnostic = [f for f in deduped_all if f.get("disease") or f.get("pest")]
    print(f"deduped diagnostic corpus: {len(diagnostic)}")
    print(f"deduped full corpus (diagnostic + context): {len(deduped_all)}")
    return diagnostic, deduped_all


def embed_texts(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    return emb.cpu().numpy().astype("float32")


def latency_stats(fn, queries, k=K):
    times = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q, k)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)), "max_ms": float(times.max())}


def bench_numpy(corpus_emb, query_emb):
    def search(q, k):
        sims = corpus_emb @ q
        idx = np.argpartition(-sims, k)[:k]
        return idx[np.argsort(-sims[idx])]
    t0 = time.perf_counter()
    # "index build" for numpy is just holding the matrix -- no separate structure
    build_ms = (time.perf_counter() - t0) * 1000
    stats = latency_stats(search, [query_emb[i] for i in range(len(query_emb))])
    return build_ms, stats


def bench_faiss(corpus_emb, query_emb):
    import faiss
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    build_ms = (time.perf_counter() - t0) * 1000

    def search(q, k):
        return index.search(q.reshape(1, -1), k)
    stats = latency_stats(search, [query_emb[i] for i in range(len(query_emb))])
    return build_ms, stats


def bench_qdrant(corpus_emb, corpus_meta, query_emb, collection_name):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(url="http://localhost:6333", timeout=60)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    t0 = time.perf_counter()
    client.create_collection(collection_name, vectors_config=VectorParams(size=corpus_emb.shape[1], distance=Distance.COSINE))
    batch = 256
    for i in range(0, len(corpus_emb), batch):
        points = [
            PointStruct(id=i + j, vector=corpus_emb[i + j].tolist(), payload={"disease": corpus_meta[i + j].get("disease", "")})
            for j in range(min(batch, len(corpus_emb) - i))
        ]
        client.upsert(collection_name, points=points)
    build_ms = (time.perf_counter() - t0) * 1000

    def search(q, k):
        return client.query_points(collection_name, query=q.tolist(), limit=k)
    stats = latency_stats(search, [query_emb[i] for i in range(len(query_emb))])
    return build_ms, stats


def run_scale(name, corpus, n_query_trials=N_QUERY_TRIALS):
    print(f"\n{'=' * 20} scale: {name} (n={len(corpus)}) {'=' * 20}")
    texts = [retrieval_text(f) for f in corpus]
    print("embedding corpus...")
    corpus_emb = embed_texts(texts)

    rng = np.random.default_rng(42)
    query_idx = rng.choice(len(corpus), size=min(n_query_trials, len(corpus)), replace=False)
    query_emb = corpus_emb[query_idx]  # query with real corpus embeddings (self-retrieval-ish, fine for pure latency)

    results = {}

    print("benchmarking numpy...")
    build_ms, stats = bench_numpy(corpus_emb, query_emb)
    results["numpy"] = {"build_ms": build_ms, **stats}
    print(f"  numpy: build={build_ms:.2f}ms, query mean={stats['mean_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms")

    print("benchmarking faiss...")
    build_ms, stats = bench_faiss(corpus_emb, query_emb)
    results["faiss"] = {"build_ms": build_ms, **stats}
    print(f"  faiss: build={build_ms:.2f}ms, query mean={stats['mean_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms")

    print("benchmarking qdrant (real server)...")
    build_ms, stats = bench_qdrant(corpus_emb, corpus, query_emb, f"q6_{name}")
    results["qdrant"] = {"build_ms": build_ms, **stats}
    print(f"  qdrant: build={build_ms:.2f}ms, query mean={stats['mean_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms")

    return results


def main():
    diagnostic, full_corpus = build_corpora()

    all_results = {}
    all_results["diagnostic_corpus"] = run_scale("diagnostic", diagnostic)
    all_results["full_corpus"] = run_scale("full", full_corpus)

    with open("/home/minura/agrivision-rag/data/q6_benchmark_results.json", "w") as f:
        json.dump({"n_diagnostic": len(diagnostic), "n_full": len(full_corpus), "n_query_trials": N_QUERY_TRIALS, "k": K, "results": all_results}, f, indent=2)

    print("\n\n=== FINAL SUMMARY ===")
    for scale_name, results in all_results.items():
        print(f"\n{scale_name}:")
        for backend, r in results.items():
            print(f"  {backend:8s} build={r['build_ms']:9.1f}ms  query_mean={r['mean_ms']:7.3f}ms  query_p95={r['p95_ms']:7.3f}ms")


if __name__ == "__main__":
    main()
