#!/usr/bin/env python3
"""
Experiment 2: Retrieval Decomposition. For the same 465 Phase 2 primary cases, compares
retrieval quality (R@1/R@5/R@20/MRR) between:
  A. Qwen-generated query (the actual query_text already produced during the main
     ablation run -- reused here, no new VLM generation needed)
  B. Oracle query (the ground-truth disease name itself)
against the frozen diagnostic corpus with BGE. Pure embedding + retrieval, no VLM calls,
so this runs on any machine with the corpus + BGE (built to run on qbits in parallel with
devon's oracle-diagnosis experiment).
"""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "agrivision-rag" / "data"
FROZEN_DIR = Path.home() / "agrivision-rag" / "frozen"
BGE_MODEL = "TaylorAI/bge-micro-v2"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
K_LIST = (1, 3, 5, 10, 20)


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


def embed(model, texts, batch_size=256):
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
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
    from sentence_transformers import SentenceTransformer

    print("loading diagnostic corpus...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    for f_ in corpus:
        if "retrieval_text" not in f_:
            f_["retrieval_text"] = retrieval_text(f_)
    print(f"  {len(corpus)} facts")

    print("loading eval set + Qwen-generated queries...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = {c["case_id"]: c for c in json.load(f)}
    qwen_queries = {}
    with open(DATA_DIR / "phase2_full_v2_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qwen_queries[r["case_id"]] = r["query_text"]

    primary_cases = [c for c in evalset.values() if c["eval_group"] in PRIMARY_GROUPS and c["case_id"] in qwen_queries]
    print(f"  {len(primary_cases)} primary cases with a Qwen query available")

    print("loading BGE...")
    model = SentenceTransformer(BGE_MODEL, device="cuda" if __import__("torch").cuda.is_available() else "cpu")

    print("embedding corpus...")
    corpus_texts = [f["retrieval_text"] for f in corpus]
    corpus_emb = embed(model, corpus_texts)

    print("embedding Qwen queries...")
    qwen_texts = [qwen_queries[c["case_id"]] for c in primary_cases]
    qwen_emb = embed(model, qwen_texts)

    print("embedding oracle queries...")
    oracle_texts = [c["ground_truth_disease"] for c in primary_cases]
    oracle_emb = embed(model, oracle_texts)

    results = {"qwen": {}, "oracle": {}}
    per_case_ranks = {"qwen": {}, "oracle": {}}

    for label, q_emb in [("qwen", qwen_emb), ("oracle", oracle_emb)]:
        print(f"\n=== computing ranks: {label} ===")
        ranks = []
        for i, c in enumerate(primary_cases):
            r = full_rank(q_emb[i], corpus_emb, corpus, c["ground_truth_disease"])
            ranks.append(r)
            per_case_ranks[label][c["case_id"]] = r
        results[label]["overall"] = metrics_from_ranks(ranks)
        print(results[label]["overall"])

        by_group = {}
        for group in PRIMARY_GROUPS:
            sub_idx = [i for i, c in enumerate(primary_cases) if c["eval_group"] == group]
            sub_ranks = [ranks[i] for i in sub_idx]
            by_group[group] = metrics_from_ranks(sub_ranks)
        results[label]["by_group"] = by_group

    out = {"results": results, "per_case_ranks": per_case_ranks, "n_cases": len(primary_cases)}
    out_path = DATA_DIR / "retrieval_decomposition_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n\n=== FINAL SUMMARY ===")
    for label in ["qwen", "oracle"]:
        m = results[label]["overall"]
        print(f"{label:10s} R@1={m['recall@1']:.3f} R@3={m['recall@3']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} R@20={m['recall@20']:.3f} MRR={m['mrr']:.3f}")
        for group, gm in results[label]["by_group"].items():
            print(f"  {group:28s} n={gm['n']:3d} R@1={gm['recall@1']:.3f} R@5={gm['recall@5']:.3f} R@20={gm['recall@20']:.3f} MRR={gm['mrr']:.3f}")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
