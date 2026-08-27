#!/usr/bin/env python3
"""
Stage F groundwork: hybrid text+visual retrieval, retrieval-only (no diagnosis).
Combines two independently-computed, differently-scaled similarity signals per case:
  - S_T: BGE cosine similarity, Qwen-generated text query vs BGE text corpus embeddings
  - S_V: SigLIP cosine similarity, query IMAGE vs SigLIP text corpus embeddings
Since BGE (384-dim) and SigLIP (768-dim) are unrelated embedding spaces, raw similarity
scores aren't directly comparable -- each is min-max normalized per-query before
combining: S_H = alpha * norm(S_T) + (1-alpha) * norm(S_V).

Sweeps alpha from 0.0 to 1.0 and reports R@1/R@5/R@20/MRR at each value. No formal
held-out dev/test split exists for this project (the 465-case set has been the single
fixed evaluation set throughout) -- flagging that explicitly rather than pretending a
dev-set-selected alpha, per the plan's own methodological standard. All alpha results
reported transparently instead.
"""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "agrivision-rag" / "data"
FROZEN_DIR = Path.home() / "agrivision-rag" / "frozen"
PV_IMAGES_DIR = DATA_DIR / "pv_images"
BGE_MODEL = "TaylorAI/bge-micro-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
K_LIST = (1, 3, 5, 10, 20)
ALPHAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


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


def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def full_rank_from_scores(scores, corpus, target_disease):
    order = np.argsort(-scores)
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
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoProcessor, AutoModel
    from PIL import Image

    print("loading diagnostic corpus + existing corpus embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    for f_ in corpus:
        if "retrieval_text" not in f_:
            f_["retrieval_text"] = retrieval_text(f_)
    bge_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")
    siglip_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy")
    print(f"  corpus: {len(corpus)} facts, BGE emb {bge_corpus_emb.shape}, SigLIP emb {siglip_corpus_emb.shape}")

    print("loading eval set + Qwen queries...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = {c["case_id"]: c for c in json.load(f)}
    qwen_queries = {}
    with open(DATA_DIR / "phase2_full_v2_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qwen_queries[r["case_id"]] = r["query_text"]

    primary_cases = [c for c in evalset.values() if c["eval_group"] in PRIMARY_GROUPS and c["case_id"] in qwen_queries]
    print(f"  {len(primary_cases)} primary cases")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading BGE, embedding Qwen queries...")
    bge_model = SentenceTransformer(BGE_MODEL, device=device)
    qwen_texts = [qwen_queries[c["case_id"]] for c in primary_cases]
    S_T_query_emb = bge_model.encode(qwen_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).cpu().numpy().astype("float32")

    print("loading SigLIP, embedding query IMAGES...")
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32).to(device)
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    siglip_model.eval()

    query_embs_v = []
    batch_size = 16
    for i in range(0, len(primary_cases), batch_size):
        batch = primary_cases[i:i + batch_size]
        images = [Image.open(PV_IMAGES_DIR / Path(c["image_path"]).name).convert("RGB") for c in batch]
        inputs = siglip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = siglip_model.get_image_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        query_embs_v.append(feat.cpu())
        if i % 80 == 0:
            print(f"  {i}/{len(primary_cases)}")
    S_V_query_emb = torch.cat(query_embs_v, dim=0).numpy().astype("float32")

    print("\ncomputing per-case similarity matrices...")
    S_T_all = S_T_query_emb @ bge_corpus_emb.T   # (n_query, n_corpus)
    S_V_all = S_V_query_emb @ siglip_corpus_emb.T

    results = {}
    per_alpha_ranks = {}
    for alpha in ALPHAS:
        ranks = []
        for i, c in enumerate(primary_cases):
            s_t = minmax_norm(S_T_all[i])
            s_v = minmax_norm(S_V_all[i])
            s_h = alpha * s_t + (1 - alpha) * s_v
            r = full_rank_from_scores(s_h, corpus, c["ground_truth_disease"])
            ranks.append(r)
        m = metrics_from_ranks(ranks)
        results[alpha] = m
        per_alpha_ranks[alpha] = {c["case_id"]: r for c, r in zip(primary_cases, ranks)}
        print(f"alpha={alpha:.2f} (1.0=text-only, 0.0=visual-only): R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} R@20={m['recall@20']:.3f} MRR={m['mrr']:.3f}")

    # also break down by group at the alpha that maximizes overall R@1
    best_alpha = max(results, key=lambda a: results[a]["recall@1"])
    print(f"\nbest alpha by R@1: {best_alpha}")
    by_group = {}
    for group in PRIMARY_GROUPS:
        idx = [i for i, c in enumerate(primary_cases) if c["eval_group"] == group]
        ranks = []
        for i in idx:
            c = primary_cases[i]
            s_t = minmax_norm(S_T_all[i])
            s_v = minmax_norm(S_V_all[i])
            s_h = best_alpha * s_t + (1 - best_alpha) * s_v
            r = full_rank_from_scores(s_h, corpus, c["ground_truth_disease"])
            ranks.append(r)
        by_group[group] = metrics_from_ranks(ranks)

    out = {"alpha_sweep": results, "best_alpha": best_alpha, "best_alpha_by_group": by_group, "n_cases": len(primary_cases)}
    out_path = DATA_DIR / "hybrid_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
