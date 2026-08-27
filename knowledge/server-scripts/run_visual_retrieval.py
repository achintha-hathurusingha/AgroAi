#!/usr/bin/env python3
"""
Stage C: Visual-first retrieval, retrieval-only evaluation (no Qwen3-VL, no diagnosis).
Embeds the 465 primary-case query IMAGES directly with SigLIP's image tower, retrieves
against the diagnostic corpus's already-built SigLIP TEXT embeddings (cross-modal
retrieval -- bypasses Qwen3-VL text generation entirely for the retrieval step). Compares
R@1/R@5/R@20/MRR against the Qwen-query and oracle-query numbers already computed in
Experiment 2. No corpus changes, no new downloads.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
SIGLIP_MODEL = "google/siglip-base-patch16-224"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
K_LIST = (1, 3, 5, 10, 20)


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
    from transformers import AutoProcessor, AutoModel

    print("loading diagnostic corpus + existing SigLIP text embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy")
    print(f"  corpus: {len(corpus)} facts, embeddings shape: {corpus_emb.shape}")

    print("loading eval set...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = json.load(f)
    primary_cases = [c for c in evalset if c["eval_group"] in PRIMARY_GROUPS]
    print(f"  {len(primary_cases)} primary cases")

    print("loading SigLIP...")
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()

    print("embedding query IMAGES with SigLIP image tower...")
    query_embs = []
    batch_size = 16
    for i in range(0, len(primary_cases), batch_size):
        batch = primary_cases[i:i + batch_size]
        images = [Image.open(c["image_path"]).convert("RGB") for c in batch]
        inputs = processor(images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        query_embs.append(feat.cpu())
        if i % 80 == 0:
            print(f"  {i}/{len(primary_cases)}")
    query_embs = torch.cat(query_embs, dim=0).numpy().astype("float32")
    print(f"  query embeddings shape: {query_embs.shape}")

    print("\ncomputing ranks...")
    ranks = []
    per_case_ranks = {}
    for i, c in enumerate(primary_cases):
        r = full_rank(query_embs[i], corpus_emb, corpus, c["ground_truth_disease"])
        ranks.append(r)
        per_case_ranks[c["case_id"]] = r

    overall = metrics_from_ranks(ranks)
    print("overall:", overall)

    by_group = {}
    for group in PRIMARY_GROUPS:
        idx = [i for i, c in enumerate(primary_cases) if c["eval_group"] == group]
        sub_ranks = [ranks[i] for i in idx]
        by_group[group] = metrics_from_ranks(sub_ranks)

    out = {"overall": overall, "by_group": by_group, "per_case_ranks": per_case_ranks, "n_cases": len(primary_cases)}
    out_path = DATA_DIR / "visual_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n\n=== FINAL SUMMARY: visual (SigLIP image-tower) retrieval ===")
    m = overall
    print(f"overall R@1={m['recall@1']:.3f} R@3={m['recall@3']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} R@20={m['recall@20']:.3f} MRR={m['mrr']:.3f}")
    for group, gm in by_group.items():
        print(f"  {group:28s} n={gm['n']:3d} R@1={gm['recall@1']:.3f} R@5={gm['recall@5']:.3f} R@20={gm['recall@20']:.3f} MRR={gm['mrr']:.3f}")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
