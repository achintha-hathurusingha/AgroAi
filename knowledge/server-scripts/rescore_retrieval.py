#!/usr/bin/env python3
"""
Rescore retrieval Recall@K for oracle-text, current-pipeline-text (Qwen query),
SigLIP-visual, and hybrid(alpha=0.25) retrieval using the alias+multi-value-aware
matching in scoring.py, vs the original strict full-string-equality matching.
Re-embeds saved query texts/images with the same models used originally and re-ranks
against the already-computed corpus embeddings -- no new generation, fast.

Also redoes the Stage A error-taxonomy category counts (RETRIEVAL_CORPUS_FAILURE /
QUERY_FAILURE / REASONING_FAILURE) using the corrected oracle_hit / current_hit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from scoring import fact_matches_ground_truth, corpus_disease_terms, ground_truth_aliases

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
PV_IMAGES_DIR = DATA_DIR / "pv_images"
BGE_MODEL = "TaylorAI/bge-micro-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
K_LIST = (1, 5, 20)
ALPHA = 0.25


def strict_rank(scores, corpus, gt):
    order = np.argsort(-scores)
    target = gt.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank
    return None


def alias_rank(scores, corpus, gt):
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        if fact_matches_ground_truth(corpus[idx].get("disease"), gt):
            return rank
    return None


def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def metrics(ranks, k_list=K_LIST):
    n = len(ranks)
    m = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in k_list}
    m["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    m["n"] = n
    return m


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoProcessor, AutoModel

    print("loading corpus + embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    bge_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")
    siglip_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy")

    print("loading evalset + query texts...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = {c["case_id"]: c for c in json.load(f)}

    qwen_queries, oracle_queries = {}, {}
    with open(DATA_DIR / "phase2_full_v2_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qwen_queries[r["case_id"]] = r["query_text"]
    with open(DATA_DIR / "oracle_retrieval_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            oracle_queries[r["case_id"]] = r["oracle_query_text"]

    cases = [c for c in evalset.values() if c["eval_group"] in PRIMARY_GROUPS
             and c["case_id"] in qwen_queries and c["case_id"] in oracle_queries]
    print(f"  {len(cases)} cases")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading BGE, embedding oracle + qwen queries...")
    bge_model = SentenceTransformer(BGE_MODEL, device=device)
    oracle_texts = [oracle_queries[c["case_id"]] for c in cases]
    qwen_texts = [qwen_queries[c["case_id"]] for c in cases]
    S_oracle = bge_model.encode(oracle_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).cpu().numpy().astype("float32")
    S_qwen = bge_model.encode(qwen_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).cpu().numpy().astype("float32")

    print("loading SigLIP, embedding query images...")
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32).to(device)
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    siglip_model.eval()
    query_embs_v = []
    batch_size = 16
    for i in range(0, len(cases), batch_size):
        batch = cases[i:i + batch_size]
        images = [Image.open(PV_IMAGES_DIR / Path(c["image_path"]).name).convert("RGB") for c in batch]
        inputs = siglip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = siglip_model.get_image_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        query_embs_v.append(feat.cpu())
        if i % 80 == 0:
            print(f"  {i}/{len(cases)}")
    S_visual = torch.cat(query_embs_v, dim=0).numpy().astype("float32")

    print("\ncomputing similarity matrices...")
    Oracle_all = S_oracle @ bge_corpus_emb.T
    Qwen_all = S_qwen @ bge_corpus_emb.T
    Visual_all = S_visual @ siglip_corpus_emb.T

    modes = {}
    for name, sim_all in [("oracle_text", Oracle_all), ("qwen_text_current_pipeline", Qwen_all), ("siglip_visual", Visual_all)]:
        strict_ranks, alias_ranks = [], []
        for i, c in enumerate(cases):
            gt = c["ground_truth_disease"]
            strict_ranks.append(strict_rank(sim_all[i], corpus, gt))
            alias_ranks.append(alias_rank(sim_all[i], corpus, gt))
        modes[name] = {
            "strict": metrics(strict_ranks),
            "alias": metrics(alias_ranks),
            "per_case_alias_rank": {c["case_id"]: r for c, r in zip(cases, alias_ranks)},
        }
        print(f"{name:30s} strict R@1={modes[name]['strict']['recall@1']:.3f}  alias R@1={modes[name]['alias']['recall@1']:.3f}")

    # hybrid alpha=0.25
    strict_ranks, alias_ranks = [], []
    for i, c in enumerate(cases):
        gt = c["ground_truth_disease"]
        s_t = minmax_norm(Qwen_all[i])
        s_v = minmax_norm(Visual_all[i])
        s_h = ALPHA * s_t + (1 - ALPHA) * s_v
        strict_ranks.append(strict_rank(s_h, corpus, gt))
        alias_ranks.append(alias_rank(s_h, corpus, gt))
    modes["hybrid_alpha025"] = {
        "strict": metrics(strict_ranks),
        "alias": metrics(alias_ranks),
        "per_case_alias_rank": {c["case_id"]: r for c, r in zip(cases, alias_ranks)},
    }
    print(f"{'hybrid_alpha025':30s} strict R@1={modes['hybrid_alpha025']['strict']['recall@1']:.3f}  alias R@1={modes['hybrid_alpha025']['alias']['recall@1']:.3f}")

    # by-group breakdown (confident_match only, the group that matters)
    print("\n=== By eval_group breakdown (primary_confident_match) ===")
    for name in modes:
        idx = [i for i, c in enumerate(cases) if c["eval_group"] == "primary_confident_match"]
        pass  # already computed above per-case; recompute filtered below

    by_group = {}
    for name, sim_all in [("oracle_text", Oracle_all), ("qwen_text_current_pipeline", Qwen_all), ("siglip_visual", Visual_all)]:
        by_group[name] = {}
        for group in PRIMARY_GROUPS:
            idx = [i for i, c in enumerate(cases) if c["eval_group"] == group]
            sr, ar = [], []
            for i in idx:
                gt = cases[i]["ground_truth_disease"]
                sr.append(strict_rank(sim_all[i], corpus, gt))
                ar.append(alias_rank(sim_all[i], corpus, gt))
            by_group[name][group] = {"strict": metrics(sr), "alias": metrics(ar)}
    idx = [i for i, c in enumerate(cases) if c["eval_group"] == "primary_confident_match"]
    by_group["hybrid_alpha025"] = {"primary_confident_match": {}}
    sr, ar = [], []
    for i in idx:
        gt = cases[i]["ground_truth_disease"]
        s_t = minmax_norm(Qwen_all[i])
        s_v = minmax_norm(Visual_all[i])
        s_h = ALPHA * s_t + (1 - ALPHA) * s_v
        sr.append(strict_rank(s_h, corpus, gt))
        ar.append(alias_rank(s_h, corpus, gt))
    by_group["hybrid_alpha025"]["primary_confident_match"] = {"strict": metrics(sr), "alias": metrics(ar)}

    for name in by_group:
        cm = by_group[name].get("primary_confident_match", {})
        if cm:
            print(f"{name:30s} confident_match strict R@1={cm['strict']['recall@1']:.3f} R@5={cm['strict']['recall@5']:.3f}  "
                  f"alias R@1={cm['alias']['recall@1']:.3f} R@5={cm['alias']['recall@5']:.3f}")

    # error taxonomy recompute
    print("\n=== Error taxonomy recompute (alias-aware oracle_hit / current_hit) ===")
    with open(DATA_DIR / "phase2_full_v2_results.jsonl") as f:
        main_rows = {json.loads(l)["case_id"]: json.loads(l) for l in f}
    oracle_alias_ranks = modes["oracle_text"]["per_case_alias_rank"]
    qwen_alias_ranks = modes["qwen_text_current_pipeline"]["per_case_alias_rank"]

    from collections import Counter
    cm_cases = [c for c in cases if c["eval_group"] == "primary_confident_match"]
    cats = []
    for c in cm_cases:
        cid = c["case_id"]
        gt = c["ground_truth_disease"]
        arm2 = main_rows[cid]["arms"]["2_rag_bge"]
        diag = arm2.get("diagnosis")
        from scoring import is_correct_aliased
        correct = is_correct_aliased(diag, gt)
        oracle_hit = oracle_alias_ranks.get(cid) is not None and oracle_alias_ranks[cid] <= 5
        current_hit = qwen_alias_ranks.get(cid) is not None and qwen_alias_ranks[cid] <= 5
        if correct:
            cat = "CORRECT"
        elif not oracle_hit:
            cat = "RETRIEVAL_CORPUS_FAILURE"
        elif oracle_hit and not current_hit:
            cat = "QUERY_FAILURE"
        else:
            cat = "REASONING_FAILURE"
        cats.append(cat)
    cat_counts = Counter(cats)
    print(f"total confident_match cases: {len(cm_cases)}")
    for cat, n in cat_counts.most_common():
        print(f"  {cat:28s} {n:3d}  ({n/len(cm_cases)*100:.1f}%)")

    out = {
        "n_cases": len(cases),
        "retrieval_modes": {k: {"strict": v["strict"], "alias": v["alias"]} for k, v in modes.items()},
        "by_group_confident_match": {k: v.get("primary_confident_match") for k, v in by_group.items()},
        "error_taxonomy_recount": dict(cat_counts),
    }
    with open(DATA_DIR / "rescore_retrieval_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved rescore_retrieval_report.json")


if __name__ == "__main__":
    main()
