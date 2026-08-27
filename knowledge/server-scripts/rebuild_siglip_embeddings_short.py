#!/usr/bin/env python3
"""
Stage G-1: test the SigLIP-truncation hypothesis. SigLIP's text tower caps at 64
tokens (confirmed: corpus facts average 43.8 tokens, 17.8% exceed 64 and get silently
cut). This re-embeds the corpus with a SHORTENED, truncation-safe text per fact
(disease/pest name + a short symptom snippet, prioritized so the most useful signal
survives truncation) and re-runs the Stage C-style visual retrieval-only eval, to see
whether truncation was actually suppressing SigLIP's retrieval quality or not.
CPU-friendly (SigLIP-base text tower over ~17.6K short strings), no VLM needed.
"""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
PV_IMAGES_DIR = DATA_DIR / "pv_images"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
MAX_WORDS = 40  # ~64 tokens at this corpus's ~1.4 tokens/word ratio, with margin


def short_retrieval_text(fact):
    """Disease/pest name first (highest-value, token-cheap), then as much symptom
    text as fits, prioritized over species/management which matter less for matching
    an image to a specific disease."""
    parts = []
    disease = fact.get("disease") or fact.get("pest") or ""
    if disease:
        parts.append(f"Disease: {disease}.")
    symptoms = fact.get("symptoms") or ""
    if symptoms:
        parts.append(f"Symptoms: {symptoms}")
    text = " ".join(parts)
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    return text


def strict_rank(scores, corpus, gt):
    order = np.argsort(-scores)
    target = gt.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank
    return None


def alias_rank(scores, corpus, gt):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from scoring import fact_matches_ground_truth
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        if fact_matches_ground_truth(corpus[idx].get("disease"), gt):
            return rank
    return None


def metrics(ranks, k_list=(1, 5, 20)):
    n = len(ranks)
    m = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in k_list}
    m["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    m["n"] = n
    return m


def main():
    import torch
    from transformers import AutoProcessor, AutoModel
    from PIL import Image

    print("loading corpus...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)

    print("building shortened retrieval texts...")
    short_texts = [short_retrieval_text(f) for f in corpus]
    lens = [len(t.split()) for t in short_texts]
    print(f"  mean words: {sum(lens)/len(lens):.1f}  max: {max(lens)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("loading SigLIP...")
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32).to(device)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()

    # verify truncation is actually fixed now
    tok = processor.tokenizer
    tok_lens = [len(tok(t)["input_ids"]) for t in short_texts[:2000]]
    print(f"  sample token lengths: mean={sum(tok_lens)/len(tok_lens):.1f} max={max(tok_lens)} "
          f"fraction>=64: {sum(1 for l in tok_lens if l>=64)/len(tok_lens):.1%}")

    print("embedding shortened corpus texts with SigLIP text tower...")
    batch_size = 64
    embs = []
    for i in range(0, len(short_texts), batch_size):
        batch = short_texts[i:i + batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feat = torch.nn.functional.normalize(out.float(), dim=-1)
        embs.append(feat.cpu())
        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(short_texts)}")
    corpus_emb_short = torch.cat(embs, dim=0).numpy().astype("float32")
    np.save(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb_short.npy", corpus_emb_short)
    print(f"saved shortened SigLIP corpus embeddings: {corpus_emb_short.shape}")

    print("\nloading evalset + embedding query images (same as Stage C)...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = {c["case_id"]: c for c in json.load(f)}
    cases = [c for c in evalset.values() if c["eval_group"] in PRIMARY_GROUPS]
    print(f"  {len(cases)} cases")

    query_embs = []
    batch_size_img = 16
    for i in range(0, len(cases), batch_size_img):
        batch = cases[i:i + batch_size_img]
        images = [Image.open(PV_IMAGES_DIR / Path(c["image_path"]).name).convert("RGB") for c in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        query_embs.append(feat.cpu())
        if i % 80 == 0:
            print(f"  {i}/{len(cases)}")
    query_emb = torch.cat(query_embs, dim=0).numpy().astype("float32")

    print("\ncomputing similarity + ranks (short-text corpus embeddings)...")
    sim_all = query_emb @ corpus_emb_short.T
    strict_ranks, alias_ranks = [], []
    for i, c in enumerate(cases):
        gt = c["ground_truth_disease"]
        strict_ranks.append(strict_rank(sim_all[i], corpus, gt))
        alias_ranks.append(alias_rank(sim_all[i], corpus, gt))

    overall = {"strict": metrics(strict_ranks), "alias": metrics(alias_ranks)}
    print(f"overall: strict R@1={overall['strict']['recall@1']:.3f} R@5={overall['strict']['recall@5']:.3f}  "
          f"alias R@1={overall['alias']['recall@1']:.3f} R@5={overall['alias']['recall@5']:.3f}")

    by_group = {}
    for group in PRIMARY_GROUPS:
        idx = [i for i, c in enumerate(cases) if c["eval_group"] == group]
        sr = [strict_ranks[i] for i in idx]
        ar = [alias_ranks[i] for i in idx]
        by_group[group] = {"strict": metrics(sr), "alias": metrics(ar)}
        print(f"{group:26s} strict R@1={by_group[group]['strict']['recall@1']:.3f} R@5={by_group[group]['strict']['recall@5']:.3f}  "
              f"alias R@1={by_group[group]['alias']['recall@1']:.3f} R@5={by_group[group]['alias']['recall@5']:.3f}")

    out = {"n_cases": len(cases), "overall": overall, "by_group": by_group,
           "per_case_alias_rank": {c["case_id"]: r for c, r in zip(cases, alias_ranks)},
           "corpus_word_stats": {"mean": sum(lens) / len(lens), "max": max(lens)}}
    with open(DATA_DIR / "siglip_short_retrieval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved siglip_short_retrieval_results.json")


if __name__ == "__main__":
    main()
