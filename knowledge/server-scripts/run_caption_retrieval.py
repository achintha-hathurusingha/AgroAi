#!/usr/bin/env python3
"""
Stage G-2: Caption-then-retrieve, retrieval-only evaluation. Literature-precedented
alternative to raw cross-modal SigLIP embedding (which truncates ~18% of corpus facts
at 64 tokens): have Qwen3-VL describe the visible symptoms in a short sentence, embed
that caption with BGE (384-token capacity, no truncation risk for our corpus), and
retrieve against the existing BGE text corpus embeddings. Tests whether bridging
image->short-text->text-retrieve beats direct image-embedding cross-modal retrieval
(Stage C: R@1 3.3%, R@5 20.0% alias-corrected on confident_match) or the current
pipeline's full free-form Qwen query (R@1 1.3%, R@5 6.2% alias-corrected).

This is deliberately NOT the same as the current pipeline's query construction (Arm 2):
that prompt asks for a full diagnostic description; this one asks ONLY for a short,
neutral symptom description, closer to what caption-then-retrieve literature uses, and
short enough to stay a cheap, focused BGE query.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from scoring import fact_matches_ground_truth

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BGE_MODEL = "TaylorAI/bge-micro-v2"
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
K_LIST = (1, 5, 20)

CAPTION_PROMPT = (
    "Describe only the visible symptoms on this leaf in one short sentence (under 25 "
    "words): spots, discoloration, texture, holes, wilting, powdery/fuzzy coating, or "
    "other abnormalities. Do not name a disease or species, just describe what you see. "
    "If the leaf looks healthy and unmarked, say \"no visible symptoms.\""
)


def generate_caption(model, processor, image):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": CAPTION_PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


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


def metrics(ranks, k_list=K_LIST):
    n = len(ranks)
    if n == 0:
        return {f"recall@{k}": 0.0 for k in k_list} | {"mrr": 0.0, "n": 0}
    m = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in k_list}
    m["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    m["n"] = n
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    print("loading corpus + BGE embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    bge_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")

    print("loading eval set...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = json.load(f)
    cases = [c for c in evalset if c["eval_group"] in PRIMARY_GROUPS]
    if args.limit:
        cases = cases[:args.limit]
    print(f"  {len(cases)} cases")

    print("loading Qwen3-VL (for captioning only)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    print("generating short symptom captions...")
    captions = {}
    cap_path = DATA_DIR / "caption_retrieval_captions.jsonl"
    with open(cap_path, "w") as cap_f:
        for i, c in enumerate(cases):
            image = Image.open(c["image_path"]).convert("RGB")
            cap = generate_caption(model, processor, image)
            captions[c["case_id"]] = cap
            cap_f.write(json.dumps({"case_id": c["case_id"], "caption": cap}) + "\n")
            cap_f.flush()
            if i % 50 == 0:
                print(f"  [{i}/{len(cases)}] {c['case_id'][:40]}: {cap[:70]}")

    del model
    torch.cuda.empty_cache()

    print("\nloading BGE, embedding captions...")
    bge_model = SentenceTransformer(BGE_MODEL, device="cuda")
    texts = [captions[c["case_id"]] for c in cases]
    query_emb = bge_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).cpu().numpy().astype("float32")

    print("\ncomputing ranks...")
    sim_all = query_emb @ bge_corpus_emb.T
    strict_ranks, alias_ranks = [], []
    for i, c in enumerate(cases):
        gt = c["ground_truth_disease"]
        strict_ranks.append(strict_rank(sim_all[i], corpus, gt))
        alias_ranks.append(alias_rank(sim_all[i], corpus, gt))

    overall = {"strict": metrics(strict_ranks), "alias": metrics(alias_ranks)}
    print(f"overall: strict R@1={overall['strict']['recall@1']:.3f}  alias R@1={overall['alias']['recall@1']:.3f} R@5={overall['alias']['recall@5']:.3f}")

    by_group = {}
    for group in PRIMARY_GROUPS:
        idx = [i for i, c in enumerate(cases) if c["eval_group"] == group]
        sr = [strict_ranks[i] for i in idx]
        ar = [alias_ranks[i] for i in idx]
        by_group[group] = {"strict": metrics(sr), "alias": metrics(ar)}
        print(f"  {group:26s} strict R@1={by_group[group]['strict']['recall@1']:.3f}  "
              f"alias R@1={by_group[group]['alias']['recall@1']:.3f} R@5={by_group[group]['alias']['recall@5']:.3f}")

    out = {"n_cases": len(cases), "overall": overall, "by_group": by_group,
           "per_case_alias_rank": {c["case_id"]: r for c, r in zip(cases, alias_ranks)}}
    out_path = DATA_DIR / "caption_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
