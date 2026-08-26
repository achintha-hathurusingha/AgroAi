#!/usr/bin/env python3
"""
Phase 1B step 2: run the frozen Q1/Q2 experiment.
1. Fill in the 7 vlm_caption queries using real Qwen3-VL output on real PlantVillage images.
2. Embed the ~400-fact pool + 47 queries with three candidate embedders.
3. Retrieve top-5 per query per embedder, log full records.
4. Score Recall@1/@3/@5 and MRR, overall and per query_type.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

SPEC_PATH = Path("/home/minura/agrivision-rag/q1q2_query_spec.json")
POOL_PATH = Path("/home/minura/agrivision-rag/data/q1q2_pool.json")
PV_DIR = Path("/home/minura/agrivision-rag/data/plantvillage/raw/color")
OUT_DIR = Path("/home/minura/agrivision-rag/data")
SEED = 42

SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "TaylorAI/bge-micro-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def fill_vlm_captions(spec):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    random.seed(SEED)
    print("loading Qwen3-VL for caption generation...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    prompt = (
        "You are an expert plant pathologist. Look at this leaf image and describe what you "
        "see: the plant species, and any visible disease symptoms or damage. Be specific and "
        "descriptive about the visual appearance. Do not just name a disease -- describe what "
        "you actually observe."
    )

    for q in spec["queries"]:
        if q["query_type"] != "vlm_caption":
            continue
        cls_dir = PV_DIR / q["source_image_class"]
        images = list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg"))
        img_path = random.choice(images)
        image = Image.open(img_path).convert("RGB")

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        caption = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        q["query_text"] = caption
        q["source_image_file"] = str(img_path.name)
        print(f"  {q['query_id']} ({q['source_image_class']}): {caption[:150]}")

    del model
    torch.cuda.empty_cache()
    return spec


def embed_sentence_transformers(model_name, texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    del model
    torch.cuda.empty_cache()
    return emb.cpu()


def embed_siglip_text(texts):
    from transformers import AutoProcessor, AutoModel
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()
    embs = []
    batch_size = 32
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
    return torch.cat(embs, dim=0)


def retrieve_and_score(embedder_name, query_embs, pool_embs, queries, pool, k_list=(1, 3, 5)):
    sims = query_embs @ pool_embs.T  # cosine sim since both normalized
    records = []
    per_query_hits = {}

    for qi, q in enumerate(queries):
        scores = sims[qi]
        topk_vals, topk_idx = torch.topk(scores, k=min(20, len(pool)))
        target = q["target_disease"].strip().lower()
        first_hit_rank = None
        for rank, (idx, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
            fact = pool[idx]
            is_relevant = (fact.get("disease") or "").strip().lower() == target
            records.append({
                "query_id": q["query_id"], "embedder": embedder_name, "rank": rank,
                "fact_id": fact["fact_id"], "similarity": val, "disease": fact.get("disease", ""),
                "relevant": is_relevant,
            })
            if is_relevant and first_hit_rank is None:
                first_hit_rank = rank
        per_query_hits[q["query_id"]] = first_hit_rank

    metrics = {}
    for k in k_list:
        hits = sum(1 for r in per_query_hits.values() if r is not None and r <= k)
        metrics[f"recall@{k}"] = hits / len(queries)
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in per_query_hits.values()) / len(queries)
    metrics["mrr"] = mrr

    by_type = defaultdict(list)
    for q in queries:
        by_type[q["query_type"]].append(per_query_hits[q["query_id"]])
    type_metrics = {}
    for qtype, ranks in by_type.items():
        tm = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in k_list}
        tm["mrr"] = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)
        tm["n"] = len(ranks)
        type_metrics[qtype] = tm

    return records, metrics, type_metrics


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)
    with open(POOL_PATH) as f:
        pool = json.load(f)

    spec = fill_vlm_captions(spec)
    with open(OUT_DIR / "q1q2_query_spec_filled.json", "w") as f:
        json.dump(spec, f, indent=2)

    queries = spec["queries"]
    query_texts = [q["query_text"] for q in queries]
    pool_texts = [f["retrieval_text"] for f in pool]

    all_records = []
    all_metrics = {}
    all_type_metrics = {}

    print("\n=== embedding with sentence-transformers (all-MiniLM-L6-v2) ===")
    q_emb = embed_sentence_transformers(SENTENCE_TRANSFORMERS_MODEL, query_texts)
    p_emb = embed_sentence_transformers(SENTENCE_TRANSFORMERS_MODEL, pool_texts)
    records, metrics, type_metrics = retrieve_and_score("sentence-transformers", q_emb, p_emb, queries, pool)
    all_records += records
    all_metrics["sentence-transformers"] = metrics
    all_type_metrics["sentence-transformers"] = type_metrics
    print(metrics)

    print("\n=== embedding with BGE-micro-v2 ===")
    q_emb = embed_sentence_transformers(BGE_MODEL, query_texts)
    p_emb = embed_sentence_transformers(BGE_MODEL, pool_texts)
    records, metrics, type_metrics = retrieve_and_score("bge-micro-v2", q_emb, p_emb, queries, pool)
    all_records += records
    all_metrics["bge-micro-v2"] = metrics
    all_type_metrics["bge-micro-v2"] = type_metrics
    print(metrics)

    print("\n=== embedding with SigLIP text tower ===")
    q_emb = embed_siglip_text(query_texts)
    p_emb = embed_siglip_text(pool_texts)
    records, metrics, type_metrics = retrieve_and_score("siglip-text", q_emb, p_emb, queries, pool)
    all_records += records
    all_metrics["siglip-text"] = metrics
    all_type_metrics["siglip-text"] = type_metrics
    print(metrics)

    with open(OUT_DIR / "q1q2_results_records.json", "w") as f:
        json.dump(all_records, f, indent=2)
    with open(OUT_DIR / "q1q2_results_summary.json", "w") as f:
        json.dump({"overall": all_metrics, "by_query_type": all_type_metrics}, f, indent=2)

    print("\n=== SUMMARY (overall) ===")
    for name, m in all_metrics.items():
        print(f"{name:20s}  R@1={m['recall@1']:.3f}  R@3={m['recall@3']:.3f}  R@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}")

    print("\n=== SUMMARY (by query type) ===")
    for name, tm in all_type_metrics.items():
        print(f"\n{name}:")
        for qtype, m in tm.items():
            print(f"  {qtype:20s} n={m['n']:2d}  R@1={m['recall@1']:.3f}  R@3={m['recall@3']:.3f}  R@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}")


if __name__ == "__main__":
    main()
