#!/usr/bin/env python3
"""
Milestone 3 acceptance test, per the agreed spec:
1. Qwen generates the Prompt C structured query on a known test image.
2. BGE generates its embedding.
3. NumPy searches the deduplicated diagnostic corpus.
4. Return Top-5 facts.
5. Record fact_id, disease, similarity, rank, query text.
6. Determine whether the correct disease appears in Top-1/3/5.

Test image: the same Cedar Apple Rust PlantVillage image used throughout Phase 1
(first the raw zero-shot test, then Q1/Q2 experiments) -- for direct before/after
comparability against the original un-grounded Qwen3-VL result.
"""
import json
from pathlib import Path

import torch
from PIL import Image

import agrivision_pipeline as pipeline

TEST_IMAGE = Path(
    "/home/minura/agrivision-rag/data/plantvillage/raw/color/Apple___Cedar_apple_rust/"
    "96a1d021-2c27-46a0-9891-41e449e4910e___FREC_C.Rust 3610.JPG"
)
GROUND_TRUTH_DISEASE = "cedar apple rust"
OUT_PATH = Path("/home/minura/agrivision-rag/data/milestone3_acceptance_result.json")


def main():
    print("building/loading diagnostic corpus...")
    corpus = pipeline.build_diagnostic_corpus()
    print(f"  {len(corpus)} facts")

    print("embedding corpus with BGE-micro-v2 (cached if available)...")
    corpus_emb = pipeline.embed_corpus(corpus)
    print(f"  embedding matrix shape: {corpus_emb.shape}")

    print("\nloading Qwen3-VL...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(pipeline.QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(pipeline.QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    image = Image.open(TEST_IMAGE).convert("RGB")
    print(f"\nstep 1: generating Prompt C query for {TEST_IMAGE.name} ...")
    query_text = pipeline.generate_prompt_c_query(model, processor, image)
    print(f"query text:\n{query_text}\n")

    del model
    torch.cuda.empty_cache()

    print("step 2: embedding query with BGE-micro-v2...")
    query_emb = pipeline.embed_query(query_text)

    print("step 3: NumPy exact cosine retrieval over deduplicated diagnostic corpus...")
    results = pipeline.retrieve(query_emb, corpus_emb, corpus, k=5)

    print("\nstep 4/5: Top-5 results")
    for r in results:
        print(f"  rank={r['rank']}  disease={r['disease']!r}  species={r['species_raw']!r}  sim={r['similarity']:.4f}  fact_id={r['fact_id']}")

    target = GROUND_TRUTH_DISEASE.strip().lower()
    hit_rank = next((r["rank"] for r in results if r["disease"].strip().lower() == target), None)
    top1 = hit_rank == 1
    top3 = hit_rank is not None and hit_rank <= 3
    top5 = hit_rank is not None and hit_rank <= 5

    print(f"\nstep 6: ground truth = {GROUND_TRUTH_DISEASE!r}")
    print(f"  hit rank: {hit_rank}")
    print(f"  Top-1: {top1}   Top-3: {top3}   Top-5: {top5}")

    output = {
        "test_image": str(TEST_IMAGE), "ground_truth_disease": GROUND_TRUTH_DISEASE,
        "query_text": query_text, "top5_results": results,
        "hit_rank": hit_rank, "top1": top1, "top3": top3, "top5": top5,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
