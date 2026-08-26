#!/usr/bin/env python3
"""
Milestone 3 control case: same pipeline, same code, no changes -- just a different test
image (powdery mildew, a known Prompt-C success from the isolation experiment) to
distinguish "Cedar Apple Rust is a known upstream query-information limitation" from
"something is wrong with the full 17,583-fact pipeline implementation."
"""
import json
from pathlib import Path

import torch
from PIL import Image

import agrivision_pipeline as pipeline

TEST_IMAGE = Path(
    "/home/minura/agrivision-rag/data/plantvillage/raw/color/Cherry_(including_sour)___Powdery_mildew/"
    "3c726192-540f-4ce5-9590-0aa8500e195c___FREC_Pwd.M 0547.JPG"
)
GROUND_TRUTH_DISEASE = "powdery mildew"
OUT_PATH = Path("/home/minura/agrivision-rag/data/milestone3_control_result.json")


def main():
    print("loading diagnostic corpus (cached)...")
    corpus = pipeline.build_diagnostic_corpus()
    print(f"  {len(corpus)} facts")

    print("loading corpus embeddings (cached)...")
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
