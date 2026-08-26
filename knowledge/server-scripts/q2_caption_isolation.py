#!/usr/bin/env python3
"""
Phase 1B follow-up: Q2 caption-failure isolation experiment.
Same 7 PlantVillage images as the original vlm_caption queries. Generates 3 new prompt
variants (B: visual-evidence-only, forbids species/disease naming; C: structured,
explicit anti-inference; D: same prompt as the original A, but 400 tokens instead of 128
-- isolates the truncation hypothesis from the prompt-wording hypothesis). Scores all
variants with BGE-micro-v2 (the Q1 winner) against the same frozen pool used in Phase 1B.
"""
import json
from pathlib import Path

import torch
from PIL import Image

PV_DIR = Path("/home/minura/agrivision-rag/data/plantvillage/raw/color")
POOL_PATH = Path("/home/minura/agrivision-rag/data/q1q2_pool.json")
OUT_PATH = Path("/home/minura/agrivision-rag/data/q2_isolation_results.json")
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BGE_MODEL = "TaylorAI/bge-micro-v2"

# same 7 (query_id, target_disease, image_class, exact_filename) as the original vlm_caption queries
CASES = [
    ("D1", "cedar apple rust", "Apple___Cedar_apple_rust", "96a1d021-2c27-46a0-9891-41e449e4910e___FREC_C.Rust 3610.JPG"),
    ("D2", "apple scab", "Apple___Apple_scab", "c27b45af-c362-4d37-ab5b-ee197a3670de___FREC_Scab 3521.JPG"),
    ("D3", "black rot", "Apple___Black_rot", "1cdac481-1275-4228-a2ca-e3deaf268ffc___JR_FrgE.S 2897.JPG"),
    ("D4", "powdery mildew", "Cherry_(including_sour)___Powdery_mildew", "3c726192-540f-4ce5-9590-0aa8500e195c___FREC_Pwd.M 0547.JPG"),
    ("D5", "septoria leaf spot", "Tomato___Septoria_leaf_spot", "9b52dfbf-f3d7-4777-a620-7eaa6d3086aa___Matt.S_CG 6156.JPG"),
    ("D6", "early blight", "Potato___Early_blight", "51e028c1-7aec-46e9-a389-391bc38b46f3___RS_Early.B 8505.JPG"),
    ("D7", "bacterial leaf spot", "Peach___Bacterial_spot", "81d21a0e-8551-41e5-b8d6-b5840fc6cda9___Rut._Bact.S 1400.JPG"),
]

PROMPT_B = (
    "Describe only the visible visual evidence in this leaf image. Do not identify the "
    "plant species or name any disease. Mention leaf shape, color, lesions, spots, "
    "patterns, texture, and any other observable abnormalities."
)
PROMPT_C = (
    "Examine this leaf image and respond in this exact format:\n"
    "Species: unknown\n"
    "Visible symptoms:\n- \n- \n- \n"
    "Do not infer or guess the species or disease name unless it is visually unambiguous. "
    "List only what you directly observe (color, shape, spots, texture, patterns, damage)."
)
PROMPT_D = (
    "You are an expert plant pathologist. Look at this leaf image and describe what you "
    "see: the plant species, and any visible disease symptoms or damage. Be specific and "
    "descriptive about the visual appearance. Do not just name a disease -- describe what "
    "you actually observe."
)


def generate(model, processor, image, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def embed_bge(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    del model
    torch.cuda.empty_cache()
    return emb.cpu()


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    print("loading Qwen3-VL...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    results = []
    for query_id, target_disease, cls, fname in CASES:
        img_path = PV_DIR / cls / fname
        image = Image.open(img_path).convert("RGB")
        print(f"\n=== {query_id} ({target_disease}) ===")

        cap_b = generate(model, processor, image, PROMPT_B, 128)
        print(f"  B (visual-only): {cap_b[:200]}")
        cap_c = generate(model, processor, image, PROMPT_C, 128)
        print(f"  C (structured):  {cap_c[:200]}")
        cap_d = generate(model, processor, image, PROMPT_D, 400)
        print(f"  D (longer, 400tok): {cap_d[:200]}")

        results.append({
            "query_id": query_id, "target_disease": target_disease, "image_class": cls,
            "prompt_b_text": cap_b, "prompt_c_text": cap_c, "prompt_d_text": cap_d,
        })

    del model
    torch.cuda.empty_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved raw captions to {OUT_PATH}")


if __name__ == "__main__":
    main()
