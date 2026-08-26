#!/usr/bin/env python3
"""
Small Q2 validation: does explicitly prompting for disease-diagnostic visual features
(not just "describe what you see") close the gap on the hard cases from the isolation
experiment? 4 cases: two known-hard misses (cedar apple rust, early blight), one where a
wrong species claim previously hurt retrieval (bacterial spot), one known success as a
control (powdery mildew, where Prompt C already hit rank 1). Compares new Prompt E
against the already-known Prompt C results for the same 4 images.
"""
import json
from pathlib import Path

import torch
from PIL import Image

PV_DIR = Path("/home/minura/agrivision-rag/data/plantvillage/raw/color")
POOL_PATH = Path("/home/minura/agrivision-rag/data/q1q2_pool.json")
OUT_PATH = Path("/home/minura/agrivision-rag/data/q2_prompt_e_results.json")
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BGE_MODEL = "TaylorAI/bge-micro-v2"

CASES = [
    ("D1", "cedar apple rust", "Apple___Cedar_apple_rust", "96a1d021-2c27-46a0-9891-41e449e4910e___FREC_C.Rust 3610.JPG", "missing distinctive feature (orange tubes)"),
    ("D6", "early blight", "Potato___Early_blight", "51e028c1-7aec-46e9-a389-391bc38b46f3___RS_Early.B 8505.JPG", "missing distinctive feature (concentric rings)"),
    ("D7", "bacterial leaf spot", "Peach___Bacterial_spot", "81d21a0e-8551-41e5-b8d6-b5840fc6cda9___Rut._Bact.S 1400.JPG", "wrong species previously hurt retrieval"),
    ("D4", "powdery mildew", "Cherry_(including_sour)___Powdery_mildew", "3c726192-540f-4ce5-9590-0aa8500e195c___FREC_Pwd.M 0547.JPG", "control (already succeeded with prompt C)"),
]

PROMPT_E = (
    "Examine this leaf image and respond in this exact format:\n"
    "Species: unknown\n"
    "Visible diagnostic features:\n"
    "- Lesion color: \n"
    "- Lesion shape: \n"
    "- Lesion distribution/pattern: \n"
    "- Presence of halos: \n"
    "- Presence of concentric rings: \n"
    "- Surface texture/structures: \n"
    "- Underside structures (if visible): \n"
    "- Leaf deformation/curling: \n"
    "- Necrosis/tissue death: \n"
    "- Fruit symptoms (if visible): \n"
    "Only report what is visually observable -- write 'not observed' for any feature that "
    "is not visible or not applicable. Do not infer or guess the species or disease name."
)

# known Prompt C results for the same 4 images, from the prior isolation experiment
PROMPT_C_BASELINE = {
    "D1": {"first_relevant_rank": None, "top1_disease": "bacterial leaf spot"},
    "D6": {"first_relevant_rank": 12, "top1_disease": "septoria leaf spot"},
    "D7": {"first_relevant_rank": 1, "top1_disease": "bacterial leaf spot"},
    "D4": {"first_relevant_rank": 1, "top1_disease": "powdery mildew"},
}


def generate_caption(model, processor, image, prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def embed(texts):
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
    for query_id, target_disease, cls, fname, failure_type in CASES:
        img_path = PV_DIR / cls / fname
        image = Image.open(img_path).convert("RGB")
        caption = generate_caption(model, processor, image, PROMPT_E)
        print(f"\n=== {query_id} ({target_disease}) [{failure_type}] ===")
        print(caption)
        results.append({"query_id": query_id, "target_disease": target_disease, "failure_type": failure_type, "prompt_e_text": caption})

    del model
    torch.cuda.empty_cache()

    with open(POOL_PATH) as f:
        pool = json.load(f)
    pool_texts = [f["retrieval_text"] for f in pool]
    p_emb = embed(pool_texts)
    q_emb = embed([r["prompt_e_text"] for r in results])
    sims = q_emb @ p_emb.T

    print("\n=== E vs C comparison ===")
    for i, r in enumerate(results):
        scores = sims[i]
        topk_vals, topk_idx = torch.topk(scores, k=20)
        target = r["target_disease"].strip().lower()
        first_rank = None
        for rank, idx in enumerate(topk_idx.tolist(), start=1):
            if (pool[idx].get("disease") or "").strip().lower() == target:
                first_rank = rank
                break
        top1_disease = pool[topk_idx[0].item()]["disease"]
        r["prompt_e_first_relevant_rank"] = first_rank
        r["prompt_e_top1_disease"] = top1_disease
        c = PROMPT_C_BASELINE[r["query_id"]]
        print(f"{r['query_id']} ({r['failure_type']}): C_rank={c['first_relevant_rank']} C_top1={c['top1_disease']!r}  ->  E_rank={first_rank} E_top1={top1_disease!r}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
