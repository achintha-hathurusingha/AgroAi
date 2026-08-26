#!/usr/bin/env python3
"""
Tests a CPJ-style (Caption-Prompt-Judge) iterative refinement loop against the same 4
hard cases from the bottleneck diagnosis. Unlike every prior Q2 experiment (single-shot
caption generation), this gives Qwen3-VL a chance to revise a caption an LLM-judge scores
as low-quality, before it's used as the retrieval query -- following arxiv.org/html/2512.24947.
Uses Qwen3-VL as both captioner and judge (consistent with CPJ's own tested setup, which
uses a VLM of comparable class for caption generation).
"""
import json
from pathlib import Path

import torch
from PIL import Image

import agrivision_pipeline as pipeline

OUT_PATH = Path("/home/minura/agrivision-rag/data/cpj_refinement_results.json")
JUDGE_THRESHOLD = 8.0
MAX_REFINEMENTS = 2

CASES = [
    ("D1_cedar_apple_rust", "cedar apple rust", "Apple___Cedar_apple_rust",
     "96a1d021-2c27-46a0-9891-41e449e4910e___FREC_C.Rust 3610.JPG"),
    ("D6_early_blight", "early blight", "Potato___Early_blight",
     "51e028c1-7aec-46e9-a389-391bc38b46f3___RS_Early.B 8505.JPG"),
    ("D7_bacterial_leaf_spot", "bacterial leaf spot", "Peach___Bacterial_spot",
     "81d21a0e-8551-41e5-b8d6-b5840fc6cda9___Rut._Bact.S 1400.JPG"),
    ("D4_powdery_mildew", "powdery mildew", "Cherry_(including_sour)___Powdery_mildew",
     "3c726192-540f-4ce5-9590-0aa8500e195c___FREC_Pwd.M 0547.JPG"),
]

CAPTION_PROMPT = (
    "Examine this leaf image and respond in this exact format:\n"
    "Species: unknown\n"
    "Visible symptoms:\n- \n- \n- \n"
    "Do not infer or guess the species or disease name unless it is visually unambiguous. "
    "List only what you directly observe (color, shape, spots, texture, patterns, damage)."
)

JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating the quality of a plant-symptom description that will be used as "
    "a search query in a disease-diagnosis retrieval system. Look at the image and the "
    "description below, then rate the description from 1 to 10 based on: (1) accuracy -- "
    "does it match what's visible, (2) completeness -- does it capture ALL notable visual "
    "details, especially any distinctive patterns, textures, colors, or structures, (3) "
    "specificity -- is it precise rather than vague/generic.\n\n"
    "Description:\n{caption}\n\n"
    "Respond in exactly this format:\nScore: <number 1-10>\nMissing: <one sentence on what "
    "specific visual detail, if any, is missing or under-described>"
)

REFINEMENT_PROMPT_TEMPLATE = (
    "Your previous description of this leaf image was rated {score}/10. "
    "Feedback: {feedback}\n\n"
    "Provide an improved, more detailed and specific description of this leaf's visible "
    "symptoms, paying special attention to the feedback above -- especially any "
    "distinctive patterns, textures, colors, or structures that may have been missed. "
    "Respond in this exact format:\nSpecies: unknown\nVisible symptoms:\n- \n- \n- \n"
    "Do not name the plant species or disease."
)


def generate(model, processor, image, prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_judge_output(text):
    score = None
    feedback = ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip().split("/")[0].strip())
            except ValueError:
                pass
        elif line.lower().startswith("missing:"):
            feedback = line.split(":", 1)[1].strip()
    return score, feedback


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(pipeline.BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return emb.cpu().numpy().astype("float32")


def full_rank(query_emb, corpus_emb, corpus, target_disease):
    import numpy as np
    sims = corpus_emb @ query_emb
    order = np.argsort(-sims)
    target = target_disease.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank
    return None


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    print("loading Qwen3-VL...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(pipeline.QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(pipeline.QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    print("loading diagnostic corpus + cached BGE embeddings...")
    corpus = pipeline.build_diagnostic_corpus()
    import numpy as np
    corpus_emb = np.load(pipeline.EMB_CACHE)

    results = {}
    for case_id, target_disease, cls, fname in CASES:
        img_path = Path(f"/home/minura/agrivision-rag/data/plantvillage/raw/color/{cls}/{fname}")
        image = Image.open(img_path).convert("RGB")
        print(f"\n{'=' * 20} {case_id} (target: {target_disease}) {'=' * 20}")

        history = []
        caption = generate(model, processor, image, CAPTION_PROMPT)
        print(f"initial caption:\n{caption}\n")

        for iteration in range(MAX_REFINEMENTS + 1):
            judge_text = generate(model, processor, image, JUDGE_PROMPT_TEMPLATE.format(caption=caption), max_new_tokens=100)
            score, feedback = parse_judge_output(judge_text)
            print(f"  iteration {iteration}: judge_score={score} feedback={feedback!r}")
            history.append({"iteration": iteration, "caption": caption, "judge_score": score, "judge_feedback": feedback})

            if score is not None and score >= JUDGE_THRESHOLD:
                print("  score meets threshold, stopping refinement")
                break
            if iteration == MAX_REFINEMENTS:
                print("  max refinements reached, stopping")
                break

            refine_prompt = REFINEMENT_PROMPT_TEMPLATE.format(score=score if score is not None else "unknown", feedback=feedback or "no specific feedback given")
            caption = generate(model, processor, image, refine_prompt)
            print(f"  refined caption:\n{caption}\n")

        initial_caption = history[0]["caption"]
        final_caption = history[-1]["caption"]

        q_emb_initial = embed([initial_caption])[0]
        q_emb_final = embed([final_caption])[0]
        rank_initial = full_rank(q_emb_initial, corpus_emb, corpus, target_disease)
        rank_final = full_rank(q_emb_final, corpus_emb, corpus, target_disease)
        print(f"  RESULT: initial_rank={rank_initial}  final_rank={rank_final}")

        results[case_id] = {
            "target_disease": target_disease, "history": history,
            "initial_caption": initial_caption, "final_caption": final_caption,
            "rank_initial": rank_initial, "rank_final": rank_final,
            "n_refinements": len(history) - 1,
        }

    del model
    torch.cuda.empty_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY: initial vs CPJ-refined rank ===")
    for case_id, r in results.items():
        print(f"{case_id:28s} initial={str(r['rank_initial']):8s} final={str(r['rank_final']):8s} n_refinements={r['n_refinements']}")


if __name__ == "__main__":
    main()
