#!/usr/bin/env python3
"""
Experiment 1: Oracle Retrieval. Same 465 primary Phase 2 cases, same diagnosis prompt
(v2), same diagnostic corpus + BGE retrieval -- but the retrieval QUERY is the
ground-truth disease name itself, not a Qwen3-VL-generated description. This measures
the ceiling: if retrieval is guaranteed to surface the right evidence, does Qwen3-VL
actually make a better diagnosis, or does it still get it wrong?

CRITICAL: the ground truth is used ONLY to construct the retrieval query. It is never
passed to the final diagnosis call -- that call sees only the image and the retrieved
evidence text, identical to how Arm 2 works, so this is a fair test of "what if retrieval
were perfect" not "what if the model were told the answer."
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import agrivision_pipeline as pipeline

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
EVALSET_PATH = DATA_DIR / "phase2_evalset.json"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TOP_K = 5
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}

# identical diagnosis prompt to the v2 fix in run_phase2_ablation.py, for direct comparability
DIAGNOSIS_PROMPT_WITH_EVIDENCE_TMPL = (
    "You are an expert plant pathologist reviewing a leaf image submitted specifically "
    "because the grower suspected a problem. Most such images do show some sign of "
    "disease, pest damage, or stress -- carefully inspect for spots, lesions, "
    "discoloration, powdery or fuzzy coatings, holes, wilting, curling, or any other "
    "abnormality before concluding the leaf is healthy. Any spots or discoloration should "
    "be treated as a likely disease sign, not dismissed as natural wear, unless clearly "
    "just an artifact of leaf age near the stem/edges. Here is retrieved reference "
    "information that may or may not be relevant:\n\n{evidence}\n\n"
    "Based on the image and, where relevant, the reference information above, respond in "
    "exactly this format:\n"
    "Observed signs: <describe every visible spot, lesion, discoloration, texture change, "
    "or abnormality, or state \"none\" only if the leaf is genuinely uniform and unmarked>\n"
    "Diagnosis: <a specific disease name if any signs were observed; \"healthy\" ONLY if "
    "you stated \"none\" above. If signs were observed, give your single best-guess "
    "specific diagnosis even if uncertain -- a specific guess is more useful than "
    "declining to answer. Only answer \"unknown\" if signs were observed and you truly "
    "have no plausible guess at all.>\n"
    "Confidence: <a number from 0 to 100>\n"
    "Reasoning: <one to two sentences explaining your diagnosis and whether the reference information supported it>"
)


def generate(model, processor, image, prompt, max_new_tokens=200):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_diagnosis(text):
    import re
    d = {"observed_signs": None, "diagnosis": None, "confidence": None, "reasoning": None}
    for line in text.splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith("observed signs:"):
            d["observed_signs"] = line.split(":", 1)[1].strip()
        elif low.startswith("diagnosis:"):
            d["diagnosis"] = line.split(":", 1)[1].strip().strip('"')
        elif low.startswith("confidence:"):
            m = re.search(r"\d+(\.\d+)?", line.split(":", 1)[1])
            if m:
                d["confidence"] = float(m.group())
        elif low.startswith("reasoning:"):
            d["reasoning"] = line.split(":", 1)[1].strip()
    return d


def retrieve(query_emb, corpus_emb, corpus, k):
    sims = corpus_emb @ query_emb
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [{"rank": r + 1, "fact_id": corpus[i].get("fact_id"), "disease": corpus[i].get("disease", ""),
              "similarity": float(sims[i]), "retrieval_text": corpus[i]["retrieval_text"]}
             for r, i in enumerate(idx)]


def evidence_block(facts):
    return "\n\n".join(f"[{i+1}] {f['retrieval_text']}" for i, f in enumerate(facts))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()

    print("loading eval set (primary cases only)...")
    with open(EVALSET_PATH) as f:
        all_cases = json.load(f)
    cases = [c for c in all_cases if c["eval_group"] in PRIMARY_GROUPS]
    if args.stride:
        cases = cases[::args.stride]
    if args.limit:
        cases = cases[:args.limit]
    print(f"  {len(cases)} primary cases")

    print("loading corpus + BGE embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        diagnostic = json.load(f)
    bge_diag_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")

    print("loading Qwen3-VL...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    print("loading BGE (query embedder)...")
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer(pipeline.BGE_MODEL, device="cuda")

    out_path = DATA_DIR / "oracle_retrieval_results.jsonl"
    error_path = out_path.with_suffix(".errors.log")
    n_done = 0
    n_errors = 0
    with open(out_path, "w") as out_f, open(error_path, "w") as err_f:
        for case in cases:
            case_id = case["case_id"]
            try:
                image = Image.open(case["image_path"]).convert("RGB")
                gt = case["ground_truth_disease"]

                # ORACLE QUERY: the ground-truth disease name itself, used only for retrieval
                oracle_query_text = gt
                q_emb = bge_model.encode([oracle_query_text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).cpu().numpy().astype("float32")[0]
                retrieved = retrieve(q_emb, bge_diag_emb, diagnostic, TOP_K)
                evidence = evidence_block(retrieved)

                diag_text = generate(model, processor, image, DIAGNOSIS_PROMPT_WITH_EVIDENCE_TMPL.format(evidence=evidence))
                parsed = parse_diagnosis(diag_text)

                result = {
                    "case_id": case_id, "ground_truth_disease": gt, "eval_group": case["eval_group"],
                    "oracle_query_text": oracle_query_text, "retrieved": retrieved,
                    "diagnosis_raw": diag_text, **parsed,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_done += 1
                correct = (parsed.get("diagnosis") or "").strip().lower() == gt.strip().lower()
                print(f"[{n_done}/{len(cases)}] {case_id}: oracle_diag={parsed.get('diagnosis')!r} gt={gt!r} correct={correct}")
            except Exception as e:
                n_errors += 1
                import traceback
                err_f.write(f"{case_id}: {e}\n{traceback.format_exc()}\n\n")
                err_f.flush()
                continue

    print(f"\ndone. {n_done} completed, {n_errors} errors. Output: {out_path}")


if __name__ == "__main__":
    main()
