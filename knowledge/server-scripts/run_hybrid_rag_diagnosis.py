#!/usr/bin/env python3
"""
Stage E follow-up: Hybrid RAG end-to-end diagnosis. Extends Stage D (pure visual
cross-modal retrieval -> diagnosis) the same way Stage F extended Stage C: combine BGE
text similarity (Qwen-generated query, same queries as the main ablation) with SigLIP
visual similarity (query image) via per-query min-max-normalized weighted sum,
S_H = alpha * norm(S_T) + (1-alpha) * norm(S_V), at alpha=0.25 (Stage F's best R@1 value).
Same 465 primary cases, same corpus, same v2 diagnosis prompt as Stage D and the main
ablation, for direct comparability. Only the retrieval representation changes.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
BGE_MODEL = "TaylorAI/bge-micro-v2"
TOP_K = 5
ALPHA = 0.25  # Stage F best alpha by R@1
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}

# identical to Stage D / main ablation v2 fix, for direct comparability
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

JUDGE_PROMPT_TMPL = (
    "You are auditing an AI plant-disease diagnosis for faithfulness to its supporting "
    "evidence. Below is the evidence that was retrieved and the diagnosis that was "
    "produced from it.\n\nEvidence:\n{evidence}\n\nDiagnosis reasoning:\n{reasoning}\n\n"
    "Break the diagnosis reasoning into its individual factual claims. For each claim, "
    "classify it as ENTAILED (directly supported by the evidence), CONTRADICTED "
    "(conflicts with the evidence), or NEUTRAL (not addressed by the evidence, e.g. a "
    "direct visual observation). Respond in exactly this format:\n"
    "Claims: <total number of claims>\nEntailed: <number entailed>\n"
    "Contradicted: <number contradicted>\nNeutral: <number neutral>\n"
    "Grounded: <YES or NO -- does the overall diagnosis follow reasonably from the evidence?>"
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


def parse_judge(text):
    import re
    d = {"claims": None, "entailed": None, "contradicted": None, "neutral": None, "grounded": None}
    for line in text.splitlines():
        line = line.strip()
        low = line.lower()
        for key in ("claims", "entailed", "contradicted", "neutral"):
            if low.startswith(key + ":"):
                m = re.search(r"\d+", line.split(":", 1)[1])
                if m:
                    d[key] = int(m.group())
        if low.startswith("grounded:"):
            v = line.split(":", 1)[1].strip().upper()
            d["grounded"] = v.startswith("Y")
    return d


def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def retrieve_hybrid(s_t, s_v, corpus, k, alpha=ALPHA):
    s_h = alpha * minmax_norm(s_t) + (1 - alpha) * minmax_norm(s_v)
    idx = np.argpartition(-s_h, k)[:k]
    idx = idx[np.argsort(-s_h[idx])]
    return [{"rank": r + 1, "fact_id": corpus[i].get("fact_id"), "disease": corpus[i].get("disease", ""),
              "similarity": float(s_h[i]), "retrieval_text": corpus[i]["retrieval_text"]}
             for r, i in enumerate(idx)]


def evidence_block(facts):
    return "\n\n".join(f"[{i+1}] {f['retrieval_text']}" for i, f in enumerate(facts))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("loading diagnostic corpus + BGE/SigLIP text embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        corpus = json.load(f)
    bge_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")
    siglip_corpus_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy")

    print("loading eval set + Qwen-generated queries (from main ablation)...")
    with open(DATA_DIR / "phase2_evalset.json") as f:
        evalset = {c["case_id"]: c for c in json.load(f)}
    qwen_queries = {}
    with open(DATA_DIR / "phase2_full_v2_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qwen_queries[r["case_id"]] = r["query_text"]

    cases = [c for c in evalset.values() if c["eval_group"] in PRIMARY_GROUPS and c["case_id"] in qwen_queries]
    if args.stride:
        cases = cases[::args.stride]
    if args.limit:
        cases = cases[:args.limit]
    print(f"  {len(cases)} cases, alpha={ALPHA}")

    print("loading BGE...")
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer(BGE_MODEL, device="cuda")

    print("loading SigLIP...")
    from transformers import AutoProcessor as SP, AutoModel as SM
    siglip_model = SM.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    siglip_processor = SP.from_pretrained(SIGLIP_MODEL)
    siglip_model.eval()

    print("loading Qwen3-VL...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    out_path = Path(args.out) if args.out else DATA_DIR / "hybrid_rag_diagnosis_results.jsonl"
    error_path = out_path.with_suffix(".errors.log")
    n_done = 0
    n_errors = 0
    with open(out_path, "w") as out_f, open(error_path, "w") as err_f:
        for case in cases:
            case_id = case["case_id"]
            try:
                image = Image.open(case["image_path"]).convert("RGB")
                gt = case["ground_truth_disease"]

                query_text = qwen_queries[case_id]
                q_emb_t = bge_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                s_t = bge_corpus_emb @ q_emb_t

                siglip_inputs = siglip_processor(images=image, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    siglip_out = siglip_model.get_image_features(**siglip_inputs)
                feat = siglip_out.pooler_output if hasattr(siglip_out, "pooler_output") else siglip_out
                q_emb_v = torch.nn.functional.normalize(feat.float(), dim=-1).cpu().numpy().astype("float32")[0]
                s_v = siglip_corpus_emb @ q_emb_v

                retrieved = retrieve_hybrid(s_t, s_v, corpus, TOP_K)
                evidence = evidence_block(retrieved)

                diag_text = generate(model, processor, image, DIAGNOSIS_PROMPT_WITH_EVIDENCE_TMPL.format(evidence=evidence))
                parsed = parse_diagnosis(diag_text)
                judge_text = generate(model, processor, None, JUDGE_PROMPT_TMPL.format(evidence=evidence, reasoning=parsed.get("reasoning") or ""), max_new_tokens=150)
                judge = parse_judge(judge_text)

                result = {
                    "case_id": case_id, "ground_truth_disease": gt, "eval_group": case["eval_group"],
                    "retrieved": retrieved, "diagnosis_raw": diag_text, **parsed,
                    "judge_raw": judge_text, "judge": judge,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_done += 1
                correct = (parsed.get("diagnosis") or "").strip().lower() == gt.strip().lower()
                print(f"[{n_done}/{len(cases)}] {case_id}: diag={parsed.get('diagnosis')!r} gt={gt!r} correct={correct}")
            except Exception as e:
                n_errors += 1
                import traceback
                err_f.write(f"{case_id}: {e}\n{traceback.format_exc()}\n\n")
                err_f.flush()
                continue

    print(f"\ndone. {n_done} completed, {n_errors} errors. Output: {out_path}")


if __name__ == "__main__":
    main()
