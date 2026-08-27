#!/usr/bin/env python3
"""
Phase 2 main ablation runner, per phase2-execution-plan.md.
Usage: python3 run_phase2_ablation.py --limit 10   (pilot)
       python3 run_phase2_ablation.py               (full run, all primary+supplementary cases)

Checkpointed: appends one JSON line per completed case to the output JSONL immediately,
so a crash loses at most the in-progress case. Per-case errors are caught and logged,
never kill the run.
"""
import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = Path("/home/minura/agrivision-rag/frozen")
EVALSET_PATH = DATA_DIR / "phase2_evalset.json"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BGE_MODEL = "TaylorAI/bge-micro-v2"
TOP_K = 5
TOP_N_RERANK = 20

QUERY_PROMPT = (
    "Examine this leaf image and respond in this exact format:\n"
    "Species: unknown\nVisible symptoms:\n- \n- \n- \n"
    "Do not infer or guess the species or disease name unless it is visually unambiguous. "
    "List only what you directly observe (color, shape, spots, texture, patterns, damage)."
)

DIAGNOSIS_PROMPT_NO_EVIDENCE = (
    "You are an expert plant pathologist reviewing a leaf image submitted specifically "
    "because the grower suspected a problem. Most such images do show some sign of "
    "disease, pest damage, or stress -- carefully inspect for spots, lesions, "
    "discoloration, powdery or fuzzy coatings, holes, wilting, curling, or any other "
    "abnormality before concluding the leaf is healthy. Any spots or discoloration should "
    "be treated as a likely disease sign, not dismissed as natural wear, unless clearly "
    "just an artifact of leaf age near the stem/edges.\n\n"
    "Respond in exactly this format:\n"
    "Observed signs: <describe every visible spot, lesion, discoloration, texture change, "
    "or abnormality, or state \"none\" only if the leaf is genuinely uniform and unmarked>\n"
    "Diagnosis: <a specific disease name if any signs were observed; \"healthy\" ONLY if "
    "you stated \"none\" above. If signs were observed, give your single best-guess "
    "specific diagnosis even if uncertain -- a specific guess is more useful than "
    "declining to answer. Only answer \"unknown\" if signs were observed and you truly "
    "have no plausible guess at all.>\n"
    "Confidence: <a number from 0 to 100>\n"
    "Reasoning: <one to two sentences explaining your diagnosis>"
)

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

RERANK_PROMPT_TMPL = (
    "A plant symptom query is being matched against candidate reference facts. Query:\n"
    "{query}\n\nCandidate facts:\n{candidates}\n\n"
    "List the numbers of the {k} candidates most relevant to diagnosing the query, most "
    "relevant first, comma-separated (e.g. \"3, 1, 7, 12, 5\")."
)


def retrieval_text(fact):
    lines = []
    if fact.get("species_raw"):
        lines.append(f"Species: {fact['species_raw']}")
    if fact.get("disease"):
        lines.append(f"Disease/Issue: {fact['disease']}")
    if fact.get("pest"):
        lines.append(f"Pest: {fact['pest']}")
    if fact.get("symptoms"):
        lines.append(f"Symptoms: {fact['symptoms']}")
    if fact.get("management"):
        lines.append(f"Management: {fact['management']}")
    return "\n".join(lines)


def generate(model, processor, image, prompt, max_new_tokens=200):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        # NOTE: do_sample=False (greedy) was tried first for reproducibility; Arm 1
        # predicted "healthy" in 80% of 610 cases (true rate 29.5%). Reverting to sampling
        # barely changed this (78.4% on a diverse 51-case cross-section) -- the actual
        # cause was the diagnosis prompt itself offering "healthy"/"unknown" as easy
        # escape hatches, causing the model to rationalize real visible symptoms away as
        # "natural wear." Fixed by rewriting the prompt (see DIAGNOSIS_PROMPT_* above),
        # not by decoding settings. Kept sampling since Phase 1B never showed collapse.
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_diagnosis(text):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="pilot mode: only run first N cases")
    ap.add_argument("--stride", type=int, default=None, help="take every Nth case for a diverse cross-section instead of the first N sequential (same-class) cases")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else DATA_DIR / (f"phase2_pilot_results.jsonl" if args.limit else "phase2_full_results.jsonl")
    error_log_path = out_path.with_suffix(".errors.log")

    print("loading eval set...")
    with open(EVALSET_PATH) as f:
        cases = json.load(f)
    if args.stride:
        cases = cases[::args.stride]
    if args.limit:
        cases = cases[:args.limit]
    print(f"running {len(cases)} cases, output -> {out_path}")

    print("loading corpora + embeddings...")
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        diagnostic = json.load(f)
    with open(FROZEN_DIR / "agmmu_phase2_context_v1.json") as f:
        context = json.load(f)
    full_corpus = diagnostic + context

    bge_diag_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy")
    bge_full_emb = np.load(FROZEN_DIR / "agmmu_phase2_full_v1_bge_emb.npy")
    siglip_diag_emb = np.load(FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy")

    print("loading Qwen3-VL...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL, quantization_config=bnb_config, device_map="cuda:0")

    print("loading BGE (query embedder)...")
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer(BGE_MODEL, device="cuda")

    print("loading SigLIP (query embedder for arm 3)...")
    from transformers import AutoProcessor as AutoProc2, AutoModel as AutoModel2
    siglip_model = AutoModel2.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float16).to("cuda")
    siglip_processor = AutoProc2.from_pretrained("google/siglip-base-patch16-224")

    def embed_bge_query(text):
        return bge_model.encode([text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).cpu().numpy().astype("float32")[0]

    def embed_siglip_query(text):
        inputs = siglip_processor(text=[text], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = siglip_model.get_text_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        return feat.cpu().numpy().astype("float32")[0]

    n_done = 0
    n_errors = 0
    with open(out_path, "a") as out_f, open(error_log_path, "a") as err_f:
        for case in cases:
            case_id = case["case_id"]
            try:
                image = Image.open(case["image_path"]).convert("RGB")
                result = {"case_id": case_id, "ground_truth_disease": case["ground_truth_disease"],
                          "ground_truth_source": case["ground_truth_source"], "eval_group": case["eval_group"],
                          "arms": {}}

                # shared query generation
                query_text = generate(model, processor, image, QUERY_PROMPT, max_new_tokens=150)
                result["query_text"] = query_text

                # Arm 1: VLM only
                diag_text = generate(model, processor, image, DIAGNOSIS_PROMPT_NO_EVIDENCE)
                result["arms"]["1_vlm_only"] = {"diagnosis_raw": diag_text, **parse_diagnosis(diag_text)}

                q_emb_bge = embed_bge_query(query_text)
                q_emb_siglip = embed_siglip_query(query_text)

                arm_configs = {
                    "2_rag_bge": (bge_diag_emb, diagnostic),
                    "3_rag_siglip": (siglip_diag_emb, diagnostic),
                    "4_rag_full_corpus": (bge_full_emb, full_corpus),
                }
                for arm_name, (emb_matrix, corpus_list) in arm_configs.items():
                    q_emb = q_emb_siglip if arm_name == "3_rag_siglip" else q_emb_bge
                    retrieved = retrieve(q_emb, emb_matrix, corpus_list, TOP_K)
                    evidence = evidence_block(retrieved)
                    diag_text = generate(model, processor, image, DIAGNOSIS_PROMPT_WITH_EVIDENCE_TMPL.format(evidence=evidence))
                    judge_text = generate(model, processor, None, JUDGE_PROMPT_TMPL.format(evidence=evidence, reasoning=parse_diagnosis(diag_text).get("reasoning") or ""), max_new_tokens=150)
                    result["arms"][arm_name] = {
                        "retrieved": retrieved, "diagnosis_raw": diag_text, **parse_diagnosis(diag_text),
                        "judge_raw": judge_text, "judge": parse_judge(judge_text),
                    }

                # Arm 5: rerank top-20 (BGE) down to top-k via LLM reranker
                retrieved_20 = retrieve(q_emb_bge, bge_diag_emb, diagnostic, TOP_N_RERANK)
                candidates_text = "\n".join(f"{i+1}. {f['retrieval_text']}" for i, f in enumerate(retrieved_20))
                rerank_text = generate(model, processor, None, RERANK_PROMPT_TMPL.format(query=query_text, candidates=candidates_text, k=TOP_K), max_new_tokens=60)
                nums = [int(x) for x in re.findall(r"\d+", rerank_text)][:TOP_K]
                reranked = [retrieved_20[n - 1] for n in nums if 1 <= n <= len(retrieved_20)]
                if not reranked:
                    reranked = retrieved_20[:TOP_K]
                evidence5 = evidence_block(reranked)
                diag_text5 = generate(model, processor, image, DIAGNOSIS_PROMPT_WITH_EVIDENCE_TMPL.format(evidence=evidence5))
                judge_text5 = generate(model, processor, None, JUDGE_PROMPT_TMPL.format(evidence=evidence5, reasoning=parse_diagnosis(diag_text5).get("reasoning") or ""), max_new_tokens=150)
                result["arms"]["5_rag_rerank"] = {
                    "retrieved_pool": retrieved_20, "reranked": reranked, "rerank_raw": rerank_text,
                    "diagnosis_raw": diag_text5, **parse_diagnosis(diag_text5),
                    "judge_raw": judge_text5, "judge": parse_judge(judge_text5),
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_done += 1
                print(f"[{n_done}/{len(cases)}] {case_id}: arm1={result['arms']['1_vlm_only']['diagnosis']!r} arm2={result['arms']['2_rag_bge']['diagnosis']!r} gt={case['ground_truth_disease']!r}")

            except Exception as e:
                n_errors += 1
                err_f.write(f"{case_id}: {e}\n{traceback.format_exc()}\n\n")
                err_f.flush()
                print(f"ERROR on {case_id}: {e}", file=sys.stderr)
                continue

    print(f"\ndone. {n_done} completed, {n_errors} errors. Output: {out_path}")


if __name__ == "__main__":
    main()
