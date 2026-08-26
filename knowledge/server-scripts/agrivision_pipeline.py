#!/usr/bin/env python3
"""
Milestone 3: the first real, minimal end-to-end AgriVision-RAG retrieval pipeline.

    Image -> Qwen3-VL (Prompt C) -> structured query text
                                        |
                                        v
                                  BGE-micro-v2 embedding
                                        |
                                        v
                        NumPy exact cosine vs deduplicated diagnostic corpus
                                        |
                                        v
                                    Top-k evidence

Deliberately minimal per the Q1/Q2/Q5/Q6 decisions: no Qdrant, no FastAPI, no reranker.
Exposes a small retrieve(query_embedding, k) interface so the backend (currently NumPy)
can be swapped later without touching the rest of the pipeline.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAW_PATH = Path("/home/minura/agrivision-rag/data/agmmu/agmmu_ft_hf1.json")
CORPUS_CACHE = Path("/home/minura/agrivision-rag/data/diagnostic_corpus.json")
EMB_CACHE = Path("/home/minura/agrivision-rag/data/diagnostic_corpus_bge_emb.npy")
BGE_MODEL = "TaylorAI/bge-micro-v2"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT_C = (
    "Examine this leaf image and respond in this exact format:\n"
    "Species: unknown\n"
    "Visible symptoms:\n- \n- \n- \n"
    "Do not infer or guess the species or disease name unless it is visually unambiguous. "
    "List only what you directly observe (color, shape, spots, texture, patterns, damage)."
)


def get_str(qa, field):
    v = qa.get(field, {})
    a = v.get("a", "") if isinstance(v, dict) else ""
    return a if isinstance(a, str) else ""


def normalize_entry(e):
    qa = e.get("finetuning qa", {})
    species = get_str(qa, "species")
    dict_fields = {}
    for field in qa:
        a = qa[field].get("a") if isinstance(qa[field], dict) else None
        if isinstance(a, dict):
            dict_fields[field] = a
    if not dict_fields:
        return [{
            "species_raw": species, "disease": get_str(qa, "disease/issue identification"),
            "pest": get_str(qa, "insect/pest"), "symptoms": get_str(qa, "symptom description"),
            "management": get_str(qa, "management instructions"),
        }]
    species_keys = set()
    for a in dict_fields.values():
        species_keys.update(a.keys())
    facts = []
    for sk in species_keys:
        facts.append({
            "species_raw": sk,
            "disease": dict_fields.get("disease/issue identification", {}).get(sk, "") if "disease/issue identification" in dict_fields else get_str(qa, "disease/issue identification"),
            "pest": dict_fields.get("insect/pest", {}).get(sk, "") if "insect/pest" in dict_fields else get_str(qa, "insect/pest"),
            "symptoms": dict_fields.get("symptom description", {}).get(sk, "") if "symptom description" in dict_fields else get_str(qa, "symptom description"),
            "management": dict_fields.get("management instructions", {}).get(sk, "") if "management instructions" in dict_fields else get_str(qa, "management instructions"),
        })
    return facts


def fingerprint(fact):
    parts = [fact.get(k, "") for k in ("species_raw", "disease", "pest", "symptoms", "management")]
    return "|".join(str(p).strip().lower() for p in parts)


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


def build_diagnostic_corpus():
    if CORPUS_CACHE.exists():
        with open(CORPUS_CACHE) as f:
            return json.load(f)
    with open(RAW_PATH) as f:
        raw = json.load(f)
    all_facts = []
    for e in raw:
        all_facts.extend(normalize_entry(e))
    seen = {}
    for f in all_facts:
        fp = fingerprint(f)
        if fp not in seen:
            f["fact_id"] = f"fact_{len(seen):06d}"
            seen[fp] = f
    deduped = list(seen.values())
    diagnostic = [f for f in deduped if f.get("disease") or f.get("pest")]
    for f in diagnostic:
        f["retrieval_text"] = retrieval_text(f)
    with open(CORPUS_CACHE, "w") as f:
        json.dump(diagnostic, f, indent=2)
    return diagnostic


def embed_corpus(corpus):
    if EMB_CACHE.exists():
        return np.load(EMB_CACHE)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL, device="cuda")
    texts = [f["retrieval_text"] for f in corpus]
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    emb = emb.cpu().numpy().astype("float32")
    np.save(EMB_CACHE, emb)
    del model
    torch.cuda.empty_cache()
    return emb


_bge_model = None


def embed_query(text):
    global _bge_model
    from sentence_transformers import SentenceTransformer
    if _bge_model is None:
        _bge_model = SentenceTransformer(BGE_MODEL, device="cuda")
    emb = _bge_model.encode([text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return emb.cpu().numpy().astype("float32")[0]


def retrieve(query_embedding, corpus_emb, corpus, k=5):
    """The swappable retrieval interface -- currently NumPy exact cosine (Q6 decision)."""
    sims = corpus_emb @ query_embedding
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [{"rank": r + 1, "fact_id": corpus[i]["fact_id"], "disease": corpus[i].get("disease", ""),
              "species_raw": corpus[i].get("species_raw", ""), "similarity": float(sims[i])}
             for r, i in enumerate(idx)]


def generate_prompt_c_query(model, processor, image):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_C}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
