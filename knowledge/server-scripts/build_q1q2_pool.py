#!/usr/bin/env python3
"""
Phase 1B step 1: build the frozen ~400-fact sample pool per phase1a-preprocessing-decisions.md
and phase1b-q1q2-experiment-design.md. Deterministic (fixed seed) so the pool is reproducible.
"""
import json
import random
from collections import Counter
from pathlib import Path

RAW_PATH = Path("/home/minura/agrivision-rag/data/agmmu/agmmu_ft_hf1.json")
OUT_PATH = Path("/home/minura/agrivision-rag/data/q1q2_pool.json")
SEED = 42
TARGET_DISEASES = [
    "cedar apple rust", "apple scab", "black rot", "powdery mildew",
    "septoria leaf spot", "early blight", "bacterial leaf spot", "fire blight",
]
CAP_PER_DISEASE = 20
TOTAL_POOL_SIZE = 400


def get_str(qa, field):
    v = qa.get(field, {})
    a = v.get("a", "") if isinstance(v, dict) else ""
    return a if isinstance(a, str) else ""


def normalize_entry(e):
    """Turn one raw AgMMU entry into a list of canonical facts (usually 1; >1 if the
    answer was a dict keyed by species, per the Phase 1A decision to split, not exclude)."""
    qa = e.get("finetuning qa", {})
    species = get_str(qa, "species")

    # detect dict-shaped (multi-species) answers on any field
    dict_fields = {}
    for field in qa:
        a = qa[field].get("a") if isinstance(qa[field], dict) else None
        if isinstance(a, dict):
            dict_fields[field] = a

    if not dict_fields:
        disease = get_str(qa, "disease/issue identification")
        pest = get_str(qa, "insect/pest")
        return [{
            "source_faq_id": e.get("faq-id"),
            "species_raw": species,
            "disease": disease,
            "pest": pest,
            "symptoms": get_str(qa, "symptom description"),
            "management": get_str(qa, "management instructions"),
            "image_description": get_str(qa, "image description"),
        }]

    # split: one fact per species key found in any dict-shaped field
    species_keys = set()
    for a in dict_fields.values():
        species_keys.update(a.keys())
    facts = []
    for sk in species_keys:
        facts.append({
            "source_faq_id": e.get("faq-id"),
            "species_raw": sk,
            "disease": dict_fields.get("disease/issue identification", {}).get(sk, "") if "disease/issue identification" in dict_fields else get_str(qa, "disease/issue identification"),
            "pest": dict_fields.get("insect/pest", {}).get(sk, "") if "insect/pest" in dict_fields else get_str(qa, "insect/pest"),
            "symptoms": dict_fields.get("symptom description", {}).get(sk, "") if "symptom description" in dict_fields else get_str(qa, "symptom description"),
            "management": dict_fields.get("management instructions", {}).get(sk, "") if "management instructions" in dict_fields else get_str(qa, "management instructions"),
            "image_description": get_str(qa, "image description"),
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


def main():
    random.seed(SEED)
    with open(RAW_PATH) as f:
        raw = json.load(f)

    all_facts = []
    for e in raw:
        all_facts.extend(normalize_entry(e))

    diagnostic = [f for f in all_facts if f.get("disease") or f.get("pest")]
    print(f"raw entries: {len(raw)}, normalized facts: {len(all_facts)}, diagnostic facts: {len(diagnostic)}")

    # dedupe diagnostic corpus by fingerprint, assign fact_id, keep first occurrence + collect source ids
    seen = {}
    for f in diagnostic:
        fp = fingerprint(f)
        if fp not in seen:
            f["fact_id"] = f"fact_{len(seen):06d}"
            f["source_faq_ids"] = [f["source_faq_id"]]
            seen[fp] = f
        else:
            seen[fp]["source_faq_ids"].append(f["source_faq_id"])
    deduped = list(seen.values())
    print(f"deduped diagnostic facts: {len(deduped)}")

    def disease_of(f):
        return (f.get("disease") or "").strip().lower()

    target_pool = []
    for td in TARGET_DISEASES:
        matches = [f for f in deduped if disease_of(f) == td]
        random.shuffle(matches)
        capped = matches[:CAP_PER_DISEASE]
        print(f"  target disease {td!r}: {len(matches)} unique facts available, using {len(capped)}")
        target_pool.extend(capped)

    target_fact_ids = {f["fact_id"] for f in target_pool}
    distractor_candidates = [f for f in deduped if disease_of(f) not in TARGET_DISEASES and f["fact_id"] not in target_fact_ids]
    random.shuffle(distractor_candidates)
    n_distractors = max(0, TOTAL_POOL_SIZE - len(target_pool))
    distractors = distractor_candidates[:n_distractors]
    print(f"distractors: {len(distractors)} (from {len(distractor_candidates)} candidates)")

    pool = target_pool + distractors
    for f in pool:
        f["retrieval_text"] = retrieval_text(f)
        f.pop("source_faq_id", None)

    print(f"final pool size: {len(pool)}")

    with open(OUT_PATH, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
