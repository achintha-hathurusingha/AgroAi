#!/usr/bin/env python3
"""
Diagnostic experiment: is the full-scale retrieval failure caused by (a) the VLM never
extracting the disease-discriminating visual feature, or (b) something about the
corpus/embedding representation itself? 4 cases, 4 conditions each, BGE-micro-v2 vs the
full 17,583-fact corpus. No new Qwen3-VL calls -- all text below is either the actual
cached Prompt C output, the actual cached hand-authored benchmark text, or a documented
edit of the Prompt C output (never fabricated from scratch).
"""
import json
from pathlib import Path

import numpy as np

import agrivision_pipeline as pipeline

OUT_PATH = Path("/home/minura/agrivision-rag/data/q2_bottleneck_diagnosis.json")

# condition "qwen": actual cached Prompt C output (verbatim, from q2_isolation_results.json)
# condition "human": actual cached hand-authored species_symptoms (Q3) text (verbatim, from q1q2_query_spec.json)
# condition "hybrid": the qwen text with ONE line appended -- the disease's known missing
#   discriminative feature, drawn from the same hand-authored symptom descriptions already
#   reviewed earlier in this project (not new invented claims)
# condition "oracle_species": the qwen text with "Species: unknown" replaced by the correct species

CASES = {
    "D1_cedar_apple_rust": {
        "target_disease": "cedar apple rust",
        "qwen": (
            "Species: unknown\nVisible symptoms:\n- Leaf is green with irregular brown spots\n"
            "- Some spots appear as small, scattered lesions\n- Leaf margin is slightly serrated\n"
            "- Veins are visible and appear darker than surrounding leaf tissue\n"
            "- Leaf tip is pointed\n- Leaf base has a short petiole attached"
        ),
        "human": "apple tree leaves with bright orange-yellow spots and orange tubular growths underneath",
        "hybrid_extra": "- Orange gelatinous tube-like structures visible on the underside of the leaf",
        "correct_species": "apple",
    },
    "D6_early_blight": {
        "target_disease": "early blight",
        "qwen": (
            "Species: unknown\nVisible symptoms:\n- Leaf is green with visible veins\n"
            "- Numerous dark brown to black spots scattered across the surface\n"
            "- Some spots are clustered, others are isolated\n"
            "- Leaf margin appears slightly curled or irregular\n"
            "- Some areas of the leaf show necrotic or discolored tissue\n"
            "- Leaf surface appears somewhat wrinkled or textured\n"
            "- Tip of leaf is slightly curled or damaged\n- Overall leaf shape is ovate with a pointed tip"
        ),
        "human": "potato plant with bullseye-patterned brown spots on the older lower leaves",
        "hybrid_extra": "- Spots show concentric target-like rings (bullseye pattern)",
        "correct_species": "potato",
    },
    "D7_bacterial_leaf_spot": {
        "target_disease": "bacterial leaf spot",
        "qwen": (
            "Species: unknown\nVisible symptoms:\n- elongated leaf with wavy margins\n"
            "- surface appears textured or puckered\n- small brown spots near midrib\n"
            "- tip of leaf is curled upward\n- leaf is green with some browning at tip and along margin"
        ),
        "human": "peach tree with small dark angular leaf spots and a shot-hole appearance, and sunken cracked spots on the fruit",
        "hybrid_extra": "- Some spots have fallen out of the leaf tissue, leaving a shot-hole appearance",
        "correct_species": "peach",
    },
    "D4_powdery_mildew": {
        "target_disease": "powdery mildew",
        "qwen": (
            "Species: unknown\nVisible symptoms:\n- Leaf is heart-shaped with a pointed tip\n"
            "- Surface has mottled green and lighter green patterns\n"
            "- Veins are visible and prominent\n- Edges appear slightly curled or irregular\n"
            "- Small brownish spot near the base of the petiole\n"
            "- Overall texture appears somewhat wrinkled or puckered"
        ),
        "human": "cherry tree leaves covered in a white powdery coating with some curling and distortion",
        "hybrid_extra": "- Surface has a white to grayish powdery coating",
        "correct_species": "cherry",
    },
}


def build_conditions():
    conditions = {}
    for case_id, c in CASES.items():
        qwen = c["qwen"]
        hybrid = qwen + "\n" + c["hybrid_extra"]
        oracle = qwen.replace("Species: unknown", f"Species: {c['correct_species']}", 1)
        conditions[case_id] = {
            "target_disease": c["target_disease"],
            "qwen": qwen,
            "human": c["human"],
            "hybrid": hybrid,
            "oracle_species": oracle,
        }
    return conditions


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(pipeline.BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return emb.cpu().numpy().astype("float32")


def full_rank(query_emb, corpus_emb, corpus, target_disease):
    sims = corpus_emb @ query_emb
    order = np.argsort(-sims)
    target = target_disease.strip().lower()
    for rank, idx in enumerate(order, start=1):
        if (corpus[idx].get("disease") or "").strip().lower() == target:
            return rank, float(sims[order[0]]), corpus[order[0]].get("disease", "")
    return None, None, None


def main():
    print("loading diagnostic corpus + cached BGE embeddings...")
    corpus = pipeline.build_diagnostic_corpus()
    corpus_emb = np.load(pipeline.EMB_CACHE)
    print(f"  corpus: {len(corpus)} facts")

    conditions = build_conditions()
    results = {}
    for case_id, c in conditions.items():
        print(f"\n=== {case_id} (target: {c['target_disease']}) ===")
        results[case_id] = {"target_disease": c["target_disease"], "conditions": {}}
        for cond_name in ["qwen", "human", "hybrid", "oracle_species"]:
            text = c[cond_name]
            q_emb = embed([text])[0]
            rank, top1_sim, top1_disease = full_rank(q_emb, corpus_emb, corpus, c["target_disease"])
            results[case_id]["conditions"][cond_name] = {
                "text": text, "rank": rank, "top1_sim": top1_sim, "top1_disease": top1_disease,
            }
            print(f"  {cond_name:16s} rank={str(rank):8s} top1={top1_disease!r}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY: rank by case x condition ===")
    print(f"{'case':28s} {'qwen':>8s} {'human':>8s} {'hybrid':>8s} {'oracle_sp':>10s}")
    for case_id, r in results.items():
        c = r["conditions"]
        print(f"{case_id:28s} {str(c['qwen']['rank']):>8s} {str(c['human']['rank']):>8s} {str(c['hybrid']['rank']):>8s} {str(c['oracle_species']['rank']):>10s}")


if __name__ == "__main__":
    main()
