#!/usr/bin/env python3
"""
Phase 2 eval-set construction, per phase2-execution-plan.md section 1.
186 primary cases (16 confident-match diseased classes + 12 healthy + 3 no-match
negative-control classes, 6 images each) + 20 supplementary AgMMU-sourced cases.
Deterministic (seed=42). Excludes images already used in prior Phase 1 hard-case
experiments.
"""
import json
import random
from pathlib import Path

PV_DIR = Path("/home/minura/agrivision-rag/data/plantvillage/raw/color")
AGMMU_EVAL_JSON = Path("/home/minura/agrivision-rag/data/agmmu/agmmu_e_filtered_hf1.json")
AGMMU_IMAGES_DIR = Path("/home/minura/agrivision-rag/data/agmmu")
COVERAGE_PATH = Path("/home/minura/agrivision-rag/data/pv_corpus_coverage.json")
OUT_PATH = Path("/home/minura/agrivision-rag/data/phase2_evalset.json")

SEED = 42
N_PER_CLASS = 15  # scaled up from the plan's 6 after pilot timing came in at ~21s/case (vs. 30-60s estimated)
N_AGMMU_SUPPLEMENTARY = 40

CONFIDENT_MATCH_MIN_COUNT = 5

# images already used in prior Phase 1 experiments -- excluded from Phase 2 sampling
PREVIOUSLY_STUDIED = {
    "96a1d021-2c27-46a0-9891-41e449e4910e___FREC_C.Rust 3610.JPG",
    "c27b45af-c362-4d37-ab5b-ee197a3670de___FREC_Scab 3521.JPG",
    "1cdac481-1275-4228-a2ca-e3deaf268ffc___JR_FrgE.S 2897.JPG",
    "3c726192-540f-4ce5-9590-0aa8500e195c___FREC_Pwd.M 0547.JPG",
    "9b52dfbf-f3d7-4777-a620-7eaa6d3086aa___Matt.S_CG 6156.JPG",
    "51e028c1-7aec-46e9-a389-391bc38b46f3___RS_Early.B 8505.JPG",
    "81d21a0e-8551-41e5-b8d6-b5840fc6cda9___Rut._Bact.S 1400.JPG",
    "1a67f47d-d35d-4c47-a336-af8ec6141113___FREC_C.Rust 4404.JPG",
}

NO_MATCH_CLASSES = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Esca_(Black_Measles)",
]


def parse_ground_truth(pv_class):
    """PV class -> (species, disease_or_healthy)"""
    parts = pv_class.split("___")
    species = parts[0].replace("_", " ").strip()
    label = parts[1].replace("_", " ").strip() if len(parts) > 1 else "unknown"
    is_healthy = label.lower() == "healthy"
    return species, label, is_healthy


def main():
    random.seed(SEED)
    with open(COVERAGE_PATH) as f:
        coverage = json.load(f)

    confident_classes = [c for c, v in coverage.items() if v["matched_disease"] and v["corpus_count"] >= CONFIDENT_MATCH_MIN_COUNT]
    healthy_classes = [d.name for d in PV_DIR.iterdir() if d.name.split("___")[-1].lower() == "healthy"]
    weak_match_classes = [c for c, v in coverage.items() if v["matched_disease"] and 0 < v["corpus_count"] < CONFIDENT_MATCH_MIN_COUNT]

    print(f"confident-match classes: {len(confident_classes)}")
    print(f"healthy classes: {len(healthy_classes)}")
    print(f"no-match (negative control) classes: {len(NO_MATCH_CLASSES)}")
    print(f"weak-match (supplementary) classes: {len(weak_match_classes)}")

    def sample_class(cls, group):
        cls_dir = PV_DIR / cls
        images = [p for p in (list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg"))) if p.name not in PREVIOUSLY_STUDIED]
        chosen = random.sample(images, min(N_PER_CLASS, len(images)))
        species, label, is_healthy = parse_ground_truth(cls)
        cases = []
        for img in chosen:
            cases.append({
                "case_id": f"{cls}__{img.stem}",
                "image_path": str(img),
                "pv_class": cls,
                "species": species,
                "ground_truth_disease": "healthy" if is_healthy else label.lower(),
                "ground_truth_source": "plantvillage",
                "eval_group": group,
            })
        return cases

    all_cases = []
    for cls in confident_classes:
        all_cases += sample_class(cls, "primary_confident_match")
    for cls in healthy_classes:
        all_cases += sample_class(cls, "primary_healthy")
    for cls in NO_MATCH_CLASSES:
        all_cases += sample_class(cls, "primary_negative_control")
    for cls in weak_match_classes:
        all_cases += sample_class(cls, "supplementary_weak_match")

    # AgMMU supplementary cases
    with open(AGMMU_EVAL_JSON) as f:
        agmmu_eval = json.load(f)
    random.shuffle(agmmu_eval)
    agmmu_cases = []
    for entry in agmmu_eval:
        if len(agmmu_cases) >= N_AGMMU_SUPPLEMENTARY:
            break
        images = entry.get("images", [])
        if not images:
            continue
        img_rel = images[0].replace("./images/", "copied_images/")
        img_path = AGMMU_IMAGES_DIR / img_rel
        if not img_path.exists():
            continue
        answer = entry.get("answer", "")
        agmmu_cases.append({
            "case_id": f"agmmu_{entry.get('faq-id')}",
            "image_path": str(img_path),
            "pv_class": None,
            "species": None,
            "ground_truth_disease": answer.strip().lower(),
            "ground_truth_source": "agmmu_eval",
            "eval_group": "supplementary_agmmu",
            "question": entry.get("question"),
            "options": entry.get("options"),
        })

    all_cases += agmmu_cases

    print(f"\ntotal cases: {len(all_cases)}")
    from collections import Counter
    print(Counter(c["eval_group"] for c in all_cases))

    with open(OUT_PATH, "w") as f:
        json.dump(all_cases, f, indent=2)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
