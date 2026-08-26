#!/usr/bin/env python3
"""Cross-reference PlantVillage's 38 classes against the frozen diagnostic corpus's
disease vocabulary, to ground Phase 2 eval-set construction in real coverage numbers."""
import json
from collections import Counter
from pathlib import Path

with open("frozen/agmmu_phase2_diagnostic_v1.json") as f:
    corpus = json.load(f)
disease_counts = Counter((f.get("disease") or "").strip().lower() for f in corpus if f.get("disease"))

pv_classes = sorted(d.name for d in Path("data/plantvillage/raw/color").iterdir())

mapping_hints = {
    "Apple___Apple_scab": ["apple scab"],
    "Apple___Black_rot": ["black rot"],
    "Apple___Cedar_apple_rust": ["cedar apple rust"],
    "Apple___healthy": ["healthy"],
    "Blueberry___healthy": ["healthy"],
    "Cherry_(including_sour)___healthy": ["healthy"],
    "Cherry_(including_sour)___Powdery_mildew": ["powdery mildew"],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["gray leaf spot", "cercospora"],
    "Corn_(maize)___Common_rust_": ["common rust", "corn rust"],
    "Corn_(maize)___healthy": ["healthy"],
    "Corn_(maize)___Northern_Leaf_Blight": ["northern leaf blight", "northern corn leaf blight"],
    "Grape___Black_rot": ["black rot"],
    "Grape___Esca_(Black_Measles)": ["esca", "black measles"],
    "Grape___healthy": ["healthy"],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["isariopsis", "leaf blight"],
    "Orange___Haunglongbing_(Citrus_greening)": ["citrus greening", "huanglongbing"],
    "Peach___Bacterial_spot": ["bacterial spot", "bacterial leaf spot"],
    "Peach___healthy": ["healthy"],
    "Pepper,_bell___Bacterial_spot": ["bacterial spot", "bacterial leaf spot"],
    "Pepper,_bell___healthy": ["healthy"],
    "Potato___Early_blight": ["early blight"],
    "Potato___healthy": ["healthy"],
    "Potato___Late_blight": ["late blight"],
    "Raspberry___healthy": ["healthy"],
    "Soybean___healthy": ["healthy"],
    "Squash___Powdery_mildew": ["powdery mildew"],
    "Strawberry___healthy": ["healthy"],
    "Strawberry___Leaf_scorch": ["leaf scorch"],
    "Tomato___Bacterial_spot": ["bacterial spot", "bacterial leaf spot"],
    "Tomato___Early_blight": ["early blight"],
    "Tomato___healthy": ["healthy"],
    "Tomato___Late_blight": ["late blight"],
    "Tomato___Leaf_Mold": ["leaf mold"],
    "Tomato___Septoria_leaf_spot": ["septoria leaf spot", "septoria"],
    "Tomato___Spider_mites Two-spotted_spider_mite": ["spider mite"],
    "Tomato___Target_Spot": ["target spot"],
    "Tomato___Tomato_mosaic_virus": ["mosaic virus", "tomato mosaic"],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["yellow leaf curl", "curl virus"],
}

print(f"{'PV class':55s} {'best match':30s} count")
matched = 0
results = {}
for cls in pv_classes:
    hints = mapping_hints.get(cls, [])
    best = None
    bestcount = 0
    for term in hints:
        for d, c in disease_counts.items():
            if term in d and c > bestcount:
                best = d
                bestcount = c
    if best:
        matched += 1
    results[cls] = {"matched_disease": best, "corpus_count": bestcount}
    print(f"{cls:55s} {str(best):30s} {bestcount}")

print()
print("matched classes:", matched, "/", len(pv_classes))

with open("data/pv_corpus_coverage.json", "w") as f:
    json.dump(results, f, indent=2)
