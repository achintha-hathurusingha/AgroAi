# Sri Lanka–specific Data Sources for AgriVision-RAG

Web research 2026-08-26 into Sri Lanka–specific agricultural data, to complement the
general CVPR-paper datasets in [data-finding-plan.md](data-finding-plan.md). Relevant
if the project should demonstrate local/regional applicability (a differentiator from
generic PlantVillage-style projects).

## Findings

### Weligama Coconut Leaf Wilt Disease (WCWLD) + Coconut Caterpillar Infestation
- Paper: "Early Diagnosis and Severity Assessment of Weligama Coconut Leaf Wilt Disease
  and Coconut Caterpillar Infestation using Deep Learning-based Image Processing
  Techniques" — arXiv: https://arxiv.org/abs/2501.18835
- Authors: Samitha Vidhanaarachchi, Janaka L. Wijekoon, W. A. Shanaka P.
  Abeysiriwardhana, Malitha Wijesundara (2025).
- Methods: transfer-learning CNN + Mask R-CNN for early diagnosis/severity assessment,
  YOLO for caterpillar counting.
- **Dataset access not confirmed by search** — no public dataset link found; would need
  to read the paper directly or contact the authors. Coconut is a major Sri Lankan
  export crop, so this is the most locally-relevant disease-severity paper found.
- Related: a second paper on UAV-based multispectral detection of the same disease —
  https://link.springer.com/article/10.1007/s41348-025-01115-z (object-based
  classification from drone imagery, likely not public dataset either — worth checking).

### data.gov.lk — Sri Lanka Open Data Portal (ICTA)
- Agriculture & Livelihood category: https://data.gov.lk/search/field_topic/agriculture-and-livelihood-18
- 45 datasets total: Production (24), Agriculture (17), Fertilizer (4), Performance (2), Crop (1).
- Formats: mostly Excel (41), PDF (29), CSV (26) — **statistical/administrative data
  (crop forecasts, production figures), not image data.**
- Use: not for training vision models, but a plausible source of *structured facts* for
  the RAG knowledge base (e.g. crop production stats, seasonal forecasts) if the
  project wants Sri Lanka–specific retrieval facts alongside AgMMU's general corpus.
- Example: "Crop Forecast: Maha 2022/23" dataset.

### General fruit/disease datasets surfaced (not Sri Lanka–specific, but usable)
- Fruits-360 (Kaggle) — 90,380 images, 131 fruit/vegetable classes, 100×100px RGB,
  67,692 train / 22,688 test.
- Guava leaf & fruit disease dataset (PMC) — anthracnose, scab, styler root end, leaf disease.
- Pomegranate disease datasets (Halabja, Iraq — not Sri Lanka, but same crop family
  relevance): 2,178 original + 28,314 augmented images, 4 classes.

## Assessment

No ready-to-download Sri Lanka–specific **image** dataset was found for fruit/crop
disease — the Weligama coconut paper is the closest match but its dataset isn't
confirmed publicly available. Two realistic paths:

1. **Contact the WCWLD paper authors** (Vidhanaarachchi et al., likely reachable via a
   Sri Lankan university — check the paper's affiliation) to ask about dataset access.
   This would be the strongest "locally relevant" data point for the project.
2. **Collect original field data** — since qbits/devon are both accessible GPU servers
   and this is a Sri Lanka-based FYP, photographing local fruit/crop samples
   (mango, banana, coconut, guava — common Sri Lankan crops) would produce a genuinely
   original, differentiated dataset rather than relying entirely on existing corpora.
   Even a small (few-hundred-image) locally-collected eval set would let the project
   claim real-world, non-benchmark validation.

## Follow-ups
- [ ] Check WCWLD paper's affiliation/contact info for dataset request.
- [ ] Check the UAV multispectral WCWLD paper for a public dataset too.
- [ ] If pursuing original data collection, decide on target crops (coconut, mango,
  banana are strong candidates — all major Sri Lankan crops with existing disease
  literature to cross-reference).
