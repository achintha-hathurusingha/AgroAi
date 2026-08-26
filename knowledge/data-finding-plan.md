# AgriVision-RAG — Data Finding Plan

Sourcing plan for the datasets the [AgriVision-RAG architecture](AgriVision-RAG-architecture.md)
needs. Each pipeline stage needs different data; this plan maps stage → candidate
dataset(s) → verification status → license, so nothing gets downloaded/used without
knowing what it actually is.

## 1. What we need, by pipeline stage

| Stage | Data needed | Status |
|---|---|---|
| Fruit/crop detection | Bounding-box / detection dataset of fruits on trees or post-harvest | Found (AppleGrowthVision) |
| Disease classification | Labeled leaf/fruit images across many crops & diseases | Found (PlantVillage, AgroBench) |
| Defect/quality grading | Segmentation or region-level defect annotations on produce | Found (banana defect dataset) |
| Fine-grained fruit recognition | Large multi-category fruit image set | Found (Fruit-306), download unverified |
| Freshness/ripeness | Ordinal freshness labels per fruit | Found (Jain et al.), dataset access unverified |
| RAG knowledge base | Structured agricultural facts for retrieval | Found (AgMMU) |
| Evaluation/benchmarking | Expert-annotated VLM benchmark, ground truth for accuracy/hallucination checks | Found (AgroBench) |

## 2. Candidate datasets (verified via web search 2026-08-26)

### AgroBench (ICCV 2025) — evaluation benchmark
- 203 crop categories, 682 disease categories, 7 agricultural topics, expert-agronomist annotated.
- Dataset + code: https://dahlian00.github.io/AgroBenchPage/
- Paper: https://arxiv.org/abs/2507.20519
- Use: primary evaluation/benchmark set for the "Agricultural MLLM Benchmark" component and for sanity-checking disease-diagnosis accuracy against expert ground truth.

### PlantVillage — disease classification baseline
- 14 crop species, 38 classes (disease + healthy), 54,000+ RGB leaf images, controlled imaging conditions.
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease and https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- IEEE DataPort (Feb 2026 republish): https://ieee-dataport.org/documents/plantvillage-plant-disease-classification-dataset
- Use: cheap, well-known baseline to validate the pipeline end-to-end before touching harder data. Caveat: controlled/lab backgrounds, not representative of field conditions — needed only as a smoke-test set, not the final training data.

### Banana defect dataset (Knott et al., CVPR 2025 V4A)
- 476 field images, 1,440 annotated surface defects (bruises/scars), SAM-assisted panoptic masks.
- Data: HuggingFace, **license CC-BY-NC-4.0 (non-commercial only)**
- Code: https://github.com/manuelknott/banana-defect-segmentation
- Paper: https://arxiv.org/abs/2411.16219
- Use: defect/grading pipeline reference implementation + small real-world validation set. **Non-commercial license — do not use for anything beyond research/portfolio demo purposes.**

### AppleGrowthVision (CVPR 2025 V4A)
- 9,317 high-res stereo images, 1,125 densely annotated, six BBCH growth stages, MinneApple-compatible.
- GitHub: https://github.com/fraunhoferhhi/AppleGrowthVision
- Project page: https://fraunhoferhhi.github.io/AppleGrowthVision/
- Paper: https://arxiv.org/abs/2505.14029
- Use: fruit detection pretraining/fine-tuning data; also usable for the growth-stage angle if the project scope expands there.

### Fruit-306 / FruitEnsemble dataset (Yu et al.)
- 116,233 images, 306 fine-grained fruit categories, 7:1:2 train/val/test split, includes category-level text descriptions (useful for RAG facts).
- Paper: https://arxiv.org/pdf/2605.20892
- **Download link not located in search — follow up directly on the paper's project page/appendix or contact authors before relying on this.**

### Jain et al. fruit freshness/quality dataset (CVPR 2025 MTF)
- Ordinal freshness labels, 92.71% avg accuracy baseline reported.
- Paper: https://arxiv.org/abs/2511.01449
- **Dataset access not located in search — check paper's data availability statement directly.**

### SMART dataset (Aug 2026)
- 37,586 images, 48 disease categories, structured captions, long-tail robustness study.
- Could not locate an arXiv/GitHub/HF page in search — **paper itself not yet located; re-search by exact title once more context (venue, authors) is available.**

### AgMMU — RAG knowledge base source (already in [README.md](README.md))
- Project: https://agmmu.github.io/ · GitHub: https://github.com/AgMMU/AgMMU · HF: https://huggingface.co/datasets/AgMMU/AgMMU_v1
- Use: primary source for the "expert-verified facts" retrieval corpus (AgriRAG used 74,611 such facts) and for MCQ-style evaluation of the RAG pipeline.

### General-purpose fruit datasets (need re-verification before use — not confirmed this session)
- MinneApple (referenced as a baseline in the AppleGrowthVision results, not independently searched here)
- Fruits-360 (commonly cited fruit-classification dataset, not independently searched here)
- Mendeley "Fruits Dataset for Classification": https://data.mendeley.com/datasets/rg254yr63x/1
- Kaggle "Fruits and Vegetables Image Recognition": https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

## 3. Priority order

1. **PlantVillage** — smallest, cleanest, fastest to get the SigLIP→RAG→Qwen3-VL pipeline working end-to-end on qbits before dealing with harder data.
2. **AgroBench** — download early even before heavy training, since it doubles as both eval set and a source of expert-verified facts for the knowledge base.
3. **AgMMU** — wire up the retrieval knowledge base next; this is the core of the "RAG" half of the architecture.
4. **AppleGrowthVision** — first real-world detection dataset, moderate size (fits comfortably in available disk).
5. **Banana defect dataset** — for the quality/grading branch; remember the CC-BY-NC-4.0 restriction.
6. **Fruit-306 / Jain freshness dataset** — pursue once their access paths are confirmed (see follow-ups below).

## 4. Storage plan on qbits

- `/home` on qbits is at 99%+ usage shared across many users — **do not** bulk-download datasets there without checking free space first (`df -h /home`).
- Model weights already live under `/home/minura/.cache/huggingface` (18GB) — datasets should go in a separate subdirectory, e.g. `~/agrivision-rag/data/<dataset-name>/`, so they can be independently cleaned up.
- Before each dataset download: check `df -h /home`, and prefer downloading directly on qbits (not proxying through the local machine) since qbits has verified internet access.
- Revisit the earlier NFS-mount discussion if dataset sizes exceed the ~27GB currently free.

## 5. Follow-ups / open questions

- [ ] Locate the actual download path for the Jain et al. freshness dataset (check paper's data-availability section).
- [ ] Locate the actual download path for Fruit-306 (FruitEnsemble paper appendix/project page).
- [ ] Re-search for the SMART paper itself (title/venue only known from the user's research so far — arXiv ID not yet found).
- [ ] Verify MinneApple and Fruits-360 sources directly before use (not confirmed in this session).
- [ ] Check licensing on every dataset before any commercial framing of the resume project — the banana defect dataset is explicitly non-commercial (CC-BY-NC-4.0); others need checking too.
