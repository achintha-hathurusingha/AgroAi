# AgriVision-RAG — Project Architecture & Direction

Extended research beyond the initial CVPR paper list (see [README.md](README.md)), covering
the broader 2025–2026 trend: **CNN classifier → VLM → MLLM → RAG → agentic/robust
multimodal system.** Key justification: AgroBench (ICCV 2025) evaluates VLMs across 203
crop categories and 682 disease categories and finds current VLMs still struggle with
fine-grained agricultural identification — motivating something beyond a basic VLM demo.

This is the **chosen direction** for the AgroAi resume project (see project memory
`project-vlm-rag-resume-project`): not a single paper reproduction, but an original
engineering project combining ideas from AgriRAG + FruitEnsemble + AgroBench + SMART +
lightweight inference + disease segmentation + quality assessment.

---

## Candidate project ideas (ranked)

| Rank | Project | Resume strength | Difficulty |
|---|---|---|---|
| 1 | Fruit Disease + Quality RAG | ★★★★★ | High |
| 2 | Agricultural MLLM Benchmark | ★★★★★ | Medium/High |
| 3 | Few-shot / Unknown Disease Detection | ★★★★★ | High |
| 4 | VLM Reliability / Hallucination Detector | ★★★★★ | High |
| 5 | Edge Agricultural AI | ★★★★½ | Medium |
| 6 | Disease + Segmentation + Severity | ★★★★½ | High |
| 7 | Agri Foundation Model | ★★★★½ | Very High |
| 8 | Fruit Ripeness Multimodal AI | ★★★★ | High |
| 9 | Qwen-VL LoRA + RAG | ★★★★ | Medium/High |

### 1. Agricultural Vision RAG (production-style AgriRAG)
```
Crop/Fruit Image → SigLIP → Vector Database → Retrieve expert facts → Qwen3-VL
                                                        │
                                        ┌───────────────┼───────────────┐
                                        ▼               ▼               ▼
                                     Disease         Severity       Treatment
```
Tech: SigLIP, Qwen3-VL, FAISS/Qdrant, RAG, FastAPI, Docker, GPU inference.

### 2. Fruit Quality + Disease Multimodal AI
```
Fruit Image
    │
    ├── Disease Pipeline (type, severity, localization)
    └── Quality Pipeline (freshness, defects, grade)
              │
       Multimodal Reasoning → Final Report
```
Example output:
```
Fruit: Mango
Disease: Anthracnose | Confidence: 91%
Disease severity: Moderate
Defect area: 13.7%
Freshness: 78%
Commercial grade: B
Explanation: ...
```
CVPR 2025 fruit-quality paper baseline: 92.71% avg accuracy for fine-grained freshness classification.

### 3. Agricultural MLLM Benchmark
Instead of building another model, build an evaluation platform comparing Qwen3-VL, LLaVA,
Gemini, GPT-4o-vision, InternVL, and smaller VLMs on accuracy, hallucination, robustness,
and reasoning quality. AgroBench (expert agronomist annotations, seven agricultural topics)
is the reference benchmark. Demonstrates LLM evaluation + multimodal AI + experiment
automation, stronger than "trained ResNet on PlantVillage."

### 4. VLM Hallucination / Reliability Detector
Motivated by CVPR 2025 finding that VLMs show "blind faith in text" — disproportionately
trusting textual claims over visual evidence when the two conflict. Build an "Agricultural
Multimodal Reliability Engine": feed image + a (possibly false) text claim, detect
image/text conflict.
```
Image: Healthy leaf
Text: "This leaf has severe fungal infection."
→ Visual evidence: HEALTHY | Textual claim: DISEASED | Conflict detected: YES | Confidence: 94%
```

### 5. Edge Agricultural AI
Based on a CVPR 2025 lightweight crop-disease paper (up to 90% data reduction, 93% model
compression, <2% accuracy loss). Pipeline: Camera → Edge device → MobileNet/EfficientNet →
Quantization → ONNX/TensorRT → Disease detection → API/Cloud. Benchmark accuracy/latency/RAM/size
across ResNet, MobileNet, INT8, INT4. Shows AI + optimization + deployment + hardware skills.

### 6. Agri-FM — Agricultural Foundation Model
Domain-adapt a self-supervised foundation model (DINOv2/SigLIP) to close-field agricultural
vision, inspired by CVPR 2025's Agri-FM+ (147K-image agricultural dataset, 8 benchmarks for
detection/segmentation). Connects to prior interest in DINOv2 and anomaly detection.

### 7. Few-shot Agricultural Disease Detection
Based on a 2024 paper on visual-information-guided multimodal plant disease anomaly
detection (93.81% AUROC, 2-shot setting) — detects unknown/unseen diseases rather than only
classes seen during training. Realistic: real-world systems can't assume every disease was
in the training set. Unknown cases fall back to VLM/RAG knowledge-base search.

### 8. Multimodal Fruit Ripeness (vision + e-nose)
Based on a 2025 banana ripeness study combining computer vision (peel appearance) with an
electronic nose (volatile organic compounds) — genuinely multimodal, not just "image model
called multimodal."

### 9. Disease Detection + Segmentation + Severity
Based on recent date-palm disease work combining CLIP + PaliGemma2 + Grounding DINO + SAM 2.1
+ ViT regression for classification → detection → segmentation → severity estimation.
Generalizable pipeline: Image → Grounding DINO (find fruit/leaf) → SAM 2 (disease region) →
Severity estimation → MLLM explanation.

### 10. Qwen-VL LoRA Fine-tuning
Based on a 2026 study on LoRA fine-tuning of Qwen2.5-VL for plant disease diagnosis (LoRA on
both vision encoder and language model). Compare: Qwen zero-shot vs Qwen+RAG vs Qwen+LoRA vs
Qwen+LoRA+RAG.

---

## Chosen combined architecture: AgriVision-RAG

**Multimodal Agricultural Intelligence Platform** — combines ideas from AgriRAG,
FruitEnsemble, AgroBench, and SMART rather than reproducing a single paper.

```
                         USER IMAGE
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vision Encoder  │
                    │    SigLIP       │
                    └────────┬────────┘
                             │
               ┌─────────────┼──────────────┐
               ▼             ▼              ▼
          Fruit/Crop      Disease        Quality
          Detection      Detection       Analysis
               │             │              │
               └─────────────┼──────────────┘
                             ▼
                      Vector Retrieval
                             │
                             ▼
                    Agricultural RAG
                             │
                             ▼
                       Qwen3-VL
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
              Diagnosis  Severity   Recommendation
                             │
                             ▼
                       JSON Response
                             │
                             ▼
                         FastAPI
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
                 Web UI             Monitoring
```

### Evaluation layer
```
                  AgriVision-RAG
                        │
                        ▼
                Evaluation Engine
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
    Accuracy        Hallucination      Latency
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                  ML Experiment DB
```

### Deployment pipeline
```
GitHub → GitHub Actions → Docker → GPU server (qbits) → FastAPI → Qdrant → Qwen3-VL → Monitoring
```

### Target resume line
> **AgriVision-RAG — Multimodal Agricultural Intelligence Platform**
> Developed a multimodal AI system combining SigLIP visual retrieval, Qwen3-VL, vector-based
> RAG, disease/defect analysis, and quality assessment; deployed GPU inference through
> FastAPI and Docker with automated evaluation of diagnostic accuracy, retrieval quality,
> latency, and hallucination.

---

## Supporting papers referenced (not yet downloaded as PDFs)

- **AgroBench** (ICCV 2025) — VLM evaluation across 203 crop / 682 disease categories, expert
  agronomist annotations, seven agricultural topics.
- **SMART** (Aug 2026) — Structured multimodal agricultural retrieval-augmented transformer
  for plant disease diagnosis; 37,586 images, 48 disease categories; studies structured
  captions, retrieval robustness, long-tailed classes, human-in-the-loop diagnosis.
- CVPR 2025 lightweight crop-disease paper — data/model compression for low-resource edge
  deployment (~90% data reduction, ~93% model compression, <2% accuracy loss).
- CVPR 2025 Agri-FM+ — self-supervised foundation model adapted to close-field agricultural
  vision (147K images, 8 benchmarks).
- 2024 few-shot/unseen-disease anomaly detection paper — 93.81% AUROC in 2-shot setting.
- 2025 banana ripeness study — vision + electronic nose (VOC) multimodal fusion.
- Date-palm disease pipeline — CLIP + PaliGemma2 + Grounding DINO + SAM 2.1 + ViT regression.
- 2026 Qwen2.5-VL LoRA fine-tuning study for plant disease diagnosis.
- CVPR 2025 VLM "blind faith in text" finding — motivates the hallucination/reliability
  detector idea.

*(These have not yet been located/downloaded as PDFs into `knowledge/papers/` — follow up
by searching each title on openaccess.thecvf.com / arXiv and adding them the same way as the
original five.)*

## Infrastructure notes

- **Primary GPU server: `devon` (192.248.10.68)**, user `minura`, RTX 4090 (24GB VRAM),
  Intel i9-14900K, 62GB RAM, Ubuntu 24.04.4, 561GB free disk. This is minura's main dev
  machine (existing conda envs: `fyp`, `adair`, `pmrf`, etc. — left untouched; work happens
  in a cloned `agrivision` env). SSH key: `Achintha` (kept out of this repo via
  `.gitignore` — never commit it).
- `qbits` (192.248.10.67) — a shared multi-user server, RTX 4080 SUPER (16GB), `/home` at
  99% usage (only ~27-51GB free, shared across ~9 users). Used for initial setup/testing
  and, since it has spare GPU capacity, cheap parallel jobs (e.g. Experiment 2's retrieval
  decomposition) alongside devon's heavier work. GPU freed there (`~/agrivision-rag/stop.sh`)
  once no longer in active use, since it's shared.
  **Update 2026-08-27: `minura` was added to the `sudo` group** (previously blocked —
  "not in the sudoers file"). With sudo now working, mounted an NFS share for extra
  storage: `sudo mount -t nfs -o vers=4 10.8.158.19:/srv/nfs/gpu-share /mnt/gpu-nfs-share`
  (needed `sudo apt-get install nfs-common` first — the mount helper wasn't installed;
  this triggered Ubuntu's standard post-install notice about an already-pending kernel
  upgrade, unrelated to the install — did **not** reboot, since qbits has other active
  users). Result: 93GB free at `/mnt/gpu-nfs-share`, resolving the disk constraint that
  limited qbits work throughout Phase 1. **Not yet in `/etc/fstab`** — this mount will
  not survive a reboot until added there. `minura` *does* appear to be a real sudoer on
  devon too (untested — no root action has been needed there yet).
- On `devon`: models cached under `~/.cache/huggingface` (SigLIP, Qwen3-VL-8B-Instruct —
  22GB+), datasets under `~/agrivision-rag/data/` (PlantVillage color split extracted,
  923MB, 38 classes/54,305 images; AgMMU JSON facts + eval images extracted, 17GB, 770
  entries), source repos cloned under `~/agrivision-rag/repos/` (AgMMU, AppleGrowthVision,
  banana-defect-segmentation).
  Skip AgMMU's `images_ft.tar.gz` (~500GB fine-tuning image corpus, split into 11 parts) —
  not needed for RAG retrieval, only the JSON facts + smaller `images.tar.gz` (17GB) are.
  `agmmu_ft_hf1.json` (45,096 entries) is the real knowledge-base content: per-entry
  species / disease-identification / symptom-description / management-instructions facts
  — exactly what the RAG retrieval corpus should be built from, and it's usable as pure
  text without needing the paired images. **Path note:** the JSON's `images` field points
  at `./images/<id>/...`, but the actual extracted folder is `copied_images/<id>/...` —
  remap this when writing data-loading code.
- First real-data pipeline test (`test_real_pipeline.py`) on a genuine PlantVillage image:
  ground truth `Apple___Cedar_apple_rust`; Qwen3-VL zero-shot got species (*Malus
  domestica*) and disease-vs-healthy right, but guessed the wrong specific disease (Apple
  Scab/Blotch) — concrete evidence for why the RAG-grounding step matters, not just a
  synthetic-noise smoke test.
