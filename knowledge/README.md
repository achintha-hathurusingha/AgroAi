# CVPR 2025–2026 Knowledge Base: Fruit Quality, Defect Detection & Agricultural Disease Diagnosis

Local copies of five CVPR workshop papers relevant to fruit quality assessment, defect detection,
and agricultural disease diagnosis, plus notes on how they inform an AgroAi project direction.

Each paper has both the abstract page (`.html`) and the full paper (`.pdf`) saved in
[papers/](papers/).

---

## 1. Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction

- **Year:** 2025 · **Venue:** CVPR 2025 Workshop (MTF)
- **Focus:** Fine-grained fruit freshness and quality prediction
- **Core ideas:** Vision-Language Models (VLMs), ordinal regression, meta-learning, privacy-preserving learning
- **Reported result:** 92.71% average accuracy across fruits
- **Why it matters:** Directly relevant to automated fruit quality assessment and retail inspection.
- **Source:** https://openaccess.thecvf.com/content/CVPR2025W/MTF/html/Jain_Privacy_Preserving_Ordinal-Meta_Learning_with_VLMs_for_Fine-Grained_Fruit_Quality_CVPRW_2025_paper.html
- **Local files:** `papers/Jain_Privacy_Preserving_Ordinal-Meta_Learning_with_VLMs_for_Fine-Grained_Fruit_Quality_CVPRW_2025_paper.{html,pdf}`

## 2. Weakly Supervised Panoptic Segmentation for Defect-Based Grading of Fresh Produce

- **Year:** 2025 · **Venue:** CVPR 2025 Workshop (V4A)
- **Focus:** Automated defect detection and grading of fresh produce, particularly bananas
- **Tasks:** Defect localization, segmentation, defect counting/measurement, grading
- **Dataset:** 476 field images with 1,440 annotated defects
- **Core ideas:** Weak supervision, panoptic segmentation, SAM-assisted annotation
- **Why it matters:** Very relevant to industrial fruit inspection and automated quality grading.
- **Source:** https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/Knott_Weakly_Supervised_Panoptic_Segmentation_for_Defect-Based_Grading_of_Fresh_Produce_CVPRW_2025_paper.html
- **Local files:** `papers/Knott_Weakly_Supervised_Panoptic_Segmentation_for_Defect-Based_Grading_of_Fresh_Produce_CVPRW_2025_paper.{html,pdf}`

## 3. AppleGrowthVision: A Large-Scale Stereo Dataset for Phenological Analysis, Fruit Detection, and 3D Reconstruction in Apple Orchards

- **Year:** 2025 · **Venue:** CVPR 2025 Workshop (V4A)
- **Focus:** Apple orchard computer vision
- **Dataset:** 9,317 high-resolution stereo images and 31,084 apple annotations
- **Tasks:** Fruit detection, growth-stage/phenological analysis, 3D reconstruction
- **Reported result:** >95% accuracy for six apple growth stages using several CNN architectures
- **Why it matters:** Useful for orchard monitoring, fruit detection, growth analysis, and future yield/quality applications.
- **Source:** https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/von_Hirschhausen_AppleGrowthVision_A_large-scale_stereo_dataset_for_phenological_analysis_fruit_detection_CVPRW_2025_paper.html
- **Local files:** `papers/von_Hirschhausen_AppleGrowthVision_A_large-scale_stereo_dataset_for_phenological_analysis_fruit_detection_CVPRW_2025_paper.{html,pdf}`

## 4. FruitEnsemble: MLLM-Guided Arbitration for Heterogeneous Ensemble in Fine-Grained Fruit Recognition

- **Year:** 2026 · **Venue:** CVPR 2026 Workshop (AI4RWC)
- **Focus:** Fine-grained fruit recognition
- **Dataset:** 306 fruit categories, 116,233 samples
- **Core ideas:** Multiple vision models, MLLM-guided arbitration, uncertainty-aware model selection
- **Reported result:** 70.49% accuracy on the fine-grained fruit-recognition task
- **Application:** Agricultural visual sorting and quality inspection
- **Why it matters:** Demonstrates how MLLMs can act as a higher-level decision-maker over multiple vision models.
- **Source:** https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/html/Yu_FruitEnsemble_MLLM-Guided_Arbitration_for_Heterogeneous_ensemble_in_Fine-Grained_Fruit_Recognition_CVPRW_2026_paper.html
- **Local files:** `papers/Yu_FruitEnsemble_MLLM-Guided_Arbitration_for_Heterogeneous_ensemble_in_Fine-Grained_Fruit_Recognition_CVPRW_2026_paper.{html,pdf}`

## 5. AgriRAG: Training-Free Retrieval-Augmented Generation for Agricultural Disease Diagnosis with Vision-Language Models

- **Year:** 2026 · **Venue:** CVPR 2026 Workshop (V4A)
- **Focus:** Agricultural disease diagnosis using multimodal AI and retrieval-augmented generation
- **Core architecture:**
  - SigLIP for image representation/retrieval
  - Agricultural knowledge base of 74,611 expert-verified facts
  - Qwen3-VL-8B as the multimodal reasoning model
  - 4-bit quantization for efficient inference
- **Benchmark:** AgMMU
- **Reported result:** 83.7% MCQ accuracy
- **Key finding:** RAG produced a large improvement over the zero-shot VLM baseline (~+40.2 pts in the ablation)
- **Hardware:** Demonstrated on an NVIDIA RTX 4070 Ti 12 GB
- **Why it matters:** The strongest paper here if the goal is to build a modern multimodal AI system rather than only a conventional disease classifier.
- **Source:** https://openaccess.thecvf.com/content/CVPR2026W/V4A/html/Marques_AgriRAG_Training-Free_Retrieval-Augmented_Generation_for_Agricultural_Disease_Diagnosis_with_Vision-Language_CVPRW_2026_paper.html
- **Local files:** `papers/Marques_AgriRAG_Training-Free_Retrieval-Augmented_Generation_for_Agricultural_Disease_Diagnosis_with_Vision-Language_CVPRW_2026_paper.{html,pdf}`

### AgMMU resources (referenced by AgriRAG)

- Project: https://agmmu.github.io/
- GitHub: https://github.com/AgMMU/AgMMU
- Hugging Face dataset: https://huggingface.co/datasets/AgMMU/AgMMU_v1

---

## VLM vs MLLM/MMLM

**VLM — Vision-Language Model**: connects visual information and language directly.

```text
Image + Text
     |
     v
    VLM
     |
     v
Classification / Similarity / Text
```
Examples: CLIP, SigLIP.

**MLLM/MMLM — Multimodal Large Language Model**: an LLM-based system reasoning over multiple modalities.

```text
Image ──┐
Text ───┤
Audio ──┤──> Multimodal LLM ──> Reasoning + Generation
Video ──┘
```

**AgriRAG pipeline:**

```text
                 Fruit / Plant Image
                         |
                         v
                      SigLIP
                         |
                  Image embedding
                         |
                         v
              Agricultural Knowledge
                     Retrieval
                         |
                         v
                  Qwen3-VL-8B
                         |
                         v
            Disease Diagnosis + Explanation
```

Resume framing: **"Vision-Language RAG system powered by a Multimodal Large Language Model (MLLM)."**

---

## Potential Resume Project Direction

**Agricultural Disease & Fruit Quality Assessment using Vision-Language RAG**

Combine ideas from the papers above:

1. Fruit/plant detection
2. Disease identification
3. Disease/defect localization
4. Disease severity estimation
5. Fruit freshness estimation
6. Commercial quality grading
7. Agricultural knowledge retrieval
8. VLM/MLLM-based explanation

Example output:

```text
Fruit: Mango
Disease: Anthracnose
Disease confidence: 91%
Severity: Moderate
Visible defect area: 14%
Estimated quality: Grade B
Recommendation: Suitable for processing
```

### Suggested architecture

```text
                    Image
                      |
              Fruit Detection
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Disease/Defect            Quality Model
     Analysis                    |
          |                       |
          +-----------+-----------+
                      |
                    SigLIP
                      |
               Vector Database
                      |
               Top-K Knowledge
                      |
                  Qwen3-VL
                      |
                      v
              Final AI Report
```

### Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Qwen3-VL
- SigLIP
- FAISS / Qdrant / pgvector
- OpenCV
- FastAPI
- Docker
- NVIDIA CUDA
- Optional: Kubernetes, MLflow, Prometheus/Grafana

### Key distinction for a resume

Avoid: *"Built a fruit disease classifier."*

Prefer: **"Built a multimodal agricultural diagnosis system using SigLIP-based visual retrieval, vector search, and Qwen3-VL to combine crop images with expert agricultural knowledge for disease diagnosis and explainable recommendations."**

This demonstrates experience with modern multimodal AI, RAG, VLM/MLLM inference, vector databases, and AI engineering.
