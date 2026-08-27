# Visual-Data Availability Audit (before building Stage B)

Per the plan's explicit instruction to check this before writing any visual-retrieval
implementation.

## What's actually available

| Asset | Status |
|---|---|
| Paired images for the 17,583 diagnostic facts (`images_ft`) | **Not available.** ~500GB corpus, deliberately never downloaded (Phase 1A/data-finding-plan decision). Confirmed absent on disk. |
| `agmmu_phase2_diagnostic_v1.json` (the frozen corpus itself) | **No image field at all** — keys are `species_raw`, `disease`, `pest`, `symptoms`, `management`, `fact_id`, `retrieval_text`. Purely text. |
| AgMMU eval images (`copied_images/`) | 1,635 files — but this is the **separate 770-entry eval set**, not the diagnostic corpus. No overlap. |
| PlantVillage images | 54,303 files — these are the **query** images (the leaf photos being diagnosed), not knowledge-base images. |
| **SigLIP text embeddings of the diagnostic corpus** | **Already built** (`agmmu_phase2_diagnostic_v1_siglip_emb.npy`, from Phase 2 setup for the old Arm 3) — this is the key enabling asset. |

**Conclusion: true image-to-image retrieval against the diagnostic corpus is not
possible without a 500GB download that's out of scope.** The plan's literal
"Image → Vision encoder → Visual retrieval → AgMMU facts" diagram assumed paired corpus
images that don't exist in what's downloaded.

## The buildable alternative: cross-modal retrieval via SigLIP, not image-to-image

SigLIP is trained so that an image and its matching text land in the *same* embedding
space (this is the entire premise of the model, and is why the earlier Q1 experiment
tested it as a candidate embedder at all). This means we don't need the corpus to have
images — we can:

1. Embed the **query image** directly with SigLIP's **image tower** — bypassing Qwen3-VL
   text generation entirely for the retrieval step, not approximating it.
2. Compare against the corpus's **already-built SigLIP text embeddings**.
3. This is genuine cross-modal retrieval (raw pixels → text knowledge base), not the old
   Arm 3 (which used SigLIP's *text* tower on a Qwen-generated description — still
   text-mediated, just with a different embedder).

This is not literally what the plan sketched, so flagging it explicitly as the proposed
design rather than silently substituting it: **"visual-first" here means "skip the
VLM-generated text description," not "compare against corpus images that don't exist."**
It directly tests the plan's actual hypothesis (information is lost in the
Image→Text→Retrieval chain) using what's genuinely available, without a new download.

## Next step

Proceed to Stage C (retrieval-only evaluation) with this design: embed the same 465
primary-case query images via SigLIP's image tower, retrieve against the existing SigLIP
corpus text embeddings, compute R@1/R@5/R@20/MRR, compare against the Qwen-query and
oracle-query numbers already in [phase2-retrieval-decomposition-results.md](phase2-retrieval-decomposition-results.md).
No new downloads, no new corpus changes, no Qwen3-VL calls needed for this step.
