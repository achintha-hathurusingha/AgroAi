#!/usr/bin/env python3
"""Ensures all embedding matrices needed for the 5-arm Phase 2 pipeline exist on disk:
- BGE over diagnostic corpus (already cached from Milestone 3)
- BGE over context corpus (new)
- BGE over full corpus = diagnostic+context concatenated (new, for Arm 4)
- SigLIP-text over diagnostic corpus (new, for Arm 3)
"""
import json
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("/home/minura/agrivision-rag/data")
FROZEN_DIR = DATA_DIR.parent / "frozen"
BGE_MODEL = "TaylorAI/bge-micro-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"


def embed_bge(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL, device="cuda")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    return emb.cpu().numpy().astype("float32")


def embed_siglip(texts, batch_size=32):
    from transformers import AutoProcessor, AutoModel
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        embs.append(feat.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(embs, dim=0).numpy().astype("float32")


def main():
    with open(FROZEN_DIR / "agmmu_phase2_diagnostic_v1.json") as f:
        diagnostic = json.load(f)
    with open(FROZEN_DIR / "agmmu_phase2_context_v1.json") as f:
        context = json.load(f)

    print(f"diagnostic: {len(diagnostic)}, context: {len(context)}")

    bge_diag_path = FROZEN_DIR / "agmmu_phase2_diagnostic_v1_bge_emb.npy"
    print(f"BGE diagnostic embeddings: {'cached' if bge_diag_path.exists() else 'MISSING'}")
    bge_diag = np.load(bge_diag_path) if bge_diag_path.exists() else embed_bge([f["retrieval_text"] for f in diagnostic])
    if not bge_diag_path.exists():
        np.save(bge_diag_path, bge_diag)

    bge_ctx_path = FROZEN_DIR / "agmmu_phase2_context_v1_bge_emb.npy"
    if bge_ctx_path.exists():
        print("BGE context embeddings: cached")
        bge_ctx = np.load(bge_ctx_path)
    else:
        print("building BGE context embeddings...")
        bge_ctx = embed_bge([f["retrieval_text"] for f in context])
        np.save(bge_ctx_path, bge_ctx)

    bge_full_path = FROZEN_DIR / "agmmu_phase2_full_v1_bge_emb.npy"
    if not bge_full_path.exists():
        print("concatenating full BGE matrix...")
        bge_full = np.concatenate([bge_diag, bge_ctx], axis=0)
        np.save(bge_full_path, bge_full)
        print(f"  full matrix shape: {bge_full.shape}")
    else:
        print("BGE full embeddings: cached")

    siglip_diag_path = FROZEN_DIR / "agmmu_phase2_diagnostic_v1_siglip_emb.npy"
    if siglip_diag_path.exists():
        print("SigLIP diagnostic embeddings: cached")
    else:
        print("building SigLIP diagnostic embeddings...")
        siglip_diag = embed_siglip([f["retrieval_text"] for f in diagnostic])
        np.save(siglip_diag_path, siglip_diag)
        print(f"  shape: {siglip_diag.shape}")

    print("\nall embeddings ready.")


if __name__ == "__main__":
    main()
