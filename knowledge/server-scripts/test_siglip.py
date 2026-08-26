#!/usr/bin/env python3
"""Smoke test: embed a synthetic test image with SigLIP and print the vector shape/stats."""
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModel

MODEL_ID = "google/siglip-base-patch16-224"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    model.eval()

    # synthetic RGB test image (no dataset needed for a smoke test)
    arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    image = Image.fromarray(arr)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    features = out.pooler_output if hasattr(out, "pooler_output") else out

    print("image_embeds shape:", tuple(features.shape))
    print("norm:", features.norm(dim=-1).item())
    print("SigLIP smoke test: OK")

if __name__ == "__main__":
    main()
