#!/usr/bin/env python3
"""
End-to-end smoke test on REAL data: pick a real PlantVillage leaf image, embed it
with SigLIP, and ask Qwen3-VL-8B (4-bit) to diagnose it -- then compare against the
ground-truth label encoded in the folder name. Loads both models once and holds the
GPU resident afterward (like serve_qwen3vl.py) so this doubles as the day's GPU hold.
"""
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)

DATA_DIR = Path("/home/minura/agrivision-rag/data/plantvillage/raw/color")
SIGLIP_ID = "google/siglip-base-patch16-224"
QWEN_ID = "Qwen/Qwen3-VL-8B-Instruct"
LOG_PATH = "/home/minura/agrivision-rag/test_real_pipeline.log"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def pick_random_image():
    class_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    class_dir = random.choice(class_dirs)
    images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
    img_path = random.choice(images)
    return img_path, class_dir.name


def main():
    img_path, ground_truth = pick_random_image()
    log(f"picked image: {img_path.name}  ground_truth: {ground_truth}")
    image = Image.open(img_path).convert("RGB")

    log(f"loading SigLIP ({SIGLIP_ID}) ...")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
    siglip_model = AutoModel.from_pretrained(SIGLIP_ID, dtype=torch.float16).to("cuda")
    siglip_model.eval()
    siglip_inputs = siglip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        siglip_out = siglip_model.get_image_features(**siglip_inputs)
    embedding = siglip_out.pooler_output if hasattr(siglip_out, "pooler_output") else siglip_out
    log(f"SigLIP embedding shape: {tuple(embedding.shape)}")

    log(f"loading Qwen3-VL ({QWEN_ID}) in 4-bit ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    qwen_processor = AutoProcessor.from_pretrained(QWEN_ID)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_ID, quantization_config=bnb_config, device_map="cuda:0"
    )
    log("Qwen3-VL loaded")

    prompt = (
        "You are an expert plant pathologist. Look at this leaf image and identify "
        "the plant species and any visible disease. Respond in the format "
        "'Species: <name> | Diagnosis: <healthy or disease name> | Reasoning: <one sentence>'."
    )
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    inputs = qwen_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(qwen_model.device)

    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=128)
    answer = qwen_processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    log(f"ground truth (folder label): {ground_truth}")
    log(f"Qwen3-VL diagnosis: {answer!r}")
    log("real-data pipeline test: OK -- holding GPU memory, sleeping until killed")

    def handle_term(signum, frame):
        log("received SIGTERM, exiting (GPU memory will be freed)")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
