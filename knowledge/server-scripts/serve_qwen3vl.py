#!/usr/bin/env python3
"""
Loads Qwen3-VL-8B-Instruct in 4-bit onto the GPU, runs one smoke-test generation,
then holds the model resident (sleeping) so the VRAM stays reserved.
Mirrors the pattern of ~/FYP/serve_dinov2.py. Stop with SIGTERM to free the GPU.
"""
import time
import signal
import sys
from datetime import datetime

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LOG_PATH = "/home/minura/agrivision-rag/serve_qwen3vl.log"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    log(f"loading {MODEL_ID} in 4-bit ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    log("model loaded")

    # synthetic smoke-test image + prompt
    arr = (np.random.rand(384, 384, 3) * 255).astype("uint8")
    image = Image.fromarray(arr)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    log(f"smoke-test generation: {text!r}")
    log("Qwen3-VL smoke test: OK -- holding GPU memory, sleeping until killed")

    def handle_term(signum, frame):
        log("received SIGTERM, exiting (GPU memory will be freed)")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
