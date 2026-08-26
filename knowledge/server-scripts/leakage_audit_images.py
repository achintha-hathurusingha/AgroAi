#!/usr/bin/env python3
"""Leakage audit: verify no PlantVillage image (used as query images throughout Phase 1)
is byte-identical to any AgMMU eval image (part of the knowledge corpus's source data)."""
import hashlib
from pathlib import Path

PV_DIR = Path("data/plantvillage/raw/color")
AGMMU_DIR = Path("data/agmmu/copied_images")


def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    agmmu_files = list(AGMMU_DIR.rglob("*.png")) + list(AGMMU_DIR.rglob("*.jpg")) + list(AGMMU_DIR.rglob("*.jpeg"))
    print(f"hashing {len(agmmu_files)} AgMMU images...")
    agmmu_hashes = {}
    for p in agmmu_files:
        agmmu_hashes[hash_file(p)] = str(p)
    print(f"  {len(agmmu_hashes)} unique hashes")

    pv_files = list(PV_DIR.rglob("*.JPG")) + list(PV_DIR.rglob("*.jpg"))
    print(f"hashing {len(pv_files)} PlantVillage images (this will take a bit)...")
    collisions = []
    for i, p in enumerate(pv_files):
        if i % 10000 == 0:
            print(f"  ...{i}/{len(pv_files)}")
        h = hash_file(p)
        if h in agmmu_hashes:
            collisions.append((str(p), agmmu_hashes[h]))

    print(f"\ncollisions found: {len(collisions)}")
    for pv, agmmu in collisions:
        print(f"  {pv}  ==  {agmmu}")

    with open("data/leakage_audit_image_result.txt", "w") as f:
        f.write(f"AgMMU images hashed: {len(agmmu_files)}\n")
        f.write(f"PlantVillage images hashed: {len(pv_files)}\n")
        f.write(f"Collisions: {len(collisions)}\n")
        for pv, agmmu in collisions:
            f.write(f"{pv} == {agmmu}\n")


if __name__ == "__main__":
    main()
