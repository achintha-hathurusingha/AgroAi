# Leakage Audit Results

Performed as part of Phase 1 closure, per the concern that evaluation images or ground
truth could leak into the retrieval corpus or query construction in a way that would
give the system an unfair, non-generalizable advantage.

## 1. Image leakage: PlantVillage (query images) vs AgMMU (knowledge corpus source)

Hashed every image in both datasets (SHA-256, byte-for-byte) and checked for any file
appearing in both:

- PlantVillage: 54,303 images hashed.
- AgMMU eval images (`copied_images/`): 1,635 files, 1,610 unique hashes (25 are internal
  duplicates within AgMMU itself — not a leakage concern, just a minor note about AgMMU's
  own data).
- **Collisions found: 0.**

No PlantVillage image used anywhere in Phase 1 testing is byte-identical to any AgMMU
image. Given the two datasets come from entirely different sources (PlantVillage is a
curated lab-photography dataset; AgMMU is sourced from a plant-ID forum/community
question set), this result is expected, but it's now verified rather than assumed.
Script: [server-scripts/leakage_audit_images.py](server-scripts/leakage_audit_images.py).

## 2. Ground-truth leakage into query construction

Code-reviewed the actual pipeline (`agrivision_pipeline.py`) and its call sites:

- `generate_prompt_c_query(model, processor, image)` — signature takes only `model`,
  `processor`, `image`. No ground-truth parameter exists.
- `PROMPT_C` is a fixed string constant with no placeholder for disease/species/ground
  truth — it's the same literal text regardless of which image is passed in.
- In both `milestone3_acceptance_test.py` and `milestone3_control_test.py`,
  `GROUND_TRUTH_DISEASE` is defined but is only referenced *after* `generate_prompt_c_query()`
  has already run, purely for scoring the retrieval result — never passed into the
  generation call itself.

**No leakage path found.** The retrieval query the system generates is structurally
incapable of containing the answer it's being tested against.

## What this does and doesn't cover

- Covers: the two concrete risks named going into this audit (image overlap, ground
  truth leaking into query text).
- Does not cover: leakage via the *choice* of which 8 diseases were tested in Q1/Q2
  (chosen from the corpus's own disease-frequency histogram — expected and necessary for
  a retrieval benchmark, not a leakage concern, but worth distinguishing from actual
  evaluation contamination).
- Any future evaluation set (e.g. drawing directly from AgMMU's own 770-entry eval images
  rather than PlantVillage) would need its own separate leakage check against whatever
  corpus is used at that time — this audit is specific to the PlantVillage-images /
  AgMMU-knowledge-base pairing used throughout Phase 1.
