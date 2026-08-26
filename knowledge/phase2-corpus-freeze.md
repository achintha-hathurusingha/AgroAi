# Phase 2 Corpus Freeze — `agmmu_phase2_v1`

Frozen so Phase 2 experiments are comparable against a fixed knowledge base — any future
change requires a new version identifier, specifically to avoid ambiguity between "the
RAG architecture improved" and "the corpus quietly changed underneath it."

Frozen on devon at `~/agrivision-rag/frozen/`, manifest at `frozen/MANIFEST.json`.

## Contents

| File | Count | SHA-256 |
|---|--:|---|
| `agmmu_phase2_diagnostic_v1.json` | 17,583 | `ae435393ac006cad5293fdcb5185e2af090606d280d1c659b7c933faee3d1c48` |
| `agmmu_phase2_context_v1.json` | 17,924 | `fbaeb690eb37b2dd1d7bd02c22d2b9b49b2143295b424fefe37ec95b27e964e5` |
| `agmmu_phase2_diagnostic_v1_bge_emb.npy` (shape 17583×384) | — | `dd9bf390c0a3bf49bf40d8c612689f4c0f592f97d31e53914a3179bb683f940a` |

- Diagnostic (17,583) + context (17,924) = 35,507, matching the full-corpus dedup count
  from the Q6 benchmark exactly — consistency check passed.
- Source: `agmmu_ft_hf1.json`, 45,096 raw entries → 45,098 normalized facts (the +2 from
  splitting the 3 multi-species dict-shaped answers into one fact per species, per the
  Phase 1A decision) → 35,507 after deduplication.
- Embedding model: `TaylorAI/bge-micro-v2` (the Q1 decision).

## What "frozen" means in practice

- These exact files are what Phase 2 experiments should load — not re-derived from the
  raw AgMMU JSON on each run, even though the derivation is deterministic (same seed,
  same code) — the frozen files are the actual source of truth to point at.
- Any content change (different dedup logic, different field handling, corpus
  additions/removals) gets a new version identifier (`agmmu_phase2_v2`, etc.), and any
  result being compared across versions needs to be re-run, not assumed transferable.
- The embeddings file is tied to both the corpus version *and* the embedding model — if
  Q1's embedder choice is ever revisited, the embeddings file needs regenerating and
  re-versioning even if the underlying corpus JSON doesn't change.
