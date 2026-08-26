# Phase 1A — Preprocessing decisions (finalized)

Decisions made from the raw findings in [agmmu-facts-inspection.md](agmmu-facts-inspection.md).
This is the user's design, not a pre-picked default — recorded here so the reasoning
survives past this conversation.

## Core principle: preprocessing ≠ filtering

These are kept as separate concerns so later experiments can still ask "does the
context-only pool help retrieval, or just add noise?" — a question that's unanswerable
if the 60.6% without disease/pest content gets silently deleted now.

- **Preprocessing** changes representation: dict-answer → normalized records,
  duplicate → canonical fact, fields → structured retrieval text.
- **Filtering** changes the retrieval population: 45,096 → a chosen subset for a given
  experiment.

## Decisions

| Issue | Decision | Reason |
|---|---|---|
| 60.6% without disease/pest field | Keep, but split into a separate pool | May still contain useful contextual evidence (species/symptom/management text can help reasoning even without an explicit disease label) |
| Diagnostic corpus | ~17,760 disease/pest-bearing records — primary retrieval corpus for the disease-diagnosis experiment | Directly aligned with the research question |
| Context corpus | ~27,336 remaining records — kept as a separate pool, not deleted | Enables a later experiment: disease/pest-only vs all-records vs hybrid retrieval |
| Exact duplicates (13.7%) | Deduplicate the *retrieval* corpus, never the raw dataset | Prevents Top-k retrieval from returning near-identical repeats of the same fact |
| Deduplication metadata | Preserve `original_faq_ids` per canonical fact | Traceability — lets the writeup state "X unique knowledge records derived from Y original entries" instead of silently dropping data |
| 3 dict-shaped (multi-species) answers | Normalize/split into one fact per species-within-entry, not excluded | These are valid multi-species information, not corrupted records — splitting preserves the knowledge instead of discarding it |
| `species` free-text field | Keep raw (`species_raw`), no normalization in Phase 1 | It's not a controlled taxonomy (10,325 unique values); collapsing "tree"/"apple tree"/a specific cultivar together would be fabricating a taxonomy that isn't actually in the data |
| Raw dataset | Never modified | Preserves source of truth / reproducibility |

## Canonical fact representation (target schema)

```json
{
  "fact_id": "...",
  "source_faq_ids": ["..."],
  "species_raw": "apple tree",
  "disease": "cedar apple rust",
  "pest": null,
  "symptoms": "...",
  "management": "...",
  "image_description": "..."
}
```

A retrieval text representation is generated from this schema (not embedded directly —
embedding is Q1, a separate decision):

```
Species: apple tree
Disease/Issue: cedar apple rust
Symptoms: ...
Management: ...
```

## Pipeline shape

```
Raw AgMMU (45,096 entries)
       │
       ▼
Parse records — normalize dict-shaped (multi-species) answers into one record each
       │
       ▼
Canonical fact representation
       │
       ▼
Exact-content deduplication (retrieval corpus only; raw dataset untouched)
       │
       ▼
Split: Diagnostic corpus (~17,760) vs Context corpus (~27,336)
       │
       ▼
Retrieval experiments (Phase 1B+)
```

## What's still open

This resolves former "Still open" item 5 in [phase-1-plan.md](phase-1-plan.md). Items 1–4
and 6 there remain open, most immediately **Q1: embedding model / embedding space**,
which is now sharper given this schema: the question isn't just "which embedding model,"
but whether the text knowledge base should share an embedding space with SigLIP's image
embeddings (relevant to whether the "SigLIP retrieval" ablation arm is architecturally
meaningful) or use a text-specialized embedder instead.
