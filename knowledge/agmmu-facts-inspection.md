# AgMMU 45,096-fact dataset — inspection findings (Phase 1A)

Investigation of `agmmu_ft_hf1.json` before any embedding/preprocessing decisions, per
[phase-1-plan.md](phase-1-plan.md) open question 5. Findings only — no preprocessing
decisions made here; that's still yours to make from this data.

Script: [server-scripts/inspect_agmmu_facts.py](server-scripts/inspect_agmmu_facts.py).
Run on devon: `python3 inspect_agmmu_facts.py`.

## What one entry actually contains

Every entry has three top-level keys (100% coverage): `faq-id`, `images`, `finetuning qa`.

`finetuning qa` is a dict of up to 6 possible sub-fields, each `{"q": ..., "a": ...}`.
**Coverage is very uneven — this is the single most important finding:**

| Field | Entries with this field | Coverage |
|---|---:|---:|
| species | 45,096 | 100% |
| management instructions | 23,390 | 51.9% |
| symptom description | 16,720 | 37.1% |
| image description | 15,526 | 34.4% |
| insect/pest | 9,209 | 20.4% |
| disease/issue identification | 8,551 | 19.0% |

**Only 19% of entries have an explicit disease field, and only 39.4% (17,760 entries)
have *either* a disease or pest field** — i.e. actually identify what's wrong with the
plant. The other 60.6% are species-only or species+management/symptom records with no
named issue. **19.2% (8,671 entries) are species-only records with nothing else at all**
— e.g. "this is a hydrangea" with no diagnostic content whatsoever.

This directly matters for Phase 1: if the research question is about *disease diagnosis*
retrieval, the effectively usable core of this dataset is closer to **~17,760 entries**
(disease-or-pest-bearing), not the headline 45,096.

## Completeness / malformed records

- 0 entries missing `finetuning qa` entirely.
- 247 entries have missing/empty `images` list.
- 2 entries have an empty `species` answer.
- **3 entries have a fundamentally different shape**: their answer (`"a"`) is a *dict*
  keyed by species, not a plain string — because the image shows multiple plant species
  at once. Example (faq-id 634638):
  ```json
  "disease/issue identification": {"a": {
    "apple": "possibly appleleaf blister mite",
    "ash": "ash anthracnose",
    "hazelnut": "unknown"
  }}
  ```
  Any preprocessing/embedding code needs to explicitly handle (or explicitly exclude)
  this shape — treating `"a"` as always-a-string will crash or silently corrupt these
  records.

## Duplicates

- 39,835 unique fact fingerprints (all QA-field answers concatenated) out of 45,096
  entries → **6,182 entries (13.7%) are involved in exact duplication** (921 groups of
  identical answer-content entries, presumably multiple images/questions pointing at the
  same underlying fact).

## Species field is messy free text, not a controlled vocabulary

10,325 unique species *values* — clearly user-entered free text, not normalized:
- Most common values are vague, not species names: `grass` (1,521), `none` (1,251,
  meaning no plant identified at all), `tree` (812), `plant` (553).
- Real species do appear with real volume: `tomato` (995), `apple tree` (833),
  `maple tree` (788), `rose` (532), `hydrangea` (445), etc.
- Because it's free text, the same real species can appear under multiple strings
  (e.g. "apple tree" vs "apple" vs a specific cultivar) — no dedup/normalization has been
  attempted here; a naive groupby on the raw string will undercount how many entries
  actually cover a given species.
- 16,207 unique (species, disease) pairs — but most of the top pairs have an *empty*
  disease value, again reflecting the 81% of entries with no disease field.

## Image paths point at the corpus we deliberately didn't download

Sample entries confirm image paths are `./images_ft/<faq-id>/...` — i.e. these facts'
paired images live in the ~500GB `images_ft.tar.gz` that [data-finding-plan.md](data-finding-plan.md)
already decided to skip. This confirms (doesn't just assume) that using this 45K set
means **text-only** facts — there is no way to do image-to-image retrieval against this
corpus without downloading ~500GB, which is a separate, real infrastructure decision if
it's ever reconsidered.

## Open, unresolved for you (not decided here)

- Whether to keep the 60.6% of entries with no disease/pest field at all (they still
  have species + sometimes management/symptom text — arguably still useful context, or
  arguably noise for a disease-diagnosis retrieval corpus).
- Whether/how to dedupe the 13.7% duplicate-content entries before embedding.
- How to handle the 3 multi-species dict-shaped answer entries (exclude, or split into
  one fact per species-within-entry).
- Whether the free-text `species` field needs any normalization before being used as a
  retrieval filter or ground-truth signal.
