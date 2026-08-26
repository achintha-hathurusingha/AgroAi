# Phase 1B — Q1/Q2 Experiment Results

Executed exactly per the frozen spec in [phase1b-q1q2-experiment-design.md](phase1b-q1q2-experiment-design.md)
— no changes to disease list, pool, or query text after seeing results. Raw artifacts in
[results/](results/): `q1q2_pool.json` (the 400-fact pool), `q1q2_query_spec_filled.json`
(47 queries with the 7 real VLM captions filled in), `q1q2_results_records.json` (every
top-20 retrieval per query per embedder, with similarity scores), `q1q2_results_summary.json`.

## Overall results (47 queries, 400-fact pool)

| Embedder | R@1 | R@3 | R@5 | MRR |
|---|--:|--:|--:|--:|
| sentence-transformers (`all-MiniLM-L6-v2`) | 0.574 | 0.702 | 0.745 | 0.663 |
| **bge-micro-v2** | **0.681** | **0.809** | **0.830** | **0.751** |
| siglip-text | 0.532 | 0.723 | 0.787 | 0.648 |

BGE-micro-v2 wins on every overall metric. (Model choice for the "sentence-transformers"
candidate: `sentence-transformers/all-MiniLM-L6-v2`, the standard general-purpose model —
this specific checkpoint wasn't pinned down earlier, noting it here since it's a concrete
choice that affects the result.)

## By query type

| Query type | sentence-transformers R@1/R@5/MRR | bge-micro-v2 R@1/R@5/MRR | siglip-text R@1/R@5/MRR |
|---|---|---|---|
| disease_name (n=8) | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| symptoms (n=8) | 0.625 / 0.750 / 0.664 | 0.750 / 0.875 / 0.812 | 0.375 / 0.875 / 0.578 |
| species_symptoms (n=8) | 0.500 / 0.875 / 0.681 | 0.750 / 0.875 / 0.807 | 0.375 / 0.750 / 0.583 |
| description_no_name (n=8) | 0.625 / 0.875 / 0.762 | **0.875** / 0.875 / **0.883** | 0.625 / 0.875 / 0.698 |
| management (n=8) | 0.625 / 0.875 / 0.737 | 0.625 / **1.000** / 0.812 | **0.750** / 0.875 / **0.812** |
| **vlm_caption (n=7)** | **0.000 / 0.000 / 0.060** | **0.000 / 0.286 / 0.111** | **0.000 / 0.286 / 0.153** |

`disease_name` is a trivial sanity check (all three get it perfectly — expected, it's
just literal string matching in disguise). BGE-micro-v2 leads on most hand-authored
types; siglip-text is notably weaker specifically on `symptoms`/`species_symptoms` but
competitive on `management`.

## The headline finding: real VLM captions retrieve catastrophically badly

**All three embedders scored 0% Recall@1 on the `vlm_caption` queries**, and R@5 only
reached 0–29% — versus 50–100% R@1 on every hand-authored query type. This is the most
important result in the experiment, and it isn't a retrieval-mechanism failure — reading
the actual captions ([results/q1q2_query_spec_filled.json](results/q1q2_query_spec_filled.json))
shows why:

**Qwen3-VL confidently misidentified the plant species in most of these captions**, via
elaborate (but wrong) leaf-morphology reasoning:

| Query | Actual species | Qwen3-VL's claimed species |
|---|---|---|
| D1 (cedar apple rust) | Apple | *Sorbus aucuparia* (rowan / mountain ash) |
| D2 (apple scab) | Apple | *Prunus persica* (peach) |
| D3 (black rot) | Apple | Hedged: "*Malus* (apple) or *Prunus* (cherry/plum)" — partially right |
| D4 (powdery mildew) | Cherry | *Prunus persica* (peach) — right genus family, wrong species |
| D6 (early blight) | Potato | *Acer* (maple) or *Quercus* (oak) |
| D7 (bacterial spot) | Peach | *Quercus* species (oak) |

D5 (tomato) avoided a wrong species claim and focused on actual symptom description
("small, discrete, dark brown to blackish spots") — and its retrieval, while still not
R@1, did rank the correct fact noticeably higher (rank 7–13 across embedders) than the
wrong-species queries (several of which never surfaced the correct fact in the top 20 at
all). This is direct evidence the *content* of the caption, not the retrieval mechanism,
is the cause.

A second compounding factor: `max_new_tokens=128` in the captioning call frequently cut
the response off mid-sentence — often right as it was starting to describe actual visible
symptoms, after spending most of the budget on the (often wrong) species-identification
preamble. So even where symptom description was coming, it may not have made it into the
embedded text at all.

**This is a different, worse outcome than the original Phase 1 real-pipeline test**
(the Cedar Apple Rust case, which correctly identified *Malus domestica* and got
disease-vs-healthy right). That test used a shorter, more direct prompt
("identify the plant species and any visible disease... respond in the format
Species: ... | Diagnosis: ... | Reasoning: ..."). This experiment's prompt asked for a
more open-ended, verbose description ("describe what you see... be specific and
descriptive... do not just name a disease"), which appears to have elicited confident
but incorrect botanical taxonomy reasoning rather than a direct, grounded answer. This
is a genuine, measured hallucination-rate finding — a concrete instance of exactly the
"hallucination rate" metric the ablation design already commits to tracking — not
something staged to make a point.

One secondary pattern, worth noting but not over-reading given only 7 data points:
siglip-text found the correct fact within the top 20 for 4 of 7 vlm_caption queries
(ranks 2, 4, 6, 10) while sentence-transformers and bge-micro-v2 each only found it for
3 of 7, often at worse ranks. Whether SigLIP's text tower is genuinely more robust to
this kind of noisy, verbose, partially-wrong text — or this is small-sample noise —
isn't something this experiment size can settle on its own.

## What this does and doesn't resolve

- **Q1 (embedding model)**: bge-micro-v2 wins on the hand-authored-query numbers. Whether
  that holds once a working Q2 answer is in place (i.e. once the real retrieval query
  isn't a raw, unedited VLM caption) is a fair question — the current results are
  strongest evidence for how each embedder handles *clean* text, not necessarily for the
  final end-to-end system.
- **Q2 (retrieval query construction)**: the experiment strongly suggests a **raw,
  unprocessed Qwen3-VL caption is a bad retrieval query** — but it does not by itself
  tell you why (prompt wording vs. token budget vs. some other issue), nor what the fix
  should be (a shorter/more targeted captioning prompt, extracting just a symptom
  fragment, using the model's own disease-candidate guess instead of a full description,
  something else). That's a real decision still ahead of you, now backed by concrete
  evidence of what happens if you don't address it.
