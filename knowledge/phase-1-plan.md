# AgriVision-RAG — Phase 1 Plan

**How we're working from here:** you learn the concepts and make the design decisions;
I explain things in depth on request, execute infra/ops you delegate (SSH, downloads,
environment setup), and review your calls rather than making them for you. Nothing below
is a decision already made on your behalf — where a choice is needed, it's presented as
an open question with tradeoffs.

## What Phase 1 is actually for

Everything so far (papers, architecture doc, data, models, servers) is scaffolding. Phase
1 has one concrete goal: **build the smallest possible working RAG loop and find out
whether retrieval-grounding actually improves diagnosis accuracy** over a bare VLM. The
real-data test we already ran (Cedar Apple Rust — Qwen3-VL got species and "diseased"
right, but guessed the wrong specific disease) is the exact gap this phase should try to
close. If retrieval doesn't measurably help, that's an important — and honestly more
interesting — finding too.

## What already exists (context, not homework)

- Downloaded: SigLIP (`google/siglip-base-patch16-224`), Qwen3-VL-8B-Instruct (4-bit),
  PlantVillage (54,305 images, 38 classes), AgMMU eval set (770 image+MCQ entries) and
  its 45,096-entry fact set (`agmmu_ft_hf1.json` — species/disease/symptoms/management
  per entry, text-only, no image dependency).
- Running on `devon` (192.248.10.68), `agrivision` conda env, RTX 4090.
- All of this documented in [AgriVision-RAG-architecture.md](AgriVision-RAG-architecture.md)
  and [data-finding-plan.md](data-finding-plan.md).

## Concepts worth understanding before/while building this

You don't need to master these up front — read enough to make each decision below
consciously, then go deeper as you build. Ask me to explain any of these in as much
depth as you want.

1. **Embeddings & similarity search** — what a vector embedding actually represents,
   why cosine/dot-product similarity finds "similar" items, why this works for retrieval.
2. **RAG (Retrieval-Augmented Generation)** — the two-stage idea: retrieve relevant text
   first, then hand it to the LLM as context so it can ground its answer instead of
   relying purely on what it memorized during training. The [AgriRAG paper](papers/Marques_AgriRAG_Training-Free_Retrieval-Augmented_Generation_for_Agricultural_Disease_Diagnosis_with_Vision-Language_CVPRW_2026_paper.pdf)
   is the direct reference for this exact use case.
3. **Vector databases** — what FAISS/Qdrant actually do differently from just looping
   over embeddings in Python (indexing structures, approximate vs exact search).
4. **Quantization** — what `bitsandbytes` 4-bit (NF4) loading is actually doing to the
   model weights, and the accuracy/memory tradeoff it implies.
5. **Prompt construction for grounded generation** — how retrieved facts get formatted
   and inserted into the Qwen3-VL chat prompt (what goes in vs what gets left out).
6. **Evaluation methodology** — why the AgMMU eval set uses MCQ format (easy, objective
   scoring) vs open-ended generation (harder to score, closer to real use) — and what
   the [SMART paper's finding](AgriVision-RAG-architecture.md) that "structured captions
   outperform LLM-generated narratives" implies for how you evaluate.

## Decisions you need to make

These are genuinely open — I'm laying out the tradeoffs, not a recommendation to rubber-stamp.

### 1. What gets embedded for retrieval — image or text?
- **Image-to-image**: embed the query photo with SigLIP, search against embeddings of
  the *images* paired with each fact (only 770 AgMMU eval images have this; the 45K fact
  set's images are in the un-downloaded 500GB `images_ft` corpus).
- **Caption-then-text**: ask Qwen3-VL to describe/caption the query image first, embed
  that caption as text, search against the 45,096 text facts.
- **Hybrid**: both signals combined somehow.
- This is the single most consequential design decision in the whole pipeline — it
  determines what data you can actually use (you have all 45K facts as text, but almost
  none of their images).

### 2. What embeds the text facts?
- `sentence-transformers` is already installed in the `fyp`/`agrivision` env on devon.
- `TaylorAI/bge-micro-v2` is already sitting in devon's HF cache (pre-existing, not
  something I downloaded for this project) — a small BGE text-embedding model.
- SigLIP itself also has a text tower (it's trained for text-image alignment) — could
  embed both facts and captions in the same space, which might be architecturally
  cleaner than a separate text embedder.

### 3. Vector store: FAISS vs Qdrant vs "just numpy for now"?
- FAISS: a local index file, no server process, simplest to get running.
- Qdrant: a real server, supports metadata filtering (e.g. filter by species before
  similarity search), closer to what a "production-style" system would use.
- At 45K facts, even brute-force cosine similarity in numpy would run in well under a
  second — worth deciding whether Phase 1 needs real infrastructure yet, or whether
  that's premature.

### 4. What counts as success for Phase 1?
- A target accuracy uplift number on the AgMMU eval set (770 MCQs) comparing zero-shot
  Qwen3-VL vs RAG-grounded Qwen3-VL?
- Or a smaller set of qualitative case studies (like the Cedar Apple Rust one) that make
  the point convincingly without a full benchmark run?
- This affects how much of Phase 1's effort goes into evaluation tooling vs the pipeline
  itself.

### 5. Which facts to index?
- All 45,096 entries as-is, or filtered/deduped first (some entries may be near-duplicate
  or low-quality)? Worth spot-checking a sample before deciding.

## Suggested milestones (a sequence to work through, not a deadline)

1. Explain the RAG loop back in your own words/diagram — if you can't, that's a signal
   to slow down before writing code.
2. Decide questions 1–5 above, and write down *why* (even briefly) — this record is
   useful for the resume writeup later.
3. Build the smallest possible retrieval step (embed the facts, embed one query, get
   top-k results) and sanity-check the results make sense by eye.
4. Wire retrieval output into the Qwen3-VL prompt and re-run the Cedar Apple Rust test
   (or a similar case) — does grounding fix the wrong-diagnosis problem?
5. Run the chosen evaluation (milestone 4 in miniature, or the full AgMMU eval set) and
   compare zero-shot vs RAG-grounded accuracy.
6. Write up what you found — including if RAG didn't help as much as expected, since
   that's a legitimate and interesting result too.

## How I'll help vs what's yours

- **Mine on request:** deep explanations of any concept above, code review, running
  infra/data tasks you tell me to run, catching bugs, keeping the knowledge base updated.
- **Yours:** picking the retrieval strategy, embedding model, vector store, success
  criteria, and writing the actual pipeline code and interpreting results.
