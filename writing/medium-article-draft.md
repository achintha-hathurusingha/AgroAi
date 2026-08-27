# The Bottleneck Nobody Talks About in Agricultural RAG

### Why bolting retrieval onto a vision-language model barely helped — until we changed one thing

I fed a photo of an apple leaf with cedar apple rust to Qwen3-VL, a strong open vision-language model. It got the species right (*Malus domestica*). It got "diseased, not healthy" right. Then it guessed apple scab.

Close, but wrong — and wrong in exactly the way that matters in agriculture, where the treatment for one fungal disease can do nothing for another. That single failure was the seed of a research project: **can retrieval-augmented generation (RAG) fix this?** Give the model access to a real agricultural knowledge base at inference time, and see if grounding beats guessing.

The honest answer turned out to be more interesting than "yes" or "no."

## The setup

- **Knowledge base**: 17,583 disease/pest facts extracted from AgMMU, a public agricultural multimodal dataset — species, symptoms, management, no images attached to the facts themselves.
- **Query images**: 465 leaf photos from PlantVillage, split into cases where the disease genuinely exists in the corpus, cases that are just healthy, and negative controls where the disease is *not* in the corpus at all (to catch hallucination).
- **Model**: Qwen3-VL-8B-Instruct, 4-bit quantized, running on a single consumer GPU.
- **Baseline RAG pipeline**: ask the VLM to describe the image, embed that description, retrieve the top-5 most similar facts, hand them back to the VLM for a final diagnosis.

Standard stuff. It's the architecture most people reach for first.

## It barely worked

On the subset where the disease actually exists in the knowledge base — the only place RAG could structurally help — accuracy went from **0.4%** (zero-shot) to **4.2%** (with retrieval). Statistically indistinguishable from noise across the full evaluation set.

This is the point where a lot of writeups would conclude "RAG doesn't help for this domain" and move on. That conclusion would have been wrong, and the reason why is the actual finding.

## The ceiling experiment

Before giving up on retrieval, I ran one more test: what if the retrieval query were *perfect* — literally the ground-truth disease name, used only to fetch evidence, never shown to the model as an answer?

**Accuracy jumped to 78.7%.**

Same corpus. Same model. Same diagnosis prompt. The only thing that changed was the query used to find evidence. McNemar's test comparing oracle retrieval against the naive pipeline: p = 2.2 × 10⁻⁴⁰, with the oracle winning on every single discordant case — not once did the naive pipeline get a case right that the oracle got wrong.

That's not a subtle effect. It means the knowledge base was never the problem. **The bottleneck was entirely in how the query got constructed.**

## Quantifying the failure

I classified every wrong diagnosis into three buckets, using evidence already collected rather than a new judging pass:

- **Query failure** — a good query would have found the right fact; the actual one didn't: **86.2%**
- **Retrieval/corpus failure** — not findable even with a perfect query: **6.2%**
- **Reasoning failure** — the right evidence was retrieved, and the model still got it wrong anyway: **3.3%**

Six out of every seven fixable failures traced back to one step: turning a photo into a search query. Not the corpus. Not the model's reasoning. The bridge between them.

## What was actually broken

The naive pipeline's query construction went `image → VLM generates a text description → embed that text → search`. Somewhere in that first arrow, information was getting lost. Reading the actual generated descriptions confirmed it: generic language like "brown spots, irregular margin" — accurate, but not specific enough to distinguish the true disease from thousands of textually similar alternatives in the corpus. No amount of prompt engineering fixed this; I tested five distinct prompt strategies at full corpus scale and none of them reliably won.

The fix wasn't a better prompt. It was skipping text generation for the retrieval step entirely.

## Cross-modal retrieval, and the comparison that proves it

SigLIP — the same vision-language model used elsewhere in the pipeline — embeds images and text into a shared space. Instead of generating a text description and then embedding *that*, I embedded the **image directly** and searched against the corpus's own text embeddings. No paired images in the knowledge base needed — just the existing text facts, embedded once.

```
Naive:   image → VLM caption → text embed → search
Direct:  image → image embed ─────────────→ search
```

Result: **16.7%** accuracy on the disease-in-corpus subset — a 4x improvement over the naive pipeline, p = 3.0 × 10⁻⁷.

Here's the comparison that actually isolates *why*: I also ran a variant that used SigLIP's text tower on the VLM-generated caption — same embedding model as the direct-image approach, still text-mediated. It scored **4.2%** — identical to the original pipeline. Same model. Only the representation changed. Only the representation mattered.

| Approach | Representation | Accuracy |
|---|---|---|
| VLM-only, no retrieval | — | 0.4% |
| Text RAG (BGE embedder) | text-mediated | 4.2% |
| Text RAG (SigLIP embedder) | text-mediated | 4.2% |
| **Visual RAG** | **direct image embedding** | **16.7%** |
| Oracle (ceiling, not deployable) | ground-truth text | 78.7% |

## The parts that didn't work

Not every follow-up idea panned out, and I think those results matter as much as the ones that did.

I tried blending text and visual retrieval scores together — it improved the retrieval *ranking* metric substantially, but that improvement didn't survive into diagnosis accuracy (p = 0.81, no significant difference from visual retrieval alone). The lesson: a retrieval metric getting better doesn't automatically mean the downstream task gets better, especially once the model already sees enough candidates that reordering them barely matters.

I also tried two literature-backed fixes for a real, measured problem — SigLIP silently truncates long text at 64 tokens, and 18% of the corpus facts exceeded that. Shortening every fact to guarantee no truncation actually made retrieval *worse*, because it threw away more real signal than the truncation had cost. A caption-then-retrieve bridge helped on some diseases and failed badly on others, netting out below the direct-embedding approach. Both negative results, both worth reporting, neither worth hiding.

## Where this leaves things

The honest scorecard: **16.7% accuracy, up from 0.4%, against a ceiling of 78.7%.** That's a real, statistically robust improvement and a long way from solved.

But the more useful output of this project isn't a single number — it's the diagnosis. The knowledge was always valuable (the oracle proved that beyond reasonable doubt). The bottleneck was never the model's capacity to reason, or the corpus's coverage. It was one specific, identifiable, partially fixable step: converting a photograph into something a retrieval system can search on. Fix that step, even partially, and the rest of the system — unchanged — gets dramatically better.

If you're building RAG on top of a vision-language model and it isn't helping, that's worth checking before concluding retrieval doesn't work for your domain.

---

*This is one result from an ongoing research project applying retrieval-augmented VLMs to agricultural disease diagnosis. Full experimental writeup, statistics, and code: [link to repo].*
