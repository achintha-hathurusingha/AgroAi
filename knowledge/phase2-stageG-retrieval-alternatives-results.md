# Stage G: Testing Retrieval Alternatives (SigLIP Truncation Fix & Caption-then-Retrieve)

Two experiments testing whether the SigLIP truncation finding (confirmed: corpus facts
average 43.8 tokens, 17.8% exceed SigLIP's 64-token text-tower limit and get silently
cut) is actually suppressing retrieval quality, and whether the literature's
best-precedented mitigation (caption-then-retrieve) beats what we already have. Both run
retrieval-only (no diagnosis), same pattern as Stage C, on the 465-case primary set. All
numbers alias-corrected (see [phase2-scoring-correction.md](phase2-scoring-correction.md)).

- G-1: [server-scripts/rebuild_siglip_embeddings_short.py](server-scripts/rebuild_siglip_embeddings_short.py)
  -- re-embeds the corpus with a deliberately shortened, truncation-safe text per fact
  (disease/pest name + symptom snippet, capped at 40 words) and re-runs SigLIP visual
  retrieval with the new embeddings.
- G-2: [server-scripts/run_caption_retrieval.py](server-scripts/run_caption_retrieval.py)
  -- Qwen3-VL generates a short neutral symptom caption from the query image, embedded
  with BGE and retrieved against the existing full-text BGE corpus embeddings.

## Result: neither beats the existing hybrid or pure-visual retrieval on R@5

| Method | confident_match R@1 | confident_match R@5 |
|---|--:|--:|
| Current pipeline (Qwen free-text query, BGE) | 1.3% | 6.2% |
| SigLIP visual, full-length text (Stage C, existing) | 3.3% | **20.0%** |
| SigLIP visual, shortened/truncation-safe text (G-1, new) | 3.7% | 10.0% |
| Caption-then-retrieve (Qwen caption -> BGE) (G-2, new) | 6.2% | 10.8% |
| Hybrid α=0.25, text+visual (Stage F, existing) | **8.3%** | **24.6%** |
| Oracle (ground-truth-as-query, ceiling) | 87.5% | 93.8% |

**G-1 (truncation fix) made things worse, not better**: R@1 barely moved (3.3%→3.7%)
but R@5 dropped by half (20.0%→10.0%). The shortened text averaged only 8.9 words per
fact (much less than the 64-token/~45-word budget available) because ~51% of corpus
facts are pest-only entries with an empty `symptoms` field, so "disease/pest name + a
short snippet" often reduced to just the name with nothing else. **The truncation
itself was not the dominant problem — cutting off ~18% of facts at 64 tokens was
apparently less costly than aggressively shortening all of them to guarantee no
truncation.** The original long-text embeddings, though silently truncated for a
minority of facts, retained more usable signal on average.

**G-2 (caption-then-retrieve) improved R@1 but not R@5**, and has high variance by
disease type: caption-then-retrieve nails visually distinctive symptoms very precisely
(Cherry/Squash Powdery Mildew: retrieval ranks 1-24, near-perfect) but fails badly on
diseases whose visible symptoms don't caption distinctively (Cedar Apple Rust: ranks
662-2172; Tomato Mosaic Virus: ranks 533-2055 -- generic captions like "mottled
discoloration" don't distinguish mosaic virus from many other conditions). Net effect:
better than pure-text or shortened-SigLIP, but still well below hybrid's R@5.

## Conclusion: hybrid (Stage F) remains the best retrieval method found; diminishing returns on further embedding tricks

Across five retrieval variants tested since Stage C, **hybrid α=0.25 text+visual
blending is still the strongest non-oracle retrieval method** on the metric (R@5) that
actually predicts diagnosis accuracy (per Stage D's finding that diagnosis accuracy
tracks R@5, not R@1, since the model reads all top-5 evidence regardless of rank).
However, Stage E's follow-up already showed hybrid's retrieval advantage over
pure-visual does **not** translate into a diagnosis-accuracy advantage (McNemar p=0.81,
see [phase2-hybrid-diagnosis-results.md](phase2-hybrid-diagnosis-results.md)) -- so
neither hybrid nor these two new alternatives currently beat **Stage D's pure-visual RAG
diagnosis result** (16.7% confident_match accuracy, alias-corrected) in practice.

This is a reasonable stopping point for retrieval-representation tinkering: three
distinct approaches (hybrid blending, truncation-safe shortening, caption-bridging) have
now been tried beyond the original SigLIP cross-modal design, and none has produced a
diagnosis-level improvement over Stage D. The binding constraint is not embedding
representation at the margin -- it's the ~35-45 percentage point gap between best
non-oracle retrieval (R@5 ~20-25%) and Oracle's true ceiling (93.8% R@5), which no
representation trick closes; that gap reflects a harder problem (bridging visual
symptom appearance to textual disease description without paired training data) that
would need either paired training data (not available) or a fundamentally different
approach (e.g. fine-tuning, per the literature review's GME/Long-CLIP pointers) rather
than another off-the-shelf embedding swap.
