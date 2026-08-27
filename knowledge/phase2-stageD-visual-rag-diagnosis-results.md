# Stage D — Visual RAG Diagnosis: Full Run Results

Full 465-case run on devon, using SigLIP cross-modal retrieval (query image -> SigLIP text
embeddings of the corpus, per Stage C) to select evidence, then Qwen3-VL-8B-Instruct
diagnosis with that evidence injected (same diagnosis prompt/template as the main
ablation's Arm 2). Runtime: 465/465 completed, 0 errors, ~35 minutes on devon's RTX 4090.
Script: [server-scripts/run_visual_rag_diagnosis.py](server-scripts/run_visual_rag_diagnosis.py).
Analysis: [server-scripts/analyze_stageD.py](server-scripts/analyze_stageD.py). Raw data:
[results/phase2/visual_rag_diagnosis_results.jsonl](results/phase2/visual_rag_diagnosis_results.jsonl).

This is the first experiment since Oracle Retrieval to show a **statistically significant,
practically real improvement over both baselines** — not just at the retrieval level
(Stage C already showed that), but where it actually matters: final diagnosis accuracy.

## Headline numbers (n=465, same primary eval set used throughout Phase 2)

| Arm | Accuracy | 95% CI |
|---|--:|--:|
| VLM-only (Arm 1, main ablation) | 37.8% | [33.6%, 42.3%] |
| RAG, BGE text query (Arm 2, current pipeline) | 39.4% | [35.0%, 43.9%] |
| **Visual RAG (Stage D)** | **44.5%** | **[40.1%, 49.1%]** |
| Oracle retrieval (Experiment 1, ceiling) | 75.9% | — |

## On the subgroup that actually tests diagnostic ability (primary_confident_match, n=240)

This is the group where ground truth is a specific disease actually present in the corpus
— the only group where "RAG helping" is even a coherent question (primary_healthy and
primary_negative_control are near-ceiling/near-floor regardless of retrieval, as they've
been throughout Phase 2).

| Arm | Accuracy | 95% CI |
|---|--:|--:|
| VLM-only | 0.4% (1/240) | [0.1%, 2.3%] |
| RAG, BGE text query | 3.7% (9/240) | [2.0%, 7.0%] |
| **Visual RAG (Stage D)** | **15.4% (37/240)** | **[11.4%, 20.5%]** |
| Oracle retrieval | 73.3% | — |

## Statistical significance (McNemar's test, paired, continuity-corrected)

| Comparison | Discordant pairs (D right/other wrong vs other right/D wrong) | p-value |
|---|---|--:|
| Stage D vs VLM-only, overall (n=465) | 38 vs 7 | 7.7×10⁻⁶ |
| Stage D vs RAG-BGE, overall (n=465) | 31 vs 7 | 1.9×10⁻⁴ |
| Stage D vs VLM-only, confident_match (n=240) | 37 vs 1 | 1.4×10⁻⁸ |
| Stage D vs RAG-BGE, confident_match (n=240) | 28 vs 0 | 3.4×10⁻⁷ |

On confident_match, Stage D corrected 37 cases that VLM-only got wrong while breaking only
1; against RAG-BGE it corrected 28 while breaking 0. This is a clean, one-sided win, not a
noisy trade — consistent with the main ablation's null result being a retrieval-query
problem specifically, not evidence that RAG can't help.

## Groups that didn't move (as expected)

| Group | VLM-only | RAG-BGE | Stage D |
|---|--:|--:|--:|
| primary_healthy (n=180) | 97.2% | 96.7% | 94.4% |
| primary_negative_control (n=45) | 0.0% | 0.0% | 0.0% |

Healthy cases dipped very slightly (97.2%→94.4%) — plausibly a little evidence-injection
noise nudging a few borderline "healthy" calls toward an unnecessary diagnosis, within
overlapping CIs. Negative control stayed at floor across all arms, as designed (these
cases have no correct answer in the corpus by construction).

## Faithfulness / groundedness (RAG-Triad judge, same methodology as main ablation)

grounded_rate = 60.0%, mean_faithfulness (entailed/claims) = 39.8% (n=465). Both lower than
would be ideal, but not the focus here — Stage D's purpose was fixing the retrieval
bottleneck the error taxonomy identified, and it clearly did that.

## Interpretation

This closes the loop the Phase 2 Extension plan opened:

1. Main ablation found no significant RAG benefit.
2. Oracle Retrieval proved RAG helps enormously *if* retrieval succeeds (p≈10⁻³⁸).
3. Error taxonomy found 74.6% of failures were query-construction failures, not corpus or
   reasoning failures.
4. Stage C showed SigLIP cross-modal (image-as-query) retrieval beats Qwen-text-query
   retrieval by ~2.5x on R@1 (3.3% vs 1.3%).
5. **Stage D now confirms that retrieval improvement survives into diagnosis accuracy**:
   swapping only the retrieval query mechanism (text query -> image-based cross-modal
   query), with everything else about the pipeline unchanged, takes confident_match
   accuracy from 3.7% to 15.4% — a 4.2x improvement, p=3.4×10⁻⁷.

The gap to Oracle (15.4% vs 73.3%) is still large, meaning retrieval quality is still the
binding constraint — R@1 on confident_match is only 3.3% even with visual retrieval (Stage
C), so most cases still don't get the right fact at rank 1. But this experiment
demonstrates the mechanism works end-to-end and the direction of further investment
(better retrieval, not better prompting/reasoning) is correct.

## Where this leaves Stage E (decision point) and Stage F/G

Per the plan's Stage E decision point: visual RAG **did** substantially improve diagnosis
accuracy (statistically significant, one-sided, large effect on the group that matters).
That justifies extending Stage F's retrieval-only hybrid result (α=0.25: R@1 5.8% vs
Stage C's 3.3% pure-visual, see
[phase2-stageF-hybrid-retrieval-results.md](phase2-stageF-hybrid-retrieval-results.md))
into an actual hybrid-retrieval diagnosis run, the same way Stage C's retrieval result was
extended into this Stage D diagnosis run. Given the Stage D pattern (retrieval R@1 3.3% ->
diagnosis accuracy 15.4%, roughly 4.7x), hybrid's higher R@1 (5.8%) suggests a further,
though likely more modest incremental, diagnosis improvement — worth confirming with an
actual run rather than assuming.

Stage G (reranking) remains deferred per the plan, pending Recall@50 analysis to confirm
there's enough headroom in the candidate pool to make reranking worthwhile.
