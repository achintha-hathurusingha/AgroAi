# Phase 1B.5 — Bottleneck Diagnosis: Qwen vs Human vs Hybrid vs Oracle-Species

4 hard/representative cases, 4 query conditions each, BGE-micro-v2 against the full
17,583-fact corpus. No new Qwen3-VL calls — all text is either the actual cached Prompt C
output, the actual cached hand-authored benchmark text, or a documented one-line edit of
the Prompt C output. Full text of every query used is in
[results/q2_bottleneck_diagnosis.json](results/q2_bottleneck_diagnosis.json). Script:
[server-scripts/q2_bottleneck_diagnosis.py](server-scripts/q2_bottleneck_diagnosis.py).

## Conditions

- **qwen**: actual cached Prompt C output (the pipeline's real current query).
- **human**: actual cached hand-authored species+symptoms text from the original benchmark.
- **hybrid**: qwen's text with exactly one line appended — the disease's known
  discriminative feature (e.g. cedar apple rust's orange tube structures), drawn from the
  same hand-authored descriptions already reviewed earlier in the project, not newly invented.
- **oracle_species**: qwen's text with `Species: unknown` replaced by the correct species
  name, symptom content otherwise unchanged.

## Result

| Case | qwen | human | hybrid | oracle_species |
|---|--:|--:|--:|--:|
| Cedar apple rust | 1,755 | **3** | 1,689 | 43 |
| Early blight | 228 | **1** | 214 | 5 |
| Bacterial leaf spot | 16 | 65 | 18 | 93 |
| Powdery mildew | 40 | **1** | 12 | 58 |

## Finding 1: the corpus/embedding representation is not fundamentally broken

The **human** condition reaches rank 1–3 for three of four cases (cedar apple rust,
early blight, powdery mildew) — a dramatic ceiling compared to the qwen baseline (1,755;
228; 40 respectively). This rules out "the retrieval representation itself can't
discriminate this disease" as the primary explanation for those three cases: given a
good textual query, the system finds the right answer easily. The bottleneck for those
cases is squarely what Qwen's actual generated text contains, not the corpus or embedder.

## Finding 2: simply adding the missing feature (hybrid) barely helps

This is the most surprising and important result. Appending exactly one line naming the
disease's known discriminative feature to Qwen's existing generic description moved the
rank almost nothing — cedar apple rust 1,755→1,689, early blight 228→214, bacterial spot
16→18 (slightly worse), powdery mildew 40→12 (the one real improvement, still far short
of the human condition's rank 1). **A correct, specific fact buried inside several lines
of generic surrounding text ("leaf is green with irregular brown spots... veins visible...
margin serrated") doesn't shift BGE's embedding enough to meaningfully change the
ranking.** This is a finding about how the embedding aggregates information, not just
about what Qwen fails to mention — it suggests the *overall phrasing/gestalt* of the
query matters as much as whether a key fact is present at all, which "just tell Qwen to
also mention X" prompt strategies wouldn't fix on their own.

## Finding 3: species context is a powerful but double-edged retrieval signal

Substituting the correct species name produced the largest single effect in this whole
experiment — but in **both directions**:
- Cedar apple rust: 1,755 → **43** (40x improvement)
- Early blight: 228 → **5** (45x improvement)
- Bacterial leaf spot: 16 → **93** (6x *worse*)
- Powdery mildew: 40 → **58** (slightly worse)

Reading the actual top-1 results explains why it sometimes hurts: for bacterial leaf spot,
providing "peach" as species pulled in *other* peach diseases ("possibly peach leaf curl
or brown rot") as stronger matches than the correct one. For powdery mildew, providing
"cherry" pulled in "cherry leaf spot" instead. **Species context helps when the correct
disease is textually distinctive within that species' disease-space in the corpus, but
actively hurts when a different, more textually-similar disease shares the same species.**
This directly confirms and sharpens the hypothesis raised before running this experiment
(species may be acting as a retrieval anchor) — but the effect isn't uniformly positive,
so "always tell the retrieval step the species" isn't a safe fix on its own either.

## What this means, without deciding the next step

- The retrieval/embedding system works when given good input — this is not primarily an
  "the corpus doesn't encode the right information" problem.
- The bottleneck is upstream, in what Qwen3-VL's description actually contains and how
  it's phrased — but the fix isn't as simple as "add the missing feature" (Finding 2
  shows that alone doesn't work) or "always add species" (Finding 3 shows that can
  backfire depending on the disease).
- A viable direction this data points toward, not yet tested: species context conditioned
  on some notion of confidence or corpus-awareness (only supply species when it would
  help disambiguate rather than mislead) — the architecture idea raised before running
  this experiment, now with concrete supporting evidence rather than just a hypothesis.
- This remains a 4-case diagnostic, not a validated general rule — the same caution that
  applied to the earlier 7-image isolation experiment (which didn't generalize to full
  scale) applies here too before treating any of these three findings as settled.
