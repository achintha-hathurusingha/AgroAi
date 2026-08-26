# Phase 1 — Q6: NumPy vs FAISS vs Qdrant Benchmark Results

Real latency benchmark at real corpus scale, using BGE-micro-v2 (the Q1 winner) to
embed retrieval text. Qdrant is a genuine standalone-binary server process (downloaded
directly from GitHub releases, no Docker/root needed — devon has no Docker installed and
no sudo access was used), not an embedded/in-memory shortcut. Raw data:
[results/q6_benchmark_results.json](results/q6_benchmark_results.json). Script:
[server-scripts/q6_vector_store_benchmark.py](server-scripts/q6_vector_store_benchmark.py).

## Two corpus scales tested

- **Diagnostic corpus**: 17,583 deduped facts (matches Phase 1A's figure).
- **Full corpus (diagnostic + context, deduped together)**: 35,507 unique facts — notably,
  deduping the *combined* pool collapses 45,098 raw normalized facts by 21.3%, higher than
  the diagnostic-only dedup rate (13.7% from Phase 1A). The context corpus (species-only /
  partial entries) apparently has even more internal duplication than the diagnostic one —
  e.g. many entries that just say "Species: grass" and nothing else collide trivially.

## Results

| Backend | Diagnostic (17,583) build | Diagnostic query mean / p95 | Full (35,507) build | Full query mean / p95 |
|---|--:|--:|--:|--:|
| **NumPy** (brute-force cosine) | **0.0ms** | **0.343ms / 0.176ms** | **0.0ms** | **0.390ms / 0.436ms** |
| FAISS (exact flat index) | 13.4ms | 0.724ms / 0.955ms | 28.3ms | 2.060ms / 11.753ms |
| Qdrant (real server, HTTP) | 4,471ms | 2.945ms / 4.944ms | 8,817ms | 3.648ms / 5.755ms |

(One measurement quirk worth flagging honestly: at diagnostic scale, NumPy's p95 came in
*below* its mean — an artifact of only 50 query trials with one slow early outlier
skewing the mean upward, not a real distributional inversion. Not worth over-reading;
the qualitative ranking is unambiguous regardless.)

## The result is unambiguous at this scale

**NumPy brute-force cosine similarity is the fastest option on every measure, at both
scales tested** — faster to "build" (trivially, it's just holding a matrix) and faster
per query than both FAISS and Qdrant. FAISS's usual advantage (index structures for very
large corpora, or approximate search) doesn't materialize here — an exact flat FAISS
index has more per-query call overhead than a raw BLAS matrix multiply for a corpus this
size, so it's consistently ~2–5x slower than NumPy rather than faster. Qdrant's latency
is dominated by network/server round-trip overhead (even to localhost) and its build
time is ~4.5–8.8 seconds vs. NumPy's effectively-zero, since every point upload is a real
HTTP request to a separate process.

**In absolute terms, none of the three would meaningfully bottleneck the real pipeline.**
Even Qdrant's ~3–4ms query latency is negligible next to a Qwen3-VL generation call
(which will take on the order of seconds) — so this isn't a case where the "slow" option
is actually too slow to use, it's genuinely fast in every case, just relatively slower.

## What this means for the Q6 decision

The quantitative case for NumPy/FAISS over Qdrant *purely on speed* is clear at this
scale — Qdrant's complexity does not buy latency here. The decision therefore comes down
to the qualitative tradeoffs Qdrant *does* offer that a raw NumPy matrix doesn't, weighed
against real setup/operational cost now measured, not assumed:

**What Qdrant provides that NumPy/FAISS don't:**
- Persistence across process restarts (a NumPy array or FAISS index lives only in the
  Python process's memory — losing it on every restart unless you build your own
  save/load layer).
- Metadata filtering integrated into the search itself (e.g. restrict a search to a
  given species without a separate filtering pass).
- A separate server process, so multiple workers/processes could share one index without
  duplicating memory or re-embedding.

**Real, now-measured cost of that:**
- ~4.5–8.8 second index build time (vs. instant) at this data scale.
- 2.945–3.648ms per query (vs. 0.343–0.390ms) — still fast in absolute terms, but a real
  ~10x latency multiple.
- An extra running process to manage (currently running on devon at `localhost:6333`,
  PID recorded in `~/agrivision-rag/qdrant-bin/qdrant.pid` — left running for Milestone 3
  in case it's the chosen backend; can be stopped with `kill $(cat qdrant.pid)` if not
  needed).

This experiment answers "does Qdrant's complexity buy speed at this scale" — no, clearly
not. Whether persistence/filtering/multi-process access are worth that cost for Phase 1's
actual needs (a single-process retrieval step feeding into the ablation study, not a
concurrent multi-user service yet) is the decision still left to you.
