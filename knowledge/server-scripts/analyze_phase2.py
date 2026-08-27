#!/usr/bin/env python3
"""Full Phase 2 analysis: exact accuracy, McNemar pairwise tests, Wilson CIs, ECE,
faithfulness/groundedness from judge output. Excludes supplementary_agmmu (contaminated
ground truth). Pure Python, no scipy dependency."""
import json
import math
from collections import Counter, defaultdict

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\phase2_full_v2_results.jsonl") as f:
    rows = [json.loads(l) for l in f]

ARMS = ["1_vlm_only", "2_rag_bge", "3_rag_siglip", "4_rag_full_corpus", "5_rag_rerank"]
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}

primary = [r for r in rows if r["eval_group"] in PRIMARY_GROUPS]
weak_match = [r for r in rows if r["eval_group"] == "supplementary_weak_match"]
agmmu = [r for r in rows if r["eval_group"] == "supplementary_agmmu"]
print(f"total: {len(rows)}  primary(valid): {len(primary)}  weak_match(supplementary): {len(weak_match)}  agmmu(EXCLUDED, contaminated): {len(agmmu)}")


def is_correct(r, arm):
    d = (r["arms"][arm].get("diagnosis") or "").strip().lower()
    gt = r["ground_truth_disease"].strip().lower()
    return d == gt


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (p, max(0, center - margin), min(1, center + margin))


def mcnemar(a_correct, b_correct):
    """paired binary: a_correct, b_correct are lists of bool of same length"""
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)  # a right, b wrong
    c = sum(1 for a, bb in zip(a_correct, b_correct) if not a and bb)  # a wrong, b right
    n = b + c
    if n == 0:
        return b, c, None, None
    chi2 = (abs(b - c) - 1) ** 2 / n  # continuity-corrected
    # p-value from chi2 with 1 df, using complementary error function approx
    p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, chi2, p


print("\n=== Overall accuracy per arm (primary, n=%d) ===" % len(primary))
correct_by_arm = {}
for arm in ARMS:
    correct = [is_correct(r, arm) for r in primary]
    correct_by_arm[arm] = correct
    k = sum(correct)
    p, lo, hi = wilson_ci(k, len(primary))
    print(f"{arm:20s} {k:3d}/{len(primary)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== By eval group ===")
for group in ["primary_confident_match", "primary_healthy", "primary_negative_control"]:
    sub = [r for r in primary if r["eval_group"] == group]
    print(f"\n{group} (n={len(sub)}):")
    for arm in ARMS:
        k = sum(1 for r in sub if is_correct(r, arm))
        p, lo, hi = wilson_ci(k, len(sub))
        print(f"  {arm:20s} {k:3d}/{len(sub)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== Supplementary: weak-match corpus coverage (n=%d) ===" % len(weak_match))
for arm in ARMS:
    k = sum(1 for r in weak_match if is_correct(r, arm))
    print(f"  {arm:20s} {k:3d}/{len(weak_match)} = {k/len(weak_match):.3f}")

print("\n=== McNemar's test: Arm 1 (VLM only) vs each RAG arm (primary, n=%d) ===" % len(primary))
for arm in ARMS[1:]:
    b, c, chi2, pval = mcnemar(correct_by_arm["1_vlm_only"], correct_by_arm[arm])
    print(f"1_vlm_only vs {arm:20s} discordant(1right,other wrong)={b:3d} discordant(other right,1 wrong)={c:3d} chi2={chi2}  p={pval}")

print("\n=== McNemar's test: Arm 2 (RAG BGE, main treatment) vs Arm 3/4/5 ===")
for arm in ["3_rag_siglip", "4_rag_full_corpus", "5_rag_rerank"]:
    b, c, chi2, pval = mcnemar(correct_by_arm["2_rag_bge"], correct_by_arm[arm])
    print(f"2_rag_bge vs {arm:20s} b={b:3d} c={c:3d} chi2={chi2}  p={pval}")

print("\n=== ECE (10 bins), primary cases ===")
for arm in ARMS:
    bins = defaultdict(list)
    for r in primary:
        conf = r["arms"][arm].get("confidence")
        if conf is None:
            continue
        b_idx = min(int(conf // 10), 9)
        bins[b_idx].append(is_correct(r, arm))
    ece = 0.0
    n_total = sum(len(v) for v in bins.values())
    for b_idx, vals in bins.items():
        if not vals:
            continue
        bin_conf = (b_idx * 10 + 5) / 100  # bin midpoint as proxy
        bin_acc = sum(vals) / len(vals)
        ece += (len(vals) / n_total) * abs(bin_conf - bin_acc)
    print(f"{arm:20s} ECE={ece:.3f}  (n_with_confidence={n_total})")

print("\n=== Faithfulness / groundedness (arms 2-5, from judge calls, primary cases) ===")
for arm in ARMS[1:]:
    grounded = [r["arms"][arm]["judge"].get("grounded") for r in primary if "judge" in r["arms"][arm]]
    grounded_valid = [g for g in grounded if g is not None]
    entailed_props = []
    for r in primary:
        j = r["arms"][arm].get("judge", {})
        claims = j.get("claims")
        entailed = j.get("entailed")
        if claims and claims > 0 and entailed is not None:
            entailed_props.append(entailed / claims)
    grounded_rate = sum(grounded_valid) / len(grounded_valid) if grounded_valid else None
    mean_faithfulness = sum(entailed_props) / len(entailed_props) if entailed_props else None
    print(f"{arm:20s} grounded_rate={grounded_rate:.3f} (n={len(grounded_valid)})  mean_faithfulness(entailed/claims)={mean_faithfulness:.3f} (n={len(entailed_props)})")

print("\n=== Diagnosis distribution sanity check (primary) ===")
for arm in ARMS:
    diag = Counter((r["arms"][arm].get("diagnosis") or "").strip().lower() for r in primary)
    h = diag.get("healthy", 0)
    u = diag.get("unknown", 0)
    print(f"{arm:20s} healthy={h}/{len(primary)}={h/len(primary):.3f}  unknown={u}/{len(primary)}={u/len(primary):.3f}  unique_diagnoses={len(diag)}")

# save summary
summary = {
    "n_primary": len(primary), "n_weak_match": len(weak_match), "n_agmmu_excluded": len(agmmu),
    "accuracy": {arm: sum(correct_by_arm[arm]) / len(primary) for arm in ARMS},
}
with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\phase2_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved summary json")
