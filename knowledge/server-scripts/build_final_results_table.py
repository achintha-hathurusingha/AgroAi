#!/usr/bin/env python3
"""
Consolidation script (thesis writeup pass). Recomputes every number in the final
authoritative results table directly from the raw saved experiment outputs already in
the repo, using the corrected alias-aware scoring (scoring.py). Produces
results/phase2/final_results_table.json as the single reproducible source backing
final-research-findings.md's tables. No new model calls, no new experiments -- pure
recomputation, so this script is the reproducibility check for the consolidation.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import is_correct_aliased

RESULTS_DIR = Path(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2")
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (p, max(0, center - margin), min(1, center + margin))


def mcnemar(a_correct, b_correct):
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct) if not a and bb)
    n = b + c
    if n == 0:
        return b, c, None, None
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, chi2, p


def accuracy_block(rows, get_diag, get_gt, get_group):
    primary = [r for r in rows if get_group(r) in PRIMARY_GROUPS]
    out = {}
    correct_all = [is_correct_aliased(get_diag(r), get_gt(r)) for r in primary]
    k = sum(correct_all)
    p, lo, hi = wilson_ci(k, len(primary))
    out["overall"] = {"k": k, "n": len(primary), "accuracy": p, "ci95": [lo, hi]}
    for group in ["primary_confident_match", "primary_healthy", "primary_negative_control"]:
        sub = [r for r in primary if get_group(r) == group]
        if not sub:
            continue
        kk = sum(1 for r in sub if is_correct_aliased(get_diag(r), get_gt(r)))
        p, lo, hi = wilson_ci(kk, len(sub))
        out[group] = {"k": kk, "n": len(sub), "accuracy": p, "ci95": [lo, hi]}
    out["_correct_flags_confident_match"] = {
        r["case_id"]: is_correct_aliased(get_diag(r), get_gt(r))
        for r in primary if get_group(r) == "primary_confident_match"
    }
    return out


print("loading raw experiment outputs...")
with open(RESULTS_DIR / "phase2_full_v2_results.jsonl") as f:
    main_rows = [json.loads(l) for l in f]
main_rows = [r for r in main_rows if r["eval_group"] in PRIMARY_GROUPS]

with open(RESULTS_DIR / "oracle_retrieval_results.jsonl") as f:
    oracle_rows = [json.loads(l) for l in f]

with open(RESULTS_DIR / "visual_rag_diagnosis_results.jsonl") as f:
    stageD_rows = [json.loads(l) for l in f]

with open(RESULTS_DIR / "hybrid_rag_diagnosis_results.jsonl") as f:
    hybrid_rows = [json.loads(l) for l in f]

results = {}

print("VLM-only (arm 1)...")
results["vlm_only"] = accuracy_block(
    main_rows, lambda r: r["arms"]["1_vlm_only"].get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("Text RAG / BGE (arm 2)...")
results["text_rag_bge"] = accuracy_block(
    main_rows, lambda r: r["arms"]["2_rag_bge"].get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("SigLIP-text RAG (arm 3, text-mediated, NOT cross-modal visual-first)...")
results["siglip_text_rag_arm3"] = accuracy_block(
    main_rows, lambda r: r["arms"]["3_rag_siglip"].get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("RAG full corpus (arm 4)...")
results["rag_full_corpus_arm4"] = accuracy_block(
    main_rows, lambda r: r["arms"]["4_rag_full_corpus"].get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("RAG + rerank (arm 5)...")
results["rag_rerank_arm5"] = accuracy_block(
    main_rows, lambda r: r["arms"]["5_rag_rerank"].get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("Visual-first RAG diagnosis (Stage D, SigLIP cross-modal image-tower retrieval)...")
results["visual_rag_stageD"] = accuracy_block(
    stageD_rows, lambda r: r.get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("Hybrid RAG diagnosis (Stage E follow-up, alpha=0.25 text+visual)...")
results["hybrid_rag"] = accuracy_block(
    hybrid_rows, lambda r: r.get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("Oracle RAG (ceiling)...")
results["oracle_rag"] = accuracy_block(
    oracle_rows, lambda r: r.get("diagnosis"),
    lambda r: r["ground_truth_disease"], lambda r: r["eval_group"])

print("\ncomputing McNemar's tests on primary_confident_match (n=240)...")
flags = {name: results[name].pop("_correct_flags_confident_match") for name in results}
cm_ids = list(flags["vlm_only"].keys())
for name in results:
    assert set(flags[name].keys()) == set(cm_ids), f"case-id mismatch for {name}"

pairs = [
    ("text_rag_bge", "vlm_only"),
    ("visual_rag_stageD", "vlm_only"),
    ("visual_rag_stageD", "text_rag_bge"),
    ("hybrid_rag", "visual_rag_stageD"),
    ("hybrid_rag", "text_rag_bge"),
    ("oracle_rag", "vlm_only"),
    ("oracle_rag", "text_rag_bge"),
]
mcnemar_results = {}
for a, b in pairs:
    a_flags = [flags[a][c] for c in cm_ids]
    b_flags = [flags[b][c] for c in cm_ids]
    disc_a, disc_b, chi2, p = mcnemar(a_flags, b_flags)
    key = f"{a}_vs_{b}"
    mcnemar_results[key] = {
        f"{a}_right_{b}_wrong": disc_a, f"{b}_right_{a}_wrong": disc_b,
        "chi2": chi2, "p_value": p, "n": len(cm_ids),
    }
    print(f"  {key:45s} {a}={disc_a:3d} {b}={disc_b:3d}  p={p}")

results["mcnemar_confident_match"] = mcnemar_results

print("\nloading retrieval-side numbers (from Stage F/G qbits recompute)...")
with open(RESULTS_DIR / "rescore_retrieval_report.json") as f:
    retrieval_report = json.load(f)
results["retrieval_recall"] = {
    "source": "rescore_retrieval_report.json (alias-corrected, re-embedded on qbits from saved query texts)",
    "confident_match": retrieval_report["by_group_confident_match"],
}

with open(RESULTS_DIR / "rescore_retrieval_report.json") as f:
    tax = json.load(f)["error_taxonomy_recount"]
results["error_taxonomy_confident_match_n240"] = tax

out_path = RESULTS_DIR / "final_results_table.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out_path}")

print("\n=== SUMMARY: primary_confident_match accuracy (n=240), alias-corrected ===")
for name in ["vlm_only", "text_rag_bge", "siglip_text_rag_arm3", "rag_full_corpus_arm4",
             "rag_rerank_arm5", "visual_rag_stageD", "hybrid_rag", "oracle_rag"]:
    b = results[name]["primary_confident_match"]
    print(f"  {name:22s} {b['k']:3d}/{b['n']} = {b['accuracy']:.3f}  95% CI [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
