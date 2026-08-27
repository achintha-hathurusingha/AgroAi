#!/usr/bin/env python3
"""
Rescore diagnosis accuracy for the main ablation, Oracle Retrieval, and Stage D
(visual RAG diagnosis) using the corrected alias-aware matching (scoring.py), fixing
the PV-label-vs-corpus-vocabulary mismatch bug. Pure recompute on already-collected
data -- no GPU, no new experiments. Reports old (strict) vs new (alias) side by side.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import is_correct_aliased, ground_truth_aliases

RESULTS_DIR = Path(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2")
PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (p, max(0, center - margin), min(1, center + margin))


def strict_correct(diagnosis, gt):
    return (diagnosis or "").strip().lower() == gt.strip().lower()


def report(label, rows, get_diag):
    primary = [r for r in rows if r["eval_group"] in PRIMARY_GROUPS]
    print(f"\n{'='*70}\n{label}  (n={len(primary)})\n{'='*70}")
    old_k = sum(1 for r in primary if strict_correct(get_diag(r), r["ground_truth_disease"]))
    new_k = sum(1 for r in primary if is_correct_aliased(get_diag(r), r["ground_truth_disease"]))
    op, olo, ohi = wilson_ci(old_k, len(primary))
    np_, nlo, nhi = wilson_ci(new_k, len(primary))
    print(f"  overall   strict={old_k:3d}/{len(primary)}={op:.3f} [{olo:.3f},{ohi:.3f}]   "
          f"alias={new_k:3d}/{len(primary)}={np_:.3f} [{nlo:.3f},{nhi:.3f}]   delta={np_-op:+.3f}")

    for group in ["primary_confident_match", "primary_healthy", "primary_negative_control"]:
        sub = [r for r in primary if r["eval_group"] == group]
        if not sub:
            continue
        ok = sum(1 for r in sub if strict_correct(get_diag(r), r["ground_truth_disease"]))
        nk = sum(1 for r in sub if is_correct_aliased(get_diag(r), r["ground_truth_disease"]))
        op, olo, ohi = wilson_ci(ok, len(sub))
        np_, nlo, nhi = wilson_ci(nk, len(sub))
        print(f"  {group:26s} strict={ok:3d}/{len(sub)}={op:.3f}   alias={nk:3d}/{len(sub)}={np_:.3f}   delta={np_-op:+.3f}")

    rescued = [r for r in primary if not strict_correct(get_diag(r), r["ground_truth_disease"])
               and is_correct_aliased(get_diag(r), r["ground_truth_disease"])]
    print(f"  cases rescued by alias fix: {len(rescued)}")
    return {"strict": old_k / len(primary), "alias": new_k / len(primary), "n": len(primary), "rescued": len(rescued)}


summary = {}

print("Loading main ablation...")
with open(RESULTS_DIR / "phase2_full_v2_results.jsonl") as f:
    main_rows = [json.loads(l) for l in f]
main_rows = [r for r in main_rows if r["eval_group"] in PRIMARY_GROUPS]  # excludes contaminated supplementary_agmmu
for arm in ["1_vlm_only", "2_rag_bge", "3_rag_siglip", "4_rag_full_corpus", "5_rag_rerank"]:
    summary[f"main_ablation_{arm}"] = report(
        f"Main ablation -- {arm}", main_rows, lambda r, a=arm: r["arms"][a].get("diagnosis"))

print("\nLoading Oracle Retrieval...")
with open(RESULTS_DIR / "oracle_retrieval_results.jsonl") as f:
    oracle_rows = [json.loads(l) for l in f]
summary["oracle"] = report("Oracle Retrieval", oracle_rows, lambda r: r.get("diagnosis"))

print("\nLoading Stage D (visual RAG diagnosis)...")
with open(RESULTS_DIR / "visual_rag_diagnosis_results.jsonl") as f:
    stageD_rows = [json.loads(l) for l in f]
summary["stageD_visual_rag"] = report("Stage D -- Visual RAG diagnosis", stageD_rows, lambda r: r.get("diagnosis"))

with open(RESULTS_DIR / "rescore_diagnosis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved rescore_diagnosis_summary.json")
