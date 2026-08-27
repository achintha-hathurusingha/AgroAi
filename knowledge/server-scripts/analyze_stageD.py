#!/usr/bin/env python3
"""Stage D analysis: full visual RAG diagnosis (SigLIP cross-modal retrieval -> Qwen3-VL
diagnosis) accuracy, with Wilson CIs and McNemar's tests against VLM-only (Arm 1) and
current-pipeline RAG (Arm 2 BGE-text) from the main ablation, using the same 465-case
primary eval set. Pure Python, no scipy dependency (matches analyze_phase2.py methodology)."""
import json
import math
from collections import Counter

RESULTS_DIR = r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2"

with open(f"{RESULTS_DIR}\\visual_rag_diagnosis_results.jsonl") as f:
    stageD = {json.loads(l)["case_id"]: json.loads(l) for l in open(f"{RESULTS_DIR}\\visual_rag_diagnosis_results.jsonl")}

with open(f"{RESULTS_DIR}\\phase2_full_v2_results.jsonl") as f:
    main_rows = {json.loads(l)["case_id"]: json.loads(l) for l in f}

PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}
common_ids = [cid for cid in stageD if cid in main_rows and stageD[cid]["eval_group"] in PRIMARY_GROUPS]
print(f"Stage D cases: {len(stageD)}  matched to main ablation: {len(common_ids)}")


def is_correct_D(cid):
    d = (stageD[cid].get("diagnosis") or "").strip().lower()
    gt = stageD[cid]["ground_truth_disease"].strip().lower()
    return d == gt


def is_correct_main(cid, arm):
    d = (main_rows[cid]["arms"][arm].get("diagnosis") or "").strip().lower()
    gt = main_rows[cid]["ground_truth_disease"].strip().lower()
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
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct) if not a and bb)
    n = b + c
    if n == 0:
        return b, c, None, None
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, chi2, p


ARMS_MAIN = {"1_vlm_only": "VLM-only", "2_rag_bge": "RAG (BGE text, current pipeline)"}

print(f"\n=== Overall accuracy, Stage D (visual RAG) vs main-ablation arms (n={len(common_ids)}) ===")
correctD = [is_correct_D(c) for c in common_ids]
k = sum(correctD)
p, lo, hi = wilson_ci(k, len(common_ids))
print(f"{'D_visual_rag':30s} {k:3d}/{len(common_ids)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
correct_main = {}
for arm, label in ARMS_MAIN.items():
    correct = [is_correct_main(c, arm) for c in common_ids]
    correct_main[arm] = correct
    k = sum(correct)
    p, lo, hi = wilson_ci(k, len(common_ids))
    print(f"{arm:30s} {k:3d}/{len(common_ids)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  ({label})")

print(f"\n=== By eval group ===")
for group in ["primary_confident_match", "primary_healthy", "primary_negative_control"]:
    sub = [c for c in common_ids if stageD[c]["eval_group"] == group]
    print(f"\n{group} (n={len(sub)}):")
    k = sum(is_correct_D(c) for c in sub)
    p, lo, hi = wilson_ci(k, len(sub))
    print(f"  {'D_visual_rag':30s} {k:3d}/{len(sub)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    for arm, label in ARMS_MAIN.items():
        k = sum(is_correct_main(c, arm) for c in sub)
        p, lo, hi = wilson_ci(k, len(sub))
        print(f"  {arm:30s} {k:3d}/{len(sub)} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  ({label})")

print(f"\n=== McNemar's test: Stage D (visual RAG) vs VLM-only / RAG-BGE (n={len(common_ids)}) ===")
for arm, label in ARMS_MAIN.items():
    b, c, chi2, pval = mcnemar(correctD, correct_main[arm])
    print(f"D vs {arm:20s} D_right_other_wrong={b:3d} other_right_D_wrong={c:3d} chi2={chi2}  p={pval}  ({label})")

print(f"\n=== McNemar's test on confident_match subgroup only (n=240, the diagnosable subset) ===")
sub = [c for c in common_ids if stageD[c]["eval_group"] == "primary_confident_match"]
correctD_sub = [is_correct_D(c) for c in sub]
for arm, label in ARMS_MAIN.items():
    correct_sub = [is_correct_main(c, arm) for c in sub]
    b, cc, chi2, pval = mcnemar(correctD_sub, correct_sub)
    print(f"D vs {arm:20s} D_right_other_wrong={b:3d} other_right_D_wrong={cc:3d} chi2={chi2}  p={pval}  ({label})")

print(f"\n=== Diagnosis distribution sanity check, Stage D ===")
diag = Counter((stageD[c].get("diagnosis") or "").strip().lower() for c in common_ids)
h = diag.get("healthy", 0)
u = diag.get("unknown", 0)
print(f"healthy={h}/{len(common_ids)}={h/len(common_ids):.3f}  unknown={u}/{len(common_ids)}={u/len(common_ids):.3f}  unique_diagnoses={len(diag)}")

print(f"\n=== Faithfulness / groundedness, Stage D (from judge output) ===")
grounded = [stageD[c]["judge"].get("grounded") for c in common_ids if "judge" in stageD[c]]
grounded_valid = [g for g in grounded if g is not None]
entailed_props = []
for c in common_ids:
    j = stageD[c].get("judge", {})
    claims = j.get("claims")
    entailed = j.get("entailed")
    if claims and claims > 0 and entailed is not None:
        entailed_props.append(entailed / claims)
grounded_rate = sum(grounded_valid) / len(grounded_valid) if grounded_valid else None
mean_faithfulness = sum(entailed_props) / len(entailed_props) if entailed_props else None
print(f"grounded_rate={grounded_rate:.3f} (n={len(grounded_valid)})  mean_faithfulness={mean_faithfulness:.3f} (n={len(entailed_props)})")

summary = {
    "n": len(common_ids),
    "accuracy_stageD": sum(correctD) / len(common_ids),
    "accuracy_vlm_only": sum(correct_main["1_vlm_only"]) / len(common_ids),
    "accuracy_rag_bge": sum(correct_main["2_rag_bge"]) / len(common_ids),
}
with open(f"{RESULTS_DIR}\\stageD_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved summary json")
