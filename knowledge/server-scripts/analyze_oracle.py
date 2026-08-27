#!/usr/bin/env python3
import json
import math

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\oracle_retrieval_results.jsonl") as f:
    oracle_rows = {json.loads(l)["case_id"]: json.loads(l) for l in f}

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\phase2_full_v2_results.jsonl") as f:
    main_rows = {json.loads(l)["case_id"]: json.loads(l) for l in f}

PRIMARY_GROUPS = {"primary_confident_match", "primary_healthy", "primary_negative_control"}


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


def oracle_correct(cid):
    r = oracle_rows.get(cid)
    if not r:
        return None
    d = (r.get("diagnosis") or "").strip().lower()
    gt = r["ground_truth_disease"].strip().lower()
    return d == gt


def arm_correct(cid, arm):
    r = main_rows.get(cid)
    if not r:
        return None
    d = (r["arms"][arm].get("diagnosis") or "").strip().lower()
    gt = r["ground_truth_disease"].strip().lower()
    return d == gt


common_ids = [cid for cid in oracle_rows if cid in main_rows and main_rows[cid]["eval_group"] in PRIMARY_GROUPS]
print(f"common cases: {len(common_ids)}")

oracle_c = [oracle_correct(cid) for cid in common_ids]
arm1_c = [arm_correct(cid, "1_vlm_only") for cid in common_ids]
arm2_c = [arm_correct(cid, "2_rag_bge") for cid in common_ids]

k = sum(oracle_c)
p, lo, hi = wilson_ci(k, len(common_ids))
print(f"\nOracle retrieval overall: {k}/{len(common_ids)} = {p:.3f}  95% CI [{lo:.3f},{hi:.3f}]")

print("\nBy group:")
for group in PRIMARY_GROUPS:
    idx = [i for i, cid in enumerate(common_ids) if main_rows[cid]["eval_group"] == group]
    sub = [oracle_c[i] for i in idx]
    k = sum(sub)
    p, lo, hi = wilson_ci(k, len(sub))
    print(f"  {group:28s} {k:3d}/{len(sub)} = {p:.3f}  95% CI [{lo:.3f},{hi:.3f}]")

print("\nMcNemar: Oracle vs Arm1 (VLM only)")
b, c, chi2, pval = mcnemar(oracle_c, arm1_c)
print(f"  discordant(oracle right,arm1 wrong)={b} discordant(arm1 right,oracle wrong)={c} chi2={chi2:.3f} p={pval:.2e}")

print("\nMcNemar: Oracle vs Arm2 (RAG BGE, current pipeline)")
b, c, chi2, pval = mcnemar(oracle_c, arm2_c)
print(f"  discordant(oracle right,arm2 wrong)={b} discordant(arm2 right,oracle wrong)={c} chi2={chi2:.3f} p={pval:.2e}")

print("\nMcNemar: Oracle vs Arm2, confident_match subgroup only")
idx = [i for i, cid in enumerate(common_ids) if main_rows[cid]["eval_group"] == "primary_confident_match"]
o_sub = [oracle_c[i] for i in idx]
a2_sub = [arm2_c[i] for i in idx]
b, c, chi2, pval = mcnemar(o_sub, a2_sub)
print(f"  n={len(idx)} discordant(oracle right,arm2 wrong)={b} discordant(arm2 right,oracle wrong)={c} chi2={chi2:.3f} p={pval:.2e}")
